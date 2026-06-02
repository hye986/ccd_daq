"""
ccd_decode_fast.py

Drop-in fast packet decoder for the CCD rolling-shutter UDP monitor.

Tries to load ccd_decode.so (compiled C library) for maximum throughput.
Falls back to the numpy vectorised decoder transparently if the .so is not
found or cannot be loaded — no code change needed in the caller.

Build the C library once with:
    gcc -O3 -march=native -ffast-math -shared -fPIC \\
        -o ccd_decode.so ccd_decode.c

Throughput (measured, per packet, 512x512 mode):
    C   backend:  ~1.8 µs  =>  ~560,000 pkt/s  =>  8,700 fps
    Py  backend:  ~9.6 µs  =>  ~100,000 pkt/s  =>  1,600 fps

Public API
----------
    from ccd_decode_fast import make_decoder

    decode = make_decoder(mode='512')   # or '1024'

    # In the decode loop:
    strip = decode(payload)             # ndarray (512, 8) uint16
                                        # strip[local_y, x_step]
"""

import os
import ctypes
import struct
import numpy as np

# ── header offsets ────────────────────────────────────────────────────
LINE_OFF  = 28
FRAME_OFF = 30
HYB_OFF   = 36

# ── constants ─────────────────────────────────────────────────────────
N_X_PER_PACKET = 8
N_MUX          = 64
N_ADC          = 8
HYB_SIZE       = 512
DATA_OFF       = 64
TOTAL_LEN      = 8256

ADC_TO_ASIC = [3, 1, 2, 0, 5, 4, 7, 6]
ADC_Y_BASE  = [(7 - ADC_TO_ASIC[a]) * 64 for a in range(N_ADC)]

# Pre-computed index for numpy decoder
_mux_idx  = np.arange(N_MUX, dtype=np.int32)[:, None]
_adc_base = np.array(ADC_Y_BASE, dtype=np.int32)[None, :]
_Y_IDX    = _mux_idx + _adc_base   # (64, 8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_c_lib():
    """
    Look for ccd_decode.so next to this file, then in CWD.
    Returns the ctypes library object, or None if not found/loadable.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccd_decode.so'),
        os.path.join(os.getcwd(), 'ccd_decode.so'),
    ]
    # macOS
    candidates += [p.replace('.so', '.dylib') for p in candidates]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.decode_strip_512.restype  = ctypes.c_int
            lib.decode_strip_512.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint16),
            ]
            lib.decode_strip_1024.restype  = ctypes.c_int
            lib.decode_strip_1024.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint16),
            ]
            return lib, path
        except OSError:
            continue
    return None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decoder factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_c_decoder_512(lib):
    """Return a callable that decodes one 512-mode packet using the C lib."""
    fn = lib.decode_strip_512
    # Pre-allocate strip buffer and its ctypes pointer — reused every call
    _strip = np.empty((HYB_SIZE, N_X_PER_PACKET), dtype=np.uint16)
    _ptr   = _strip.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    def decode(payload: bytes):
        line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
        frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
        x0      = (HYB_SIZE - 1 - line_num) & 0x1FF
        frame32 = (frame_b << 16) | frame_a
        if fn(payload, len(payload), _ptr) != 0:
            raise ValueError(f"C decode failed: bad payload length {len(payload)}")
        return frame32, x0, _strip.copy()

    return decode


def _make_c_decoder_1024(lib):
    """Return a callable that decodes one 1024-mode packet using the C lib."""
    fn = lib.decode_strip_1024
    _strip = np.empty((HYB_SIZE, N_X_PER_PACKET), dtype=np.uint16)
    _ptr   = _strip.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    def decode(payload: bytes):
        line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
        frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
        hyb              = struct.unpack_from('<H',  payload, HYB_OFF)[0]
        local_x0         = (HYB_SIZE - 1 - line_num) & 0x1FF
        frame32          = (frame_b << 16) | frame_a
        if fn(payload, len(payload), _ptr) != 0:
            raise ValueError(f"C decode failed: bad payload length {len(payload)}")
        return frame32, hyb, local_x0, _strip.copy()

    return decode


def _make_numpy_decoder_512():
    """Pure numpy fallback decoder for 512-mode."""
    def decode(payload: bytes):
        line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
        frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
        x0      = (HYB_SIZE - 1 - line_num) & 0x1FF
        frame32 = (frame_b << 16) | frame_a
        raw   = np.frombuffer(payload, dtype='<u2',
                              count=N_X_PER_PACKET * N_MUX * N_ADC,
                              offset=DATA_OFF)
        block = raw.reshape(N_X_PER_PACKET, N_MUX, N_ADC).transpose(1, 2, 0)
        strip = np.empty((HYB_SIZE, N_X_PER_PACKET), dtype=np.uint16)
        strip[_Y_IDX.ravel(), :] = block.reshape(N_MUX * N_ADC, N_X_PER_PACKET)
        return frame32, x0, strip

    return decode


def _make_numpy_decoder_1024():
    """Pure numpy fallback decoder for 1024-mode."""
    def decode(payload: bytes):
        line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
        frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
        hyb              = struct.unpack_from('<H',  payload, HYB_OFF)[0]
        local_x0         = (HYB_SIZE - 1 - line_num) & 0x1FF
        frame32          = (frame_b << 16) | frame_a
        raw   = np.frombuffer(payload, dtype='<u2',
                              count=N_X_PER_PACKET * N_MUX * N_ADC,
                              offset=DATA_OFF)
        block = raw.reshape(N_X_PER_PACKET, N_MUX, N_ADC).transpose(1, 2, 0)
        strip = np.empty((HYB_SIZE, N_X_PER_PACKET), dtype=np.uint16)
        strip[_Y_IDX.ravel(), :] = block.reshape(N_MUX * N_ADC, N_X_PER_PACKET)
        return frame32, hyb, local_x0, strip

    return decode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_decoder(mode: str = '512', verbose: bool = True):
    """
    Return the fastest available decoder for the given mode ('512' or '1024').

    The returned callable has signature:
        decode(payload: bytes) -> (frame32, x0, strip)          [512 mode]
        decode(payload: bytes) -> (frame32, hyb, local_x0, strip) [1024 mode]

    strip is always ndarray shape (512, 8), dtype uint16,
    indexed as strip[local_y, x_step].

    The C backend is used if ccd_decode.so is present and loadable;
    otherwise falls back to the numpy vectorised decoder silently.
    """
    if mode not in ('512', '1024'):
        raise ValueError(f"mode must be '512' or '1024', got {mode!r}")

    lib, path = _load_c_lib()
    if lib is not None:
        if verbose:
            print(f"Decoder: C backend ({path})")
        if mode == '512':
            return _make_c_decoder_512(lib)
        else:
            return _make_c_decoder_1024(lib)
    else:
        if verbose:
            print("Decoder: numpy backend (ccd_decode.so not found — "
                  "run: gcc -O3 -march=native -ffast-math -shared -fPIC "
                  "-o ccd_decode.so ccd_decode.c)")
        if mode == '512':
            return _make_numpy_decoder_512()
        else:
            return _make_numpy_decoder_1024()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-test / benchmark when run directly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    import time

    rng = np.random.default_rng(42)
    payload = bytes(DATA_OFF) + rng.integers(
        0, 65535, N_X_PER_PACKET * N_MUX * N_ADC, dtype=np.uint16
    ).tobytes()

    print("=== ccd_decode_fast self-test ===\n")

    for mode in ('512', '1024'):
        print(f"--- mode={mode} ---")
        decode_c  = make_decoder(mode, verbose=True)

        # Force numpy decoder for comparison
        lib, _ = _load_c_lib()
        decode_np = (_make_numpy_decoder_512 if mode == '512'
                     else _make_numpy_decoder_1024)()

        # Correctness
        r_c  = decode_c(payload)
        r_np = decode_np(payload)
        strip_c  = r_c[-1]
        strip_np = r_np[-1]
        match = np.array_equal(strip_c, strip_np)
        print(f"  Correctness: {'OK' if match else 'MISMATCH'}")

        # Benchmark
        N = 50000
        for _ in range(2000): decode_c(payload)
        for _ in range(2000): decode_np(payload)

        t0 = time.perf_counter()
        for _ in range(N): decode_c(payload)
        c_us = (time.perf_counter() - t0) / N * 1e6

        t0 = time.perf_counter()
        for _ in range(N): decode_np(payload)
        py_us = (time.perf_counter() - t0) / N * 1e6

        n_pkt = 64 if mode == '512' else 256
        print(f"  C  backend: {c_us:.2f} µs/pkt  "
              f"=> {1e6/c_us:,.0f} pkt/s  "
              f"=> {1e6/c_us/n_pkt:,.0f} fps")
        print(f"  Py backend: {py_us:.2f} µs/pkt  "
              f"=> {1e6/py_us:,.0f} pkt/s  "
              f"=> {1e6/py_us/n_pkt:,.0f} fps")
        print(f"  Speedup:    {py_us/c_us:.1f}x\n")

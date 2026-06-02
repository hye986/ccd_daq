#!/usr/bin/env python3
"""
udp_mock_sender_1024.py

1024x1024 CCD mock UDP sender composed of four 512x512 hybrids (HYBs).

Global coordinate system
------------------------
  (X, Y):  X = 0..1023  horizontal  (left=0,   right=1023)
            Y = 0..1023  vertical    (bottom=0, top=1023)

HYB layout and local-to-global mapping
---------------------------------------
  All HYBs share the same local readout convention:
    local_x = 0..511  (511 = ASIC side, 0 = far side)
    local_y = 0..511  (ASIC k covers local_y = (7-k)*64 .. (7-k)*64+63)
    ASIC0 -> local_y = 448-511  (local top)
    ASIC7 -> local_y = 0-63    (local bottom)
    line_num = 511 - local_x0   (0 = first packet, ASIC side)
    Rolling shutter: local_x0 starts at 511 (line_num=0), steps down to 0.

  HYB0: X=512-1023, Y=512-1023   ASICs on RIGHT (X=1023)
    global X =  512 + local_x
    global Y =  512 + local_y

  HYB1: X=512-1023, Y=0-511      ASICs on RIGHT (X=1023)
    global X =  512 + local_x
    global Y =    0 + local_y

  HYB2: X=0-511,   Y=0-511       ASICs on LEFT (X=0), 180-deg rotated
    global X =  511 - local_x
    global Y =  511 - local_y

  HYB3: X=0-511,   Y=512-1023    ASICs on LEFT (X=0), 180-deg rotated
    global X =  511 - local_x
    global Y = 1023 - local_y

  Verification (ASIC0, local_y=448-511, line_num=0 i.e. local_x=511):
    HYB0: X=1023,   Y=960-1023  (top-right quadrant, top strip)      ✓
    HYB1: X=1023,   Y=448-511   (bottom-right quadrant, top strip)   ✓
    HYB2: X=0,      Y=0-63      (bottom-left quadrant, bottom strip)  ✓
    HYB3: X=0,      Y=512-575   (top-left quadrant, bottom strip)    ✓

UDP payload  (8256 bytes = 64-byte header + 8192-byte data)
-----------------------------------------------------------
  Header (64 bytes):
    Bytes  0-16  PREFIX       (17 bytes, fixed magic)
    Bytes 17-27  MID_PADDING  (11 bytes, fixed)
    Bytes 28-29  line_num     uint16-LE = 511 - local_x0
    Bytes 30-33  frame32      uint16-LE x2: frame_a=lo16, frame_b=hi16
    Bytes 34-35  2 zero bytes
    Bytes 36-37  hyb_num      uint16-LE (0-3)
    Bytes 38-63  HEADER_END   (28 zero bytes)

  Data (8192 bytes):
    for x_step = 0..7:   local_x = local_x0 - x_step
      for mux = 0..63:
        ADC0..ADC7  (8 x uint16-LE)
    ADC wiring: ADC_TO_ASIC = [3,1,2,0,5,4,7,6]
    ADC_Y_BASE[adc] = (7 - ADC_TO_ASIC[adc]) * 64

UDP packet order per frame (256 packets total):
  For line_num = 0, 8, 16, ..., 504:
    HYB0, HYB1, HYB2, HYB3
"""

import argparse
import socket
import struct
import time

import numpy as np

#  header constants 
PREFIX = (
    b"\x50\x41\x43\x5f\x5f\x5f\x5f\x5f"
    b"\x41\x51\x54\x41\x44\x41\x54\x45\x4d"
)  # 17 bytes
MID_PADDING = b"\x01\x07\x9d\x33\x05\x00\x00\x00\xba\xda\xde"  # 11 bytes
HEADER_END  = b"\x00\x00" * 13                                   # 28 bytes

#  fixed constants 
HYB_SIZE          = 512
N_HYBS            = 4
N_X_PER_PACKET    = 8
N_MUX             = 64
N_ADC             = 8

HEADER_LEN        = 64
ADC_DATA_LEN      = N_X_PER_PACKET * N_MUX * N_ADC * 2   # = 8192 bytes
TOTAL_PAYLOAD_LEN = HEADER_LEN + ADC_DATA_LEN              # = 8256 bytes

# ADC-to-ASIC wiring (identical for all HYBs in local coordinates)
ADC_TO_ASIC = [3, 1, 2, 0, 5, 4, 7, 6]
ADC_Y_BASE  = [(7 - ADC_TO_ASIC[adc]) * 64 for adc in range(N_ADC)]

# Pre-computed coordinate lookup tables (shape (4, 512)):
#   GY_LUT[hyb, local_y] = global Y
#   GX_LUT[hyb, local_x] = global X
_ly = np.arange(HYB_SIZE, dtype=np.int32)
GY_LUT = np.stack([
     512 + _ly,   # HYB0
       0 + _ly,   # HYB1
     511 - _ly,   # HYB2
    1023 - _ly,   # HYB3
])  # shape (4, 512)

_lx = np.arange(HYB_SIZE, dtype=np.int32)
GX_LUT = np.stack([
    512 + _lx,   # HYB0  local_x=511 -> X=1023 (ASIC side)
    512 + _lx,   # HYB1  local_x=511 -> X=1023 (ASIC side)
    511 - _lx,   # HYB2  local_x=511 -> X=0    (ASIC side)
    511 - _lx,   # HYB3  local_x=511 -> X=0    (ASIC side)
])  # shape (4, 512)


def local_to_global_X(hyb: int, local_x: int) -> int:
    if hyb in (0, 1):
        return 512 + local_x    # local_x=511 -> X=1023 (ASIC side)
    else:
        return 511 - local_x    # local_x=511 -> X=0    (ASIC side)


#  header builder 
def build_header(local_x0: int, frame32: int, hyb: int) -> bytes:
    """line_num = 511 - local_x0  (0 when local_x0=511, i.e. ASIC side first)."""
    line_num = (HYB_SIZE - 1 - local_x0) & 0xFFFF
    frame_a  = frame32 & 0xFFFF
    frame_b  = (frame32 >> 16) & 0xFFFF
    return (
        PREFIX
        + MID_PADDING
        + struct.pack("<H",  line_num)
        + struct.pack("<HH", frame_a, frame_b)
        + b"\x00\x00"
        + struct.pack("<H",  hyb & 0xFFFF)
        + HEADER_END
    )


# Pre-compute y-index array for vectorised packing:
# _Y_IDX[mux, adc] = ADC_Y_BASE[adc] + mux   shape (N_MUX, N_ADC)
_mux_idx  = np.arange(N_MUX, dtype=np.int32)[:, None]      # (64, 1)
_adc_base = np.array(ADC_Y_BASE, dtype=np.int32)[None, :]  # (1, 8)
_Y_IDX    = _mux_idx + _adc_base                            # (64, 8)


#  image data packer 
def pack_image_data(mat: np.ndarray, hyb: int, local_x0: int,
                    noise_sigma: float = 0.0,
                    cm_sigma: float = 0.0) -> bytes:
    """
    Vectorised pack of N_X_PER_PACKET local columns for one HYB.
    mat[Y, X]: global 1024x1024, Y=0 bottom, Y=1023 top.

    Builds the full 8192-byte block as a single numpy array:
      block[x_step, mux, adc] = mat[gY[local_y], gX[local_x]]
                                  where local_y = _Y_IDX[mux, adc]
    ~50-100x faster than the equivalent Python triple-loop.
    """
    assert mat.shape == (1024, 1024), f"Expected (1024,1024), got {mat.shape}"

    # local_x values for all x_steps
    local_xs = local_x0 - np.arange(N_X_PER_PACKET, dtype=np.int32)
    if local_xs.min() < 0 or local_xs.max() >= HYB_SIZE:
        raise ValueError(f"local_x range out of bounds for local_x0={local_x0}")

    # Map local_x -> global X for all x_steps: shape (N_X_PER_PACKET,)
    gX_arr = GX_LUT[hyb, local_xs]   # (8,)
    # Map local_y -> global Y for all local_y: shape (HYB_SIZE,)
    gY_arr = GY_LUT[hyb]             # (512,)

    # Gather full columns from global frame: cols[local_y, x_step]
    cols  = mat[gY_arr[:, None], gX_arr[None, :]]  # (512, 8)

    # Reindex by _Y_IDX to get [mux, adc, x_step], then transpose to [x_step, mux, adc]
    block = cols[_Y_IDX, :]                         # (64, 8, 8) [mux, adc, x_step]
    block = np.transpose(block, (2, 0, 1))          # (8, 64, 8) [x_step, mux, adc]
    block = block.astype(np.float32)

    if cm_sigma > 0:
        # per-x common-mode: (N_X_PER_PACKET, 1, 1) broadcasts over mux+adc
        cm = np.random.normal(0.0, cm_sigma,
                              size=(N_X_PER_PACKET, 1, 1)).astype(np.float32)
        block += cm

    if noise_sigma > 0:
        block += np.random.normal(0.0, noise_sigma,
                                  size=block.shape).astype(np.float32)

    return np.clip(block, 0, 65535).astype('<u2').tobytes()


#  test-pattern data generator 
def lfsr16_step(val: int) -> int:
    val &= 0xFFFF
    bit = ((val >> 0) ^ (val >> 2) ^ (val >> 3) ^ (val >> 5)) & 1
    return ((val >> 1) | (bit << 15)) & 0xFFFF


def make_pattern_data(pattern: str, state: dict) -> bytes:
    """Generate 8192 bytes of test pattern in ADC-stream order."""
    n_steps = N_X_PER_PACKET * N_MUX
    out = bytearray()

    if pattern == "ramp":
        state.setdefault("counters", [0] * N_ADC)
        ctr = state["counters"]
        for _ in range(n_steps):
            out += struct.pack("<8H", *[c & 0xFFFF for c in ctr])
            for ch in range(N_ADC):
                ctr[ch] = (ctr[ch] + 1) & 0xFFFF

    elif pattern == "walking1":
        state.setdefault("bitpos", [0] * N_ADC)
        bp = state["bitpos"]
        for _ in range(n_steps):
            out += struct.pack("<8H", *[(1 << bp[ch]) & 0xFFFF for ch in range(N_ADC)])
            for ch in range(N_ADC):
                bp[ch] = (bp[ch] + 1) % 16

    elif pattern == "aaaa55":
        state.setdefault("toggle", 0)
        toggle = state["toggle"]
        for _ in range(n_steps):
            v = 0xAAAA if toggle == 0 else 0x5555
            out += struct.pack("<8H", v, v, v, v, v, v, v, v)
            toggle ^= 1
        state["toggle"] = toggle

    elif pattern == "channel_id":
        consts = [ch * 0x1111 for ch in range(N_ADC)]
        for _ in range(n_steps):
            out += struct.pack("<8H", *consts)

    elif pattern == "prbs16":
        state.setdefault("lfsr", [0xACE1, 0xBEEF, 0x1234, 0x00F1,
                                   0xCAFE, 0xBABE, 0x0BAD, 0xC0DE])
        lfsr = state["lfsr"]
        for _ in range(n_steps):
            out += struct.pack("<8H", *[v & 0xFFFF for v in lfsr])
            for ch in range(N_ADC):
                lfsr[ch] = lfsr16_step(lfsr[ch])

    elif pattern == "mux_ramp":
        # Value = mux index (0-63), identical across all ADC channels and x_steps.
        # Expected decoded image: 8 horizontal ASIC bands per HYB, each showing a
        # 0->63 ramp along local_y within the band (64 pixels per band).
        # Verifies MUX sequencing, independent of ADC wiring.
        for _ in range(N_X_PER_PACKET):
            for mux in range(N_MUX):
                out += struct.pack("<8H", *([mux] * N_ADC))

    elif pattern == "local_y_ramp":
        # Value = local_y = ADC_Y_BASE[adc] + mux (0-511).
        # Expected decoded image: clean 0->511 ramp along local_y for every HYB.
        # Verifies both MUX sequencing AND the ADC->ASIC wiring map together.
        for _ in range(N_X_PER_PACKET):
            for mux in range(N_MUX):
                vals = [ADC_Y_BASE[adc] + mux for adc in range(N_ADC)]
                out += struct.pack("<8H", *vals)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    assert len(out) == ADC_DATA_LEN
    return bytes(out)


#  main 
def main():
    ap = argparse.ArgumentParser(
        description="Mock UDP sender for 1024x1024 CCD (4 HYBs)")
    ap.add_argument("--dst-ip",        default="127.0.0.1")
    ap.add_argument("--dst-port",      type=int,   default=5000)
    ap.add_argument("--interval",      type=float, default=0.001,
                    help="Sleep between packets (s)")
    ap.add_argument("--pattern",
                    choices=["image", "ramp", "walking1", "aaaa55",
                             "channel_id", "prbs16", "mux_ramp", "local_y_ramp"],
                    default="image")
    ap.add_argument("--npy",           default="../image/fsp_tng_mpg_hll_1024x1024.npy")
    ap.add_argument("--noise-sigma",   type=float, default=10.0)
    ap.add_argument("--cm-sigma",      type=float, default=500.0)
    ap.add_argument("--log-every",     type=float, default=5.0)
    ap.add_argument("--start-frame32", type=int,   default=0)
    args = ap.parse_args()

    print("1024x1024 CCD, 4 HYBs")
    print("  HYB0: X=512-1023 Y=512-1023  ASICs right (X=1023), local_x=511->0 maps X=1023->512")
    print("  HYB1: X=512-1023 Y=0-511     ASICs right (X=1023), local_x=511->0 maps X=1023->512")
    print("  HYB2: X=0-511   Y=0-511      ASICs left  (X=0),    local_x=511->0 maps X=0->511  (180-deg)")
    print("  HYB3: X=0-511   Y=512-1023   ASICs left  (X=0),    local_x=511->0 maps X=0->511  (180-deg)")
    print(f"  ADC_TO_ASIC={ADC_TO_ASIC}  payload={TOTAL_PAYLOAD_LEN} bytes")
    print(f"  Packet order: line_num=0..504 step 8, each line: HYB0 HYB1 HYB2 HYB3")

    mat = None
    if args.pattern == "image":
        mat = np.load(args.npy)
        if mat.shape != (1024, 1024):
            raise ValueError(f"Expected (1024,1024), got {mat.shape}")
        mat = mat.astype(np.uint16)
        print(f"Loaded: {args.npy}  min={mat.min()} max={mat.max()}")

    sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frame32  = args.start_frame32 & 0xFFFFFFFF
    local_x0 = HYB_SIZE - 1   # start at local_x=511 (ASIC side, line_num=0)

    pattern_states = [{} for _ in range(N_HYBS)]
    sent_total = 0
    sent_since = 0
    t0     = time.perf_counter()
    t_last = time.time()

    # Rate pacing: schedule is advanced by interval after each packet.
    # Using perf_counter avoids the ~1 ms sleep floor for high packet rates.
    next_send = time.perf_counter()

    while True:
        for hyb in range(N_HYBS):
            hdr = build_header(local_x0=local_x0, frame32=frame32, hyb=hyb)

            if args.pattern == "image":
                adc_data = pack_image_data(
                    mat, hyb=hyb, local_x0=local_x0,
                    noise_sigma=args.noise_sigma,
                    cm_sigma=args.cm_sigma,
                )
            else:
                adc_data = make_pattern_data(args.pattern, pattern_states[hyb])

            payload = hdr + adc_data
            assert len(payload) == TOTAL_PAYLOAD_LEN, \
                f"payload len={len(payload)} expected={TOTAL_PAYLOAD_LEN}"

            now_pc = time.perf_counter()
            if next_send > now_pc:
                time.sleep(next_send - now_pc)
            sock.sendto(payload, (args.dst_ip, args.dst_port))

            sent_total += 1
            sent_since += 1
            next_send  += args.interval

        local_x0 -= N_X_PER_PACKET
        if local_x0 < 0:
            local_x0 = HYB_SIZE - 1
            frame32  = (frame32 + 1) & 0xFFFFFFFF

        now = time.time()
        if now - t_last >= args.log_every:
            line_num = HYB_SIZE - 1 - local_x0
            dt   = now - t_last
            rate = sent_since / dt
            print(f"[{now-t0:8.1f}s] pattern={args.pattern:9s}  "
                  f"sent={sent_total}  rate={rate:.1f} pkt/s  "
                  f"frame={frame32}  next_local_x0={local_x0}  "
                  f"next_line_num={line_num}")
            sent_since = 0
            t_last = now


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
udp_recorder.py

High-throughput, zero-loss CCD frame recorder to HDF5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design for no data loss — four layers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 1 — OS socket buffer (SO_RCVBUF = 256 MB)
    The kernel buffers incoming UDP packets before userspace even wakes
    up.  A large buffer absorbs multi-millisecond OS scheduling jitter.
    Check the actual value granted with --show-config.

  Layer 2 — RX thread (single responsibility: drain the socket)
    Calls recvfrom() in a tight loop and puts raw bytes onto pkt_queue.
    Does nothing else.  Uses drop-oldest eviction if the queue fills so
    the socket is always drained and the kernel buffer never overflows.

  Layer 3 — Decode thread (uses C decoder, releases GIL)
    Reads from pkt_queue, decodes using ccd_decode_fast (C via ctypes —
    releases the Python GIL during the decode, so RX thread runs freely).
    Assembles strips into frames and pushes each assembled frame to
    frm_queue, along with completeness metadata (complete flag,
    missing_x count).  Incomplete frames are passed through so the
    writer can decide whether to keep or discard them.

  Layer 4 — Writer thread (batch writes to HDF5 and/or RAW)
    Accumulates WRITE_BATCH frames in RAM, then writes them in a single
    contiguous slice.  This amortises the per-write fixed overhead
    (~2 ms) across many frames, keeping average disk latency well below
    the frame inter-arrival time even at 300 fps.
    With --keep-incomplete suppressed (the default), any frame whose
    complete flag is 0 (at least one UDP strip missing) is silently
    dropped before writing.  This reliably discards the partial first
    frame that occurs when the recorder starts mid-stream, as well as
    any later frames affected by packet loss.  Pass --keep-incomplete
    to override and write all frames regardless.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
How to verify no data was lost
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python udp_recorder.py --check run001.h5

  Three independent indicators:

  1. pkt_drop / frm_drop (runtime counters, printed every 5 s)
     Application-level drops from queue overflow.  Should always be 0.

  2. complete / missing_x (per saved frame in HDF5)
     The decode thread tracks which x-strips arrived.  complete=0 means
     at least one UDP packet was dropped before it could be decoded.

  3. frame32 gaps  (/gaps/ dataset in HDF5)
     Hardware frame counter is sequential.  A jump of N means N-1 frames
     were lost at the network or OS socket layer.

  Zero loss: pkt_drop=0, frm_drop=0, gaps=0, complete=1 for all frames.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HDF5 layout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /frames/data          uint16 or int16  (N, H, W)   frame[y, x]
  /frames/frame32       uint32  (N,)     hardware counter
  /frames/timestamp     float64 (N,)     host wall time (epoch s)
  /frames/complete      uint8   (N,)     1=all strips received
  /frames/missing_x     uint16  (N,)     missing strip count
  /frames/trigger_idx   uint32  (N,)     sequential decoder index
  /gaps/frame32_before  uint32  (M,)
  /gaps/frame32_after   uint32  (M,)
  /gaps/n_skipped       uint32  (M,)
  /pedestal             float32 (H, W)   copy of input pedestal (optional)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Build C decoder first (strongly recommended):
  gcc -O3 -march=native -ffast-math -shared -fPIC -o ccd_decode.so ccd_decode.c

  # Record all frames:
  python udp_recorder.py --output run001.h5

  # 2048x1024 mode, Poisson trigger at 10 Hz:
  python udp_recorder.py --mode 1024 --trigger poisson --trigger-rate 10 --output run002.h5

  # With pedestal subtraction (better compression):
  python udp_recorder.py --pedestal pedestal.npy --output run003.h5

  # Record all frames, skipping incomplete ones (default behaviour):
  python udp_recorder.py --output run004.h5

  # Keep incomplete frames (e.g. for diagnostics):
  python udp_recorder.py --keep-incomplete --output run004.h5

  # RAW-only output (incomplete frames skipped by default):
  python udp_recorder.py --raw-only --raw-output run004.raw

  # RAW-only, keeping incomplete frames:
  python udp_recorder.py --raw-only --raw-output run004.raw --keep-incomplete

  # Per-HYB 1024x512 RAW output for all four HYBs (alongside HDF5):
  python udp_recorder.py --output run005.h5 --hyb-raw-output run005_{H}.raw
  # Produces: run005_H0.raw, run005_H1.raw, run005_H2.raw, run005_H3.raw

  # Per-HYB RAW for selected HYBs only (H0 top-right and H2 bottom-left):
  python udp_recorder.py --output run005.h5 --hyb-raw-output run005_{H}.raw --hyb-select H0,H2

  # Per-HYB RAW only, no HDF5, no full-frame RAW:
  python udp_recorder.py --raw-only --hyb-raw-output run006_{H}.raw --hyb-select H1,H3

  # HYB layout (top view, charge transfer direction →← ):
  #   +----------+----------+
  #   |  H3      |  H0      |
  #   | top-left | top-right|
  #   +----------+----------+
  #   |  H2      |  H1      |
  #   | bot-left | bot-right|
  #   +----------+----------+
  # H0=hyb0, H1=hyb1, H2=hyb2, H3=hyb3.
  # Each 1024×512 RAW file: line 0 = frame_marker (0xFFFF) + ADC + 512 pixels,
  #                        lines 1-511 = line_marker (0xFFFE) + ADC + 512 pixels.

  # Check for data loss:
  python udp_recorder.py --check run001.h5

  # Show throughput analysis for your setup:
  python udp_recorder.py --show-config
"""

import argparse
import struct
import os
import queue
import socket
import threading
import time

import numpy as np
import h5py

try:
    import hdf5plugin
    _BLOSC = True
except ImportError:
    _BLOSC = False

import sys
sys.path.append('./ccd_decode')
from ccd_decode_fast import make_decoder, _load_c_lib


def _ensure_dir(path: str):
    """Create output directory if it doesn't exist."""
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
        print(f"  Created output directory: {d}")


#  protocol constants 
LINE_OFF  = 28
FRAME_OFF = 30
HYB_OFF   = 36
DATA_OFF  = 64

N_X_PER_PACKET = 8
N_MUX          = 64
N_ADC          = 8
HYB_LOCAL_Y    = 512   # local_y range (unchanged)
HYB_LOCAL_X     = 1024  # local_x range (doubled for 2048x1024 mode)
TOTAL_LEN      = 8256   # 64-byte header + 8192-byte data

ADC_TO_ASIC = [3, 1, 2, 0, 5, 4, 7, 6]
ADC_Y_BASE  = [(7 - ADC_TO_ASIC[a]) * 64 for a in range(N_ADC)]

_ly = np.arange(HYB_LOCAL_Y, dtype=np.int32)
GY_LUT = np.stack([ 512+_ly, _ly, 511-_ly, 1023-_ly ])  # (4, 512)
_lx = np.arange(HYB_LOCAL_X, dtype=np.int32)
GX_LUT = np.stack([ 1024+_lx, 1024+_lx, 1023-_lx, 1023-_lx ])  # (4, 1024)

#  tunable defaults 
SOCK_RCVBUF    = 256 * 1024 * 1024   # 256 MB kernel buffer request
PKT_QUEUE_SIZE = 32768               # ~256 MB at 8256 B/pkt
FRM_QUEUE_SIZE = 1024                # assembled frames waiting to write
WRITE_BATCH    = 16                  # frames per h5py slice write

RAW_HEADER   = b'16BU0000'
RAW_FRAME_MARKER = 0xFFFF
RAW_LINE_MARKER  = 0xFFFE


def _serialise_raw_frames(frames):
    """Serialise (B, H, W) uint16 frames to the converter RAW layout.

    Each RAW record is one vertical readout line: marker + ADC + H pixels.
    The first record of each frame gets 0xFFFF; following records get 0xFFFE.
    """
    B, H, W = frames.shape
    rec_dtype = np.dtype([
        ('marker', '<u2'),
        ('adc', '<u4'),
        ('pix', '<u2', (H,)),
    ])
    records = np.empty(B * W, dtype=rec_dtype)
    records['marker'] = RAW_LINE_MARKER
    records['marker'][::W] = RAW_FRAME_MARKER
    records['adc'] = 0
    records['pix'] = frames.transpose(0, 2, 1).reshape(B * W, H)
    return records.tobytes()


# HYB sub-matrix extraction helpers
# ------------------------------------
# The 2048×1024 frame is assembled from 4 HYBs (hyb 0-3):
#   H0 (hyb=0): top-right    — rows 512-1023, cols 1024-2047
#   H1 (hyb=1): bottom-right — rows 0-511,    cols 1024-2047
#   H2 (hyb=2): bottom-left  — rows 0-511,    cols 0-1023
#   H3 (hyb=3): top-left     — rows 512-1023, cols 0-1023
#
# GX_LUT[hyb] and GY_LUT[hyb] map local HYB line index 0..1023 to global
# frame column and row respectively.  Using these LUTs to index the frame
# guarantees correct readout order (including any mirroring) regardless of
# the physical charge-transfer direction.
#
# In the RAW sub-file each "line" is one local HYB column (512 pixels tall),
# read out in local line order 0→1023 (matching the LUT index).  Line 0 gets
# the frame marker (0xFFFF); subsequent lines get the line marker (0xFFFE).

def _serialise_raw_hyb(frames, hyb):
    """Serialise (B, 2048, 1024) frames to RAW for a single 1024×512 HYB.

    Parameters
    ----------
    frames : ndarray, shape (B, 2048, 1024), dtype uint16
        Full assembled frames.
    hyb : int
        HYB index 0-3 (H0=top-right, H1=bottom-right,
                         H2=bottom-left, H3=top-left).

    Returns
    -------
    bytes
        RAW-format bytes for B frames of this HYB.
    """
    B = frames.shape[0]
    N_LINE = HYB_LOCAL_X   # 1024 lines (columns = local_x) (columns in HYB coordinates)
    N_PIX  = HYB_LOCAL_Y   # 512 pixels per line (rows = local_y) per line (rows in HYB coordinates)

    # GY_LUT[hyb]: shape (1024,) — global row index for each HYB local line
    # GX_LUT[hyb]: shape (1024,) — global col index for each HYB local line
    gy = GY_LUT[hyb]        # (1024,)  global row indices, pixel order 0→1023
    gx = GX_LUT[hyb][::-1] # (1024,)  global col indices, REVERSED so that
    # local line 0 = outermost column = first UDP line_num=0x0000:
    #   H0/H1: line 0 → x=2047, line 1023 → x=1024
    #   H2/H3: line 0 → x=0,    line 1023 → x=1023

    # Extract (B, N_LINE, N_PIX): for each frame, for each local line l,
    # pixel p is frames[b, gy[p], gx[l]]   -- rows vary along pixel axis,
    # cols vary along line axis.
    # Advanced indexing: frames[:, gy[:, None], gx[None, :]]
    #   → shape (B, N_PIX, N_LINE)  (gy indexes rows, gx indexes cols)
    # We want shape (B, N_LINE, N_PIX) so transpose axes 1 and 2.
    sub = frames[:, gy[:, None], gx[None, :]]   # (B, N_PIX, N_LINE)
    sub = sub.transpose(0, 2, 1)                 # (B, N_LINE, N_PIX)

    rec_dtype = np.dtype([
        ('marker', '<u2'),
        ('adc',    '<u4'),
        ('pix',    '<u2', (N_PIX,)),
    ])
    records = np.empty(B * N_LINE, dtype=rec_dtype)
    records['marker']       = RAW_LINE_MARKER
    records['marker'][::N_LINE] = RAW_FRAME_MARKER
    records['adc']          = 0
    records['pix']          = sub.reshape(B * N_LINE, N_PIX)
    return records.tobytes()



# Thread 1 — RX: drain the socket, nothing else


def rx_thread_fn(sock, pkt_queue, stop_event, stats):
    """
    Pure receive loop.  Single responsibility: recvfrom → queue.
    Drop-oldest on full queue so the socket is always drained.
    """
    put = pkt_queue.put_nowait
    get = pkt_queue.get_nowait

    while not stop_event.is_set():
        try:
            payload, _ = sock.recvfrom(65535)
        except OSError:
            break
        if len(payload) != TOTAL_LEN:
            stats['rx_bad'] += 1
            continue
        stats['rx_ok'] += 1
        try:
            put(payload)
        except queue.Full:
            stats['pkt_drop'] += 1
            try:
                get()
            except queue.Empty:
                pass
            put(payload)



# Thread 2 — Decode + frame assembly (with subsampling)
#
# Two variants:
#
#   _make_decode_thread()      — standard path.
#     Assembles complete 1024×1024 or 2048×1024 (or 512×512) frames and pushes them to
#     frm_queue for the writer thread to handle (.h5 / full-frame .raw /
#     per-HYB .raw via _serialise_raw_hyb).
#
#   _make_decode_thread_direct() — UDP-direct HYB path (1024 mode only).
#     Splits the pkt_queue fan-out across two co-operating threads:
#       • hyb_writer_thread  reads pkt_queue, writes per-HYB .raw files
#         directly from decoded strips — (no 1024×1024 or 2048×1024 frame is assembled,
#         non-selected HYBs are skipped after header-only parsing.
#       • frame_assemble_thread  reads the same pkt_queue via a second
#         queue (relay_queue) and assembles full frames for frm_queue only
#         when .h5 or full-frame .raw output is also requested.
#     When neither .h5 nor full-frame .raw is needed the relay_queue is
#     not used at all and frame_assemble_thread is not started.
#
# Per-HYB RAW record format (same as full-frame RAW):
#   line 0      → frame_marker (0xFFFF) + adc(0) + 512 × uint16
#   lines 1-511 → line_marker  (0xFFFE) + adc(0) + 512 × uint16
#
# strip[local_y, x_step] from ccd_decode_fast maps directly to the RAW
# pixel column:  RAW_pix[local_y] at line_num + x_step.
# Because local_y = (7 - ADC_TO_ASIC[adc]) * 64 + mux, the strip already
# has the correct row ordering for the RAW file — no LUT needed.


def _make_decode_thread(mode, decode_pkt, pkt_queue, frm_queue,
                        stop_event, stats, save_every):
    """
    Standard decode thread factory for '512', '1024', or '2048' mode.

    Frame subsampling (save_every=N):
      Every packet: parse 6-byte header only (~2 µs) to track frame32
                    and detect real hardware gaps.
      Every Nth frame: full decode + strip placement.
      All decoded frames (complete or not) are pushed to frm_queue with
      completeness metadata (complete flag, missing_x count).  Filtering
      of incomplete frames is the writer's responsibility.

    Gap detection:
      prev_any_f32 tracks the last frame32 seen in the stream (not just
      saved ones).  A jump > save_every means real packet loss; a jump
      of exactly save_every is the expected subsampling gap.
    """
    is_1024 = (mode == '1024')
    is_2048 = (mode == '2048')
    H = 1024 if (is_1024 or is_2048) else HYB_SIZE
    W = 2048 if is_2048 else (1024 if is_1024 else HYB_SIZE)

    def _thread():
        work         = np.zeros((H, W), dtype=np.uint16)
        got_x        = (np.zeros((4, HYB_LOCAL_X), dtype=bool) if (is_1024 or is_2048)
                        else np.zeros(HYB_LOCAL_Y, dtype=bool))
        cur_f32      = None
        cur_ts       = None
        prev_any_f32 = None
        frame_idx    = 0
        saving       = False

        def _push_frame():
            nonlocal prev_any_f32
            missing  = int((~got_x).sum())
            complete = 1 if missing == 0 else 0
            gap = None
            if prev_any_f32 is not None:
                delta = int((cur_f32 - prev_any_f32) & 0xFFFFFFFF)
                real_gap = delta - save_every
                if real_gap > 0:
                    gap = (int(prev_any_f32), int(cur_f32), real_gap)
                    stats['gaps'] += real_gap
            prev_any_f32 = cur_f32
            rec = dict(
                data        = work.copy(),
                frame32     = cur_f32,
                timestamp   = cur_ts,
                complete    = complete,
                missing_x   = missing,
                trigger_idx = frame_idx,
                gap         = gap,
            )
            stats['frm_seen'] += 1
            try:
                frm_queue.put_nowait(rec)
            except queue.Full:
                stats['frm_drop'] += 1

        while not stop_event.is_set():
            try:
                payload = pkt_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
                line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
            except Exception:
                stats['decode_err'] += 1
                continue

            frame32 = (frame_b << 16) | frame_a
            stats['decoded'] += 1

            if cur_f32 is None:
                cur_f32   = frame32
                cur_ts    = time.time()
                saving    = (frame_idx % save_every == 0)

            if frame32 != cur_f32:
                if saving:
                    _push_frame()
                else:
                    if prev_any_f32 is not None:
                        delta = int((cur_f32 - prev_any_f32) & 0xFFFFFFFF)
                        real_gap = delta - save_every
                        if real_gap > 0:
                            stats['gaps'] += real_gap
                    prev_any_f32 = cur_f32
                    stats['skipped'] = stats.get('skipped', 0) + 1
                work[:] = 0
                got_x[:] = False
                frame_idx += 1
                cur_f32   = frame32
                cur_ts    = time.time()
                saving    = (frame_idx % save_every == 0)

            if not saving:
                continue

            try:
                if is_1024 or is_2048:
                    _, hyb, local_x0, strip = decode_pkt(payload)
                    if not (0 <= hyb < 4):
                        continue
                    gy  = GY_LUT[hyb]
                    lxs = local_x0 - np.arange(N_X_PER_PACKET, dtype=np.int32)
                    vld = (lxs >= 0) & (lxs < HYB_LOCAL_X)
                    gxs = GX_LUT[hyb, lxs[vld]]
                    work[gy[:,None], gxs[None,:]] = strip[:, vld]
                    got_x[hyb, lxs[vld]] = True
                else:
                    _, x0, strip = decode_pkt(payload)
                    lxs = x0 - np.arange(N_X_PER_PACKET, dtype=np.int32)
                    vld = (lxs >= 0) & (lxs < HYB_LOCAL_X)
                    work[:, lxs[vld]] = strip[:, vld]
                    got_x[lxs[vld]]   = True
            except Exception:
                stats['decode_err'] += 1

    return _thread


def _make_decode_thread_direct(decode_pkt, pkt_queue, frm_queue,
                               stop_event, stats, save_every,
                               selected_hybs, hyb_files,
                               skip_incomplete, max_frames,
                               need_frames):
    """
    UDP-direct HYB writer + optional frame assembler (1024/2048 mode only).

    Returns (hyb_writer_fn, frame_assemble_fn | None, relay_queue | None).

    Parameters
    ----------
    max_frames : int
        Stop after writing this many complete frames per HYB file.
        Passed explicitly so this factory has no dependency on args.
    need_frames : bool
        If True, every packet is also relayed to relay_queue so the
        frame assembler can build full 1024×1024 or 2048×1024 frames for .h5 /
        full-frame .raw output.  If False, relay_queue is None and
        the frame assembler thread is not started.
    """
    selected_hybs = {hyb for _, hyb in selected_hybs}

    # relay_queue: pkt_queue → hyb_writer → relay_queue → frame_assembler
    # Only created when the frame assembler is also needed.
    relay_queue = queue.Queue(maxsize=PKT_QUEUE_SIZE) if need_frames else None

    # ── per-HYB line buffers ──────────────────────────────────────────
    # buf[hyb] : shape (1024, 1024) uint16 for 2048 mode
    #   axis 0 = local_y (row), axis 1 = line_num (column/line index)
    hyb_bufs  = {hyb: np.zeros((HYB_LOCAL_Y, HYB_LOCAL_X), dtype=np.uint16)
                  for hyb in selected_hybs}
    # Track which line_nums have been received for completeness check
    hyb_got   = {hyb: np.zeros(HYB_LOCAL_X, dtype=bool)
                  for hyb in selected_hybs}
    # Map hyb → (file_handle, name) for writing
    hyb_to_file = {hyb: (fh, name)
                   for name, (fh, hyb) in hyb_files.items()}

    # RAW record dtype: marker(2) + adc(4) + 1024×uint16 pixels (2048 mode)
    _rec_dtype = np.dtype([('marker', '<u2'), ('adc', '<u4'),
                           ('pix', '<u2', (HYB_LOCAL_Y,))])
    max_f = max_frames

    def _flush_hyb(hyb):
        """Write one complete (or partial) HYB frame to its RAW file."""
        fh, name = hyb_to_file[hyb]
        buf = hyb_bufs[hyb]
        # Note: pedestal subtraction is not supported in the UDP-direct HYB
        # path. Use the .h5 path if pedestal correction is needed.
        records = np.empty(HYB_LOCAL_X, dtype=_rec_dtype)
        records['marker']    = RAW_LINE_MARKER
        records['marker'][0] = RAW_FRAME_MARKER
        records['adc']       = 0
        records['pix']       = buf.T          # (line_num, local_y)
        fh.write(records.tobytes())

    # ── hyb_writer_fn ────────────────────────────────────────────────
    def hyb_writer_fn():
        cur_f32      = None
        cur_ts       = None
        prev_any_f32 = None
        frame_idx    = 0
        saving       = False   # subsampling: decode this frame?
        written      = {hyb: 0 for hyb in selected_hybs}

        def _on_frame_boundary():
            nonlocal prev_any_f32
            if prev_any_f32 is not None:
                delta    = int((cur_f32 - prev_any_f32) & 0xFFFFFFFF)
                real_gap = delta - save_every
                if real_gap > 0:
                    stats['gaps'] += real_gap
            prev_any_f32 = cur_f32

            if not saving:
                return

            for hyb in selected_hybs:
                missing  = int((~hyb_got[hyb]).sum())
                complete = (missing == 0)
                if skip_incomplete and not complete:
                    stats.setdefault('hyb_skip_incomplete', 0)
                    stats['hyb_skip_incomplete'] += 1
                else:
                    _flush_hyb(hyb)
                    written[hyb] += 1
                    stats['written'] += 1
                hyb_bufs[hyb][:] = 0
                hyb_got[hyb][:]  = False

            # Periodic progress print (same cadence as writer_thread_fn)
            rep_hyb = next(iter(selected_hybs))
            w = written[rep_hyb]
            if w <= 5 or w % 200 == 0:
                elapsed = time.time() - stats['t_start']
                skipped = stats.get('hyb_skip_incomplete', 0)
                names   = ', '.join(
                    f"{n}:{written[h]}" for n, h in
                    sorted((n, h) for n, (_, h) in hyb_files.items())
                )
                print(f"  [{elapsed:7.1f}s] wrote={w:6d}  "
                      f"({names})  "
                      f"skip_incomplete={skipped}  "
                      f"gaps={stats['gaps']}  "
                      f"pkt_drop={stats['pkt_drop']}")

            # --max-frames: stop once any HYB file has reached the limit.
            # All selected HYBs are written in lockstep so written counts
            # are always equal; use the first one as the representative.
            rep_hyb = next(iter(selected_hybs))
            if written[rep_hyb] >= max_f:
                print(f"\nReached --max-frames={max_f}, stopping.")
                stop_event.set()

        while not stop_event.is_set():
            try:
                payload = pkt_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Relay to frame assembler before any processing
            if relay_queue is not None:
                try:
                    relay_queue.put_nowait(payload)
                except queue.Full:
                    stats['pkt_drop'] += 1
                    try:
                        relay_queue.get_nowait()
                    except queue.Empty:
                        pass
                    relay_queue.put_nowait(payload)

            # Header parse — always
            try:
                frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
                line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
                hyb              = struct.unpack_from('<H',  payload, HYB_OFF)[0]
            except Exception:
                stats['decode_err'] += 1
                continue

            if not (0 <= hyb < 4):
                continue

            frame32 = (frame_b << 16) | frame_a
            stats['decoded'] += 1

            if cur_f32 is None:
                cur_f32 = frame32
                cur_ts  = time.time()
                saving  = (frame_idx % save_every == 0)

            if frame32 != cur_f32:
                _on_frame_boundary()
                frame_idx += 1
                cur_f32   = frame32
                cur_ts    = time.time()
                saving    = (frame_idx % save_every == 0)

            # Skip non-selected HYBs and non-saving frames immediately
            if hyb not in selected_hybs or not saving:
                continue

            # Full strip decode for selected HYB
            try:
                _, _, _, strip = decode_pkt(payload)
            except Exception:
                stats['decode_err'] += 1
                continue

            # Place 8 columns into the line buffer
            # strip shape: (HYB_LOCAL_Y, N_X_PER_PACKET) = (512, 8)
            # line_num is the first column of this group (0, 8, 16 ... 504)
            col_end = min(line_num + N_X_PER_PACKET, HYB_LOCAL_X)
            n_cols  = col_end - line_num
            hyb_bufs[hyb][:, line_num:col_end] = strip[:, :n_cols]
            hyb_got[hyb][line_num:col_end]      = True

        # Flush last in-progress frame on shutdown
        if cur_f32 is not None and saving:
            _on_frame_boundary()

        for hyb in selected_hybs:
            fh, name = hyb_to_file[hyb]
            fh.flush()
            print(f"  HYB direct writer closed {name} ({written[hyb]} frames)")

    # ── frame_assemble_fn (optional) ──────────────────────────────────
    frame_assemble_fn = None
    if need_frames:
        def frame_assemble_fn():
            """
            Read relay_queue (fed by hyb_writer_fn) and assemble full
            1024×1024 or 2048×1024 frames for frm_queue (→ writer_thread_fn).
            Identical logic to _make_decode_thread for 1024/2048 mode.
            """
            work         = np.zeros((1024, 2048), dtype=np.uint16)
            got_x        = np.zeros((4, HYB_SIZE), dtype=bool)
            cur_f32      = None
            cur_ts       = None
            prev_any_f32 = None
            frame_idx    = 0
            saving       = False

            def _push_frame():
                nonlocal prev_any_f32
                missing  = int((~got_x).sum())
                complete = 1 if missing == 0 else 0
                gap = None
                if prev_any_f32 is not None:
                    delta = int((cur_f32 - prev_any_f32) & 0xFFFFFFFF)
                    real_gap = delta - save_every
                    if real_gap > 0:
                        gap = (int(prev_any_f32), int(cur_f32), real_gap)
                        # gaps already counted by hyb_writer; don't double-count
                prev_any_f32 = cur_f32
                rec = dict(
                    data        = work.copy(),
                    frame32     = cur_f32,
                    timestamp   = cur_ts,
                    complete    = complete,
                    missing_x   = missing,
                    trigger_idx = frame_idx,
                    gap         = gap,
                )
                stats['frm_seen'] += 1
                try:
                    frm_queue.put_nowait(rec)
                except queue.Full:
                    stats['frm_drop'] += 1

            while not (stop_event.is_set() and relay_queue.empty()):
                try:
                    payload = relay_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
                    line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
                    hyb_h            = struct.unpack_from('<H',  payload, HYB_OFF)[0]
                except Exception:
                    continue

                if not (0 <= hyb_h < 4):
                    continue

                frame32 = (frame_b << 16) | frame_a

                if cur_f32 is None:
                    cur_f32 = frame32
                    cur_ts  = time.time()
                    saving  = (frame_idx % save_every == 0)

                if frame32 != cur_f32:
                    if saving:
                        _push_frame()
                    else:
                        if prev_any_f32 is not None:
                            delta = int((cur_f32 - prev_any_f32) & 0xFFFFFFFF)
                            real_gap = delta - save_every
                            # gaps already counted by hyb_writer
                        prev_any_f32 = cur_f32
                        stats['skipped'] = stats.get('skipped', 0) + 1
                    work[:] = 0
                    got_x[:] = False
                    frame_idx += 1
                    cur_f32   = frame32
                    cur_ts    = time.time()
                    saving    = (frame_idx % save_every == 0)

                if not saving:
                    continue

                try:
                    _, hyb_d, local_x0, strip = decode_pkt(payload)
                    if not (0 <= hyb_d < 4):
                        continue
                    gy  = GY_LUT[hyb_d]
                    lxs = local_x0 - np.arange(N_X_PER_PACKET, dtype=np.int32)
                    vld = (lxs >= 0) & (lxs < HYB_LOCAL_X)
                    gxs = GX_LUT[hyb_d, lxs[vld]]
                    work[gy[:, None], gxs[None, :]] = strip[:, vld]
                    got_x[hyb_d, lxs[vld]] = True
                except Exception:
                    stats['decode_err'] += 1

    return hyb_writer_fn, frame_assemble_fn, relay_queue



# HDF5 helpers


def _ckwargs(compress, pedestal_applied=False):
    if _BLOSC:
        sh = (hdf5plugin.Blosc.BITSHUFFLE if pedestal_applied
              else hdf5plugin.Blosc.SHUFFLE)
        if compress == 'blosc-lz4':
            return hdf5plugin.Blosc(cname='lz4', clevel=9, shuffle=sh)
        if compress == 'blosc-zstd':
            return hdf5plugin.Blosc(cname='zstd', clevel=6, shuffle=sh)
    if compress in ('blosc-lz4', 'blosc-zstd'):
        print("Warning: hdf5plugin missing, using gzip-4. "
              "Install with: pip install hdf5plugin")
    return dict(compression='gzip', compression_opts=4)


def _create_file(path, height, width, max_frames, chunk_f,
                 compress, pedestal_applied, pedestal, args):
    ck          = _ckwargs(compress, pedestal_applied)
    pixel_dtype = 'uint16'  # always uint16; pedestal-sub uses modular wrap
    f = h5py.File(path, 'w', libver='latest')
    f.attrs['created']          = time.strftime('%Y-%m-%dT%H:%M:%S')
    f.attrs['mode']             = args.mode
    f.attrs['trigger']          = args.trigger
    f.attrs['trigger_rate']     = (args.trigger_rate
                                   if args.trigger == 'poisson' else -1)
    f.attrs['compress']         = compress
    f.attrs['pedestal_applied'] = int(pedestal_applied)
    if pedestal_applied:
        f.attrs['pedestal_file'] = str(args.pedestal)
        f.create_dataset('pedestal', data=pedestal,
                         compression='gzip', compression_opts=4)

    grp = f.create_group('frames')
    grp.create_dataset('data',
                       shape=(max_frames, height, width),
                       dtype=pixel_dtype,
                       chunks=(chunk_f, height, width),
                       **ck)
    sck = dict(chunks=(min(chunk_f * 8, 512),), **ck)
    for name, dt in [('frame32','uint32'), ('timestamp','float64'),
                     ('complete','uint8'), ('missing_x','uint16'),
                     ('trigger_idx','uint32')]:
        grp.create_dataset(name, shape=(max_frames,), dtype=dt, **sck)

    gaps = f.create_group('gaps')
    gck = dict(compression='gzip', compression_opts=1)
    for name in ('frame32_before', 'frame32_after', 'n_skipped'):
        gaps.create_dataset(name, shape=(0,), maxshape=(None,),
                            dtype='uint32', chunks=(512,), **gck)
    f.flush()
    return f, grp, gaps



# Thread 3 — writers


def raw_writer_thread_fn(frm_queue, stop_event, stats, args, height, width,
                         pedestal, skip_incomplete=False, selected_hybs=None):
    """Write selected frames directly to RAW, without creating an HDF5 file.

    By default (skip_incomplete=True) any frame with complete=0 (i.e. at
    least one UDP strip was not received) is discarded before writing.
    This covers the partial first frame that typically appears when the
    recorder starts mid-stream.  Pass --keep-incomplete to override.

    If selected_hybs is non-empty, per-HYB 512×512 RAW files are also
    written alongside the full-frame RAW (or instead of it when --raw-only
    is combined with --hyb-raw-output without --raw-output).
    """
    if not args.raw_output and not selected_hybs:
        raise ValueError("--raw-only requires --raw-output and/or --hyb-raw-output")

    pedestal_applied = pedestal is not None
    max_f = args.max_frames

    next_save = time.time()
    if args.trigger == 'poisson':
        next_save += np.random.exponential(1.0 / args.trigger_rate)

    B = WRITE_BATCH
    batch_data = np.empty((B, height, width), dtype=np.uint16)
    bpos = 0
    written = 0
    skipped_incomplete = 0

    # Open per-HYB file handles
    hyb_files = {}  # name -> file handle
    if selected_hybs:
        for name, hyb in selected_hybs:
            path = args.hyb_raw_output.replace('{H}', name)
            _ensure_dir(path)
            fh = open(path, 'wb', buffering=1024 * 1024)
            fh.write(RAW_HEADER)
            hyb_files[name] = (fh, hyb)

    def _write_batch():
        nonlocal written, bpos
        if bpos == 0:
            return
        chunk = batch_data[:bpos]
        if raw_f is not None:
            raw_f.write(_serialise_raw_frames(chunk))
        for name, (fh, hyb) in hyb_files.items():
            fh.write(_serialise_raw_hyb(chunk, hyb))
        written += bpos
        bpos = 0

    raw_f = None
    try:
        if args.raw_output:
            _ensure_dir(args.raw_output)
            raw_f = open(args.raw_output, 'wb', buffering=1024 * 1024)
            raw_f.write(RAW_HEADER)

        last_flush_t = time.time()

        while not (stop_event.is_set() and frm_queue.empty()):
            try:
                rec = frm_queue.get(timeout=0.2)
            except queue.Empty:
                if bpos > 0 and time.time() - last_flush_t > 1.0:
                    _write_batch()
                    if raw_f is not None:
                        raw_f.flush()
                    for name, (fh, hyb) in hyb_files.items():
                        fh.flush()
                    last_flush_t = time.time()
                continue

            stats['frm_seen'] += 1

            if skip_incomplete and rec['complete'] == 0:
                skipped_incomplete += 1
                if skipped_incomplete <= 3 or skipped_incomplete % 100 == 0:
                    print(f"  [skip-incomplete] RAW: dropped frame hw={rec['frame32']}  "
                          f"miss={rec['missing_x']}  (total skipped: {skipped_incomplete})")
                continue

            now = time.time()
            if args.trigger == 'poisson':
                if now < next_save:
                    continue
                while next_save <= now:
                    next_save += np.random.exponential(1.0 / args.trigger_rate)

            if written + bpos >= max_f:
                print(f"\nReached --max-frames={max_f}, stopping.")
                stop_event.set()
                break

            if pedestal_applied:
                batch_data[bpos] = ((rec['data'].astype(np.int32)
                                     - pedestal.astype(np.int32)) % 65536
                                    ).astype(np.uint16)
            else:
                batch_data[bpos] = rec['data']

            bpos += 1
            stats['written'] += 1

            if bpos == B:
                _write_batch()
                if written % args.flush_every < B:
                    if raw_f is not None:
                        raw_f.flush()
                    for name, (fh, hyb) in hyb_files.items():
                        fh.flush()
                    last_flush_t = time.time()

            if stats['written'] <= 5 or stats['written'] % 200 == 0:
                elapsed = now - stats['t_start']
                print(f"  [{elapsed:7.1f}s] wrote={stats['written']:6d}  "
                      f"hw={rec['frame32']}  complete={rec['complete']}  "
                      f"miss={rec['missing_x']}  "
                      f"pkt_drop={stats['pkt_drop']}  "
                      f"frm_drop={stats['frm_drop']}  "
                      f"gaps={stats['gaps']}")

        _write_batch()
        if raw_f is not None:
            raw_f.flush()
        for name, (fh, hyb) in hyb_files.items():
            fh.flush()

    finally:
        if raw_f is not None:
            raw_f.close()
        for name, (fh, hyb) in hyb_files.items():
            fh.close()

    if skipped_incomplete:
        print(f"  --skip-incomplete: {skipped_incomplete} incomplete frame(s) "
              f"dropped from RAW output.")
    if raw_f is not None:
        print(f"Closed {args.raw_output}  ({written} frames)")
    for name, (fh, hyb) in hyb_files.items():
        path = args.hyb_raw_output.replace('{H}', name)
        print(f"Closed {path}  ({written} frames, HYB {name}/hyb{hyb})")


def writer_thread_fn(frm_queue, stop_event, stats, args, height, width,
                     pedestal, skip_incomplete=False, selected_hybs=None):
    """
    Batch-write assembled frames to HDF5, and optionally to full-frame RAW.

    In the direct-HYB path, selected_hybs is handled upstream by
    hyb_writer_fn; this function only needs to write .h5 and/or full-frame
    .raw from the already-assembled frames in frm_queue.

    selected_hybs is accepted but ignored here when --hyb-raw-output is
    active with the direct path; it is only used in the legacy (non-direct)
    path where _serialise_raw_hyb is called inside _write_batch.
    """
    pedestal_applied = pedestal is not None
    max_f   = args.max_frames
    chunk_f = max(1, min(WRITE_BATCH * 2, 64))

    # Poisson schedule
    next_save = time.time()
    if args.trigger == 'poisson':
        next_save += np.random.exponential(1.0 / args.trigger_rate)

    written   = 0
    gap_count = 0
    skipped_incomplete = 0

    # Pre-allocated batch buffers
    B           = WRITE_BATCH
    px_dtype    = np.uint16   # always uint16; pedestal-sub uses % 65536
    batch_data  = np.empty((B, height, width), dtype=px_dtype)
    batch_f32   = np.empty(B, dtype=np.uint32)
    batch_ts    = np.empty(B, dtype=np.float64)
    batch_cmp   = np.empty(B, dtype=np.uint8)
    batch_miss  = np.empty(B, dtype=np.uint16)
    batch_tidx  = np.empty(B, dtype=np.uint32)
    bpos        = 0

    raw_f = None
    if args.raw_output:
        _ensure_dir(args.raw_output)
        raw_f = open(args.raw_output, 'wb', buffering=1024 * 1024)
        raw_f.write(RAW_HEADER)

    # Note: per-HYB RAW files are handled by hyb_writer_fn in the direct
    # path. In the legacy path (no --hyb-raw-output or non-1024 mode) we
    # still support them here via _serialise_raw_hyb for compatibility.
    hyb_files = {}
    if selected_hybs and not getattr(args, '_direct_hyb', False):
        for name, hyb in selected_hybs:
            path = args.hyb_raw_output.replace('{H}', name)
            _ensure_dir(path)
            fh = open(path, 'wb', buffering=1024 * 1024)
            fh.write(RAW_HEADER)
            hyb_files[name] = (fh, hyb)

    with h5py.File(args.output, 'w', libver='latest') as f:
        # Inline creation so we have f in scope for the whole function
        ck          = _ckwargs(args.compress, pedestal_applied)
        pixel_dtype = 'uint16'  # always uint16; pedestal-sub uses modular wrap
        f.attrs['created']          = time.strftime('%Y-%m-%dT%H:%M:%S')
        f.attrs['mode']             = args.mode
        f.attrs['trigger']          = args.trigger
        f.attrs['trigger_rate']     = (args.trigger_rate
                                       if args.trigger == 'poisson' else -1)
        f.attrs['compress']         = args.compress
        f.attrs['pedestal_applied'] = int(pedestal_applied)
        if pedestal_applied:
            f.attrs['pedestal_file'] = str(args.pedestal)
            f.create_dataset('pedestal', data=pedestal,
                             compression='gzip', compression_opts=4)

        grp = f.create_group('frames')
        grp.create_dataset('data',
                           shape=(max_f, height, width),
                           dtype=pixel_dtype,
                           chunks=(chunk_f, height, width),
                           **ck)
        sck = dict(chunks=(min(chunk_f * 8, 512),), **ck)
        for name, dt in [('frame32','uint32'), ('timestamp','float64'),
                         ('complete','uint8'), ('missing_x','uint16'),
                         ('trigger_idx','uint32')]:
            grp.create_dataset(name, shape=(max_f,), dtype=dt, **sck)

        gaps = f.create_group('gaps')
        gck = dict(compression='gzip', compression_opts=1)
        for name in ('frame32_before', 'frame32_after', 'n_skipped'):
            gaps.create_dataset(name, shape=(0,), maxshape=(None,),
                                dtype='uint32', chunks=(512,), **gck)
        f.flush()

        last_flush_t = time.time()

        def _write_batch():
            nonlocal written, bpos
            if bpos == 0:
                return
            chunk = batch_data[:bpos]
            if raw_f is not None:
                raw_f.write(_serialise_raw_frames(chunk))
            for name, (fh, hyb) in hyb_files.items():
                fh.write(_serialise_raw_hyb(chunk, hyb))
            i, j = written, written + bpos
            grp['data'][i:j]         = batch_data[:bpos]
            grp['frame32'][i:j]      = batch_f32[:bpos]
            grp['timestamp'][i:j]    = batch_ts[:bpos]
            grp['complete'][i:j]     = batch_cmp[:bpos]
            grp['missing_x'][i:j]    = batch_miss[:bpos]
            grp['trigger_idx'][i:j]  = batch_tidx[:bpos]
            written += bpos
            bpos = 0

        while not (stop_event.is_set() and frm_queue.empty()):
            try:
                rec = frm_queue.get(timeout=0.2)
            except queue.Empty:
                # Flush partial batch if idle for > 1 s
                if bpos > 0 and time.time() - last_flush_t > 1.0:
                    _write_batch()
                    f.flush()
                    if raw_f is not None:
                        raw_f.flush()
                    for _name, (_fh, _hyb) in hyb_files.items():
                        _fh.flush()
                    last_flush_t = time.time()
                continue

            stats['frm_seen'] += 1

            #  gap logging (always, regardless of trigger) 
            if rec['gap'] is not None:
                before, after, n = rec['gap']
                for ds_name, val in [('frame32_before', before),
                                     ('frame32_after',  after),
                                     ('n_skipped',      n)]:
                    ds = gaps[ds_name]
                    ds.resize((gap_count + 1,))
                    ds[gap_count] = val
                gap_count += 1
                stats['gaps_written'] += 1

            if skip_incomplete and rec['complete'] == 0:
                skipped_incomplete += 1
                if skipped_incomplete <= 3 or skipped_incomplete % 100 == 0:
                    print(f"  [skip-incomplete] HDF5: dropped frame hw={rec['frame32']}  "
                          f"miss={rec['missing_x']}  (total skipped: {skipped_incomplete})")
                continue

            #  trigger decision 
            now = time.time()
            if args.trigger == 'poisson':
                if now < next_save:
                    continue
                while next_save <= now:
                    next_save += np.random.exponential(1.0 / args.trigger_rate)

            #  stop if full 
            if written + bpos >= max_f:
                print(f"\nReached --max-frames={max_f}, stopping.")
                stop_event.set()
                break

            #  buffer this frame 
            if pedestal_applied:
                # Modular wrap-around: (raw - pedestal) % 65536
                # Preserves negative fluctuations as large uint16 values
                # (e.g. raw-ped = -100 => stored = 65436).
                # Stored as uint16 so the full range is usable.
                px = ((rec['data'].astype(np.int32)
                       - pedestal.astype(np.int32)) % 65536
                      ).astype(np.uint16)
                batch_data[bpos] = px
            else:
                batch_data[bpos] = rec['data']

            batch_f32[bpos]  = rec['frame32']
            batch_ts[bpos]   = rec['timestamp']
            batch_cmp[bpos]  = rec['complete']
            batch_miss[bpos] = rec['missing_x']
            batch_tidx[bpos] = rec['trigger_idx']
            bpos += 1
            stats['written'] += 1

            #  flush when batch full 
            if bpos == B:
                _write_batch()
                if written % args.flush_every < B:
                    f.flush()
                    if raw_f is not None:
                        raw_f.flush()
                    for _name, (_fh, _hyb) in hyb_files.items():
                        _fh.flush()
                    last_flush_t = time.time()

            # progress line for first few frames and every 200
            if stats['written'] <= 5 or stats['written'] % 200 == 0:
                elapsed = now - stats['t_start']
                print(f"  [{elapsed:7.1f}s] wrote={stats['written']:6d}  "
                      f"hw={rec['frame32']}  complete={rec['complete']}  "
                      f"miss={rec['missing_x']}  "
                      f"pkt_drop={stats['pkt_drop']}  "
                      f"frm_drop={stats['frm_drop']}  "
                      f"gaps={stats['gaps']}")

        #  flush remainder and finalise 
        _write_batch()
        grp.attrs['n_frames'] = written
        grp.attrs['n_gaps']   = gap_count
        gaps.attrs['n_gaps']  = gap_count
        f.flush()

    if skipped_incomplete:
        print(f"  --skip-incomplete: {skipped_incomplete} incomplete frame(s) "
              f"dropped from HDF5/RAW output.")

    if raw_f is not None:
        raw_f.flush()
        raw_f.close()
        print(f"Closed {args.raw_output}  ({written} frames)")

    for name, (fh, hyb) in hyb_files.items():
        fh.flush()
        fh.close()
        path = args.hyb_raw_output.replace('{H}', name)
        print(f"Closed {path}  ({written} frames, HYB {name}/hyb{hyb})")

    print(f"\nClosed {args.output}  ({written} frames, {gap_count} gap events)")



# Loss checker


def check_loss(path):
    print(f"\n{'='*62}")
    print(f"Loss report: {path}")
    print(f"{'='*62}")
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(path, 'r') as f:
        n = int(f['frames'].attrs.get('n_frames', f['frames/data'].shape[0]))
        if n == 0:
            print("No frames recorded.")
            return {}
        f32    = f['frames/frame32'][:n]
        cmp    = f['frames/complete'][:n]
        miss   = f['frames/missing_x'][:n]
        ts     = f['frames/timestamp'][:n]
        n_part = int((cmp == 0).sum())
        t_miss = int(miss.sum())

        print(f"\n  Frames written  : {n}")
        if len(ts) > 1:
            span = ts[-1] - ts[0]
            print(f"  Time span       : {span:.3f} s  ({n/span:.1f} fps avg)")

        print(f"\n  Within-frame packet loss:")
        print(f"    Partial frames : {n_part} / {n}  ({100*n_part/n:.2f}%)")
        print(f"    Missing strips : {t_miss} total")

        ng = int(f['gaps'].attrs.get('n_gaps', 0)) if 'gaps' in f else 0
        if ng > 0:
            sk = f['gaps/n_skipped'][:ng]
            bf = f['gaps/frame32_before'][:ng]
            af = f['gaps/frame32_after'][:ng]
            print(f"\n  Between-frame gaps:")
            print(f"    Gap events     : {ng}")
            print(f"    Frames lost    : {int(sk.sum())}")
            print(f"    Largest gap    : {int(sk.max())} frames")
            for i in range(min(5, ng)):
                print(f"      frame32 {bf[i]} -> {af[i]}  ({sk[i]} skipped)")
        else:
            print(f"\n  Between-frame gaps: none \u2713")

        jumps = int((np.diff(f32.astype(np.int64)) > 1).sum())
        trig  = f.attrs.get('trigger', '?')
        print(f"\n  frame32 non-consecutive pairs: {jumps}"
              + ("  (expected for Poisson trigger)" if jumps and trig == 'poisson'
                 else " \u2713" if jumps == 0 else " \u26a0"))

        ok = (n_part == 0 and t_miss == 0 and ng == 0)
        print(f"\n  Overall: {'NO DATA LOSS \u2713' if ok else 'DATA LOSS DETECTED \u26a0'}")
        print(f"{'='*62}\n")
        return dict(n=n, n_partial=n_part, total_missing=t_miss,
                    n_gaps=ng, ok=ok)



# Config / throughput analysis


def show_config(args):
    lib, path = _load_c_lib()
    n_pkt = {512: 64, 1024: 256, 2048: 512}.get(int(args.mode), 64)
    fps   = 300
    total_pps = fps * n_pkt

    print(f"\n{'='*62}")
    print(f"Throughput analysis — {args.mode}x{args.mode} @ {fps} fps")
    print(f"{'='*62}")
    print(f"  Packet rate     : {total_pps:,} pkt/s  ({fps} fps * {n_pkt} pkt/frame)")
    print(f"  Data rate       : {total_pps*TOTAL_LEN/1e6:.0f} MB/s")

    decode_us = 1.8 if lib else 9.6
    capacity  = 1e6 / decode_us
    margin    = capacity / total_pps
    print(f"\n  C decoder       : {'YES — ' + path if lib else 'NO (numpy fallback)'}")
    print(f"  Decode speed    : ~{decode_us:.1f} µs/pkt => {capacity:,.0f} pkt/s capacity")
    print(f"  Decode margin   : {margin:.1f}x  "
          + ("\u2713" if margin >= 2 else
             "\u26a0 tight" if margin >= 1 else
             "\u26a0 INSUFFICIENT — build C decoder"))

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_RCVBUF)
        actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        s.close()
        hold_ms = actual / (total_pps * TOTAL_LEN) * 1000
        print(f"\n  SO_RCVBUF       : {actual//1024//1024} MB "
              f"(requested {SOCK_RCVBUF//1024//1024} MB)")
        print(f"  Kernel hold time: {hold_ms:.0f} ms of full-rate data")
        if actual < SOCK_RCVBUF // 2:
            print(f"  \u26a0 Buffer capped by OS. To increase:")
            print(f"      sudo sysctl -w net.core.rmem_max={SOCK_RCVBUF}")
            print(f"      sudo sysctl -w net.core.rmem_default={SOCK_RCVBUF//4}")
    except Exception as e:
        print(f"  SO_RCVBUF check: {e}")

    print(f"\n  pkt_queue       : {args.pkt_queue_size} pkts "
          f"({args.pkt_queue_size*TOTAL_LEN//1024//1024} MB)")
    print(f"  frm_queue       : {args.frm_queue_size} frames")
    print(f"  write_batch     : {WRITE_BATCH} frames/slice")
    print(f"  Compression     : {args.compress}"
          + (" \u2713 (hdf5plugin available)" if _BLOSC
             else " (gzip fallback — pip install hdf5plugin)"))

    print(f"\n  Checklist for zero loss:")
    items = [
        (lib is not None,
         "C decoder built",
         "gcc -O3 -march=native -ffast-math -shared -fPIC -o ccd_decode.so ccd_decode.c"),
        (margin >= 2,
         f"Decode margin >= 2x (currently {margin:.1f}x)",
         "Build C decoder above"),
        (True,
         f"pkt_queue {args.pkt_queue_size} pkts = {args.pkt_queue_size*TOTAL_LEN//1024//1024} MB",
         None),
        (_BLOSC,
         "hdf5plugin installed",
         "pip install hdf5plugin"),
    ]
    for ok, label, fix in items:
        print(f"    {'[OK]' if ok else '[--]'}  {label}"
              + (f"\n         Fix: {fix}" if not ok and fix else ""))
    print(f"{'='*62}\n")



# Main


def main():
    ap = argparse.ArgumentParser(
        description="CCD UDP frame recorder — zero-loss design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--check',       metavar='FILE',
                    help='Analyse HDF5 file for data loss and exit')
    ap.add_argument('--show-config', action='store_true',
                    help='Show throughput analysis and exit')

    ap.add_argument('--mode',        choices=['512','1024','2048'], default="1024")
    ap.add_argument('--bind-ip',     default='127.0.0.1')
    ap.add_argument('--port',        type=int, default=5000)

    ap.add_argument('--trigger',     choices=['all','poisson'], default='all')
    ap.add_argument('--trigger-rate',type=float, default=1.0,
                    help='Mean save rate Hz for Poisson mode (default: 1.0)')

    ap.add_argument('--output',      default='recording.h5')
    ap.add_argument('--raw-output',  default=None, metavar='FILE',
                    help='Also write saved frames directly to converter RAW format')
    ap.add_argument('--raw-only',    action='store_true',
                    help='Write only RAW output; do not create an HDF5 file')
    ap.add_argument('--hyb-raw-output', metavar='PATTERN', default=None,
                    help='Write per-HYB 1024x512 RAW files. PATTERN must contain '
                         '{H} which is replaced by the HYB name, e.g. '
                         '"run001_H{H}.raw" produces run001_HH0.raw, run001_HH1.raw, '
                         'etc. Use --hyb-select to choose which HYBs to write '
                         '(default: all four). Requires --mode 1024.')
    ap.add_argument('--hyb-select', metavar='LIST', default=None,
                    help='Comma-separated list of HYBs to write when using '
                         '--hyb-raw-output, e.g. "H0,H2" or "H0,H1,H2,H3". '
                         'Default: all four (H0,H1,H2,H3).')
    ap.add_argument('--max-frames',  type=int, default=10000)
    ap.add_argument('--compress',    choices=['gzip','blosc-lz4','blosc-zstd'],
                    default='blosc-lz4')
    ap.add_argument('--flush-every', type=int, default=100,
                    help='f.flush() every N written frames (default: 100)')
    ap.add_argument('--pedestal',    default=None, metavar='FILE',
                    help='Per-pixel pedestal .npy — store int16 residuals '
                         'for better compression')
    ap.add_argument('--save-every',  type=int, default=1,
                    help='Save 1 out of every N frames (default: 1 = save all). '
                         'At 49k pkt/s use e.g. 10 to reduce write load by 10x. '
                         'Gap detection still runs on ALL frames regardless.')
    ap.add_argument('--keep-incomplete', action='store_true',
                    help='Write frames even if complete=0 (at least one UDP '
                         'strip was missing). By default incomplete frames are '
                         'discarded, which reliably skips the partial first '
                         'frame that occurs when the recorder starts mid-stream, '
                         'as well as any later frames affected by packet loss. '
                         'Applies to both HDF5 and RAW outputs.')
    ap.add_argument('--pkt-queue-size', type=int, default=PKT_QUEUE_SIZE)
    ap.add_argument('--frm-queue-size', type=int, default=FRM_QUEUE_SIZE)

    args = ap.parse_args()

    if args.check:
        check_loss(args.check)
        return
    if args.show_config:
        show_config(args)
        return

    height = width = 512 if args.mode == '512' else (1024 if args.mode == '1024' else 2048)

    if args.raw_only and not args.raw_output and not args.hyb_raw_output:
        print("Error: --raw-only requires --raw-output and/or --hyb-raw-output")
        return

    if not args.raw_only and os.path.exists(args.output):
        ans = input(f"'{args.output}' exists. Overwrite? [y/N] ")
        if ans.strip().lower() != 'y':
            print("Aborted."); return
    if args.raw_output and os.path.exists(args.raw_output):
        ans = input(f"'{args.raw_output}' exists. Overwrite? [y/N] ")
        if ans.strip().lower() != 'y':
            print("Aborted."); return

    # --- per-HYB RAW output validation ---
    HYB_NAMES = ['H0', 'H1', 'H2', 'H3']
    hyb_hyb_map = {'H0': 0, 'H1': 1, 'H2': 2, 'H3': 3}
    selected_hybs = []   # list of (name, hyb_index) for chosen HYBs
    if args.hyb_raw_output:
        if args.mode not in ('1024', '2048'):
            print("Error: --hyb-raw-output requires --mode 1024 "
                  "(per-HYB output is only defined for the full 1024×1024 or 2048×1024 sensor).")
            return
        if '{H}' not in args.hyb_raw_output:
            print("Error: --hyb-raw-output PATTERN must contain {H}, "
                  "e.g. \"run001_H{H}.raw\"")
            return
        # Parse --hyb-select
        if args.hyb_select:
            requested = [s.strip().upper() for s in args.hyb_select.split(',')]
            bad = [s for s in requested if s not in hyb_hyb_map]
            if bad:
                print(f"Error: unknown HYB(s) in --hyb-select: {bad}. "
                      f"Valid values: {HYB_NAMES}")
                return
            chosen_names = requested
        else:
            chosen_names = HYB_NAMES
        selected_hybs = [(name, hyb_hyb_map[name]) for name in chosen_names]
        # Check for existing files
        for name, _ in selected_hybs:
            path = args.hyb_raw_output.replace('{H}', name)
            if os.path.exists(path):
                ans = input(f"'{path}' exists. Overwrite? [y/N] ")
                if ans.strip().lower() != 'y':
                    print("Aborted."); return

    #  pedestal 
    pedestal = None
    if args.pedestal:
        pedestal = np.load(args.pedestal).astype(np.float32)
        if pedestal.shape != (height, width):
            print(f"Error: pedestal {pedestal.shape} != ({height},{width})")
            return
        print(f"Pedestal loaded: mean={pedestal.mean():.1f}  std={pedestal.std():.1f}")

    #  decoder 
    decode_pkt = make_decoder(mode=args.mode, verbose=True)

    #  socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_RCVBUF)
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if actual < SOCK_RCVBUF // 2:
            print(f"Warning: SO_RCVBUF = {actual//1024//1024} MB "
                  f"(wanted {SOCK_RCVBUF//1024//1024} MB). "
                  f"Run: sudo sysctl -w net.core.rmem_max={SOCK_RCVBUF}")
        else:
            print(f"SO_RCVBUF: {actual//1024//1024} MB")
    except OSError as e:
        print(f"Warning: SO_RCVBUF: {e}")
    sock.bind((args.bind_ip, args.port))

    #  shared state 
    pkt_queue  = queue.Queue(maxsize=args.pkt_queue_size)
    frm_queue  = queue.Queue(maxsize=args.frm_queue_size)
    stop_event = threading.Event()
    stats      = dict(rx_ok=0, rx_bad=0, pkt_drop=0,
                      decoded=0, decode_err=0, skipped=0,
                      frm_seen=0, frm_drop=0, written=0,
                      gaps=0, gaps_written=0,
                      t_start=time.time())

    #  routing decision ───────────────────────────────────────────────────
    # Direct HYB path: used when per-HYB .raw output is requested in 1024
    # mode.  Packets for selected HYBs are decoded and written directly from
    # the decode thread, bypassing the full 1024×1024 or 2048×1024 frame assembly.
    # A frame assembler relay is added only when .h5 or full-frame .raw is
    # also needed.
    #
    # Standard path: everything else (512 mode, no HYB raw, etc.).
    use_direct = (selected_hybs and args.mode in ('1024', '2048'))

    # need_frames: does the writer thread need assembled frames?
    need_frames = (not args.raw_only or args.raw_output)  # .h5 or full-frame .raw

    skip_incomplete = not args.keep_incomplete
    if skip_incomplete:
        print("  --skip-incomplete: frames with missing strips will be discarded.")
    else:
        print("  --keep-incomplete: incomplete frames will be written as-is.")

    save_every = max(1, args.save_every)

    threads = [
        threading.Thread(
            target=rx_thread_fn,
            args=(sock, pkt_queue, stop_event, stats),
            daemon=True, name='rx'),
    ]

    if use_direct:
        # Open per-HYB file handles here so both hyb_writer and (if needed)
        # writer_thread_fn can be set up cleanly.
        hyb_files = {}   # name -> (file_handle, hyb_index)
        for name, hyb in selected_hybs:
            path = args.hyb_raw_output.replace('{H}', name)
            _ensure_dir(path)
            fh = open(path, 'wb', buffering=1024 * 1024)
            fh.write(RAW_HEADER)
            hyb_files[name] = (fh, hyb)

        hyb_writer_fn, frame_assemble_fn, relay_queue = _make_decode_thread_direct(
            decode_pkt, pkt_queue, frm_queue,
            stop_event, stats, save_every,
            selected_hybs, hyb_files,
            skip_incomplete, args.max_frames,
            need_frames=need_frames,
        )
        # Mark args so writer_thread_fn knows HYB files are handled upstream
        args._direct_hyb = True

        threads.append(threading.Thread(
            target=hyb_writer_fn, daemon=True, name='hyb-writer'))

        if frame_assemble_fn is not None:
            threads.append(threading.Thread(
                target=frame_assemble_fn, daemon=True, name='frame-assembler'))

        if need_frames:
            writer_fn = raw_writer_thread_fn if args.raw_only else writer_thread_fn
            threads.append(threading.Thread(
                target=writer_fn,
                args=(frm_queue, stop_event, stats, args, height, width, pedestal,
                      skip_incomplete, None),   # selected_hybs=None: handled upstream
                daemon=True, name='writer'))
    else:
        # Standard path: one decode thread + one writer thread
        args._direct_hyb = False
        dec_fn = _make_decode_thread(
            args.mode, decode_pkt, pkt_queue, frm_queue,
            stop_event, stats, save_every)
        writer_fn = raw_writer_thread_fn if args.raw_only else writer_thread_fn
        threads.append(threading.Thread(
            target=dec_fn, daemon=True, name='decode'))
        threads.append(threading.Thread(
            target=writer_fn,
            args=(frm_queue, stop_event, stats, args, height, width, pedestal,
                  skip_incomplete, selected_hybs),
            daemon=True, name='writer'))

    for t in threads:
        t.start()

    print(f"\nRecording  mode={args.mode}  trigger={args.trigger}"
          + (f" @ {args.trigger_rate} Hz" if args.trigger == 'poisson' else ""))
    print(f"  path={'UDP-direct HYB' if use_direct else 'standard frame-assembly'}")
    print(f"  save_every={save_every}  "
          f"(saving ~{191/save_every:.1f} fps at 49k pkt/s)")
    if args.raw_only:
        print(f"  output=<raw-only>  max={args.max_frames} frames")
    else:
        print(f"  output={args.output}  max={args.max_frames} frames")
    if args.raw_output:
        print(f"  raw_output={args.raw_output}")
    if selected_hybs:
        for name, hyb in selected_hybs:
            path = args.hyb_raw_output.replace('{H}', name)
            print(f"  hyb_raw_output[{name}/hyb{hyb}]={path}")
    print(f"  pkt_queue={args.pkt_queue_size}  frm_queue={args.frm_queue_size}"
          f"  write_batch={WRITE_BATCH}")
    print(f"\nPress Ctrl-C to stop.\n")
    hdr = (f"{'time':>8}  {'rx':>8}  {'decoded':>8}  {'skipped':>8}  "
           f"{'written':>8}  {'pkt_drop':>9}  {'frm_drop':>9}  "
           f"{'dec_err':>7}  {'gaps':>6}  "
           f"{'pkt_q':>6}  {'frm_q':>6}  {'rate/s':>8}")
    print(hdr)

    t_last = time.time()
    rx_last = 0

    try:
        while not stop_event.is_set():
            time.sleep(5.0)
            now = time.time()
            dt  = now - t_last
            rate = (stats['rx_ok'] - rx_last) / dt
            rx_last = stats['rx_ok']
            t_last  = now

            loss = stats['pkt_drop'] > 0 or stats['frm_drop'] > 0 or stats['gaps'] > 0
            flag = '  *** LOSS ***' if loss else ''
            dec_err = stats['decode_err']
            print(f"{now-stats['t_start']:>8.1f}  "
                  f"{stats['rx_ok']:>8d}  "
                  f"{stats['decoded']:>8d}  "
                  f"{stats['skipped']:>8d}  "
                  f"{stats['written']:>8d}  "
                  f"{stats['pkt_drop']:>9d}  "
                  f"{stats['frm_drop']:>9d}  "
                  f"{dec_err:>6d}  "
                  f"{stats['gaps']:>6d}  "
                  f"{pkt_queue.qsize():>6d}  "
                  f"{frm_queue.qsize():>6d}  "
                  f"{rate:>8.0f}"
                  f"{flag}")

    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    for t in threads:
        if t.name != 'rx':
            t.join(timeout=30)
    sock.close()

    elapsed = time.time() - stats['t_start']
    lost = stats['pkt_drop'] + stats['frm_drop'] + stats['gaps']
    print(f"\nSummary ({elapsed:.1f}s):")
    print(f"  rx={stats['rx_ok']:,}  decoded={stats['decoded']:,}  "
          f"written={stats['written']:,}")
    print(f"  pkt_drop={stats['pkt_drop']}  frm_drop={stats['frm_drop']}  "
          f"gaps={stats['gaps']}")
    print(f"\n  {'NO DATA LOSS \u2713' if lost==0 else f'LOSS DETECTED: {lost} events \u26a0'}")

    if not args.raw_only and os.path.exists(args.output):
        check_loss(args.output)


if __name__ == '__main__':
    main()

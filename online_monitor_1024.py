#!/usr/bin/env python3
"""
online_monitor_1024.py

Real-time display of 2048x1024 CCD data from four 1024x512 HYBs over UDP.

Display modes (auto-detected, or forced with --display)
--------------------------------------------------------
  gui       Try interactive matplotlib backends in order:
              Qt5Agg -> Qt6Agg -> TkAgg -> wxAgg
            Opens a window.  Requires a display (local or via SSH -X).

  http      Headless mode — no display needed.
            Renders each frame to a PNG using the Agg backend and serves
            it via a tiny HTTP server.  Open http://<host>:<http-port>/
            in any browser on any machine.  The page auto-refreshes every
            --plot-interval seconds.

  auto      (default) Try gui first; fall back to http if no display
            is available or no GUI backend can be loaded.

Threading model
---------------
  Thread 1 — RX:      tight recvfrom → pkt_queue
  Thread 2 — Decode:  pkt_queue → assembled frames → frame_ready event
  Thread 3 — Main:    GUI event loop (gui mode) OR HTTP server (http mode)

Global coordinate system
------------------------
  frame[Y, X],  Y=0 bottom, Y=1023 top.  Displayed with origin='lower'.

HYB local-to-global mapping  (2048x1024 frame: X=0..2047, Y=0..1023)
---------------------------------------------------------------
  HYB0: global X = 1024+local_x,  global Y = 512+local_y
  HYB1: global X = 1024+local_x,  global Y =   0+local_y
  HYB2: global X = 1023-local_x,  global Y = 511-local_y
  HYB3: global X = 1023-local_x,  global Y = 1023-local_y
"""

import argparse
import io
import os
import queue
import socket
import struct
import threading
import time

import numpy as np
import sys
sys.path.append('./ccd_decode')
from ccd_decode_fast import make_decoder

#  header offsets 
LINE_OFF  = 28
FRAME_OFF = 30
HYB_OFF   = 36
DATA_OFF  = 64

#  fixed constants 
HYB_LOCAL_Y      = 512   # local_y range (unchanged)
HYB_LOCAL_X      = 1024  # local_x range (doubled)
N_HYBS            = 4
N_X_PER_PACKET    = 8
N_MUX             = 64
N_ADC             = 8

HEADER_LEN        = 64
ADC_DATA_LEN      = N_X_PER_PACKET * N_MUX * N_ADC * 2
TOTAL_PAYLOAD_LEN = HEADER_LEN + ADC_DATA_LEN   # 8256

ADC_TO_ASIC = [3, 1, 2, 0, 5, 4, 7, 6]
ADC_Y_BASE  = [(7 - ADC_TO_ASIC[adc]) * 64 for adc in range(N_ADC)]


_ly = np.arange(HYB_LOCAL_Y, dtype=np.int32)
GY_LUT = np.stack([ 512+_ly, _ly, 511-_ly, 1023-_ly ])   # (4, 512)

_lx = np.arange(HYB_LOCAL_X, dtype=np.int32)
GX_LUT = np.stack([ 1024+_lx, 1024+_lx, 1023-_lx, 1023-_lx ])  # (4, 1024)

SOCK_RCVBUF = 32 * 1024 * 1024



# Backend selection — must happen before any matplotlib import


def _probe_backend(backend: str) -> bool:
    """
    Return True only if the backend can actually open a display connection.

    Importing the backend module is not enough — Qt loads its xcb/wayland
    platform plugin lazily, so the crash happens when the first Figure is
    created, not on import.  We force that by creating and immediately
    closing a tiny test figure inside a subprocess so a crash/abort there
    cannot take down the main process.
    """
    import subprocess, sys
    code = (
        f"import matplotlib; matplotlib.use({backend!r}); "
        f"import matplotlib.pyplot as p; "
        f"p.figure(); p.close('all')"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, timeout=10
    )
    return result.returncode == 0


def _select_backend(requested: str):
    """
    Choose a matplotlib backend and return (backend_name, mode).
    mode is 'gui' or 'http'.

    Never raises — if nothing works we fall back to 'Agg' + http.

    Each candidate GUI backend is probed in a subprocess so that a
    broken Qt xcb plugin (common on Ubuntu) aborts the probe process
    rather than crashing the monitor itself.
    """
    import matplotlib
    GUI_BACKENDS = ['Qt5Agg', 'Qt6Agg', 'TkAgg', 'wxAgg']

    if requested == 'http':
        matplotlib.use('Agg')
        return 'Agg', 'http'

    # gui or auto: probe GUI backends one by one
    if requested in ('gui', 'auto'):
        has_display = bool(os.environ.get('DISPLAY') or
                           os.environ.get('WAYLAND_DISPLAY') or
                           os.name == 'nt')
        if not has_display and requested == 'gui':
            print("Warning: no DISPLAY found, falling back to http mode.")

        if has_display or requested == 'gui':
            for backend in GUI_BACKENDS:
                print(f"Probing backend {backend}...", end=' ', flush=True)
                if _probe_backend(backend):
                    matplotlib.use(backend)
                    import matplotlib.pyplot as _plt
                    _plt.switch_backend(backend)
                    print("OK")
                    print(f"Display backend: {backend}")
                    return backend, 'gui'
                else:
                    print("failed (skipping)")

        if requested == 'gui':
            print("Warning: no GUI backend available, falling back to http mode.")
        else:
            print("No working GUI backend found, using http mode.")
        matplotlib.use('Agg')
        return 'Agg', 'http'

    # explicit backend name — user knows what they want, use it directly
    matplotlib.use(requested)
    return requested, 'gui'



# Packet decode


# decode_packet is assigned in main() after mode is known
decode_packet = None



# Thread 1: RX


def receiver_thread(sock, pkt_queue, stop_event):
    while not stop_event.is_set():
        try:
            payload, _ = sock.recvfrom(65535)
        except OSError:
            break
        if len(payload) != TOTAL_PAYLOAD_LEN:
            continue
        try:
            pkt_queue.put_nowait(payload)
        except queue.Full:
            try:
                pkt_queue.get_nowait()
            except queue.Empty:
                pass
            pkt_queue.put_nowait(payload)



# Thread 2: Decode


def decode_thread(pkt_queue, buf_pair, buf_lock, active_idx,
                  frame_ready, stats, stop_event,
                  frame_timeout_s=0.5, plot_interval_s=2.0):
    """
    Two optimisations for high-rate data (10k+ pkt/s):

    1. Frame subsampling
       At 10k pkt/s / 256 pkt/frame = ~39 fps incoming with a 2 s display
       interval, 77 of every 78 frames would be decoded and immediately
       discarded.  Instead, only fully decode ONE frame per display interval
       (the one closest to the next GUI tick).  All other frames are
       header-only parsed (~2 µs) to track frame32 and detect gaps.
       CPU reduction for strip decode+place: ~78x at the above rates.

    2. Double-buffer (ping-pong) — zero-copy publish
       buf_pair = [buf_A, buf_B].  Decode writes into the INACTIVE buffer.
       On publish: swap active_idx[0] under lock (~1 µs).
       GUI reads from active_idx[0] with no waiting for a 2 MB copy.
       Lock contention: ~1 µs vs ~175 µs with a single shared buffer.

    Publish triggers (whichever fires first):
      - Frame-boundary: first packet of frame N+1 (correct on single FIFO fibre)
      - Idle timeout: no packet for frame_timeout_s s (handles silent HYBs)
    """
    def _write_buf():
        return buf_pair[1 - active_idx[0]]

    got_x           = np.zeros((N_HYBS, HYB_LOCAL_X), dtype=bool)
    hybs_seen       = set()
    current_frame32 = None
    last_pkt_t      = time.time()
    next_decode_t   = time.time()   # wall time of next wanted display frame
    active_decoding = False

    def _publish(reason='frame_boundary'):
        nonlocal next_decode_t
        hyb_info = {h: int(got_x[h].sum()) for h in range(N_HYBS)}
        with buf_lock:
            active_idx[0] = 1 - active_idx[0]   # atomic pointer swap
            stats['last_frame32']        = current_frame32
            stats['last_n_hybs']         = len(hybs_seen)
            stats['last_hyb_info']       = hyb_info
            stats['last_publish_reason'] = reason
        frame_ready.set()
        next_decode_t = time.time() + plot_interval_s

    while not stop_event.is_set():
        try:
            payload = pkt_queue.get(timeout=0.05)
        except queue.Empty:
            if (current_frame32 is not None
                    and len(hybs_seen) > 0
                    and (time.time() - last_pkt_t) >= frame_timeout_s):
                if active_decoding:
                    _publish(reason='timeout')
                    _write_buf()[:] = 0
                got_x[:]        = False
                hybs_seen.clear()
                current_frame32  = None
                active_decoding  = False
            continue

        last_pkt_t = time.time()

        #  fast header-only parse (always, ~2 µs) 
        try:
            line_num         = struct.unpack_from('<H',  payload, LINE_OFF)[0]
            frame_a, frame_b = struct.unpack_from('<HH', payload, FRAME_OFF)
            hyb              = struct.unpack_from('<H',  payload, HYB_OFF)[0]
        except Exception:
            stats['dropped'] += 1
            continue

        if not (0 <= hyb < N_HYBS):
            stats['dropped'] += 1
            continue

        frame32  = (frame_b << 16) | frame_a
        local_x0 = (HYB_LOCAL_X - 1 - line_num) & 0x3FF
        stats['decoded'] += 1

        if current_frame32 is None:
            current_frame32 = frame32

        if frame32 != current_frame32:
            # Frame boundary
            if active_decoding:
                _publish(reason='frame_boundary')
                _write_buf()[:] = 0
            got_x[:]        = False
            hybs_seen.clear()
            current_frame32 = frame32
            # Decode this frame if it falls within the display window
            active_decoding = (time.time() >= next_decode_t - plot_interval_s * 0.1)

        hybs_seen.add(hyb)

        if not active_decoding:
            stats['skipped'] = stats.get('skipped', 0) + 1
            continue

        #  full strip decode + placement (selected frames only) 
        try:
            _, _, _, strip = decode_packet(payload)
        except Exception:
            stats['dropped'] += 1
            continue

        gY    = GY_LUT[hyb]
        lxs   = local_x0 - np.arange(N_X_PER_PACKET, dtype=np.int32)
        valid = (lxs >= 0) & (lxs < HYB_LOCAL_X)
        gXs   = GX_LUT[hyb, lxs[valid]]
        wb    = _write_buf()
        wb[gY[:,None], gXs[None,:]] = strip[:, valid]
        got_x[hyb, lxs[valid]]      = True



# HTTP snapshot server (headless mode)


_HTTP_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CCD Monitor 2048x1024 — {title}</title>
  <style>
    body {{ background:#111; color:#ccc; font-family:monospace;
           display:flex; flex-direction:column; align-items:center; }}
    img  {{ max-width:98vw; image-rendering:pixelated; border:1px solid #444; }}
    p    {{ margin:4px; font-size:13px; }}
  </style>
  <meta http-equiv="refresh" content="{interval}">
</head>
<body>
  <p id="title">{title}</p>
  <img src="/frame.png?t={ts}" alt="frame">
  <p>Auto-refresh every {interval}s &nbsp;|&nbsp; {ts}</p>
</body>
</html>
"""

def _make_png(frame_arr, vmin, vmax, cmap, stats, pkt_queue, queue_size):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#111')
    im = ax.imshow(frame_arr, origin='lower', interpolation='nearest',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, 2048, 0, 1024])
    fig.colorbar(im, ax=ax).set_label('ADC value', color='#ccc')

    ax.axvline(1024, color="white", lw=0.5, ls="--", alpha=0.4)
    ax.axhline(512, color='white', lw=0.5, ls='--', alpha=0.4)
    for label, tx, ty in [('HYB0',768,768),('HYB1',768,256),
                           ('HYB2',256,256),('HYB3',256,768)]:
        ax.text(tx, ty, label, color='white', fontsize=8,
                ha='center', va='center', alpha=0.6)

    f32  = stats.get('last_frame32', '?')
    qsz  = pkt_queue.qsize()
    shown = stats.get('shown', 0)
    ax.set_title(f"Frame {f32}  |  queue={qsz}/{queue_size}  |  shown={shown}",
                 color='#eef', fontsize=9)
    ax.set_xlabel('X', color='#aaa')
    ax.set_ylabel('Y', color='#aaa')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_http_server(http_port, plot_interval, args,
                    buf_pair, buf_lock, active_idx, frame_ready,
                    stats, pkt_queue, stop_event, pedestal=None,
                    pixel_mask=None):
    import http.server

    png_lock   = threading.Lock()
    png_cache  = [b'']
    render_buf = np.zeros((1024, 2048), dtype=np.float32)

    def _render_loop():
        while not stop_event.is_set():
            if not frame_ready.wait(timeout=1.0):
                continue
            frame_ready.clear()
            with buf_lock:
                idx = active_idx[0]
            np.copyto(render_buf, buf_pair[idx])  # copy outside lock
            stats['shown'] = stats.get('shown', 0) + 1
            # Pedestal subtraction
            if pedestal is not None:
                _sub = render_buf.astype(np.int32) - pedestal.astype(np.int32)
                wrap_around = args.wrap_around
                if wrap_around:
                    _disp = (_sub % 65536).astype(np.float32)
                else:
                    _disp = np.clip(_sub, 0, 65535).astype(np.float32)
            else:
                _disp = render_buf.copy().astype(np.float32)
            # Apply pixel mask
            if pixel_mask is not None:
                _disp[pixel_mask] = 0.0
            vmin = (args.vmin if args.vmin is not None
                    else float(np.percentile(_disp, 1)))
            vmax = (args.vmax if args.vmax is not None
                    else float(np.percentile(_disp, 99)))
            png = _make_png(_disp, vmin, vmax, args.cmap,
                            stats, pkt_queue, args.queue_size)
            with png_lock:
                png_cache[0] = png

    threading.Thread(target=_render_loop, daemon=True, name='render').start()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a): pass
        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/frame.png':
                with png_lock:
                    data = png_cache[0]
                if not data:
                    self.send_response(204); self.end_headers(); return
                self.send_response(200)
                self.send_header('Content-Type',   'image/png')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control',  'no-store')
                self.end_headers()
                self.wfile.write(data)
            else:
                ts    = time.strftime('%H:%M:%S')
                f32   = stats.get('last_frame32', 'waiting...')
                title = f"2048x1024 CCD | frame {f32} | {ts}"
                body  = _HTTP_PAGE.format(
                    title=title, interval=plot_interval, ts=ts
                ).encode()
                self.send_response(200)
                self.send_header('Content-Type',   'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = http.server.HTTPServer(('0.0.0.0', http_port), Handler)
    server.timeout = 0.5
    print(f"HTTP monitor: http://localhost:{http_port}/  "
          f"(auto-refresh every {plot_interval}s)")
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()



# GUI mode


def run_gui(args, buf_pair, buf_lock, active_idx, frame_ready,
            stats, pkt_queue, stop_event, pedestal=None,
            pixel_mask=None):
    """
    GUI window with 1024x1024 hit map + marginal projections (2048x1024).

    Layout (gridspec):
      ┌┬┐
      │                 │  y    │  y-projection: mean per row,
      │   hit map       │  proj │  horizontal line, y-axis shared
      │  (2048x1024)    │       │  with image → parallel to y-axis
      ├┘       │
      │  x-projection           │
      │  mean per col, vertical │
      │  x-axis shared ┘
      └
      [colorbar strip]

    HYB boundary lines and labels are drawn on the image axes.
    Projections use set_xdata/set_ydata for fast updates (~250 µs).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation

    N = 2048
    render_buf = np.zeros((N, N), dtype=np.uint16)
    t_stat     = [time.time()]
    coords     = np.arange(N)

    #  figure layout 
    fig = plt.figure(figsize=(10, 9))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[4, 1],
        hspace=0.04,
        wspace=0.04,
    )
    ax_img  = fig.add_subplot(gs[0, 0])
    ax_yprj = fig.add_subplot(gs[0, 1], sharey=ax_img)
    ax_xprj = fig.add_subplot(gs[1, 0], sharex=ax_img)
    # gs[1,1] left empty — colorbar goes to the right of ax_yprj

    #  image 
    im    = ax_img.imshow(render_buf, origin='lower', interpolation='nearest',
                          vmin=args.vmin, vmax=args.vmax,
                          extent=[0, N, 0, N], aspect='auto')
    cbar  = fig.colorbar(im, ax=ax_yprj, location='right',
                         fraction=0.15, pad=0.04)
    cbar.set_label('ADC value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    title = ax_img.set_title('Waiting for data...', fontsize=9)
    ax_img.set_xlabel('X  (0=left, 2047=right/ASIC side for HYB0/1)', fontsize=8)
    ax_img.set_ylabel('Y  (0=bottom, 1023=top)', fontsize=8)
    ax_img.tick_params(labelsize=7)

    # HYB boundary guides and labels
    ax_img.axvline(1024, color='white', lw=0.5, ls='--', alpha=0.5)
    ax_img.axhline(512, color='white', lw=0.5, ls='--', alpha=0.5)
    for label, tx, ty in [('HYB0',768,768),('HYB1',768,256),
                           ('HYB2',256,256),('HYB3',256,768)]:
        ax_img.text(tx, ty, label, color='white', fontsize=8,
                    ha='center', va='center', alpha=0.7)

    #  y-projection (mean per row, horizontal) 
    line_yprj, = ax_yprj.plot(np.zeros(N), coords,
                               color='steelblue', lw=0.8)
    ax_yprj.set_xlabel('mean', fontsize=7)
    ax_yprj.tick_params(axis='y', labelleft=False, labelsize=7)
    ax_yprj.tick_params(axis='x', labelsize=7)
    ax_yprj.set_title('y proj', fontsize=8)
    ax_yprj.grid(True, alpha=0.3, lw=0.5)
    # HYB boundary lines on y-projection too
    ax_yprj.axhline(512, color='steelblue', lw=0.5, ls='--', alpha=0.4)

    #  x-projection (mean per col, vertical) 
    line_xprj, = ax_xprj.plot(coords, np.zeros(N),
                               color='tomato', lw=0.8)
    ax_xprj.set_ylabel('mean', fontsize=7)
    ax_xprj.tick_params(axis='x', labelbottom=False, labelsize=7)
    ax_xprj.tick_params(axis='y', labelsize=7)
    ax_xprj.set_title('x proj', fontsize=8)
    ax_xprj.grid(True, alpha=0.3, lw=0.5)
    ax_xprj.axvline(1024, color="tomato", lw=0.5, ls='--', alpha=0.4)

    def update(_):
        if not frame_ready.is_set():
            return
        frame_ready.clear()
        # Read active buffer under lock — only an index read + view,
        # no 2 MB copy while holding the lock.
        with buf_lock:
            idx = active_idx[0]
            f32 = stats['last_frame32']
        np.copyto(render_buf, buf_pair[idx])  # copy outside lock
        stats['shown'] = stats.get('shown', 0) + 1

        # Pedestal subtraction
        if pedestal is not None:
            _sub = render_buf.astype(np.int32) - pedestal.astype(np.int32)
            wrap_around = args.wrap_around  # --wrap-around flag
            if wrap_around:
                # Negative values wrap to top of uint16 range:
                # e.g. raw-ped = -100  =>  65436
                _disp = (_sub % 65536).astype(np.float32)
            else:
                # Clip negatives to 0 (standard behaviour)
                _disp = np.clip(_sub, 0, 65535).astype(np.float32)
        else:
            _disp = render_buf.astype(np.float32)
        # Apply pixel mask: set noisy/dead pixels to 0
        if pixel_mask is not None:
            _disp[pixel_mask] = 0.0

        #  image 
        im.set_data(_disp)
        if args.vmin is None or args.vmax is None:
            im.autoscale()

        #  projections 
        y_proj = _disp.mean(axis=1)   # (1024,) mean per row
        x_proj = _disp.mean(axis=0)   # (1024,) mean per col

        line_yprj.set_xdata(y_proj)
        ax_yprj.set_xlim(y_proj.min() * 0.98 or 0,
                         y_proj.max() * 1.02 or 1)

        line_xprj.set_ydata(x_proj)
        ax_xprj.set_ylim(x_proj.min() * 0.98 or 0,
                         x_proj.max() * 1.02 or 1)

        # Build HYB status string: show which HYBs sent data
        hyb_info = stats.get('last_hyb_info', {})
        n_hybs   = stats.get('last_n_hybs', 0)
        reason   = stats.get('last_publish_reason', '')
        hyb_str  = '  '.join(
            f"HYB{h}={'OK' if hyb_info.get(h,0)==HYB_LOCAL_X else hyb_info.get(h,0)}"
            for h in range(N_HYBS)
        )
        partial  = n_hybs < N_HYBS
        timeout  = reason == 'timeout'
        prefix   = ('[TIMEOUT] ' if timeout else '') + \
                   (f'[{n_hybs}/4 HYBs] ' if partial else '')
        title.set_text(
            f"{prefix}Frame {f32}  |  {hyb_str}  |  "
            f"q={pkt_queue.qsize()}/{args.queue_size}  shown={stats['shown']}")
        # Orange = partial frame; red = timeout publish
        title.set_color('#ff4444' if timeout else '#ff9944' if partial else '#eeeeff')
        fig.canvas.draw_idle()

        now = time.time()
        if now - t_stat[0] >= 10.0:
            dt = now - t_stat[0]
            skipped = stats.get('skipped', 0)
            print(f"decoded={stats['decoded']}  skipped={skipped}  "
                  f"dropped={stats.get('dropped',0)}  "
                  f"shown={stats['shown']}  rate~{stats['decoded']/dt:.0f} pkt/s")
            stats['decoded'] = 0
            stats['skipped'] = 0
            t_stat[0] = now

    fig._ani = animation.FuncAnimation(
        fig, update,
        interval=args.plot_interval * 1000,
        blit=False, cache_frame_data=False
    )
    try:
        plt.show()
    finally:
        stop_event.set()



# Main


def main():
    ap = argparse.ArgumentParser(
        description="Online monitor for 2048x1024 CCD UDP stream (4 HYBs)")
    ap.add_argument('--bind-ip',       default='127.0.0.1')
    #ap.add_argument('--bind-ip',       default='192.168.100.1')
    ap.add_argument('--port',          type=int,   default=5000)
    ap.add_argument('--plot-interval', type=float, default=2.0)
    ap.add_argument('--queue-size',    type=int,   default=4096)
    ap.add_argument('--frame-timeout', type=float, default=0.5,
                    help='Seconds of silence before publishing a partial frame '
                         '(fallback for missing HYBs). Default: 0.5 s')
    ap.add_argument('--vmin',          type=float, default=None)
    ap.add_argument('--vmax',          type=float, default=None)
    ap.add_argument('--cmap',          default='gray')
    ap.add_argument('--pedestal',    default=None, metavar='FILE',
                    help='Per-pixel pedestal .npy (float32, same shape as frame). '
                         'Subtracted before display. No impact on decode speed.')
    ap.add_argument('--wrap-around',  action='store_true', default=False,
                    help='When subtracting pedestal, add 65536 to negative values '
                         'so they wrap to the top of the uint16 range instead of '
                         'clipping. Off by default.')
    ap.add_argument('--mask-noisy',   default=None, metavar='FILE',
                    help='Boolean mask .npy of noisy pixels (True=noisy). '
                         'Masked pixels are set to 0 before display.')
    ap.add_argument('--mask-dead',    default=None, metavar='FILE',
                    help='Boolean mask .npy of dead pixels (True=dead). '
                         'Masked pixels are set to 0 before display.')
    ap.add_argument('--display',       default='auto',
                    choices=['auto', 'gui', 'http',
                             'Qt5Agg', 'Qt6Agg', 'TkAgg', 'wxAgg'],
                    help='Display mode or backend (default: auto)')
    ap.add_argument('--http-port',     type=int,   default=8080,
                    help='Port for HTTP snapshot server (default: 8080)')
    args = ap.parse_args()

    backend, mode = _select_backend(args.display)
    print(f"Mode: {mode}  backend: {backend}")

    global decode_packet
    decode_packet = make_decoder(mode='2048')

    #  pedestal and pixel masks 
    pedestal = None
    if args.pedestal:
        pedestal = np.load(args.pedestal).astype(np.float32)
        expected = (1024, 2048)
        if pedestal.shape != expected:
            print(f"Error: pedestal shape {pedestal.shape} != {expected}")
            return
        print(f"Pedestal loaded: mean={pedestal.mean():.1f}  std={pedestal.std():.1f}")

    # Combined pixel mask: True where pixel should be zeroed (noisy or dead)
    pixel_mask = None
    for attr, label in [('mask_noisy', 'noisy'), ('mask_dead', 'dead')]:
        path = getattr(args, attr, None)
        if path:
            m = np.load(path).astype(bool)
            if m.shape != (1024, 2048):
                print(f"Error: {label} mask shape {m.shape} != {EXPECTED}")
                return
            pixel_mask = m if pixel_mask is None else (pixel_mask | m)
            print(f"{label.capitalize()} mask loaded: {m.sum()} pixels")
    if pixel_mask is not None:
        print(f"Combined pixel mask: {pixel_mask.sum()} pixels will be zeroed")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_RCVBUF)
    except OSError as e:
        print(f"Warning: SO_RCVBUF: {e}")
    sock.bind((args.bind_ip, args.port))

    pkt_queue     = queue.Queue(maxsize=args.queue_size)
    # Double-buffer: two pre-allocated frames; decode writes into the
    # inactive one, then swaps active_idx[0] under lock (~1 µs).
    # GUI reads from buf_pair[active_idx[0]] — no 2 MB copy under lock.
    buf_pair      = [np.zeros((1024, 2048), dtype=np.uint16),
                     np.zeros((1024, 2048), dtype=np.uint16)]
    buf_lock      = threading.Lock()
    active_idx    = [0]   # mutable so decode thread can swap it
    frame_ready   = threading.Event()
    stop_event    = threading.Event()
    stats         = {'decoded': 0, 'dropped': 0, 'shown': 0,
                     'last_frame32': None}

    threading.Thread(target=receiver_thread,
                     args=(sock, pkt_queue, stop_event),
                     daemon=True, name='udp-rx').start()
    threading.Thread(target=decode_thread,
                     args=(pkt_queue, buf_pair, buf_lock, active_idx,
                           frame_ready, stats, stop_event,
                           args.frame_timeout, args.plot_interval),
                     daemon=True, name='decode').start()

    print(f"Listening on {args.bind_ip}:{args.port}  "
          f"payload={TOTAL_PAYLOAD_LEN} B  "
          f"queue={args.queue_size} pkts")
    print(f"ADC_TO_ASIC={ADC_TO_ASIC}")

    try:
        if mode == 'http':
            run_http_server(args.http_port, args.plot_interval, args,
                            buf_pair, buf_lock, active_idx, frame_ready,
                            stats, pkt_queue, stop_event, pedestal, pixel_mask)
        else:
            run_gui(args, buf_pair, buf_lock, active_idx, frame_ready,
                    stats, pkt_queue, stop_event, pedestal, pixel_mask)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        sock.close()


if __name__ == '__main__':
    main()

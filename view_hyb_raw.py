#!/usr/bin/env python3
"""
view_hyb_raw.py

Interactive viewer for per-HYB RAW files written by udp_recorder.py
with --hyb-raw-output.  Composites up to four 1024×512 HYB sub-frames into
their correct positions in the full 2048×1024 sensor layout and displays the
result with the same look-and-feel as view_recording.py.

Sensor layout (top view)
------------------------
  col 0                       col 2047
  +----------+----------+   row 1023
  |  H3      |  H0      |
  | top-left | top-right|
  +----------+----------+   row 512
  |  H2      |  H1      |
  | bot-left | bot-right|
  +----------+----------+   row 0

HYB RAW file format (same as full-frame RAW but 1024 wide)
----------------------------------------------------------
  header      : b'16BU0000'   (8 bytes, once at file start)
  line 0      : 0xFFFF + adc(4B) + 512×uint16   ← frame marker
  line 1-1023 : 0xFFFE + adc(4B) + 512×uint16   ← line markers

  line_num 0 = outermost column (x=2047 for H0/H1, x=0 for H2/H3)
  local_y  0 = first ADC row (as written by ccd_decode_fast)

Placement into the full frame (2048x1024)
-----------------------------------------
  H0 (hyb=0, top-right):    rows 512-1023, cols 1024-2047
                             line 0 → col 2047, line 1023 → col 1024
  H1 (hyb=1, bot-right):    rows   0-511,  cols 1024-2047
                             line 0 → col 2047, line 1023 → col 1024
  H2 (hyb=2, bot-left):     rows   0-511,  cols   0-1023
                             line 0 → col   0, line 1023 → col 1023
  H3 (hyb=3, top-left):     rows 512-1023, cols   0-1023
                             line 0 → col   0, line 1023 → col 1023

  local_y maps to the global row via GY_LUT[hyb]:
    H0: GY = 512 + local_y   (rows 512..1023)
    H1: GY = local_y          (rows   0..511)
    H2: GY = 511 - local_y   (rows 511..0, i.e. flipped)
    H3: GY = 1023 - local_y  (rows 1023..512, i.e. flipped)

Usage
-----
  # All four HYBs:
  python view_hyb_raw.py run001_HH0.raw run001_HH1.raw run001_HH2.raw run001_HH3.raw

  # Name-based shorthand (equivalent to above, order doesn't matter):
  python view_hyb_raw.py --H0 run001_HH0.raw --H1 run001_HH1.raw \\
                          --H2 run001_HH2.raw --H3 run001_HH3.raw

  # Only two HYBs (the rest of the canvas is left at 0):
  python view_hyb_raw.py --H1 run001_HH1.raw --H2 run001_HH2.raw

  # Pattern shorthand (must contain {H}):
  python view_hyb_raw.py --pattern "run001_H{H}.raw" --hybs H0,H1,H2,H3

  # Extra options:
  python view_hyb_raw.py --pattern "run001_H{H}.raw" --hybs H1,H2 \\
      --start 10 --vmin 0 --vmax 4000 --cmap plasma --every 5
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets

# ── sensor constants ─────────────────────────────────────────────────────────

# HYB dimensions for 2048x1024 frame mode
HYB_LOCAL_X = 1024  # local_x range (1024 lines per HYB in RAW)
HYB_LOCAL_Y = 512   # local_y range (512 pixels per line)
FRAME_H  = 1024      # frame height (Y)
FRAME_W  = 2048      # frame width (X)

RAW_HEADER       = b'16BU0000'
RAW_HEADER_LEN   = 8
RAW_FRAME_MARKER = 0xFFFF
RAW_LINE_MARKER  = 0xFFFE

# Record layout: marker(u16) + adc_counter(u32) + HYB_LOCAL_Y×u16 pixels
_REC_DTYPE = np.dtype([
    ('marker', '<u2'),
    ('adc',    '<u4'),
    ('pix',    '<u2', (HYB_LOCAL_Y,)),
])
REC_SIZE   = _REC_DTYPE.itemsize    # 2 + 4 + 512*2 = 1030 bytes
FRAME_RECS = HYB_LOCAL_X            # 1024 records per frame
FRAME_BYTES = FRAME_RECS * REC_SIZE  # bytes per frame in HYB RAW file

# GY_LUT[hyb, local_y] → global row in 2048×1024 frame
# local_y: 0-511 maps to global Y via:
#   H0: 512 + local_y  → rows 512-1023
#   H1: 0 + local_y    → rows 0-511
#   H2: 511 - local_y  → rows 511-0 (flipped)
#   H3: 1023 - local_y → rows 1023-512 (flipped)
_ly   = np.arange(HYB_LOCAL_Y, dtype=np.int32)
GY_LUT = np.stack([512 + _ly,        # H0: rows 512-1023
                   _ly,               # H1: rows 0-511
                   511 - _ly,         # H2: rows 511-0 (flipped)
                   1023 - _ly])       # H3: rows 1023-512 (flipped)

# GX placement: line 0 = outermost column (ASIC side)
#   H0/H1: line l → global col 2047 - l (right side, col decreases)
#   H2/H3: line l → global col l     (left side, col increases)
def _line_to_gcol(hyb: int, line: int) -> int:
    if hyb in (0, 1):
        return FRAME_W - 1 - line  # 2047, 2046, ..., 1024
    else:
        return line                  # 0, 1, ..., 1023


# ── HYB name ↔ hyb index ────────────────────────────────────────────────────

HYB_NAMES  = ['H0', 'H1', 'H2', 'H3']
NAME_TO_HYB = {'H0': 0, 'H1': 1, 'H2': 2, 'H3': 3}
HYB_TO_NAME = {v: k for k, v in NAME_TO_HYB.items()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-HYB RAW reader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybRawFile:
    """
    Memory-mapped reader for one per-HYB RAW file.

    The file layout is:
      [8-byte header] [1024 records per frame] × n_frames

    Each record: marker(u16) + adc(u32) + 512×u16 pixels.
    Records within a frame are indexed by line_num (0..511).
    Frame i starts at byte offset: HEADER_LEN + i × FRAME_BYTES.
    """

    def __init__(self, path: str, hyb: int, every: int = 1):
        self.path  = path
        self.hyb   = hyb
        self.name  = HYB_TO_NAME[hyb]
        self.every = max(1, every)

        size = os.path.getsize(path)
        hdr_len = len(RAW_HEADER)

        # Verify header
        with open(path, 'rb') as f:
            hdr = f.read(hdr_len)
        if hdr != RAW_HEADER:
            raise ValueError(
                f"{path}: bad header {hdr!r}, expected {RAW_HEADER!r}")

        data_bytes = size - hdr_len
        if data_bytes % FRAME_BYTES != 0:
            # Tolerate a partial last frame (e.g. interrupted recording)
            n_full  = data_bytes // FRAME_BYTES
            partial = data_bytes % FRAME_BYTES
            print(f"  Warning: {self.name} {path}: {partial} trailing bytes "
                  f"(partial frame) — ignoring, using {n_full} complete frames.")
            data_bytes = n_full * FRAME_BYTES

        n_total = data_bytes // FRAME_BYTES
        if n_total == 0:
            raise ValueError(f"{path}: no complete frames found.")

        # Memory-map the record array (excluding header)
        # Shape: (n_total × FRAME_RECS,)  dtype=_REC_DTYPE
        self._mmap = np.memmap(path, dtype=_REC_DTYPE, mode='r',
                               offset=hdr_len,
                               shape=(n_total * FRAME_RECS,))
        self.n_total = n_total
        # viewer indices (every-N subsampling)
        self._indices = np.arange(0, n_total, self.every)
        self.n        = len(self._indices)

    def read_hyb_frame(self, viewer_idx: int) -> np.ndarray:
        """
        Return the HYB sub-frame at viewer index as uint16 (HYB_LOCAL_X, HYB_LOCAL_Y).

        Shape: (HYB_LOCAL_X, HYB_LOCAL_Y) = (1024, 512)
          axis 0: local_y  — maps to global row via GY_LUT[hyb]
          axis 1: line_num — maps to global col via _line_to_gcol(hyb, l)

        Records are stored in line order 0→1023 (outermost col first).
        """
        raw_idx  = int(self._indices[viewer_idx])
        start    = raw_idx * FRAME_RECS
        records  = self._mmap[start : start + FRAME_RECS]   # shape (1024,)
        # records['pix'] shape: (1024, 512)  — axis0=line_num, axis1=local_y
        # We want (local_y, line_num) → transpose
        return records['pix'].T.copy()   # (HYB_LOCAL_Y, HYB_LOCAL_X)

    def completeness(self, viewer_idx: int) -> tuple:
        """Return (n_missing_lines, marker_ok) for a frame."""
        raw_idx = int(self._indices[viewer_idx])
        start   = raw_idx * FRAME_RECS
        records = self._mmap[start : start + FRAME_RECS]
        # First record should be frame marker, rest line markers
        marker_ok  = bool(records['marker'][0] == RAW_FRAME_MARKER)
        n_bad      = int(np.sum(records['marker'][1:] != RAW_LINE_MARKER))
        return n_bad, marker_ok

    def close(self):
        del self._mmap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-HYB compositor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybRecording:
    """
    Composites up to four HybRawFile objects into a full 2048×1024 frame.

    Only the loaded HYBs are populated; the rest of the canvas stays at 0.
    If only a subset of HYBs is loaded, the displayed extent is cropped to
    the bounding box of the present HYBs so the plot isn't dominated by
    empty black space.
    """

    def __init__(self, hyb_files: dict, every: int = 1):
        """
        Parameters
        ----------
        hyb_files : dict  {hyb_index: path_string}
        every      : int   show every Nth frame
        """
        if not hyb_files:
            raise ValueError("No HYB files provided.")

        self.readers = {}  # hyb → HybRawFile
        self.every   = max(1, every)

        n_per_hyb = {}
        for hyb, path in sorted(hyb_files.items()):
            r = HybRawFile(path, hyb, every=every)
            self.readers[hyb] = r
            n_per_hyb[hyb] = r.n
            print(f"  Loaded {HYB_TO_NAME[hyb]} ({path}): "
                  f"{r.n_total} frames  ({r.n} shown)")

        # n is the minimum across all HYBs (all files must provide the frame)
        self.n = min(n_per_hyb.values())
        if self.n == 0:
            raise ValueError("No displayable frames across loaded HYB files.")

        # Warn if counts differ
        if len(set(n_per_hyb.values())) > 1:
            print(f"  Warning: HYB frame counts differ: "
                  + ", ".join(f"{HYB_TO_NAME[h]}={v}"
                               for h, v in sorted(n_per_hyb.items()))
                  + f" — using min={self.n}")

        # Determine which rows/cols are covered and set crop extent
        self._compute_extent()

        # Dummy metadata fields (no per-frame metadata in RAW)
        self.mode         = 'HYB-RAW'
        self.trigger      = 'N/A'
        self.trigger_rate = -1.0
        self.created      = 'N/A'
        self.n_gaps       = 0

    def _compute_extent(self):
        """
        Compute the pixel bounding box of the loaded HYBs so the viewer
        can crop the display to the relevant region.

        Sets:
          self.row_lo, self.row_hi  — global row range [lo, hi)
          self.col_lo, self.col_hi  — global col range [lo, hi)
          self.height, self.width   — displayed sub-frame size
        """
        row_lo, row_hi = FRAME_H, 0
        col_lo, col_hi = FRAME_W, 0

        for hyb in self.readers:
            gy = GY_LUT[hyb]           # (512,) global rows for this HYB
            row_lo = min(row_lo, int(gy.min()))
            row_hi = max(row_hi, int(gy.max()) + 1)
            # cols: line 0 → outermost, line 1023 → innermost
            if hyb in (0, 1):          # right half: cols 1024-2047
                col_lo = min(col_lo, HYB_LOCAL_X)
                col_hi = max(col_hi, FRAME_W)
            else:                       # left half: cols 0-1023
                col_lo = min(col_lo, 0)
                col_hi = max(col_hi, HYB_LOCAL_X)

        self.row_lo = row_lo
        self.row_hi = row_hi
        self.col_lo = col_lo
        self.col_hi = col_hi
        self.height = row_hi - row_lo
        self.width  = col_hi - col_lo

    def frame(self, idx: int) -> np.ndarray:
        """
        Composite all loaded HYBs into a (height, width) float32 sub-frame.

        The sub-frame is cropped to the bounding box of the loaded HYBs.
        Missing HYBs are left at 0.
        """
        canvas = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

        for hyb, reader in self.readers.items():
            sub = reader.read_hyb_frame(idx)   # (HYB_LOCAL_Y, HYB_LOCAL_X) uint16
            # sub[local_y, line_num]
            # Place into canvas:
            #   global row = GY_LUT[hyb, local_y]
            #   global col = _line_to_gcol(hyb, line_num)
            gy = GY_LUT[hyb]   # (512,)
            if hyb in (0, 1):
                # cols 1024-2047, line 0→col 2047, line 1023→col 1024
                # i.e. global col = 2047 - line_num
                # sub[:, line_num] → canvas[gy, 2047-line_num]
                gcols = FRAME_W - 1 - np.arange(HYB_LOCAL_X, dtype=np.int32)
            else:
                # cols 0-1023, line 0→col 0, line 1023→col 1023
                gcols = np.arange(HYB_LOCAL_X, dtype=np.int32)

            canvas[gy[:, None], gcols[None, :]] = sub.astype(np.float32)

        return canvas[self.row_lo:self.row_hi, self.col_lo:self.col_hi]

    def meta(self, idx: int) -> dict:
        """Per-frame metadata: completeness across all loaded HYBs."""
        total_missing = 0
        marker_issues = []
        for hyb, reader in self.readers.items():
            n_miss, marker_ok = reader.completeness(idx)
            total_missing += n_miss
            if not marker_ok:
                marker_issues.append(HYB_TO_NAME[hyb])

        complete = (total_missing == 0 and len(marker_issues) == 0)
        return dict(
            idx           = idx,
            raw_idx       = idx,
            frame32       = None,
            timestamp     = None,
            complete      = int(complete),
            missing_lines = total_missing,
            marker_issues = marker_issues,
            hybs_loaded  = [HYB_TO_NAME[h] for h in sorted(self.readers)],
        )

    def percentile_clim(self, idx: int, lo: float = 1.0, hi: float = 99.0):
        img = self.frame(idx)
        # Exclude empty (zero) pixels from the percentile if only partial HYBs loaded
        vals = img[img > 0] if len(self.readers) < 4 else img.ravel()
        if len(vals) == 0:
            return 0.0, 1.0
        return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))

    def close(self):
        for r in self.readers.values():
            r.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Viewer  (same style as view_recording.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Viewer:
    def __init__(self, rec: HybRecording, args):
        self.rec        = rec
        self.args       = args
        self.idx        = max(0, min(args.start, rec.n - 1))
        self.autoscale  = (args.vmin is None and args.vmax is None)
        self._digit_buf = ''

        # initial colour limits
        if self.autoscale:
            vmin, vmax = rec.percentile_clim(self.idx)
        else:
            vmin = args.vmin if args.vmin is not None else 0
            vmax = args.vmax if args.vmax is not None else 65535

        # ── figure layout ─────────────────────────────────────────────
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#1a1a2e')

        self.ax_img  = self.fig.add_axes([0.05, 0.18, 0.72, 0.78])
        self.ax_img.set_facecolor('#0d0d1a')
        self.ax_cb   = self.fig.add_axes([0.80, 0.18, 0.03, 0.78])
        self.ax_info = self.fig.add_axes([0.85, 0.18, 0.14, 0.78])
        self.ax_info.axis('off')
        self.ax_sl   = self.fig.add_axes([0.05, 0.08, 0.90, 0.03])
        self.ax_sl.set_facecolor('#2a2a4e')
        self.ax_st   = self.fig.add_axes([0.05, 0.02, 0.90, 0.04])
        self.ax_st.axis('off')

        # ── image ─────────────────────────────────────────────────────
        # Extent in global sensor coordinates so axis labels show real pixels
        img0 = rec.frame(self.idx)
        self.im = self.ax_img.imshow(
            img0, origin='lower', interpolation='nearest',
            cmap=args.cmap, vmin=vmin, vmax=vmax,
            extent=[rec.col_lo, rec.col_hi, rec.row_lo, rec.row_hi],
        )
        self.ax_img.tick_params(colors='#aaaacc')
        for sp in self.ax_img.spines.values():
            sp.set_edgecolor('#444466')
        self.ax_img.set_xlabel('Global X (col)', color='#aaaacc')
        self.ax_img.set_ylabel('Global Y (row)', color='#aaaacc')

        # Draw HYB boundary lines on the image
        self._draw_hyb_boundaries()

        # ── colorbar ──────────────────────────────────────────────────
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cb)
        self.cbar.set_label('ADC counts', color='#aaaacc')
        self.cbar.ax.yaxis.set_tick_params(colors='#aaaacc')

        # ── title ─────────────────────────────────────────────────────
        self.title = self.ax_img.set_title('', color='#eeeeff', fontsize=11)

        # ── info panel ────────────────────────────────────────────────
        self.info_text = self.ax_info.text(
            0.05, 0.98, '', transform=self.ax_info.transAxes,
            color='#ccccee', fontsize=7.5, va='top', ha='left',
            fontfamily='monospace',
        )

        # ── slider ────────────────────────────────────────────────────
        self.slider = mwidgets.Slider(
            self.ax_sl, 'Frame', 0, rec.n - 1,
            valinit=self.idx, valstep=1,
            color='#4444aa', track_color='#2a2a4e',
        )
        self.slider.label.set_color('#aaaacc')
        self.slider.valtext.set_color('#aaaacc')
        self.slider.on_changed(self._on_slider)

        # ── status bar ────────────────────────────────────────────────
        self.status = self.ax_st.text(
            0.5, 0.5,
            '← → navigate  |  PgUp/Dn ×10  |  A auto-scale  |  '
            '+/- clim  |  S save PNG  |  Home/End  |  type+Enter jump  |  Q quit',
            transform=self.ax_st.transAxes,
            color='#888899', fontsize=8, ha='center', va='center',
        )

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event',     self._on_close)

        self._refresh()

    def _draw_hyb_boundaries(self):
        """Draw faint dividing lines at HYB boundaries within the visible area."""
        lkw = dict(color='#556688', linewidth=0.7, linestyle='--', alpha=0.6)
        r   = self.rec
        # Horizontal boundary (row 512) if both top and bottom are visible
        if r.row_lo < 512 < r.row_hi:
            self.ax_img.axhline(512, **lkw)
        # Vertical boundary (col 512) if both halves are visible
        if r.col_lo < 1024 < r.col_hi:
            self.ax_img.axvline(1024, **lkw)
        # HYB labels
        label_kw = dict(color='#8899bb', fontsize=7, alpha=0.7,
                        ha='center', va='center')
        centers = {
            'H0': (1536, 768), 'H1': (1536, 256),
            'H2': (512, 256), 'H3': (512, 768),
        }
        for name, (cx, cy) in centers.items():
            hyb = NAME_TO_HYB[name]
            if (hyb in self.rec.readers and
                    r.col_lo <= cx <= r.col_hi and
                    r.row_lo <= cy <= r.row_hi):
                self.ax_img.text(cx, cy, name,
                                 transform=self.ax_img.transData, **label_kw)

    # ── navigation ────────────────────────────────────────────────────

    def _goto(self, idx: int):
        self.idx = max(0, min(idx, self.rec.n - 1))
        self.slider.eventson = False
        self.slider.set_val(self.idx)
        self.slider.eventson = True
        self._refresh()

    def _on_slider(self, val):
        new = int(round(val))
        if new != self.idx:
            self.idx = new
            self._refresh()

    def _on_key(self, event):
        key = event.key

        if key and key.isdigit():
            self._digit_buf += key
            self._set_status(
                f'Jump to frame: {self._digit_buf}_  '
                f'(Enter to confirm, Esc to cancel)')
            return

        if key == 'enter' and self._digit_buf:
            self._goto(int(self._digit_buf))
            self._digit_buf = ''
            return

        if key == 'escape':
            if self._digit_buf:
                self._digit_buf = ''
                self._set_status_default()
            else:
                plt.close(self.fig)
            return

        self._digit_buf = ''

        if key in ('right', 'n', ' '):
            self._goto(self.idx + 1)
        elif key in ('left', 'p'):
            self._goto(self.idx - 1)
        elif key == 'home':
            self._goto(0)
        elif key == 'end':
            self._goto(self.rec.n - 1)
        elif key == 'pagedown':
            self._goto(self.idx + 10)
        elif key == 'pageup':
            self._goto(self.idx - 10)
        elif key == 'a':
            self.autoscale = not self.autoscale
            self._refresh()
        elif key in ('+', '='):
            self._scale_clim(1.1)
        elif key == '-':
            self._scale_clim(0.9)
        elif key == 's':
            self._save_png()
        elif key in ('q',):
            plt.close(self.fig)

    def _on_close(self, _):
        self.rec.close()

    # ── rendering ─────────────────────────────────────────────────────

    def _refresh(self):
        img  = self.rec.frame(self.idx)
        meta = self.rec.meta(self.idx)

        self.im.set_data(img)

        if self.autoscale:
            vals = img[img > 0] if len(self.rec.readers) < 4 else img.ravel()
            if len(vals) > 0:
                vmin = float(np.percentile(vals, 1))
                vmax = float(np.percentile(vals, 99))
            else:
                vmin, vmax = 0, 1
            self.im.set_clim(vmin, vmax)
        else:
            vmin, vmax = self.im.get_clim()

        complete = bool(meta['complete'])
        tc = '#ff9944' if not complete else '#eeeeff'
        status_str = ('complete' if complete
                      else f"PARTIAL  missing={meta['missing_lines']} lines"
                           + (f"  bad-marker={meta['marker_issues']}"
                              if meta['marker_issues'] else ''))
        self.title.set_color(tc)
        self.title.set_text(
            f"Frame {self.idx + 1} / {self.rec.n}   "
            f"HYBs: {','.join(meta['hybs_loaded'])}   {status_str}")

        hybs_str = ', '.join(meta['hybs_loaded'])
        info_lines = [
            'File info',
            '─────────',
            f"mode   : HYB-RAW",
            f"HYBs  : {hybs_str}",
            f"n_frame: {self.rec.n}",
            f"extent : X {self.rec.col_lo}-{self.rec.col_hi-1}",
            f"         Y {self.rec.row_lo}-{self.rec.row_hi-1}",
            '',
            'This frame',
            '──────────',
            f"idx    : {self.idx}",
            f"hw_num : N/A",
            f"time   : N/A",
            '',
            f"complete: {complete}",
            f"miss_ln : {meta['missing_lines']}",
            '',
            'Colour scale',
            '────────────',
            f"mode   : {'auto' if self.autoscale else 'fixed'}",
            f"vmin   : {vmin:.0f}",
            f"vmax   : {vmax:.0f}",
            '',
            'Image stats',
            '───────────',
            f"min    : {img.min():.0f}",
            f"max    : {img.max():.0f}",
            f"mean   : {img.mean():.1f}",
            f"std    : {img.std():.1f}",
        ]
        self.info_text.set_text('\n'.join(info_lines))
        self._set_status_default()
        self.fig.canvas.draw_idle()

    def _scale_clim(self, factor: float):
        self.autoscale = False
        vmin, vmax = self.im.get_clim()
        mid  = (vmin + vmax) / 2
        half = (vmax - vmin) / 2 * factor
        self.im.set_clim(mid - half, mid + half)
        self.fig.canvas.draw_idle()

    def _set_status(self, msg: str):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _set_status_default(self):
        self.status.set_text(
            '← → navigate  |  PgUp/Dn ×10  |  A auto-scale  |  '
            '+/- clim  |  S save PNG  |  Home/End  |  type+Enter jump  |  Q quit'
        )

    def _save_png(self):
        fname = f"frame_{self.idx:06d}_hyb.png"
        extent = self.ax_img.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(fname, dpi=150, bbox_inches=extent,
                         facecolor=self.ax_img.get_facecolor())
        self._set_status(f"Saved → {fname}  (press any key to continue)")
        print(f"Saved: {fname}")

    def show(self):
        plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        description='View per-HYB RAW files composited into a 2048×1024 frame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── file specification (three ways) ───────────────────────────────
    pos = ap.add_argument_group(
        'Positional (up to 4 files, auto-detected by H0/H1/H2/H3 in filename)')
    pos.add_argument('files', nargs='*', metavar='FILE',
                     help='Per-HYB RAW file(s). HYB identity is inferred '
                          'from "H0"/"H1"/"H2"/"H3" in the filename.')

    named = ap.add_argument_group('Named HYB files (explicit)')
    named.add_argument('--H0', metavar='FILE', default=None,
                       help='RAW file for H0 (top-right)')
    named.add_argument('--H1', metavar='FILE', default=None,
                       help='RAW file for H1 (bottom-right)')
    named.add_argument('--H2', metavar='FILE', default=None,
                       help='RAW file for H2 (bottom-left)')
    named.add_argument('--H3', metavar='FILE', default=None,
                       help='RAW file for H3 (top-left)')

    patt = ap.add_argument_group('Pattern shorthand')
    patt.add_argument('--pattern', metavar='PAT', default=None,
                      help='Filename pattern containing {H}, e.g. '
                           '"run001_H{H}.raw". Use with --hybs.')
    patt.add_argument('--hybs', metavar='LIST', default=None,
                      help='Comma-separated HYB names for --pattern, '
                           'e.g. "H0,H1,H2,H3" or "H1,H2". '
                           'Default: all four.')

    # ── display options ───────────────────────────────────────────────
    ap.add_argument('--start',  type=int,   default=0,
                    help='Starting frame index (default: 0)')
    ap.add_argument('--every',  type=int,   default=1,
                    help='Show every Nth frame (default: 1)')
    ap.add_argument('--vmin',   type=float, default=None,
                    help='Fixed colour scale minimum')
    ap.add_argument('--vmax',   type=float, default=None,
                    help='Fixed colour scale maximum')
    ap.add_argument('--cmap',   default='viridis',
                    help='Matplotlib colormap (default: viridis)')

    args = ap.parse_args()

    # ── resolve which files to open ───────────────────────────────────
    hyb_files = {}   # hyb_index → path

    # 1) Named flags
    for name in HYB_NAMES:
        path = getattr(args, name, None)
        if path:
            hyb_files[NAME_TO_HYB[name]] = path

    # 2) Pattern
    if args.pattern:
        if '{H}' not in args.pattern:
            print("Error: --pattern must contain {H}")
            sys.exit(1)
        names = ([s.strip().upper() for s in args.hybs.split(',')]
                 if args.hybs else HYB_NAMES)
        bad = [n for n in names if n not in NAME_TO_HYB]
        if bad:
            print(f"Error: unknown HYB(s) in --hybs: {bad}")
            sys.exit(1)
        for name in names:
            path = args.pattern.replace('{H}', name)
            hyb_files[NAME_TO_HYB[name]] = path

    # 3) Positional — infer HYB from filename
    for path in args.files:
        matched = False
        base = os.path.basename(path).upper()
        for name in HYB_NAMES:
            if name in base:
                hyb = NAME_TO_HYB[name]
                if hyb in hyb_files:
                    print(f"  Warning: {name} already specified; ignoring {path}")
                else:
                    hyb_files[hyb] = path
                matched = True
                break
        if not matched:
            print(f"  Warning: cannot determine HYB identity from filename "
                  f"'{path}' (expected H0/H1/H2/H3 in name) — skipping.")

    if not hyb_files:
        ap.print_help()
        print("\nError: no HYB files specified.")
        sys.exit(1)

    # ── check files exist ─────────────────────────────────────────────
    for hyb, path in sorted(hyb_files.items()):
        if not os.path.exists(path):
            print(f"Error: file not found: {path}  ({HYB_TO_NAME[hyb]})")
            sys.exit(1)

    # ── open and display ──────────────────────────────────────────────
    print(f"\nOpening {len(hyb_files)} HYB file(s):")
    try:
        rec = HybRecording(hyb_files, every=args.every)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n  Total frames : {rec.n}")
    print(f"  Display area : X {rec.col_lo}–{rec.col_hi-1}  "
          f"Y {rec.row_lo}–{rec.row_hi-1}  "
          f"({rec.width}×{rec.height} px)")
    if args.every > 1:
        print(f"  Showing every {args.every}th frame")
    print()

    try:
        viewer = Viewer(rec, args)
        viewer.show()
    except Exception as e:
        print(f"Viewer error: {e}")
        rec.close()
        sys.exit(1)


if __name__ == '__main__':
    main()

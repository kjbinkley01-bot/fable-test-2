"""Stacked, colour-coded stem waveforms with the beat grid and playhead.

Drawing strategy: the four lanes and the grid are painted into an off-screen
image with numpy (fast, no per-point Qt calls) that is three screens wide.
While playing, each frame just blits that image at a new offset and draws the
playhead line on top, so scrolling stays smooth. The image is rebuilt only
when the zoom changes or the view scrolls outside it.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..analysis.beats import BeatGrid
from ..analysis.waveform import BASE_BLOCK, Peaks
from ..config import SAMPLE_RATE, STEM_COLORS, STEMS

RULER_HEIGHT = 22
BACKGROUND = (0x14, 0x16, 0x1a)
LANE_ALT = (0x19, 0x1c, 0x21)
RULER_BG = (0x1f, 0x23, 0x29)
FOLLOW_POSITION = 0.3          # playhead sits 30% from the left while following
MIN_FRAMES_PER_PIXEL = float(BASE_BLOCK)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    c = QColor(hex_color)
    return c.red(), c.green(), c.blue()


class WaveformView(QWidget):
    seekRequested = Signal(int)          # frame
    viewChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(4 * 60 + RULER_HEIGHT)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._peaks: dict[str, Peaks] = {}
        self._grid: BeatGrid | None = None
        self._n_frames = 0
        self._scale = 1.0                       # amplitude scale so waveforms fill the lanes
        self._dimmed = {name: False for name in STEMS}
        self._view_start = 0.0                  # frame at the left edge
        self._fpp = 1.0                         # frames per pixel
        self._fit = True
        self.follow = True
        self._playhead = 0.0
        self._cache: QPixmap | None = None
        self._cache_start = 0.0
        self._cache_key: tuple | None = None

    # ------------------------------------------------------------------ data
    def set_track(self, peaks: dict[str, Peaks], grid: BeatGrid | None, n_frames: int) -> None:
        self._peaks = peaks
        self._grid = grid
        self._n_frames = n_frames
        peak = 1e-6
        for pk in peaks.values():
            if pk.levels:
                peak = max(peak, float(np.abs(pk.levels[-1]).max()))
        self._scale = 0.95 / peak
        self._playhead = 0.0
        self._fit = True
        self._apply_fit()
        self._invalidate()

    def set_grid(self, grid: BeatGrid | None) -> None:
        """Replace the beat grid (e.g. after the tempo was doubled) and redraw."""
        self._grid = grid
        self._invalidate()

    def clear(self) -> None:
        self._peaks = {}
        self._grid = None
        self._n_frames = 0
        self._invalidate()

    def set_dimmed(self, name: str, dimmed: bool) -> None:
        if self._dimmed.get(name) != dimmed:
            self._dimmed[name] = dimmed
            self._invalidate()

    def set_playhead(self, frame: float, playing: bool) -> None:
        self._playhead = frame
        if self.follow and playing and not self._fit:
            target = frame - FOLLOW_POSITION * self.width() * self._fpp
            self._set_view_start(target, redraw=False)
        self.update()

    # ------------------------------------------------------------------ view
    @property
    def has_track(self) -> bool:
        return self._n_frames > 0

    @property
    def frames_per_pixel(self) -> float:
        return self._fpp

    @property
    def view_start(self) -> float:
        return self._view_start

    def visible_frames(self) -> float:
        return self.width() * self._fpp

    def _fit_fpp(self) -> float:
        return max(self._n_frames / max(1, self.width()), 1.0)

    def _apply_fit(self) -> None:
        self._fpp = self._fit_fpp()
        self._view_start = 0.0

    def _clamp_start(self, start: float) -> float:
        max_start = max(0.0, self._n_frames - self.visible_frames())
        return min(max(start, 0.0), max_start)

    def _set_view_start(self, start: float, redraw: bool = True) -> None:
        start = self._clamp_start(start)
        if start != self._view_start:
            self._view_start = start
            self.viewChanged.emit()
            if redraw:
                self.update()

    def scroll_to(self, start: float) -> None:
        self._set_view_start(start)

    def zoom(self, factor: float, anchor_x: float | None = None) -> None:
        """factor > 1 zooms in. ``anchor_x`` is the pixel that stays put."""
        if not self.has_track:
            return
        if anchor_x is None:
            anchor_x = (self._playhead - self._view_start) / self._fpp
            if not 0 <= anchor_x <= self.width():
                anchor_x = self.width() / 2
        anchor_frame = self._view_start + anchor_x * self._fpp
        fit = self._fit_fpp()
        fpp = min(max(self._fpp / factor, min(MIN_FRAMES_PER_PIXEL, fit)), fit)
        self._fit = fpp >= fit * 0.999
        self._fpp = fpp
        self._view_start = self._clamp_start(anchor_frame - anchor_x * fpp)
        self._invalidate()
        self.viewChanged.emit()

    def zoom_fit(self) -> None:
        self._fit = True
        self._apply_fit()
        self._invalidate()
        self.viewChanged.emit()

    def is_fit(self) -> bool:
        return self._fit

    def _invalidate(self) -> None:
        self._cache = None
        self.update()

    # --------------------------------------------------------------- drawing
    def _lane_geometry(self) -> list[tuple[int, int]]:
        """(top, height) of each lane, below the ruler."""
        h = max(4, self.height() - RULER_HEIGHT)
        edges = [RULER_HEIGHT + round(i * h / len(STEMS)) for i in range(len(STEMS) + 1)]
        return [(edges[i], edges[i + 1] - edges[i]) for i in range(len(STEMS))]

    def _render_cache(self) -> None:
        width, height = self.width(), self.height()
        cache_w = width if self._fit else width * 3
        start = 0.0 if self._fit else self._view_start - width * self._fpp
        img = np.empty((height, cache_w, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = BACKGROUND[2], BACKGROUND[1], BACKGROUND[0]
        img[:RULER_HEIGHT, :, 0:3] = (RULER_BG[2], RULER_BG[1], RULER_BG[0])

        for i, (name, (top, lane_h)) in enumerate(zip(STEMS, self._lane_geometry())):
            if i % 2:
                img[top:top + lane_h, :, 0:3] = (LANE_ALT[2], LANE_ALT[1], LANE_ALT[0])
            pk = self._peaks.get(name)
            if pk is None:
                continue
            cols = pk.columns(start, self._fpp, cache_w) * self._scale
            mid = (lane_h - 1) / 2.0
            half = max(1.0, mid - 2)
            y_top = np.clip(np.floor(mid - cols[:, 1] * half), 0, lane_h - 1)
            y_bot = np.clip(np.ceil(mid - cols[:, 0] * half), 0, lane_h - 1)
            rows = np.arange(lane_h)[:, None]
            mask = (rows >= y_top[None, :]) & (rows <= y_bot[None, :])
            r, g, b = _rgb(STEM_COLORS[name])
            if self._dimmed.get(name):
                # muted / not soloed: draw it greyed out
                grey = 0.3 * r + 0.59 * g + 0.11 * b
                r, g, b = (int(0.35 * (0.3 * c + 0.7 * grey)) + 18 for c in (r, g, b))
            lane = img[top:top + lane_h]
            lane[mask] = (b, g, r, 255)

        self._draw_grid(img, start, cache_w)

        qimg = QImage(img.data, cache_w, height, cache_w * 4, QImage.Format.Format_RGB32)
        pixmap = QPixmap.fromImage(qimg)        # copies the pixels
        self._draw_ruler_labels(pixmap, start, cache_w)
        self._cache = pixmap
        self._cache_start = start
        self._cache_key = (width, height, self._fpp)

    def _grid_step(self) -> tuple[int, int]:
        """(show every Nth beat, show every Nth bar) for the current zoom."""
        grid = self._grid
        if grid is None or grid.bpm <= 0:
            return 0, 0
        beat_px = (60.0 / grid.bpm) * SAMPLE_RATE / self._fpp
        bar_px = beat_px * grid.beats_per_bar
        beat_step = 1 if beat_px >= 6 else 0
        bar_step = 1
        while bar_px * bar_step < 14:
            bar_step *= 2
        return beat_step, bar_step

    def _draw_grid(self, img: np.ndarray, start: float, cache_w: int) -> None:
        grid = self._grid
        if grid is None or len(grid.beats) == 0:
            return
        beat_step, bar_step = self._grid_step()
        end = start + cache_w * self._fpp

        def blend(times: np.ndarray, strength: float, top: int) -> None:
            frames = times * SAMPLE_RATE
            frames = frames[(frames >= start) & (frames < end)]
            if len(frames) == 0:
                return
            xs = np.unique(((frames - start) / self._fpp).astype(np.int64))
            xs = xs[(xs >= 0) & (xs < cache_w)]
            region = img[top:, xs, 0:3].astype(np.float32)
            img[top:, xs, 0:3] = (region + (255.0 - region) * strength).astype(np.uint8)

        if beat_step:
            blend(grid.beats, 0.10, RULER_HEIGHT)
        blend(grid.downbeats[::bar_step], 0.30, RULER_HEIGHT - 8)

    def _draw_ruler_labels(self, pixmap: QPixmap, start: float, cache_w: int) -> None:
        grid = self._grid
        painter = QPainter(pixmap)
        painter.setPen(QColor("#8a93a0"))
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)
        if grid is not None and len(grid.downbeats):
            _, bar_step = self._grid_step()
            bar_px = (60.0 / max(grid.bpm, 1e-6)) * grid.beats_per_bar * SAMPLE_RATE / self._fpp
            label_step = bar_step
            while bar_px * label_step < 44:
                label_step *= 2
            end = start + cache_w * self._fpp
            for i in range(0, len(grid.downbeats), label_step):
                frame = grid.downbeats[i] * SAMPLE_RATE
                if start <= frame < end:
                    x = (frame - start) / self._fpp
                    painter.drawText(QPointF(x + 3, 12), str(i + 1))
        painter.end()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not self.has_track:
            painter.fillRect(self.rect(), QColor(*BACKGROUND))
            painter.setPen(QColor("#6b7380"))
            font = QFont()
            font.setPixelSize(18)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Drop an audio file here\nMP3 · WAV · FLAC")
            painter.end()
            return
        width = self.width()
        key = (width, self.height(), self._fpp)
        if self._cache is None or self._cache_key != key:
            self._render_cache()
        offset = (self._view_start - self._cache_start) / self._fpp
        if offset < 0 or offset + width > self._cache.width():
            self._render_cache()
            offset = (self._view_start - self._cache_start) / self._fpp
        painter.drawPixmap(QPointF(0, 0), self._cache, QRectF(offset, 0, width, self.height()))
        # beyond the end of the song
        end_x = (self._n_frames - self._view_start) / self._fpp
        if end_x < width:
            painter.fillRect(QRectF(end_x, 0, width - end_x, self.height()), QColor(*BACKGROUND))
        # playhead
        x = (self._playhead - self._view_start) / self._fpp
        if -2 <= x <= width + 2:
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.end()

    # ------------------------------------------------------------ interaction
    def resizeEvent(self, event) -> None:
        if self._fit and self.has_track:
            self._apply_fit()
        else:
            self._view_start = self._clamp_start(self._view_start)
        self._invalidate()
        self.viewChanged.emit()

    def _frame_at(self, x: float) -> int:
        frame = self._view_start + x * self._fpp
        return int(min(max(frame, 0), max(0, self._n_frames - 1)))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.has_track:
            self.seekRequested.emit(self._frame_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self.has_track:
            self.seekRequested.emit(self._frame_at(event.position().x()))

    def wheelEvent(self, event) -> None:
        if not self.has_track:
            return
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier or delta == 0:
            # horizontal scroll
            step = event.angleDelta().x() or delta
            self.follow = False
            self.scroll_to(self._view_start - step / 120.0 * 0.15 * self.visible_frames())
            self.viewChanged.emit()
        else:
            self.zoom(1.25 ** (delta / 120.0), event.position().x())
        event.accept()

    def lane_tops(self) -> list[tuple[int, int]]:
        """Lane geometry, so the mixer strips can line up with the lanes."""
        return self._lane_geometry()

"""One mixer strip: name, mute, solo, volume fader and a level meter."""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

SLIDER_MAX = 100
SLIDER_UNITY = 80         # slider position of 0 dB
MAX_DB = 6.0
MIN_DB = -48.0


def slider_to_gain(value: int) -> float:
    """Fader position -> linear gain. Bottom is silence, 80 is 0 dB, top is +6 dB."""
    if value <= 0:
        return 0.0
    if value >= SLIDER_UNITY:
        db = MAX_DB * (value - SLIDER_UNITY) / (SLIDER_MAX - SLIDER_UNITY)
    else:
        db = MIN_DB * (SLIDER_UNITY - value) / (SLIDER_UNITY - 1)
    return 10.0 ** (db / 20.0)


def gain_to_text(gain: float) -> str:
    if gain <= 0.0:
        return "-inf dB"
    return f"{20.0 * math.log10(gain):+.1f} dB"


class LevelMeter(QWidget):
    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._level = 0.0
        self.setFixedHeight(5)

    def set_level(self, peak: float) -> None:
        # instant rise, smooth fall
        target = min(1.0, peak)
        level = target if target > self._level else self._level * 0.85
        if level < 1e-3:
            level = 0.0
        if level != self._level:
            self._level = level
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2a2f37"))
        if self._level > 0:
            # show on a dB scale: -48 dB .. 0 dB across the width
            db = 20.0 * math.log10(max(self._level, 1e-6))
            frac = min(1.0, max(0.0, (db + 48.0) / 48.0))
            painter.fillRect(QRectF(0, 0, self.width() * frac, self.height()), self._color)
        painter.end()


class ResetSlider(QSlider):
    """A slider that jumps back to unity on double-click."""

    def mouseDoubleClickEvent(self, event) -> None:
        self.setValue(SLIDER_UNITY)


class StemStrip(QFrame):
    gainChanged = Signal(int, float)     # stem index, linear gain
    muteChanged = Signal(int, bool)
    soloChanged = Signal(int, bool)

    def __init__(self, index: int, name: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("strip")
        self.index = index
        self.name = name

        swatch = QFrame()
        swatch.setFixedWidth(5)
        swatch.setStyleSheet(f"background: {color}; border: none;")

        self.label = QLabel(name.upper())
        self.label.setStyleSheet(f"color: {color}; font-weight: 700; letter-spacing: 1px;")
        self.db_label = QLabel("+0.0 dB")
        self.db_label.setObjectName("dim")
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.mute = QPushButton("M")
        self.mute.setObjectName("mute")
        self.mute.setCheckable(True)
        self.mute.setToolTip(f"Mute {name}  (key {index + 1})")
        self.mute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.solo = QPushButton("S")
        self.solo.setObjectName("solo")
        self.solo.setCheckable(True)
        self.solo.setToolTip(f"Solo {name}  (Shift+{index + 1})")
        self.solo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.fader = ResetSlider(Qt.Orientation.Horizontal)
        self.fader.setRange(0, SLIDER_MAX)
        self.fader.setValue(SLIDER_UNITY)
        self.fader.setToolTip("Volume. Double-click to reset to 0 dB.")
        self.fader.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.meter = LevelMeter(color)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.label, 1)
        top.addWidget(self.mute)
        top.addWidget(self.solo)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self.fader, 1)
        bottom.addWidget(self.db_label)
        self.db_label.setFixedWidth(58)

        body = QVBoxLayout()
        body.setContentsMargins(10, 6, 10, 6)
        body.setSpacing(4)
        body.addStretch(1)
        body.addLayout(top)
        body.addLayout(bottom)
        body.addWidget(self.meter)
        body.addStretch(1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(swatch)
        outer.addLayout(body, 1)

        self.fader.valueChanged.connect(self._on_fader)
        self.mute.toggled.connect(lambda on: self.muteChanged.emit(self.index, on))
        self.solo.toggled.connect(lambda on: self.soloChanged.emit(self.index, on))

    def _on_fader(self, value: int) -> None:
        gain = slider_to_gain(value)
        self.db_label.setText(gain_to_text(gain))
        self.gainChanged.emit(self.index, gain)

    @property
    def gain(self) -> float:
        return slider_to_gain(self.fader.value())

    def reset(self) -> None:
        self.fader.setValue(SLIDER_UNITY)
        self.mute.setChecked(False)
        self.solo.setChecked(False)

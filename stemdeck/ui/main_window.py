"""The StemDeck window: drop zone, waveforms, mixer strips, transport, export."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollBar,
    QVBoxLayout, QWidget,
)

from ..audio_io import SUPPORTED_EXTENSIONS, WAV_SUBTYPES
from ..config import MODELS, SAMPLE_RATE, STEM_COLORS, STEMS
from ..engine.client import EngineClient, EngineError
from ..settings import load_settings, save_settings
from .stem_strip import StemStrip
from .waveform_view import RULER_HEIGHT, WaveformView
from .workers import ExportWorker, LoadWorker, ProbeWorker

logger = logging.getLogger(__name__)

DECK = 0                      # milestone 1 drives a single deck
STAGE_TEXT = {
    "hash": "Identifying file",
    "decode": "Reading audio",
    "separate": "Separating stems",
    "analyze": "Detecting BPM, key and beats",
    "save": "Saving to library",
    "load": "Loading from library",
}


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class ExportDialog(QDialog):
    """Choose what to export: a mix in one file, or one file per stem."""

    def __init__(self, audible: dict[str, bool], default_subtype: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export WAV")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        self.mode_mix = QRadioButton("One file: mix of the selected stems")
        self.mode_separate = QRadioButton("Separate file for each selected stem")
        self.mode_mix.setChecked(True)
        layout.addWidget(self.mode_mix)
        layout.addWidget(self.mode_separate)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Stems to include:"))
        self.checks: dict[str, QCheckBox] = {}
        row = QHBoxLayout()
        for name in STEMS:
            box = QCheckBox(name.capitalize())
            box.setChecked(audible.get(name, True))
            box.setStyleSheet(f"color: {STEM_COLORS[name]};")
            self.checks[name] = box
            row.addWidget(box)
        layout.addLayout(row)

        self.use_faders = QCheckBox("Use the current fader levels")
        self.use_faders.setChecked(True)
        self.use_faders.setToolTip("Off: every selected stem at its original level (0 dB).")
        layout.addWidget(self.use_faders)
        self.mode_separate.toggled.connect(lambda on: self.use_faders.setEnabled(not on))

        fmt = QHBoxLayout()
        fmt.addWidget(QLabel("Format:"))
        self.subtype = QComboBox()
        for label, subtype in WAV_SUBTYPES.items():
            self.subtype.addItem(f"WAV {label}", subtype)
        index = self.subtype.findData(default_subtype)
        self.subtype.setCurrentIndex(max(0, index))
        fmt.addWidget(self.subtype, 1)
        layout.addLayout(fmt)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected(self) -> list[str]:
        return [name for name in STEMS if self.checks[name].isChecked()]


class MainWindow(QMainWindow):
    def __init__(self, engine: EngineClient | None = None) -> None:
        super().__init__()
        self.setWindowTitle("StemDeck")
        self.resize(1280, 640)
        self.setAcceptDrops(True)
        self.settings = load_settings()
        self.track = None
        self._worker: LoadWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._pending_serial = 0
        self._engine_error: str | None = None
        self._was_playing = False
        self._last_blocks = -1
        self._last_block_time = time.monotonic()
        self._stall_reported = False

        self.engine = engine or EngineClient(
            device=self.settings.get("output_device"),
            latency=float(self.settings.get("audio_latency_ms", 40)) / 1000.0,
        )
        self._build_ui()
        self._start_engine()
        self._probe = ProbeWorker(self.settings.get("device", "auto"), self)
        self._probe.result.connect(self._on_device_probed)
        QTimer.singleShot(0, self._probe.start)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ---- header: title, BPM, key ------------------------------------
        header = QHBoxLayout()
        self.title_label = QLabel("No song loaded")
        self.title_label.setObjectName("title")
        self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.title_label, 1)

        def stat(caption: str) -> QLabel:
            box = QVBoxLayout()
            box.setSpacing(0)
            cap = QLabel(caption)
            cap.setObjectName("dim")
            value = QLabel("–")
            value.setObjectName("big")
            box.addWidget(cap)
            box.addWidget(value)
            header.addLayout(box)
            header.addSpacing(24)
            return value

        self.bpm_label = stat("BPM")
        # Whether a song is "87" or "174" BPM is a convention; let the user flip it.
        self.half_button = QPushButton("÷2")
        self.double_button = QPushButton("×2")
        for button, factor, tip in ((self.half_button, 0.5, "Count the tempo at half speed"),
                                    (self.double_button, 2.0, "Count the tempo at double speed "
                                                              "(e.g. drum & bass shown as 87)")):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setObjectName("small")
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, f=factor: self._rescale_tempo(f))
        header.insertWidget(header.count() - 1, self.half_button)
        header.insertWidget(header.count() - 1, self.double_button)
        self.key_label = stat("KEY")

        self.model_combo = QComboBox()
        self.model_combo.addItem("Best quality", MODELS["quality"])
        self.model_combo.addItem("Fast", MODELS["fast"])
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(self.settings["model"])))
        self.model_combo.setToolTip(
            "Separation model for newly loaded songs.\n"
            "Best quality: four specialised networks, cleanest stems.\n"
            "Fast: one network, about four times quicker.")
        self.model_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.model_combo.currentIndexChanged.connect(
            lambda: save_settings({"model": self.model_combo.currentData()}))
        self.open_button = QPushButton("Open…")
        self.open_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.open_button.clicked.connect(self.open_dialog)
        header.addWidget(self.model_combo)
        header.addWidget(self.open_button)
        root.addLayout(header)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)

        # ---- strips + waveforms -----------------------------------------
        body = QHBoxLayout()
        body.setSpacing(0)
        strips_col = QVBoxLayout()
        strips_col.setSpacing(0)
        strips_col.setContentsMargins(0, 0, 0, 0)
        corner = QFrame()
        corner.setFixedHeight(RULER_HEIGHT)
        strips_col.addWidget(corner)
        self.strips: list[StemStrip] = []
        for i, name in enumerate(STEMS):
            strip = StemStrip(i, name, STEM_COLORS[name])
            strip.gainChanged.connect(self._on_gain)
            strip.muteChanged.connect(self._on_mute)
            strip.soloChanged.connect(self._on_solo)
            strips_col.addWidget(strip, 1)
            self.strips.append(strip)
        strips_widget = QWidget()
        strips_widget.setLayout(strips_col)
        strips_widget.setFixedWidth(230)
        body.addWidget(strips_widget)

        wave_col = QVBoxLayout()
        wave_col.setSpacing(0)
        self.waveform = WaveformView()
        self.waveform.seekRequested.connect(self._on_seek)
        self.waveform.viewChanged.connect(self._sync_scrollbar)
        wave_col.addWidget(self.waveform, 1)
        body.addLayout(wave_col, 1)
        root.addLayout(body, 1)

        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scrollbar.valueChanged.connect(self._on_scrollbar)
        scroll_row = QHBoxLayout()
        scroll_row.addSpacing(230)
        scroll_row.addWidget(self.scrollbar, 1)
        root.addLayout(scroll_row)

        # ---- transport ----------------------------------------------------
        transport = QHBoxLayout()
        self.play_button = QPushButton("▶  Play")
        self.play_button.setObjectName("play")
        self.play_button.setCheckable(True)
        self.play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_button.setToolTip("Play / pause  (Space)")
        self.play_button.clicked.connect(self.toggle_play)
        self.rewind_button = QPushButton("⏮")
        self.rewind_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rewind_button.setToolTip("Back to the start  (Home)")
        self.rewind_button.clicked.connect(lambda: self._on_seek(0))
        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setObjectName("big")
        self.time_label.setMinimumWidth(210)
        transport.addWidget(self.rewind_button)
        transport.addWidget(self.play_button)
        transport.addSpacing(12)
        transport.addWidget(self.time_label)
        transport.addStretch(1)

        self.follow_box = QCheckBox("Follow playhead")
        self.follow_box.setChecked(True)
        self.follow_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.follow_box.toggled.connect(self._on_follow)
        transport.addWidget(self.follow_box)
        for text, tip, handler in (
            ("–", "Zoom out", lambda: self.waveform.zoom(1 / 1.6)),
            ("+", "Zoom in  (or use the mouse wheel)", lambda: self.waveform.zoom(1.6)),
            ("Fit", "Show the whole song", self.waveform.zoom_fit),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(handler)
            transport.addWidget(button)
        transport.addSpacing(16)
        self.reset_button = QPushButton("Reset mix")
        self.reset_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reset_button.setToolTip("All faders to 0 dB, clear mutes and solos")
        self.reset_button.clicked.connect(self._reset_mix)
        self.export_button = QPushButton("Export…")
        self.export_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_button.setToolTip("Export stems or a custom mix as WAV  (Ctrl+E)")
        self.export_button.clicked.connect(self.export_dialog)
        transport.addWidget(self.reset_button)
        transport.addWidget(self.export_button)
        root.addLayout(transport)

        # ---- progress + status ------------------------------------------
        status_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(260)
        self.progress.hide()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cancel_button.clicked.connect(self._cancel_load)
        self.cancel_button.hide()
        self.status_label = QLabel("Drop a song on the window to begin.")
        self.status_label.setObjectName("dim")
        self.audio_label = QLabel("")
        self.audio_label.setObjectName("dim")
        self.audio_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.device_label = QLabel("Separation: checking GPU …")
        self.device_label.setObjectName("dim")
        self.device_label.setToolTip("What new songs will be separated on")
        status_row.addWidget(self.progress)
        status_row.addWidget(self.cancel_button)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.device_label)
        status_row.addSpacing(18)
        status_row.addWidget(self.audio_label)
        root.addLayout(status_row)

        # ---- keyboard ------------------------------------------------------
        def shortcut(keys, handler) -> None:
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(handler)

        shortcut(Qt.Key.Key_Space, self.toggle_play)
        shortcut(Qt.Key.Key_Home, lambda: self._on_seek(0))
        shortcut(Qt.Key.Key_Left, lambda: self._nudge(-1))
        shortcut(Qt.Key.Key_Right, lambda: self._nudge(+1))
        shortcut("Ctrl+O", self.open_dialog)
        shortcut("Ctrl+E", self.export_dialog)
        for i in range(len(STEMS)):
            shortcut(str(i + 1), lambda i=i: self.strips[i].mute.toggle())
            shortcut(f"Shift+{i + 1}", lambda i=i: self.strips[i].solo.toggle())
        self._set_track_controls(False)

    # ----------------------------------------------------------------- engine
    def _start_engine(self) -> None:
        try:
            info = self.engine.start()
        except EngineError as exc:
            self._engine_error = str(exc)
            self.audio_label.setText("Audio: not available")
            self._show_banner(f"Audio engine could not start: {exc}. "
                              "Separation, analysis and export still work.")
            return
        self._engine_error = None
        if info.get("backend") == "null":
            self.audio_label.setText("Audio: no output device")
            if info.get("note"):
                self._show_banner(f"No sound card output: {info['note']}. Playback is silent.")
        else:
            self.audio_label.setText(
                f"Audio: {info.get('device', '?')} · {info.get('host_api', '')} · "
                f"{1000 * float(info.get('latency', 0)):.0f} ms")

    def _on_device_probed(self, device, reason: str) -> None:
        """Start-up check finished: say plainly what separation will run on."""
        if device is not None and device.is_gpu:
            self.device_label.setText(f"Separation: GPU · {device.name}")
            self.device_label.setStyleSheet("")
            self.device_label.setToolTip("New songs are separated on the graphics card.")
            return
        self.device_label.setText("Separation: CPU (slow)")
        self.device_label.setStyleSheet("color: #ffb020;")
        self.device_label.setToolTip(reason or "The CPU was selected in the settings.")
        if reason:
            logger.warning("GPU not usable: %s", reason)
            self._show_banner(f"The graphics card is not being used, so separating a new song "
                              f"takes minutes instead of seconds. Reason: {reason}")

    def _show_banner(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.show()

    # ------------------------------------------------------------ drag & drop
    @staticmethod
    def _dropped_file(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in SUPPORTED_EXTENSIONS + (".m4a", ".aac"):
                return path
        return None

    def dragEnterEvent(self, event) -> None:
        if self._dropped_file(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = self._dropped_file(event)
        if path:
            event.acceptProposedAction()
            self.load_file(path)

    def open_dialog(self) -> None:
        pattern = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Open a song", "", f"Audio ({pattern});;All files (*)")
        if path:
            self.load_file(path)

    # ---------------------------------------------------------------- loading
    def load_file(self, path: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Still working on the previous song - cancel it first.")
            return
        self.banner.hide()
        self.settings = load_settings()      # pick up a configuration saved by the benchmark
        options = {
            "device": self.settings.get("device", "auto"),
            "precision": self.settings.get("precision", "fp32"),
            "batch_size": int(self.settings.get("batch_size", 0)),
            "overlap": float(self.settings.get("overlap", 0.25)),
        }
        self._worker = LoadWorker(path, self.model_combo.currentData(), options, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.status_label.setText(f"Opening {Path(path).name} …")
        self._worker.start()

    def _cancel_load(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling …")

    def _on_progress(self, stage: str, fraction: float, message: str) -> None:
        text = STAGE_TEXT.get(stage, message)
        if stage == "separate":
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * fraction))
            text += f" … {int(100 * fraction)}%"
        else:
            self.progress.setRange(0, 0)          # busy indicator for the short stages
        self.status_label.setText(text)

    def _finish_progress(self) -> None:
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.cancel_button.hide()

    def _on_cancelled(self) -> None:
        self._finish_progress()
        self.status_label.setText("Cancelled.")

    def _on_failed(self, message: str) -> None:
        self._finish_progress()
        self.status_label.setText("Could not open the song.")
        QMessageBox.warning(self, "StemDeck", f"Could not open the song:\n\n{message}")

    def _on_loaded(self, track) -> None:
        self._finish_progress()
        self.track = track
        self.engine.pause(DECK)
        stems = [track.stems[name] for name in STEMS]
        self._pending_serial = self.engine.load(DECK, stems) if self.engine.is_alive() else 0
        for strip in self.strips:      # re-send the mixer state for the new song
            self.engine.set_fader(DECK, strip.index, strip.gain)
            self.engine.set_mute(DECK, strip.index, strip.mute.isChecked())
            self.engine.set_solo(DECK, strip.index, strip.solo.isChecked())
        self.waveform.set_track(track.peaks, track.analysis.grid, track.n_frames)
        self._update_dimming()
        self._sync_scrollbar()

        analysis = track.analysis
        self.title_label.setText(track.title)
        self.setWindowTitle(f"{track.title} – StemDeck")
        self._show_tempo()
        self.key_label.setText(analysis.key.display)
        alts = ", ".join(f"{a['name']} / {a['camelot']} ({a['probability']:.0%})"
                         for a in analysis.key.alternatives[:2])
        self.key_label.setToolTip(f"Confidence {analysis.key.confidence:.0%}\nNext best: {alts}")
        self._set_track_controls(True)

        sep = track.separation or {}
        where = "GPU" if sep.get("device_kind") == "gpu" else "CPU"
        made = f"{where} in {sep.get('separate_seconds', 0):.1f} s" if sep else ""
        if track.from_cache:
            text = f"Loaded from library in {track.load_seconds:.1f} s"
            if made:
                text += f" · stems were made on {made}"
        else:
            text = (f"Separated on {where} ({sep.get('device_name', '?')}) in "
                    f"{sep.get('separate_seconds', 0):.1f} s · {sep.get('realtime_factor', 0):.1f}× realtime"
                    f" · ready in {track.load_seconds:.1f} s")
        self.status_label.setText(text)
        details = [f"{stage}: {seconds:.1f} s" for stage, seconds in (track.timings or {}).items()]
        if sep.get("fallback_reason"):
            details.append(f"GPU not used: {sep['fallback_reason']}")
        self.status_label.setToolTip("Time per step when this song was first processed\n" + "\n".join(details)
                                     if details else "")
        reason = sep.get("fallback_reason")
        if reason and not track.from_cache:
            self._show_banner(f"GPU was not used for this song, it ran on the CPU instead. Reason: {reason}")

    def _set_track_controls(self, enabled: bool) -> None:
        for widget in (self.play_button, self.rewind_button, self.export_button, self.reset_button,
                       self.half_button, self.double_button):
            widget.setEnabled(enabled)

    def _show_tempo(self) -> None:
        analysis = self.track.analysis
        self.bpm_label.setText(f"{analysis.bpm:.2f}" if analysis.bpm > 0 else "–")
        self.bpm_label.setToolTip("\n".join(
            ["Fixed tempo grid" if analysis.grid.is_constant else "Tempo varies: grid follows detected beats"]
            + analysis.grid.notes))

    def _rescale_tempo(self, factor: float) -> None:
        """÷2 / ×2: recount the beat grid at half or double time, and remember it."""
        if self.track is None or self.track.analysis.bpm <= 0:
            return
        analysis = self.track.analysis
        if not 20.0 <= analysis.bpm * factor <= 400.0:
            return
        analysis.grid = analysis.grid.rescaled(factor)
        analysis.bpm = analysis.grid.bpm
        self.waveform.set_grid(analysis.grid)
        self._show_tempo()
        try:
            from ..library import save_analysis

            save_analysis(self.track)
        except OSError as exc:
            logger.warning("could not save the tempo change: %s", exc)

    # -------------------------------------------------------------- transport
    def toggle_play(self) -> None:
        if self.track is None:
            self.play_button.setChecked(False)
            return
        status = self.engine.status().decks[DECK]
        if status.playing:
            self.engine.pause(DECK)
        else:
            self.waveform.follow = self.follow_box.isChecked()
            self.engine.play(DECK)

    def _on_seek(self, frame: int) -> None:
        if self.track is not None:
            self.engine.seek(DECK, frame)
            if not self.engine.status().decks[DECK].playing:
                self.waveform.set_playhead(frame, False)

    def _nudge(self, direction: int) -> None:
        """Left / Right: jump one bar (or 5 s when there is no grid)."""
        if self.track is None:
            return
        grid = self.track.analysis.grid
        step = 5.0
        if grid.bpm > 0:
            step = 60.0 / grid.bpm * grid.beats_per_bar
        position = self.engine.status().decks[DECK].position
        self._on_seek(int(max(0, position + direction * step * SAMPLE_RATE)))

    def _on_follow(self, on: bool) -> None:
        self.waveform.follow = on

    def _on_scrollbar(self, value: int) -> None:
        if self.scrollbar.isSliderDown():
            self.follow_box.setChecked(False)
        self.waveform.scroll_to(float(value))

    def _sync_scrollbar(self) -> None:
        bar = self.scrollbar
        bar.blockSignals(True)
        if self.track is None or self.waveform.is_fit():
            bar.setRange(0, 0)
        else:
            visible = int(self.waveform.visible_frames())
            bar.setRange(0, max(0, self.track.n_frames - visible))
            bar.setPageStep(visible)
            bar.setSingleStep(max(1, visible // 20))
            bar.setValue(int(self.waveform.view_start))
        bar.blockSignals(False)

    # ------------------------------------------------------------------ mixer
    def _on_gain(self, stem: int, gain: float) -> None:
        self.engine.set_fader(DECK, stem, gain)

    def _on_mute(self, stem: int, on: bool) -> None:
        self.engine.set_mute(DECK, stem, on)
        self._update_dimming()

    def _on_solo(self, stem: int, on: bool) -> None:
        self.engine.set_solo(DECK, stem, on)
        self._update_dimming()

    def _audible(self) -> dict[str, bool]:
        any_solo = any(s.solo.isChecked() for s in self.strips)
        return {s.name: (s.solo.isChecked() or not any_solo) and not s.mute.isChecked()
                for s in self.strips}

    def _update_dimming(self) -> None:
        for name, audible in self._audible().items():
            self.waveform.set_dimmed(name, not audible)

    def _reset_mix(self) -> None:
        for strip in self.strips:
            strip.reset()

    # ----------------------------------------------------------------- export
    def export_dialog(self) -> None:
        if self.track is None:
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        dialog = ExportDialog(self._audible(), self.settings.get("export_subtype", "PCM_24"), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        names = dialog.selected()
        if not names:
            QMessageBox.information(self, "StemDeck", "Select at least one stem to export.")
            return
        subtype = dialog.subtype.currentData()
        save_settings({"export_subtype": subtype})
        self.settings["export_subtype"] = subtype
        track = self.track
        start_dir = str(Path(track.source_path).parent) if track.source_path else ""

        from ..engine.export import export_mix, export_stems

        if dialog.mode_separate.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the stems", start_dir)
            if not folder:
                return
            job = lambda: export_stems(track.stems, folder, track.title, names, subtype)  # noqa: E731
        else:
            suffix = "mix" if len(names) == len(STEMS) else "+".join(names)
            default = str(Path(start_dir) / f"{track.title} - {suffix}.wav")
            path, _ = QFileDialog.getSaveFileName(self, "Export mix", default, "WAV (*.wav)")
            if not path:
                return
            if not path.lower().endswith(".wav"):
                path += ".wav"
            faders = {s.name: s.gain for s in self.strips} if dialog.use_faders.isChecked() else None
            mutes = {name: name not in names for name in STEMS}
            job = lambda: [export_mix(track.stems, path, faders=faders, mutes=mutes, subtype=subtype)]  # noqa: E731

        self.status_label.setText("Exporting …")
        self._export_worker = ExportWorker(job, self)
        self._export_worker.done.connect(self._on_exported)
        self._export_worker.failed.connect(
            lambda msg: QMessageBox.warning(self, "StemDeck", f"Export failed:\n\n{msg}"))
        self._export_worker.start()

    def _on_exported(self, infos) -> None:
        if not infos:
            return
        text = f"Exported {len(infos)} file{'s' if len(infos) != 1 else ''} to {Path(infos[0].path).parent}"
        quieter = max(info.attenuated_db for info in infos)
        if quieter > 0.05:
            text += f"  (turned down {quieter:.1f} dB to avoid clipping)"
        self.status_label.setText(text)

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        engine = self.engine
        for kind, payload in engine.poll_events():
            if kind == "error":
                logger.error("engine: %s", payload)
                self._show_banner(f"Audio engine: {payload}")
        if self._engine_error is None and engine._proc is not None and not engine.is_alive():
            self._engine_error = "audio engine stopped unexpectedly"
            self._show_banner("The audio engine stopped unexpectedly. Restart StemDeck to restore playback.")
        status = engine.status()
        deck = status.decks[DECK]
        if self.track is not None:
            # ignore engine state that still belongs to the previous song
            current = deck.serial == self._pending_serial
            position = deck.position if current else 0.0
            playing = bool(deck.playing and current)
            self.waveform.set_playhead(position, playing)
            self.time_label.setText(f"{format_time(position / SAMPLE_RATE)} / {format_time(self.track.duration)}")
            if playing != self._was_playing:
                self._was_playing = playing
                self.play_button.setChecked(playing)
                self.play_button.setText("❚❚  Pause" if playing else "▶  Play")
            for strip, peak in zip(self.strips, deck.stem_peaks):
                strip.meter.set_level(peak if playing else 0.0)
        if status.running and status.underflows:
            self.audio_label.setToolTip(f"Audio dropouts so far: {status.underflows}")
        # The sound card stopped asking for audio (unplugged, driver reset ...).
        now = time.monotonic()
        if status.blocks != self._last_blocks:
            self._last_blocks = status.blocks
            self._last_block_time = now
        elif (status.running and deck.playing and now - self._last_block_time > 1.5
              and not self._stall_reported):
            self._stall_reported = True
            self._show_banner("The audio device stopped responding (unplugged or changed?). "
                              "Restart StemDeck to resume playback.")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        if self._probe.isRunning():
            self._probe.wait(5000)
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(10000)
        self.engine.stop()
        super().closeEvent(event)

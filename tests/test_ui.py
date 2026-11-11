"""The window, driven off-screen: load, play, pause, seek, run to the end.

Uses the silent real-time audio backend. Exercises the 60-times-a-second
status tick in every transport state, which is where a value of the wrong
type coming back from the audio engine would surface.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from stemdeck.analysis import AnalysisResult, BeatGrid, KeyResult
from stemdeck.analysis.waveform import compute_peaks
from stemdeck.config import SAMPLE_RATE, STEMS
from stemdeck.engine.client import DeckStatus, EngineClient
from stemdeck.library import Track


def make_track(seconds: float) -> Track:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    stems = {name: np.repeat((0.2 * np.sin(2 * np.pi * f * t))[:, None], 2, 1).astype(np.float32)
             for name, f in zip(STEMS, (440, 110, 55, 330))}
    beats = np.arange(0.1, seconds, 0.5)
    grid = BeatGrid(120.0, beats, beats[::4], 4, True, 0.1, 1.0, 120.0)
    analysis = AnalysisResult(120.0, KeyResult(9, "minor", 0.8), grid, seconds, "mix")
    return Track("t" * 64, "Test Song", "", "htdemucs", stems,
                 {k: compute_peaks(v) for k, v in stems.items()}, analysis, {})


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setenv("STEMDECK_HOME", str(tmp_path / "home"))
    from PySide6.QtWidgets import QApplication

    from stemdeck.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    errors = []
    monkeypatch.setattr("sys.excepthook", lambda *exc: errors.append(exc))
    win = MainWindow(EngineClient(backend="null", block_size=512))
    win.show()
    win.errors = errors
    win.app = app
    yield win
    win.close()


def pump(win, seconds: float, until=None) -> bool:
    """Run the Qt event loop (timers, repaints) for a while."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        win.app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.005)
    return until is None


def test_status_values_are_plain_python_types(window):
    window._on_loaded(make_track(3.0))
    pump(window, 0.3)
    for playing_wanted in (True, False):
        window.toggle_play()
        assert pump(window, 2.0, lambda: window.engine.status().decks[0].playing == playing_wanted)
        status = window.engine.status()
        deck = status.decks[0]
        assert type(status.running) is bool
        assert type(deck.playing) is bool and type(deck.ended) is bool
        assert type(deck.position) is float and type(deck.n_frames) is int
        assert all(type(p) is float for p in deck.stem_peaks)


def test_play_pause_seek_and_end_of_song_raise_nothing(window):
    window._on_loaded(make_track(1.5))
    pump(window, 0.2)
    assert window.play_button.isEnabled() and not window.play_button.isChecked()

    window.toggle_play()                                        # play
    assert pump(window, 2.0, lambda: window.play_button.isChecked())
    assert "Pause" in window.play_button.text()

    window.toggle_play()                                        # pause
    assert pump(window, 2.0, lambda: not window.play_button.isChecked())
    assert "Play" in window.play_button.text()
    held = window.engine.status().decks[0].position
    pump(window, 0.3)
    assert window.engine.status().decks[0].position == held

    window._on_seek(int(0.2 * SAMPLE_RATE))                     # seek while paused
    pump(window, 0.2)
    window.toggle_play()                                        # play to the end of the song
    assert pump(window, 2.0, lambda: window.play_button.isChecked())
    assert pump(window, 5.0, lambda: window.engine.status().decks[0].ended)
    assert pump(window, 2.0, lambda: not window.play_button.isChecked())
    assert "Play" in window.play_button.text()

    window.toggle_play()                                        # play again restarts from the top
    assert pump(window, 2.0, lambda: window.play_button.isChecked())
    window.strips[0].mute.setChecked(True)
    window.strips[2].solo.setChecked(True)
    window._rescale_tempo(2.0)
    window.waveform.zoom(4.0)
    pump(window, 0.3)
    assert window.errors == [], window.errors


def test_default_deck_status_is_safe():
    deck = DeckStatus()
    assert deck.playing is False and deck.position == 0.0

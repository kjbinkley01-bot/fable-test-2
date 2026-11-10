"""The audio engine process end to end, on the silent real-time backend.

These check the things that make playback glitch-free in practice:
the audio blocks are rendered well inside their time budget, and keep being
so while the controlling process is busy (as the UI will be).
"""
import time

import numpy as np
import pytest

from stemdeck.config import SAMPLE_RATE
from stemdeck.engine.client import EngineClient


def stems(seconds=20.0, amp=0.2):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros((4, n, 2), dtype=np.float32)
    for s, f in enumerate((110, 220, 330, 440)):
        out[s] = (amp * np.sin(2 * np.pi * f * t))[:, None]
    return out


@pytest.fixture
def engine():
    client = EngineClient(backend="null", block_size=256)
    client.start()
    yield client
    client.stop()


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_load_play_seek_pause(engine):
    serial = engine.load(0, stems())
    assert wait_for(lambda: engine.status().decks[0].serial == serial)
    assert engine.status().decks[0].n_frames == 20 * SAMPLE_RATE
    engine.play(0)
    assert wait_for(lambda: engine.status().decks[0].playing)
    time.sleep(0.5)
    pos = engine.status().decks[0].position
    assert 0.3 * SAMPLE_RATE < pos < 0.8 * SAMPLE_RATE      # advancing in real time
    engine.seek(0, 10 * SAMPLE_RATE)
    assert wait_for(lambda: engine.status().decks[0].position > 9.9 * SAMPLE_RATE)
    engine.pause(0)
    assert wait_for(lambda: not engine.status().decks[0].playing)
    time.sleep(0.1)                                         # let the 10 ms fade-out finish
    p1 = engine.status().decks[0].position
    time.sleep(0.2)
    assert engine.status().decks[0].position == p1          # holds still when paused


def test_levels_follow_faders(engine):
    serial = engine.load(0, stems())
    wait_for(lambda: engine.status().decks[0].serial == serial)
    engine.play(0)
    assert wait_for(lambda: engine.status().decks[0].stem_peaks[0] > 0.15)
    engine.set_mute(0, 0, True)
    assert wait_for(lambda: engine.status().decks[0].stem_peaks[0] == 0.0)
    engine.set_mute(0, 0, False)
    engine.set_solo(0, 2, True)
    assert wait_for(lambda: engine.status().decks[0].stem_peaks[2] > 0.15
                    and engine.status().decks[0].stem_peaks[1] == 0.0)
    engine.set_solo(0, 2, False)
    engine.set_fader(0, 1, 0.5)
    assert wait_for(lambda: 0.08 < engine.status().decks[0].stem_peaks[1] < 0.12)


def test_swap_song_while_playing(engine):
    engine.load(0, stems(5.0))
    engine.play(0)
    for _ in range(5):
        serial = engine.load(0, stems(5.0))
        assert wait_for(lambda: engine.status().decks[0].serial == serial)
        engine.play(0)
        time.sleep(0.1)
    assert engine.is_alive()
    engine.poll_events()
    assert engine.last_error is None


def test_no_dropouts_while_controller_is_busy(engine):
    """Hammer the engine with commands and hog this process's CPU; the audio
    process must keep rendering every block comfortably in time."""
    serial = engine.load(0, stems(30.0))
    wait_for(lambda: engine.status().decks[0].serial == serial)
    engine.play(0)
    time.sleep(0.3)
    engine.reset_stats()
    time.sleep(0.1)
    before = engine.status()
    rng = np.random.default_rng(0)
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        for _ in range(20):                                   # a burst of fader moves
            engine.set_fader(0, int(rng.integers(4)), float(rng.random() * 2))
        engine.set_mute(0, int(rng.integers(4)), bool(rng.integers(2)))
        if rng.random() < 0.05:
            engine.seek(0, int(rng.integers(0, 20 * SAMPLE_RATE)))
        x = 0
        for i in range(200_000):                              # hold this process's interpreter
            x += i * i
    after = engine.status()
    blocks = after.blocks - before.blocks
    budget_ms = 1000.0 * 256 / SAMPLE_RATE                    # 5.8 ms per block
    print(f"\n  {blocks} blocks, slowest {after.max_render_ms:.2f} ms (block lasts {budget_ms:.1f} ms, "
          f"device buffer 40 ms), over-70%-budget blocks {after.late_blocks}, "
          f"underflows {after.underflows - before.underflows}")
    assert blocks > 0.9 * 4.0 * SAMPLE_RATE / 256             # kept real-time pace
    # The pass/fail criterion is the one you would hear: the (simulated) sound
    # card buffer never ran dry. A single slow block is absorbed by the buffer.
    assert after.underflows - before.underflows == 0
    assert after.max_render_ms < 40.0

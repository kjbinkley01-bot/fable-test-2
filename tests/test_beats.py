"""Beat grid fitting on synthetic detections (no neural network involved)."""
import numpy as np
import pytest

from stemdeck.analysis.beats import FPS, BeatGrid, fit_grid, grid_from_activations, pick_peaks


def logits_for(times, duration, strength=6.0):
    """A fake tracker output: sharp bumps at the given times."""
    n = int(duration * FPS)
    t = np.arange(n) / FPS
    out = np.full(n, -8.0)
    for bt in times:
        out = np.maximum(out, strength - 3.0 * np.abs(t - bt) * FPS)   # ~3 frames wide, like the real model
    return out


def test_peak_picking_is_sub_frame_accurate():
    times = np.array([1.003, 2.017, 3.009])
    peaks = pick_peaks(logits_for(times, 5.0))
    assert len(peaks) == 3
    assert np.abs(peaks - times).max() < 0.011      # better than the 20 ms frame step


@pytest.mark.parametrize("bpm", [90.0, 124.0, 128.0, 140.5, 174.0])
def test_constant_tempo_is_recovered_exactly(bpm):
    period = 60.0 / bpm
    rng = np.random.default_rng(int(bpm))
    beats = 0.37 + np.arange(int(170 / period)) * period
    noisy = beats + rng.normal(0, 0.006, len(beats))           # 6 ms tracker jitter
    grid = fit_grid(noisy, noisy[::4], 180.0)
    assert grid.is_constant
    assert abs(grid.bpm - bpm) < 0.02
    assert np.abs(grid.beats[5] - beats).min() < 0.004          # phase is right too
    assert grid.beats[0] < period and grid.beats[-1] > 180.0 - period   # spans the whole song
    assert np.allclose(np.diff(grid.beats), 60.0 / grid.bpm)


def test_missing_and_extra_beats_do_not_break_the_grid():
    period = 60.0 / 126.0
    beats = 0.2 + np.arange(300) * period
    rng = np.random.default_rng(1)
    keep = rng.random(len(beats)) > 0.15                         # 15% missed
    keep[100:132] = False                                        # a breakdown with no beats
    extra = beats[rng.choice(len(beats), 25, replace=False)] + period / 2   # stray off-beats
    detected = np.sort(np.concatenate([beats[keep], extra]))
    grid = fit_grid(detected, beats[::4][keep[::4]], 150.0)
    assert grid.is_constant and abs(grid.bpm - 126.0) < 0.02
    assert np.abs(grid.beats[10] - beats).min() < 0.004


def test_downbeats_and_bar_length():
    period = 0.5
    beats = 0.1 + np.arange(200) * period
    grid = fit_grid(beats, beats[2::4], 101.0)                   # the "one" is every 4th, offset 2
    assert grid.beats_per_bar == 4
    assert np.abs(grid.downbeats[3] - beats[2::4]).min() < 1e-6
    assert abs(grid.anchor - grid.downbeats[0]) < 1e-9
    waltz = fit_grid(beats, beats[::3], 101.0)
    assert waltz.beats_per_bar == 3


def test_slow_tempo_is_doubled_and_fast_halved():
    slow = 0.3 + np.arange(150) * (60.0 / 70.0)
    grid = fit_grid(slow, slow[::4], 130.0)
    assert abs(grid.bpm - 140.0) < 0.05 and abs(grid.detected_bpm - 70.0) < 0.05
    assert any("doubled" in n for n in grid.notes)
    assert np.abs(grid.downbeats[2] - slow[::4]).min() < 1e-3    # bars still start on a real "one"
    fast = 0.3 + np.arange(900) * (60.0 / 250.0)
    grid = fit_grid(fast, fast[::4], 216.0)
    assert abs(grid.bpm - 125.0) < 0.05 and any("halved" in n for n in grid.notes)


def test_variable_tempo_keeps_detected_beats():
    # a steady accelerando: no single ruler fits
    periods = np.linspace(0.60, 0.40, 300)
    beats = 0.5 + np.cumsum(periods)
    grid = fit_grid(beats, beats[::4], beats[-1] + 1)
    assert not grid.is_constant
    assert len(grid.beats) == len(beats) and np.allclose(grid.beats, beats)
    assert any("not constant" in n for n in grid.notes)
    assert 100 < grid.bpm < 150


def test_from_activations_and_half_time_hint():
    period = 60.0 / 87.0
    beats = 0.2 + np.arange(int(118 / period)) * period
    plain = grid_from_activations(logits_for(beats, 120.0), logits_for(beats[::4], 120.0), 120.0)
    assert abs(plain.bpm - 87.0) < 0.05 and not any("double time" in n for n in plain.notes)
    # same, but the tracker also half-hears beats in between (as in drum & bass)
    weak = logits_for(beats + period / 2, 120.0, strength=-0.5)
    hinted = grid_from_activations(np.maximum(logits_for(beats, 120.0), weak),
                                   logits_for(beats[::4], 120.0), 120.0)
    assert abs(hinted.bpm - 87.0) < 0.05 and any("double time" in n for n in hinted.notes)


def test_rescale_by_hand_and_roundtrip():
    beats = 0.1 + np.arange(100) * 0.5
    grid = fit_grid(beats, beats[::4], 51.0)
    double = grid.rescaled(2.0)
    assert abs(double.bpm - 2 * grid.bpm) < 1e-9 and len(double.beats) == 2 * len(grid.beats) - 1
    # every original bar start is still a bar start
    assert all(np.abs(double.downbeats - d).min() < 1e-6 for d in grid.downbeats[:-1])
    back = double.rescaled(0.5)
    assert abs(back.bpm - grid.bpm) < 1e-9 and np.allclose(back.beats, grid.beats)
    assert np.allclose(back.downbeats, grid.downbeats)
    again = BeatGrid.from_dict(double.to_dict())
    assert again.bpm == pytest.approx(double.bpm, abs=1e-3) and len(again.beats) == len(double.beats)


def test_too_few_beats():
    grid = fit_grid(np.array([1.0, 2.0]), np.array([]), 5.0)
    assert grid.bpm == 0.0 and not grid.is_constant

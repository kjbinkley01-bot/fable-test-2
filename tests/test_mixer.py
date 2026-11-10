"""Mixer correctness and click-freedom. No audio device needed.

A click is a jump in the waveform. The test signal is made of smooth sines,
so the largest sample-to-sample step it can produce is known; whatever the
mixer is asked to do (fader slams, mutes, solos, seeks, pause/play at random
moments, odd block sizes) the output must never step by more than slightly
above that.
"""
import numpy as np
import pytest

from stemdeck.config import SAMPLE_RATE
from stemdeck.engine.mixer import FADE_FRAMES, Deck, Mixer, effective_gains, render_offline, soft_clip

FREQS = (110.0, 220.0, 331.0, 440.0)
AMP = 0.2


def sine_stems(seconds: float = 6.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    stems = np.zeros((4, n, 2), dtype=np.float32)
    for s, f in enumerate(FREQS):
        wave = AMP * np.sin(2 * np.pi * f * t)
        stems[s, :, 0] = wave
        stems[s, :, 1] = wave
    return stems


def natural_max_step() -> float:
    """Largest step the four sines can make between adjacent samples."""
    return sum(AMP * 2 * np.pi * f / SAMPLE_RATE for f in FREQS)


def run(mixer: Mixer, blocks, actions=None) -> np.ndarray:
    out = []
    for i, n in enumerate(blocks):
        if actions and i in actions:
            actions[i]()
        out.append(mixer.render(n))
    return np.concatenate(out)


def test_unity_gain_is_transparent():
    stems = sine_stems(2.0)
    mixer = Mixer()
    deck = mixer.decks[0]
    deck.load(stems)
    deck.play()
    out = run(mixer, [512] * 100)
    expected = stems.sum(axis=0)[: len(out)]
    # after the 10 ms play fade-in the output must equal the plain sum of the stems
    start = FADE_FRAMES + 512
    assert np.abs(out[start:] - expected[start:]).max() < 1e-6


def test_solo_and_mute_logic():
    f = np.array([1.0, 0.5, 0.25, 2.0])
    no = np.zeros(4, bool)
    assert effective_gains(f, no, no).tolist() == [1.0, 0.5, 0.25, 2.0]
    assert effective_gains(f, np.array([0, 1, 0, 0], bool), no).tolist() == [1.0, 0.0, 0.25, 2.0]
    assert effective_gains(f, no, np.array([1, 0, 1, 0], bool)).tolist() == [1.0, 0.0, 0.25, 0.0]
    # mute wins over solo
    assert effective_gains(f, np.array([1, 0, 0, 0], bool), np.array([1, 1, 0, 0], bool)).tolist() == [0.0, 0.5, 0.0, 0.0]


@pytest.mark.parametrize("block", [64, 128, 441, 512, 1024, 4096])
def test_no_clicks_under_random_abuse(block):
    rng = np.random.default_rng(block)
    stems = sine_stems(8.0)
    mixer = Mixer()
    deck = mixer.decks[0]
    deck.load(stems)
    deck.play()
    n_blocks = int(6.0 * SAMPLE_RATE / block)
    actions = {}

    def make(kind):
        def act():
            if kind == 0:
                deck.fader[rng.integers(4)] = rng.choice([0.0, 0.3, 1.0, 2.0])
            elif kind == 1:
                s = rng.integers(4)
                deck.mute[s] = not deck.mute[s]
            elif kind == 2:
                s = rng.integers(4)
                deck.solo[s] = not deck.solo[s]
            elif kind == 3:
                deck.seek(int(rng.integers(0, 5 * SAMPLE_RATE)))
            elif kind == 4:
                deck.pause() if deck.playing else deck.play()
            elif kind == 5:
                deck.volume = float(rng.choice([0.0, 0.5, 1.0]))
            else:
                mixer.master = float(rng.choice([0.2, 1.0]))
        return act

    for i in rng.choice(n_blocks, size=min(n_blocks // 2, 400), replace=False):
        actions[int(i)] = make(int(rng.integers(7)))
    out = run(mixer, [block] * n_blocks, actions)
    assert np.isfinite(out).all()
    steps = np.abs(np.diff(out[:, 0]))
    # Gains reach at most 2 (fader) so the natural step doubles; fades add a
    # little on top. A real click would be an order of magnitude larger.
    limit = 2.0 * natural_max_step() * 1.15 + 2.0 * 4 * AMP / FADE_FRAMES
    assert steps.max() < limit, f"max step {steps.max():.4f} exceeds {limit:.4f} (click)"


def test_instant_gain_change_would_fail_the_click_test():
    """Sanity check of the test itself: an unsmoothed mute does click."""
    stems = sine_stems(1.0)
    out = stems.sum(axis=0)[:, 0].copy()
    out[20000:] = 0.0
    assert np.abs(np.diff(out)).max() > 3 * natural_max_step()


def test_pause_holds_position_and_seek_while_paused():
    stems = sine_stems(3.0)
    mixer = Mixer()
    deck = mixer.decks[0]
    deck.load(stems)
    deck.play()
    run(mixer, [512] * 10)
    deck.pause()
    run(mixer, [512] * 5)
    pos = deck.position
    run(mixer, [512] * 5)
    assert deck.position == pos                      # frozen while paused
    assert np.abs(mixer.render(512)).max() == 0.0     # and silent
    deck.seek(44100)
    mixer.render(512)
    assert deck.position == 44100


def test_end_of_track_stops_cleanly():
    stems = sine_stems(0.5)
    mixer = Mixer()
    deck = mixer.decks[0]
    deck.load(stems)
    deck.play()
    out = run(mixer, [512] * 60)
    assert deck.ended and not deck.playing
    assert deck.position == stems.shape[1]
    assert np.abs(out[stems.shape[1] + 512:]).max() == 0.0
    deck.play()                                       # play again restarts from the top
    mixer.render(512)
    assert deck.position == 512


def test_two_decks_sum():
    a, b = sine_stems(1.0) * 0.5, sine_stems(1.0) * 0.25   # stay below the soft clipper
    mixer = Mixer(n_decks=2)
    mixer.decks[0].load(a)
    mixer.decks[1].load(b)
    mixer.decks[0].play()
    mixer.decks[1].play()
    out = run(mixer, [512] * 40)
    start = 1024
    expected = (a.sum(axis=0) + b.sum(axis=0))[: len(out)]
    assert np.abs(out[start:] - expected[start:]).max() < 1e-6


def test_soft_clip_bounds_output():
    x = np.linspace(-4, 4, 1001)[:, None].repeat(2, axis=1)
    y = soft_clip(x)
    assert np.abs(y).max() <= 1.0
    quiet = np.abs(x) <= 0.89
    assert np.array_equal(y[quiet], x[quiet])          # untouched below the threshold
    assert (np.diff(y[:, 0]) >= 0).all()               # monotonic: no fold-back


def test_render_offline_matches_gains():
    stems = sine_stems(0.2)
    out = render_offline(stems, np.array([1.0, 0.0, 0.5, 0.0]))
    assert np.allclose(out, stems[0] + 0.5 * stems[2], atol=1e-7)

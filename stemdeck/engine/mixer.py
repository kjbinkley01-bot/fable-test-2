"""The real-time mixer: turns stems + fader positions into output audio.

This file is pure number-crunching with no audio device, no threads and no
UI, which is what makes it testable: ``tests/test_mixer.py`` drives it with
thousands of random fader moves, mutes, seeks and pauses and checks the
output never jumps (a jump in the waveform is what you hear as a click).

How clicks are avoided
----------------------
* Every gain - stem faders, mute, solo, deck volume, master - moves towards
  its target gradually; a full-scale change takes about 10 ms. Instant gain
  steps are what cause "zipper" noise and clicks.
* Play and pause fade in / out over the same ~10 ms.
* A seek while playing cross-fades: the old position fades out while the new
  one fades in.

Structure (and why milestone 2 fits without a rewrite)
------------------------------------------------------
``Mixer`` owns a list of ``Deck`` objects and sums them. Milestone 1 uses
one deck; a second deck is simply another entry. A deck asks its source for
audio through ``Deck._read``; tempo sync and key shifting slot in there, as
a processing stage between "fetch stems at the playhead" and "apply faders",
without touching the gain / transport / seek logic below.
"""
from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE, STEMS

#: Length of every fade / gain glide, in frames (10 ms).
FADE_FRAMES = int(0.010 * SAMPLE_RATE)

#: Level above which the master soft-clipper starts to act (about -1 dB).
CLIP_THRESHOLD = 0.89


def _slew(current, target, n_frames: int):
    """Move ``current`` towards ``target`` no faster than full scale per FADE_FRAMES."""
    max_step = n_frames / FADE_FRAMES
    return current + np.clip(target - current, -max_step, max_step)


def effective_gains(fader: np.ndarray, mute: np.ndarray, solo: np.ndarray) -> np.ndarray:
    """Fader/mute/solo -> the gain each stem is actually played at.

    If any stem is soloed only soloed stems sound. Mute always wins.
    """
    audible = solo if solo.any() else np.ones_like(solo)
    return np.where(audible & ~mute, fader, 0.0).astype(np.float64)


def soft_clip(block: np.ndarray) -> np.ndarray:
    """Round off peaks above CLIP_THRESHOLD instead of letting them clip hard."""
    mag = np.abs(block)
    over = mag > CLIP_THRESHOLD
    if not over.any():
        return block
    room = 1.0 - CLIP_THRESHOLD
    shaped = CLIP_THRESHOLD + room * np.tanh((mag[over] - CLIP_THRESHOLD) / room)
    out = block.copy()
    out[over] = np.sign(block[over]) * shaped
    return out


class Deck:
    """One playing song: stems, playhead, transport and per-stem gains."""

    def __init__(self, n_stems: int = len(STEMS)) -> None:
        self.n_stems = n_stems
        self.stems: np.ndarray | None = None   # (n_stems, frames, 2) float32
        self.n_frames = 0
        self.position = 0                      # next frame to be played
        self.playing = False
        self.ended = False
        # Targets, written by the control side.
        self.fader = np.ones(n_stems)
        self.mute = np.zeros(n_stems, dtype=bool)
        self.solo = np.zeros(n_stems, dtype=bool)
        self.volume = 1.0
        # Audio-side state.
        self._gain = np.ones(n_stems)          # gains as currently applied
        self._transport = 0.0                  # 0 = silent/stopped ... 1 = playing
        self._seek: tuple[int, int] | None = None   # (serial, frame)
        self._seek_handled = 0
        self._seek_serial = 0
        self.stem_peaks = np.zeros(n_stems)

    # ---------------------------------------------------------------- control
    def load(self, stems: np.ndarray | None) -> None:
        """Attach a song (or None to unload). Resets the transport."""
        self.playing = False
        self._transport = 0.0
        self.position = 0
        self.ended = False
        self.stems = stems
        self.n_frames = 0 if stems is None else int(stems.shape[1])
        self._gain = self._targets()

    def play(self) -> None:
        if self.stems is None:
            return
        if self.position >= self.n_frames:
            self.position = 0
        self.ended = False
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def seek(self, frame: int) -> None:
        frame = int(min(max(frame, 0), self.n_frames))
        self._seek_serial += 1
        self._seek = (self._seek_serial, frame)   # one atomic assignment

    def _targets(self) -> np.ndarray:
        return effective_gains(self.fader, self.mute, self.solo) * self.volume

    # ------------------------------------------------------------------ audio
    def _read(self, position: int, n: int) -> np.ndarray:
        """Stem audio for [position, position + n): shape (n_stems, n, 2).

        Milestone 2 seam: time-stretching / pitch-shifting replaces this.
        """
        stems = self.stems
        end = position + n
        if position >= 0 and end <= self.n_frames:
            return stems[:, position:end]
        out = np.zeros((self.n_stems, n, 2), dtype=np.float32)
        lo, hi = max(position, 0), min(end, self.n_frames)
        if hi > lo:
            out[:, lo - position:hi - position] = stems[:, lo:hi]
        return out

    def render(self, n: int) -> np.ndarray | None:
        """Produce the next ``n`` frames, shape (n, 2), or None if silent."""
        if self.stems is None:
            return None

        tail = None
        fade = 0
        seek = self._seek
        if seek is not None and seek[0] != self._seek_handled:
            self._seek_handled = seek[0]
            if self._transport > 0.0:
                # Audible: fade the old position out under the new one.
                fade = min(n, FADE_FRAMES)
                old = self._read(self.position, fade)
                gains = self._gain * self._transport
                tail = np.zeros((fade, 2))
                for s in range(self.n_stems):
                    if gains[s] != 0.0:
                        tail += old[s] * gains[s]
                # Complement of the fade-in applied below, so the two always sum to 1.
                tail *= (np.linspace(1.0, 0.0, fade, endpoint=False) - 1.0 / fade)[:, None]
            self.position = seek[1]
            self.ended = False

        t0 = self._transport
        t1 = float(_slew(t0, 1.0 if self.playing else 0.0, n))
        if t0 == 0.0 and t1 == 0.0:
            self._gain = self._targets()   # jump silently while stopped
            self.stem_peaks[:] = 0.0
            return tail if tail is None else self._pad(tail, n)

        g0 = self._gain
        g1 = _slew(g0, self._targets(), n)
        block = self._read(self.position, n)
        out = np.zeros((n, 2))
        steady = t0 == t1 and np.array_equal(g0, g1)
        if steady:
            for s in range(self.n_stems):
                if g1[s] != 0.0:
                    out += block[s] * (g1[s] * t1)
        else:
            ramp = np.linspace(0.0, 1.0, n, endpoint=False) + 1.0 / n   # ends exactly at the target
            transport = t0 + (t1 - t0) * ramp
            for s in range(self.n_stems):
                if g0[s] == 0.0 and g1[s] == 0.0:
                    continue
                out += block[s] * ((g0[s] + (g1[s] - g0[s]) * ramp) * transport)[:, None]
        for s in range(self.n_stems):
            self.stem_peaks[s] = float(np.abs(block[s]).max()) * g1[s] * t1 if g1[s] else 0.0

        if fade:
            out[:fade] *= (np.linspace(0.0, 1.0, fade, endpoint=False) + 1.0 / fade)[:, None]
            out[:fade] += tail

        self._gain = g1
        self._transport = t1
        self.position += n
        if self.position >= self.n_frames:
            # Ran off the end: stop. (_read padded the remainder with silence.)
            self.position = self.n_frames
            self.playing = False
            self._transport = 0.0
            self.ended = True
        return out

    @staticmethod
    def _pad(tail: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros((n, 2))
        out[:len(tail)] = tail
        return out


class Mixer:
    """Sums the decks and applies the master volume and soft clipper."""

    def __init__(self, n_decks: int = 2) -> None:
        self.decks = [Deck() for _ in range(n_decks)]
        self.master = 1.0
        self._master = 1.0
        self.peak = 0.0
        self.clipped = False

    def render(self, n: int) -> np.ndarray:
        """Next ``n`` frames of the master output, float32 (n, 2)."""
        out = None
        for deck in self.decks:
            block = deck.render(n)
            if block is not None:
                out = block if out is None else out + block
        m0 = self._master
        m1 = float(_slew(m0, self.master, n))
        self._master = m1
        if out is None:
            self.peak = 0.0
            self.clipped = False
            return np.zeros((n, 2), dtype=np.float32)
        if m0 != m1:
            out = out * (m0 + (m1 - m0) * (np.linspace(0.0, 1.0, n, endpoint=False) + 1.0 / n))[:, None]
        elif m1 != 1.0:
            out = out * m1
        self.peak = float(np.abs(out).max())
        self.clipped = self.peak > CLIP_THRESHOLD
        if self.clipped:
            out = soft_clip(out)
        return out.astype(np.float32, copy=False)


def render_offline(stems: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Mix (n_stems, frames, 2) with fixed linear gains -> (frames, 2) float32.

    Used for export. No soft clipping: export handles level explicitly.
    """
    out = np.zeros(stems.shape[1:], dtype=np.float32)
    for s in range(stems.shape[0]):
        if gains[s] != 0.0:
            out += stems[s] * np.float32(gains[s])
    return out

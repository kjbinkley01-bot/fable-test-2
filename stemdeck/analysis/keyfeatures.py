"""Audio -> pitch features for key detection.

What this computes, in plain terms: for each ~10 second stretch of the song,
how much energy sits on each note of the piano, from C1 (deep bass) up six
octaves. The key detector looks at that note-energy picture.

Details that matter for accuracy:

* A long analysis window (1.5 s) is used. We do not care *when* a note is
  played, only *which* notes are, and a long window separates neighbouring
  bass notes that a short one would blur together.
* Tuning. Not every track is tuned to A = 440 Hz (old vinyl rips, tracks
  that were sped up). We measure how far the track's notes sit from standard
  pitch and line the note grid up with the track before measuring.

Only numpy/scipy/soxr are used, so this runs anywhere without the GPU.
"""
from __future__ import annotations

import numpy as np

from ..config import SAMPLE_RATE

FEATURE_SR = 11025
N_FFT = 16384
HOP = 4096
MIDI_LOW = 24            # C1, 32.7 Hz
N_SEMITONES = 72         # six octaves: C1 .. B6
FILTER_HALF_WIDTH = 0.65  # in semitones

FRAMES_PER_SECOND = FEATURE_SR / HOP


def _stft_magnitude(mono: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, shape (frames, N_FFT // 2 + 1)."""
    if len(mono) < N_FFT:
        mono = np.pad(mono, (0, N_FFT - len(mono)))
    n_frames = 1 + (len(mono) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    for start in range(0, n_frames, 64):  # in blocks, to bound memory
        idx = np.arange(start, min(start + 64, n_frames))
        frames = np.stack([mono[i * HOP: i * HOP + N_FFT] for i in idx]) * window
        out[idx] = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    return out


def estimate_tuning(spectrum: np.ndarray) -> float:
    """Tuning offset of the track in semitones, in [-0.5, 0.5).

    ``spectrum`` is the time-averaged magnitude spectrum. Each spectral peak
    votes for how far it sits from the nearest equal-tempered note; votes are
    combined on a circle so that -49 cents and +49 cents count as close.
    """
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FEATURE_SR)
    lo, hi = np.searchsorted(freqs, [100.0, 2000.0])
    mag = spectrum[lo:hi]
    f = freqs[lo:hi]
    peaks = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
    if len(peaks) == 0:
        return 0.0
    # Parabolic interpolation for a sub-bin frequency estimate.
    a, b, c = mag[peaks - 1], mag[peaks], mag[peaks + 1]
    shift = 0.5 * (a - c) / np.minimum(a - 2 * b + c, -1e-12)
    peak_freq = f[peaks] + shift * (FEATURE_SR / N_FFT)
    weight = b
    keep = weight > 0.05 * weight.max()
    midi = 69.0 + 12.0 * np.log2(peak_freq[keep] / 440.0)
    deviation = midi - np.round(midi)
    vector = np.sum(weight[keep] * np.exp(2j * np.pi * deviation))
    if np.abs(vector) < 1e-9:
        return 0.0
    return float(np.angle(vector) / (2 * np.pi))


def _filterbank(tuning: float) -> np.ndarray:
    """(N_SEMITONES, n_bins) weights mapping FFT bins to semitones."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FEATURE_SR)
    with np.errstate(divide="ignore"):
        midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0) - tuning
    centers = MIDI_LOW + np.arange(N_SEMITONES)
    dist = np.abs(midi[None, :] - centers[:, None])
    weights = np.where(dist < FILTER_HALF_WIDTH,
                       np.cos(0.5 * np.pi * dist / FILTER_HALF_WIDTH) ** 2, 0.0)
    # Bass bands cover few FFT bins, treble bands many: normalise each band.
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    return weights.astype(np.float32)


def semitone_spectrogram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Return (energy per semitone per frame, tuning).

    Output shape (frames, 72), linear magnitude, ~2.7 frames per second.
    """
    import soxr

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    if sample_rate != FEATURE_SR:
        mono = soxr.resample(mono, sample_rate, FEATURE_SR, quality="HQ").astype(np.float32)
    mag = _stft_magnitude(mono)
    tuning = estimate_tuning(mag.mean(axis=0))
    return mag @ _filterbank(tuning).T, tuning


def window_features(semitones: np.ndarray, window_seconds: float = 20.0,
                    hop_seconds: float = 10.0) -> np.ndarray:
    """Summarise the semitone spectrogram over overlapping time windows.

    Returns (n_windows, 72): log-compressed mean energy per note, with the
    overall loudness divided out so quiet and loud masters look alike.
    Windows that are essentially silent are dropped.
    """
    win = max(1, int(round(window_seconds * FRAMES_PER_SECOND)))
    hop = max(1, int(round(hop_seconds * FRAMES_PER_SECOND)))
    n = semitones.shape[0]
    level = float(semitones.mean())
    if level <= 1e-9:
        return np.zeros((0, N_SEMITONES), dtype=np.float32)
    starts = list(range(0, max(1, n - win + 1), hop)) or [0]
    rows = []
    for s in starts:
        chunk = semitones[s:s + win]
        mean = chunk.mean(axis=0)
        if mean.mean() < 0.05 * level:
            continue  # near-silent stretch (intro / outro)
        rows.append(np.log1p(10.0 * mean / mean.mean()))
    if not rows:
        rows.append(np.log1p(10.0 * semitones.mean(axis=0) / level))
    return np.asarray(rows, dtype=np.float32)

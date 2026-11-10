"""Beat tracking, BPM and the beat grid.

Two steps:

1. A neural network ("Beat This!", CPJKU, ISMIR 2024) listens to the song and
   marks where it hears beats and downbeats (the "one" of each bar).
2. We turn those marks into a *grid*. Most music a DJ plays was made against
   a click at one fixed tempo, so if the detected beats line up with an
   evenly spaced ruler we use the ruler: a single BPM and a start point.
   Averaging hundreds of detections this way gives a BPM that is accurate to
   a hundredth and a grid that does not wobble. If the beats do not fit a
   ruler (a live drummer, a tempo change) we keep the detected beats as they
   are and report the typical BPM.

A fixed grid is also exactly what tempo sync needs later: beat N is at
``anchor + N * 60 / bpm``.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from ..config import SAMPLE_RATE

logger = logging.getLogger(__name__)

FPS = 50.0  # frames per second of the beat model's output

#: BPM values outside this range are doubled / halved into it. Whether a song
#: is "70" or "140" is a convention, not a fact, and DJ tools all pick a range.
#: This one was chosen on 661 annotated dance tracks: a reading near 70 almost
#: always means 140 (dubstep, trap), while readings of 80-100 are genuinely
#: slow about as often as they are half-time drum & bass, so they are left
#: alone and flagged instead (the app has x2 / /2 buttons for those).
DEFAULT_BPM_RANGE = (78.0, 200.0)

_models: dict[str, object] = {}
_models_lock = threading.Lock()


@dataclass
class BeatGrid:
    bpm: float
    beats: np.ndarray                 # beat times in seconds, ascending
    downbeats: np.ndarray             # subset of ``beats``: first beat of each bar
    beats_per_bar: int = 4
    is_constant: bool = False         # True: beats are exactly anchor + n * 60 / bpm
    anchor: float = 0.0               # time of the first downbeat
    confidence: float = 0.0           # share of detected beats the grid explains
    detected_bpm: float = 0.0         # before folding into the DJ range
    notes: list[str] = field(default_factory=list)

    def to_dict(self, include_grid: bool = True) -> dict:
        data = {
            "bpm": round(float(self.bpm), 3),
            "beats_per_bar": int(self.beats_per_bar),
            "is_constant": bool(self.is_constant),
            "anchor": round(float(self.anchor), 5),
            "confidence": round(float(self.confidence), 3),
            "detected_bpm": round(float(self.detected_bpm), 3),
            "n_beats": int(len(self.beats)),
            "notes": list(self.notes),
        }
        if include_grid:
            data["beats"] = [round(float(t), 5) for t in self.beats]
            data["downbeats"] = [round(float(t), 5) for t in self.downbeats]
        return data

    def rescaled(self, factor: float) -> "BeatGrid":
        """The same grid counted at double (factor 2) or half (factor 0.5) time."""
        if factor not in (0.5, 2.0) or len(self.beats) < 2:
            return self
        if factor == 2.0:
            mids = (self.beats[:-1] + self.beats[1:]) / 2.0
            beats = np.sort(np.concatenate([self.beats, mids]))
        else:
            # keep the beats the downbeats sit on
            start = 0
            if len(self.downbeats):
                start = int(np.abs(self.beats - self.downbeats[0]).argmin()) % 2
            beats = self.beats[start::2]
        if len(self.downbeats):
            first = int(np.abs(beats - self.downbeats[0]).argmin()) % self.beats_per_bar
            downbeats = beats[first::self.beats_per_bar]
        else:
            downbeats = beats[::self.beats_per_bar]
        notes = [n for n in self.notes if "double time" not in n and "half time" not in n]
        notes.append("tempo doubled by hand" if factor == 2.0 else "tempo halved by hand")
        return BeatGrid(
            bpm=self.bpm * factor, beats=beats, downbeats=downbeats,
            beats_per_bar=self.beats_per_bar, is_constant=self.is_constant,
            anchor=float(downbeats[0]) if len(downbeats) else float(beats[0]),
            confidence=self.confidence, detected_bpm=self.detected_bpm, notes=notes,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "BeatGrid":
        return cls(
            bpm=float(data["bpm"]),
            beats=np.asarray(data.get("beats", []), dtype=np.float64),
            downbeats=np.asarray(data.get("downbeats", []), dtype=np.float64),
            beats_per_bar=int(data.get("beats_per_bar", 4)),
            is_constant=bool(data.get("is_constant", False)),
            anchor=float(data.get("anchor", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            detected_bpm=float(data.get("detected_bpm", data["bpm"])),
            notes=list(data.get("notes", [])),
        )


# --------------------------------------------------------------------- model
def load_beat_model(device: str = "cpu"):
    """Load (and cache) the beat tracking network on ``device``."""
    with _models_lock:
        model = _models.get(device)
        if model is None:
            from beat_this.inference import Audio2Frames

            model = Audio2Frames(checkpoint_path="final0", device=device, float16=False)
            _models[device] = model
        return model


def beat_activations(audio: np.ndarray, device: str = "cpu",
                     sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Run the network. Returns (beat_logits, downbeat_logits) at 50 frames/s."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    try:
        model = load_beat_model(device)
        beat, downbeat = model(np.ascontiguousarray(mono, dtype=np.float32), sample_rate)
    except Exception as exc:
        if device == "cpu":
            raise
        logger.warning("Beat tracking on %s failed (%s); using the CPU", device, exc)
        model = load_beat_model("cpu")
        beat, downbeat = model(np.ascontiguousarray(mono, dtype=np.float32), sample_rate)
    return beat.cpu().numpy().astype(np.float64), downbeat.cpu().numpy().astype(np.float64)


# ------------------------------------------------------------- peak picking
def pick_peaks(logits: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Turn the network's frame-by-frame output into event times (seconds).

    Same rule as the model authors' post-processing (a frame is an event if
    it is the largest within +/-3 frames and the network is more than 50%
    sure), plus a refinement: the model only outputs 50 values per second
    (20 ms steps), so we fit a parabola through each peak and its neighbours
    to place the event between frames.
    """
    from scipy.ndimage import maximum_filter1d

    if len(logits) == 0:
        return np.zeros(0)
    local_max = maximum_filter1d(logits, size=7, mode="constant", cval=-1e9)
    frames = np.flatnonzero((logits == local_max) & (logits > 0))
    if len(frames) == 0:
        return np.zeros(0)
    # Merge runs of adjacent frames (a flat-topped peak) into one event.
    groups = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
    times = []
    for group in groups:
        if len(group) > 1:
            times.append(float(group.mean()))
            continue
        i = int(group[0])
        pos = float(i)
        if 0 < i < len(logits) - 1:
            a, b, c = logits[i - 1], logits[i], logits[i + 1]
            denom = a - 2 * b + c
            if denom < 0:
                pos += float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        times.append(pos)
    return np.asarray(times) / fps


def beats_from_activations(beat_logits: np.ndarray, downbeat_logits: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Logits -> (beat_times, downbeat_times, beat_weights, activation)."""
    activation = 1.0 / (1.0 + np.exp(-beat_logits))
    beats = pick_peaks(beat_logits)
    downbeats = pick_peaks(downbeat_logits)
    if len(beats) and len(downbeats):
        # A downbeat is by definition also a beat: snap each to its nearest beat.
        nearest = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
        downbeats = np.unique(beats[nearest])
    frames = np.clip(np.round(beats * FPS).astype(int), 0, max(0, len(activation) - 1))
    weights = activation[frames] if len(activation) else np.zeros(0)
    return beats, downbeats, weights, activation


def detect_beats(audio: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return (beat_times, downbeat_times) as detected, before grid fitting."""
    beats, downbeats, _, _ = beats_from_activations(*beat_activations(audio, device))
    return beats, downbeats


# -------------------------------------------------------------- grid fitting
def _mode(values: np.ndarray, default: int) -> int:
    if len(values) == 0:
        return default
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()])


def _ruler_candidates(activation: np.ndarray | None, beats: np.ndarray, weights: np.ndarray,
                      duration: float, max_candidates: int = 8) -> list[tuple[float, float]]:
    """Candidate rulers (offset a, spacing b) from the rhythm of the beat evidence.

    A steady beat is a regular pulse, and a regular pulse shows up as a sharp
    line in a frequency analysis at "beats per second". We take the strongest
    such lines as candidate tempos; the angle of each line says where its
    beats fall. This uses every bit of the tracker's evidence at once, so a
    stretch where the tracker hesitated or doubled up does not derail it.
    """
    fmin, fmax = 0.75, 4.6            # 45 .. 276 beats per minute
    if activation is not None and len(activation) > 64:
        size = 1 << int(np.ceil(np.log2(len(activation) * 32)))
        spec = np.fft.rfft(activation - activation.mean(), size)
        freqs = np.fft.rfftfreq(size, 1.0 / FPS)
    else:
        freqs = np.arange(fmin, fmax, 1.0 / (32.0 * max(duration, 1.0)))
        spec = np.zeros(len(freqs), dtype=np.complex128)
        for start in range(0, len(freqs), 4096):
            chunk = freqs[start:start + 4096]
            spec[start:start + 4096] = (weights[None, :] * np.exp(
                -2j * np.pi * chunk[:, None] * beats[None, :])).sum(axis=1)
    band = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    if len(band) < 3:
        return []
    mag = np.abs(spec[band])
    peaks = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
    peaks = peaks[mag[peaks] >= 0.25 * mag.max()]
    peaks = peaks[np.argsort(mag[peaks])[::-1]][:max_candidates]
    out = []
    for p in peaks:
        f = float(freqs[band[p]])
        angle = float(np.angle(spec[band[p]]))
        out.append(((-angle / (2 * np.pi)) / f, 1.0 / f))   # beats fall where the wave peaks
    return out


def _grid_quality(beats: np.ndarray, weights: np.ndarray, a: float, b: float,
                  tolerance: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """How well the ruler a + n*b explains the detected beats.

    precision: share of detected beats (weighted by confidence) that sit on a mark.
    coverage:  share of marks, between the first and last beat, that have a beat.
    Returns (score, precision, coverage, index of each beat, on-grid mask).
    """
    index = np.round((beats - a) / b)
    on_grid = np.abs(beats - (a + b * index)) < tolerance
    if not on_grid.any():
        return 0.0, 0.0, 0.0, index, on_grid
    precision = float(weights[on_grid].sum() / weights.sum())
    # Count the ruler marks that lie where the tracker heard anything at all.
    # A stretch with no detections (a breakdown without drums) is "no
    # evidence", not evidence against the tempo, so it is left out.
    marks = np.arange(np.round((beats[0] - a) / b), np.round((beats[-1] - a) / b) + 1)
    nearest = np.searchsorted(beats, a + marks * b)
    left = beats[np.clip(nearest - 1, 0, len(beats) - 1)]
    right = beats[np.clip(nearest, 0, len(beats) - 1)]
    gap = np.minimum(np.abs(a + marks * b - left), np.abs(right - (a + marks * b)))
    n_marks = int((gap < 1.5 * b).sum())
    coverage = float(min(1.0, len(np.unique(index[on_grid])) / max(1, n_marks)))
    score = 2 * precision * coverage / (precision + coverage)
    return score, precision, coverage, index, on_grid


def _between_beat_evidence(activation: np.ndarray, a: float, b: float, duration: float) -> float:
    """Tracker activity half way between grid beats, relative to on the beats."""
    n = np.arange(int(np.ceil(-a / b)), int(np.floor((duration - a) / b)))
    if len(n) < 8 or len(activation) < 8:
        return 0.0

    def sample(times: np.ndarray) -> np.ndarray:
        frames = np.clip(np.round(times * FPS).astype(int), 1, len(activation) - 2)
        return np.maximum(np.maximum(activation[frames - 1], activation[frames]), activation[frames + 1])

    on = float(sample(a + n * b).mean())
    return float(sample(a + (n + 0.5) * b).mean()) / on if on > 1e-6 else 0.0


def fit_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
    bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE,
    tolerance: float = 0.035,
    activation: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> BeatGrid:
    """Build the beat grid from detected beats (see module docstring).

    ``activation`` (the tracker's frame-by-frame beat probability) and
    ``weights`` (its confidence in each detected beat) are optional extras
    that make the tempo search more robust.
    """
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 4:
        grid = BeatGrid(bpm=0.0, beats=beats, downbeats=downbeats)
        grid.notes.append("too few beats detected for a tempo")
        return grid

    intervals = np.diff(beats)
    period = float(np.median(intervals))

    weights = np.ones(len(beats)) if weights is None else np.asarray(weights, dtype=np.float64)

    # Find the evenly spaced ruler (offset a, spacing b) that explains the
    # detected beats best, among the tempos the rhythm analysis proposes.
    best = None
    for cand_a, cand_b in _ruler_candidates(activation, beats, weights, duration):
        quality = _grid_quality(beats, weights, cand_a, cand_b, tolerance)
        if best is None or quality[0] > best[0][0]:
            best = (quality, cand_a, cand_b)
    if best is None:
        best = (_grid_quality(beats, weights, float(beats[0]), period, tolerance), float(beats[0]), period)
    (score, precision, coverage, index, inliers), a, b = best

    # Polish with a straight-line fit through the beats that sit on the ruler.
    for _ in range(3):
        _, first = np.unique(index[inliers], return_index=True)
        if len(first) < 8:
            break
        sel = np.flatnonzero(inliers)[first]
        b, a = (float(v) for v in np.polyfit(index[sel], beats[sel], 1))
        score, precision, coverage, index, inliers = _grid_quality(beats, weights, a, b, tolerance)

    confidence = float(score)
    n_on_grid = len(np.unique(index[inliers])) if inliers.any() else 0
    # Declare a fixed tempo only when the ruler clearly fits. On 661 annotated
    # tracks the tempo was right in over 99% of cases at a fit of 0.75 or
    # better, and in under 90% just below that.
    is_constant = n_on_grid >= 16 and score >= 0.75 and b > 0

    # Bar length and which beat is the "one".
    beat_of_down = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1) if len(downbeats) else np.zeros(0, int)
    down_index = index[beat_of_down].astype(int) if len(downbeats) else np.zeros(0, int)
    bar_lengths = np.diff(down_index)
    beats_per_bar = _mode(bar_lengths[(bar_lengths >= 2) & (bar_lengths <= 12)], 4)
    phase = _mode(down_index % beats_per_bar, 0)

    notes: list[str] = []
    if is_constant:
        detected_bpm = 60.0 / b
        bpm = detected_bpm
        if bpm < bpm_range[0]:
            while bpm < bpm_range[0]:
                bpm *= 2
                b /= 2
                phase *= 2
            notes.append(f"tempo doubled from {detected_bpm:.1f} into the DJ range")
        elif bpm >= bpm_range[1]:
            while bpm >= bpm_range[1]:
                # keep the beats that the downbeats sit on
                a += (phase % 2) * b
                phase //= 2
                bpm /= 2
                b *= 2
            notes.append(f"tempo halved from {detected_bpm:.1f} into the DJ range")
        phase %= beats_per_bar
        if 78.0 <= bpm < 100.0 and activation is not None and bpm == detected_bpm:
            # Strong evidence of beats half way between ours suggests the song
            # may really be twice as fast (typical for drum & bass).
            if _between_beat_evidence(activation, a, b, duration) >= 0.12:
                notes.append(f"could also be counted in double time ({2 * bpm:.0f} BPM), "
                             "common for drum & bass: use the x2 button if so")
        n_first = int(np.ceil((0.0 - a) / b - 1e-9))
        n_last = int(np.floor((duration - a) / b + 1e-9))
        n = np.arange(n_first, n_last + 1)
        grid_beats = a + n * b
        is_down = (n - phase) % beats_per_bar == 0
        grid_downbeats = grid_beats[is_down]
        anchor = float(grid_downbeats[0]) if len(grid_downbeats) else float(grid_beats[0])
        return BeatGrid(bpm=bpm, beats=grid_beats, downbeats=grid_downbeats,
                        beats_per_bar=beats_per_bar, is_constant=True, anchor=anchor,
                        confidence=confidence, detected_bpm=detected_bpm, notes=notes)

    # Variable tempo: keep the detected beats, report the typical tempo.
    detected_bpm = 60.0 / b if (score >= 0.4 and b > 0) else 60.0 / period
    bpm = detected_bpm
    grid_beats = beats
    grid_downbeats = downbeats
    if bpm < bpm_range[0]:
        while bpm < bpm_range[0]:
            mids = (grid_beats[:-1] + grid_beats[1:]) / 2
            grid_beats = np.sort(np.concatenate([grid_beats, mids]))
            bpm *= 2
        notes.append(f"tempo doubled from {detected_bpm:.1f} into the DJ range")
    elif bpm >= bpm_range[1]:
        while bpm >= bpm_range[1] and len(grid_beats) > 8:
            start = 0
            if len(grid_downbeats):
                start = int(np.abs(grid_beats - grid_downbeats[0]).argmin()) % 2
            grid_beats = grid_beats[start::2]
            bpm /= 2
        if len(grid_downbeats):
            nearest = np.abs(grid_beats[None, :] - grid_downbeats[:, None]).argmin(axis=1)
            grid_downbeats = np.unique(grid_beats[nearest])
        notes.append(f"tempo halved from {detected_bpm:.1f} into the DJ range")
    notes.append("tempo is not constant: grid follows the detected beats")
    anchor = float(grid_downbeats[0]) if len(grid_downbeats) else float(grid_beats[0])
    return BeatGrid(bpm=bpm, beats=grid_beats, downbeats=grid_downbeats,
                    beats_per_bar=beats_per_bar, is_constant=False, anchor=anchor,
                    confidence=confidence, detected_bpm=detected_bpm, notes=notes)


def grid_from_activations(beat_logits: np.ndarray, downbeat_logits: np.ndarray, duration: float,
                          bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> BeatGrid:
    beats, downbeats, weights, activation = beats_from_activations(beat_logits, downbeat_logits)
    return fit_grid(beats, downbeats, duration, bpm_range, activation=activation, weights=weights)


def analyze_beats(audio: np.ndarray, device: str = "cpu",
                  bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> BeatGrid:
    """Detect beats in (frames, 2) audio at 44.1 kHz and build the grid."""
    beat_logits, downbeat_logits = beat_activations(audio, device)
    return grid_from_activations(beat_logits, downbeat_logits, audio.shape[0] / SAMPLE_RATE, bpm_range)

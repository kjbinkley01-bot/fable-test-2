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

#: BPM values outside this range are doubled / halved into it, the convention
#: DJ software uses (a 65 BPM hip-hop track is shown as 130).
DEFAULT_BPM_RANGE = (70.0, 200.0)

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


def detect_beats(audio: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return (beat_times, downbeat_times) as detected, before grid fitting."""
    beat_logits, downbeat_logits = beat_activations(audio, device)
    beats = pick_peaks(beat_logits)
    downbeats = pick_peaks(downbeat_logits)
    if len(beats) and len(downbeats):
        # A downbeat is by definition also a beat: snap each to its nearest beat.
        nearest = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
        downbeats = np.unique(beats[nearest])
    return beats, downbeats


# -------------------------------------------------------------- grid fitting
def _mode(values: np.ndarray, default: int) -> int:
    if len(values) == 0:
        return default
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()])


def _best_ruler(beats: np.ndarray, period: float) -> tuple[float, float]:
    """Find offset ``a`` and spacing ``b`` so that a + n*b lands on the beats.

    For a candidate spacing, every beat is placed on a clock face whose full
    turn is one beat. If the spacing is right all beats point the same way;
    the length of their average is the score. We scan spacings within +/-6%
    of the typical gap between beats and keep the best.
    """
    length = max(beats[-1] - beats[0], period)
    f0 = 1.0 / period
    freqs = np.arange(f0 * 0.94, f0 * 1.06, 1.0 / (40.0 * length))
    best_f, best_score = f0, -1.0
    for chunk in np.array_split(freqs, max(1, len(freqs) * len(beats) // 2_000_000 + 1)):
        phases = np.exp(2j * np.pi * chunk[:, None] * beats[None, :])
        scores = np.abs(phases.sum(axis=1))
        i = int(scores.argmax())
        if scores[i] > best_score:
            best_score, best_f = float(scores[i]), float(chunk[i])
    b = 1.0 / best_f
    angle = np.angle(np.exp(2j * np.pi * best_f * beats).sum())
    a = (angle / (2 * np.pi)) * b  # a beat sits where the clock hand points
    return float(a), float(b)


def fit_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
    bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE,
    tolerance: float = 0.035,
) -> BeatGrid:
    """Build the beat grid from detected beats (see module docstring)."""
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 4:
        grid = BeatGrid(bpm=0.0, beats=beats, downbeats=downbeats)
        grid.notes.append("too few beats detected for a tempo")
        return grid

    intervals = np.diff(beats)
    period = float(np.median(intervals))

    # Find the evenly spaced ruler (period b, offset a) that the detected
    # beats agree with best. Scoring every candidate tempo against all beats
    # at once is robust to the tracker adding or dropping a beat here and
    # there, which would throw off a simple "count the beats" approach.
    a, b = _best_ruler(beats, period)
    index = np.round((beats - a) / b)
    resid = np.abs(beats - (a + b * index))
    inliers = resid < tolerance
    # Polish with a straight-line fit through the beats that sit on the ruler.
    for _ in range(3):
        _, first = np.unique(index[inliers], return_index=True)
        if len(first) < 8:
            break
        sel = np.flatnonzero(inliers)[first]
        b, a = (float(v) for v in np.polyfit(index[sel], beats[sel], 1))
        index = np.round((beats - a) / b)
        resid = np.abs(beats - (a + b * index))
        inliers = resid < tolerance

    on_grid = np.unique(index[inliers])
    precision = float(inliers.mean())                       # detected beats that fit
    span = index[inliers].max() - index[inliers].min() + 1 if inliers.any() else 1
    coverage = float(len(on_grid) / span)                   # ruler marks that have a beat
    confidence = min(precision, coverage)
    is_constant = len(on_grid) >= 16 and precision >= 0.8 and coverage >= 0.6 and b > 0

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
    detected_bpm = 60.0 / period
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


def analyze_beats(audio: np.ndarray, device: str = "cpu",
                  bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> BeatGrid:
    """Detect beats in (frames, 2) audio at 44.1 kHz and build the grid."""
    beats, downbeats = detect_beats(audio, device)
    return fit_grid(beats, downbeats, audio.shape[0] / SAMPLE_RATE, bpm_range)

"""Musical key detection.

Pipeline: note-energy features (keyfeatures.py) -> a small neural network
that scores all 24 keys, nudged by a classic music-theory key profile ->
the best key, with a confidence and runner-ups.

Why two ingredients: the network learned from dance music and is the more
accurate of the two there. The key profiles (Krumhansl & Kessler's
measurements of how well each note "fits" a key) know textbook harmony,
which dance music rarely uses. Adding them at a small weight leaves the
dance-music score unchanged and fixes plain chord progressions the network
alone got wrong, so it is a cheap safeguard for other genres.

The network's weights live in ``key_model.npz`` next to this file. It was
trained on ~1,400 dance tracks with human-checked key labels and scored on
600 different tracks (the public "GiantSteps" set, whose labels come from
Beatport and were corrected by musicians); see tools/key_model/ and the
README for the numbers. Evaluating it is a handful of matrix products in
numpy, so key detection needs no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE
from .camelot import camelot_code, display_key, key_name
from .keyfeatures import N_SEMITONES, semitone_spectrogram, window_features

MODEL_PATH = Path(__file__).with_name("key_model.npz")

#: Krumhansl-Kessler key profiles: perceived fit of each of the 12 notes
#: (starting from the key's home note) in a major and in a minor key.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
#: Weight of the profile match relative to the network (chosen on held-out data).
THEORY_WEIGHT = 2.0


@dataclass
class KeyResult:
    pitch_class: int                 # 0 = C ... 11 = B
    mode: str                        # "major" | "minor"
    confidence: float                # 0..1, the model's probability for this key
    tuning_cents: float = 0.0        # how far the track sits from A = 440 Hz
    alternatives: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return key_name(self.pitch_class, self.mode)

    @property
    def camelot(self) -> str:
        return camelot_code(self.pitch_class, self.mode)

    @property
    def display(self) -> str:
        return display_key(self.pitch_class, self.mode)

    def to_dict(self) -> dict:
        return {
            "pitch_class": int(self.pitch_class), "mode": self.mode,
            "name": self.name, "camelot": self.camelot,
            "confidence": round(float(self.confidence), 3),
            "tuning_cents": round(float(self.tuning_cents), 1),
            "alternatives": self.alternatives,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KeyResult":
        return cls(int(data["pitch_class"]), data["mode"], float(data.get("confidence", 0.0)),
                   float(data.get("tuning_cents", 0.0)), list(data.get("alternatives", [])))


@lru_cache(maxsize=1)
def _load_model() -> dict:
    data = np.load(MODEL_PATH)
    layers = []
    i = 0
    while f"w{i}" in data:
        layers.append((data[f"w{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)))
        i += 1
    return {
        "layers": layers,
        "shifts": [int(s) for s in data["shifts"]],
        "bank": str(data["meta_bank"]) if "meta_bank" in data else "linear",
        "window": float(data["meta_window"]),
        "hop": float(data["meta_hop"]),
    }


def _shift_bank(x: np.ndarray, shifts: list[int], kind: str) -> np.ndarray:
    """(B, 72) -> (B, 12, 72): row k is the note picture as seen from pitch class k.

    "octave": the twelve notes of every octave are rotated so that pitch
    class k comes first. "linear" (older models): the whole picture is slid
    so pitch class k lands on C.
    """
    if kind == "octave":
        octaves = x.reshape(x.shape[0], -1, 12)
        return np.stack([np.roll(octaves, -k, axis=2).reshape(x.shape[0], -1) for k in range(12)],
                        axis=1).astype(np.float32)
    out = np.zeros((x.shape[0], len(shifts), x.shape[1]), dtype=np.float32)
    for k, s in enumerate(shifts):
        if s == 0:
            out[:, k] = x
        elif s > 0:
            out[:, k, s:] = x[:, :-s]
        else:
            out[:, k, :s] = x[:, -s:]
    return out


def network_log_probabilities(windows: np.ndarray) -> np.ndarray:
    """Per-window features (n, 72) -> the network's log-probabilities for the 24 keys.

    Index 0-11: C..B major, 12-23: C..B minor. Windows vote by averaging.
    """
    model = _load_model()
    h = _shift_bank(windows.astype(np.float32), model["shifts"], model["bank"])
    for i, (w, b) in enumerate(model["layers"]):
        h = h @ w.T + b
        if i < len(model["layers"]) - 1:
            h = np.maximum(h, 0.0)
    logits = np.concatenate([h[:, :, 0], h[:, :, 1]], axis=1)          # (n, 24)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return logp.mean(axis=0)


def theory_scores(semitones: np.ndarray) -> np.ndarray:
    """How well the song's overall note content matches each key's profile (-1..1)."""
    mean = semitones.mean(axis=0)
    level = float(mean.mean())
    if level <= 1e-12:
        return np.zeros(24)
    chroma = np.log1p(10.0 * mean / level).reshape(-1, 12).sum(axis=0)   # fold octaves
    major = [np.corrcoef(chroma, np.roll(KK_MAJOR, k))[0, 1] for k in range(12)]
    minor = [np.corrcoef(chroma, np.roll(KK_MINOR, k))[0, 1] for k in range(12)]
    return np.nan_to_num(np.array(major + minor))


def key_probabilities(windows: np.ndarray, semitones: np.ndarray | None = None) -> np.ndarray:
    """Probabilities for the 24 keys (0-11 major, 12-23 minor).

    With ``semitones`` (the full note-energy spectrogram) the music-theory
    profile match is blended in; without, the network alone decides.
    """
    scores = network_log_probabilities(windows)
    if semitones is not None:
        scores = scores + THEORY_WEIGHT * theory_scores(semitones)
    p = np.exp(scores - scores.max())
    return p / p.sum()


def detect_key(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> KeyResult:
    """Detect the key of (frames, 2) or mono audio."""
    semis, tuning = semitone_spectrogram(audio, sample_rate)
    model = _load_model()
    windows = window_features(semis, model["window"], model["hop"])
    if len(windows) == 0:
        return KeyResult(0, "major", 0.0, 0.0, [])
    assert windows.shape[1] == N_SEMITONES
    probs = key_probabilities(windows, semis)
    order = np.argsort(probs)[::-1]

    def describe(idx: int) -> dict:
        pc, mode = int(idx % 12), "minor" if idx >= 12 else "major"
        return {"name": key_name(pc, mode), "camelot": camelot_code(pc, mode),
                "probability": round(float(probs[idx]), 3)}

    best = int(order[0])
    return KeyResult(
        pitch_class=best % 12,
        mode="minor" if best >= 12 else "major",
        confidence=float(probs[best]),
        tuning_cents=100.0 * tuning,
        alternatives=[describe(int(i)) for i in order[1:4]],
    )

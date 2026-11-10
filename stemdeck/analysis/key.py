"""Musical key detection.

Pipeline: note-energy features (keyfeatures.py) -> a small neural network
that scores all 24 keys -> the best key, with a confidence and runner-ups.

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
        "window": float(data["meta_window"]),
        "hop": float(data["meta_hop"]),
    }


def _shift_bank(x: np.ndarray, shifts: list[int]) -> np.ndarray:
    """(B, 72) -> (B, 12, 72); row k has pitch class k moved onto C."""
    out = np.zeros((x.shape[0], len(shifts), x.shape[1]), dtype=np.float32)
    for k, s in enumerate(shifts):
        if s == 0:
            out[:, k] = x
        elif s > 0:
            out[:, k, s:] = x[:, :-s]
        else:
            out[:, k, :s] = x[:, -s:]
    return out


def key_probabilities(windows: np.ndarray) -> np.ndarray:
    """Per-window features (n, 72) -> probabilities for the 24 keys.

    Index 0-11: C..B major, 12-23: C..B minor.
    """
    model = _load_model()
    h = _shift_bank(windows.astype(np.float32), model["shifts"])
    for i, (w, b) in enumerate(model["layers"]):
        h = h @ w.T + b
        if i < len(model["layers"]) - 1:
            h = np.maximum(h, 0.0)
    logits = np.concatenate([h[:, :, 0], h[:, :, 1]], axis=1)          # (n, 24)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    mean = logp.mean(axis=0)                                           # vote across the song
    p = np.exp(mean - mean.max())
    return p / p.sum()


def detect_key(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> KeyResult:
    """Detect the key of (frames, 2) or mono audio."""
    semis, tuning = semitone_spectrogram(audio, sample_rate)
    model = _load_model()
    windows = window_features(semis, model["window"], model["hop"])
    if len(windows) == 0:
        return KeyResult(0, "major", 0.0, 0.0, [])
    assert windows.shape[1] == N_SEMITONES
    probs = key_probabilities(windows)
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

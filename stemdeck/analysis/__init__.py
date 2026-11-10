"""Song analysis: BPM, key and beat grid.

Public entry points:
    analyze(stems)        - analyse an already separated song
    analyze_audio(audio)  - analyse a plain mix (no stems)
    analyze_file(path)    - convenience for the command line
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE, STEMS
from .beats import DEFAULT_BPM_RANGE, BeatGrid, analyze_beats
from .camelot import camelot_code, display_key, key_name, parse_key
from .key import KeyResult, detect_key

#: Bump when BPM / key / grid logic changes; cached songs are then re-analysed.
ANALYSIS_VERSION = 1

__all__ = [
    "ANALYSIS_VERSION", "AnalysisResult", "BeatGrid", "KeyResult",
    "analyze", "analyze_audio", "analyze_file",
    "camelot_code", "display_key", "key_name", "parse_key",
]


@dataclass
class AnalysisResult:
    bpm: float
    key: KeyResult
    grid: BeatGrid
    duration: float
    key_source: str = "mix"       # what the key detector listened to
    version: int = ANALYSIS_VERSION

    def to_dict(self, include_grid: bool = True) -> dict:
        return {
            "version": self.version,
            "bpm": round(float(self.bpm), 2),
            "key": self.key.to_dict(),
            "key_source": self.key_source,
            "duration": round(float(self.duration), 3),
            "grid": self.grid.to_dict(include_grid),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        return cls(
            bpm=float(data["bpm"]),
            key=KeyResult.from_dict(data["key"]),
            grid=BeatGrid.from_dict(data["grid"]),
            duration=float(data["duration"]),
            key_source=data.get("key_source", "mix"),
            version=int(data.get("version", 0)),
        )

    def summary(self) -> str:
        grid = self.grid
        lines = [
            f"BPM:  {self.bpm:.2f}" + ("" if grid.is_constant else "  (tempo varies; typical value)"),
            f"Key:  {self.key.display}   (confidence {self.key.confidence:.0%})",
        ]
        if self.key.alternatives:
            alts = ", ".join(f"{a['name']} / {a['camelot']} {a['probability']:.0%}"
                             for a in self.key.alternatives[:2])
            lines.append(f"      next best: {alts}")
        if abs(self.key.tuning_cents) >= 15:
            lines.append(f"      note: track is tuned {self.key.tuning_cents:+.0f} cents from A=440")
        lines.append(f"Grid: {len(grid.beats)} beats, {len(grid.downbeats)} bars of {grid.beats_per_bar}, "
                     f"first downbeat at {grid.anchor:.3f}s, "
                     f"{'fixed tempo' if grid.is_constant else 'follows detected beats'} "
                     f"(fit {grid.confidence:.0%})")
        for note in grid.notes:
            lines.append(f"      note: {note}")
        return "\n".join(lines)


def analyze(stems: dict[str, np.ndarray], device: str = "cpu",
            bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> AnalysisResult:
    """Analyse a separated song. ``stems`` maps stem name -> (frames, 2)."""
    # Both detectors listen to the full mix (the stems added back together).
    # Leaving the drums out for key detection was measured and made no
    # difference (see tools/README.md), so the simpler input is used; it is
    # also what the key model was trained on.
    mix = np.sum([stems[name] for name in STEMS], axis=0, dtype=np.float32)
    return analyze_audio(mix, device=device, bpm_range=bpm_range)


def analyze_audio(audio: np.ndarray, device: str = "cpu",
                  bpm_range: tuple[float, float] = DEFAULT_BPM_RANGE) -> AnalysisResult:
    """Analyse a mix: (frames, 2) float32 at 44.1 kHz."""
    grid = analyze_beats(audio, device=device, bpm_range=bpm_range)
    key = detect_key(audio)
    return AnalysisResult(bpm=grid.bpm, key=key, grid=grid,
                          duration=audio.shape[0] / SAMPLE_RATE, key_source="mix")


def analyze_file(path: str, use_stems: bool = True) -> AnalysisResult:
    """Analyse a file. With ``use_stems`` the song goes through the library
    (separated once, cached), exactly as the app does."""
    if use_stems:
        from ..library import load_track
        from ..settings import load_settings

        settings = load_settings()
        track = load_track(path, model=settings["model"], separation_options={
            "device": settings["device"], "precision": settings["precision"],
            "batch_size": settings["batch_size"],
        })
        return track.analysis
    from ..audio_io import load_audio

    return analyze_audio(load_audio(path))

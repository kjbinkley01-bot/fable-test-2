"""Shared constants and on-disk locations.

Everything that several subsystems must agree on lives here so that the
separation code, the cache, the playback engine and the UI cannot drift apart.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: The whole app works at one sample rate. 44.1 kHz is what the separation
#: models were trained on, so anything else is resampled once, on import.
SAMPLE_RATE = 44100
CHANNELS = 2

#: Display / storage order of the stems (top to bottom in the UI).
STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "other")

#: Colour per stem (hex), used by the waveform view and the mixer strips.
STEM_COLORS: dict[str, str] = {
    "vocals": "#ff5c8a",
    "drums": "#ffb020",
    "bass": "#4ea8ff",
    "other": "#4fd6a0",
}

#: Models we know how to run. "quality" is four fine-tuned networks (one per
#: stem) and gives the cleanest stems; "fast" is a single network, ~4x quicker.
MODELS: dict[str, str] = {
    "quality": "htdemucs_ft",
    "fast": "htdemucs",
}
DEFAULT_MODEL = "htdemucs_ft"


def data_dir() -> Path:
    """Root folder for cached stems, analysis results and settings.

    Override with the STEMDECK_HOME environment variable.
    """
    override = os.environ.get("STEMDECK_HOME")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(base) / "StemDeck"
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        root = Path(base) / "stemdeck"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir() -> Path:
    path = data_dir() / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return data_dir() / "settings.json"

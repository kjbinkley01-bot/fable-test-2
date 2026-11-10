"""Small persistent settings file (JSON).

The benchmark writes the fastest separation configuration it found on this
machine here, and the app reads it back, so tuning happens once.
"""
from __future__ import annotations

import json
from typing import Any

from .config import settings_path

DEFAULTS: dict[str, Any] = {
    "model": "htdemucs_ft",
    "device": "auto",        # auto | gpu | cpu
    "precision": "fp32",     # fp32 | fp16 (GPU only)
    "batch_size": 0,         # 0 = pick automatically
    "overlap": 0.25,
    "export_subtype": "PCM_24",
    "output_device": None,   # None = system default
}


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULTS)
    path = settings_path()
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                settings.update(stored)
        except (OSError, ValueError):
            pass  # a corrupt settings file must never stop the app from starting
    return settings


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    settings.update(updates)
    path = settings_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tmp.replace(path)
    return settings

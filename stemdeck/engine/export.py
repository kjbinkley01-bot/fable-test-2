"""Export stems or a custom mix as WAV.

The mix is rendered offline with exactly the gain rules the live mixer uses
(fader, mute, solo), so what you export is what you hear. If the result
would exceed full scale in a fixed-point WAV (16/24-bit), the whole file is
turned down just enough to avoid clipping, and the amount is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio_io import save_wav
from ..config import SAMPLE_RATE, STEMS
from .mixer import effective_gains, render_offline


@dataclass
class ExportInfo:
    path: str
    seconds: float
    peak_db: float            # peak level of the written file, dBFS
    attenuated_db: float      # how much the mix was turned down to avoid clipping (0 = not at all)


def _db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-12)))


def mix_stems(
    stems: dict[str, np.ndarray],
    faders: dict[str, float] | None = None,
    mutes: dict[str, bool] | None = None,
    solos: dict[str, bool] | None = None,
) -> np.ndarray:
    """Render a mix. Missing entries default to fader 1.0, not muted, not soloed."""
    names = [n for n in STEMS if n in stems]
    fader = np.array([(faders or {}).get(n, 1.0) for n in names], dtype=np.float64)
    mute = np.array([(mutes or {}).get(n, False) for n in names], dtype=bool)
    solo = np.array([(solos or {}).get(n, False) for n in names], dtype=bool)
    gains = effective_gains(fader, mute, solo)
    return render_offline(np.stack([stems[n] for n in names]), gains)


def write_audio(path: str | Path, audio: np.ndarray, subtype: str = "PCM_24") -> ExportInfo:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    attenuated = 0.0
    if subtype != "FLOAT" and peak > 0.999:
        audio = audio * (0.999 / peak)
        attenuated = _db(peak / 0.999)
        peak = 0.999
    save_wav(path, audio, SAMPLE_RATE, subtype)
    return ExportInfo(str(path), audio.shape[0] / SAMPLE_RATE, _db(peak), attenuated)


def export_mix(
    stems: dict[str, np.ndarray],
    path: str | Path,
    faders: dict[str, float] | None = None,
    mutes: dict[str, bool] | None = None,
    solos: dict[str, bool] | None = None,
    subtype: str = "PCM_24",
) -> ExportInfo:
    """Write one WAV containing the mix of the audible stems."""
    return write_audio(path, mix_stems(stems, faders, mutes, solos), subtype)


def export_stems(
    stems: dict[str, np.ndarray],
    folder: str | Path,
    base_name: str,
    names: list[str] | None = None,
    subtype: str = "PCM_24",
) -> list[ExportInfo]:
    """Write each chosen stem as its own WAV: '<base_name> - <stem>.wav'.

    All stems share one attenuation (if any is needed), so they still add up
    to the original mix when layered in another program.
    """
    names = names or list(STEMS)
    folder = Path(folder)
    peak = max(float(np.abs(stems[n]).max()) for n in names)
    scale = 1.0
    if subtype != "FLOAT" and peak > 0.999:
        scale = 0.999 / peak
    out = []
    for name in names:
        audio = stems[name] * scale if scale != 1.0 else stems[name]
        path = folder / f"{base_name} - {name}.wav"
        save_wav(path, audio, SAMPLE_RATE, subtype)
        out.append(ExportInfo(str(path), audio.shape[0] / SAMPLE_RATE,
                              _db(float(np.abs(audio).max())), -_db(scale)))
    return out

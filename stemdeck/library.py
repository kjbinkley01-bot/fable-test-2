"""The track library: load a song once, reuse the results forever.

A song is identified by the SHA-256 hash of the file's contents, not its
name or location, so renaming or moving a file still hits the cache and two
different files never collide.

Layout on disk (under config.cache_dir()):

    <hash[:2]>/<hash>/
        source.json                 file name, duration
        <model>/
            stems/vocals.flac ...   lossless 24-bit stems
            peaks.npz               waveform overviews for drawing
            analysis.json           BPM, key, beat grid
            meta.json               written last: marks the entry complete

Everything is written into a temporary folder and renamed into place, and
``meta.json`` is written last, so an entry that exists is always complete -
a crash or power cut half way through can not leave a broken song behind.

Analysis is versioned separately from separation: when the BPM/key code
improves, cached songs are re-analysed (a few seconds) without redoing the
expensive stem separation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .analysis import ANALYSIS_VERSION, AnalysisResult, analyze
from .analysis.waveform import Peaks, compute_peaks, peaks_from_arrays, peaks_to_arrays
from .audio_io import load_audio
from .config import DEFAULT_MODEL, SAMPLE_RATE, STEMS, cache_dir

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1

#: progress(stage, fraction, message). stage: hash | decode | separate | analyze | save | load
StageProgress = Callable[[str, float, str], None]


class LoadCancelled(Exception):
    pass


@dataclass
class Track:
    track_id: str                         # content hash
    title: str
    source_path: str
    model: str
    stems: dict[str, np.ndarray]          # name -> (frames, 2) float32
    peaks: dict[str, Peaks]
    analysis: AnalysisResult
    separation: dict = field(default_factory=dict)   # stats of the run that made the stems
    from_cache: bool = False
    load_seconds: float = 0.0

    @property
    def n_frames(self) -> int:
        return next(iter(self.stems.values())).shape[0]

    @property
    def duration(self) -> float:
        return self.n_frames / SAMPLE_RATE

    def stem_array(self) -> np.ndarray:
        """(n_stems, frames, 2) in config.STEMS order, for the playback engine."""
        return np.stack([self.stems[name] for name in STEMS])


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def entry_dir(track_id: str, model: str) -> Path:
    return cache_dir() / track_id[:2] / track_id / model


def is_cached(track_id: str, model: str) -> bool:
    return (entry_dir(track_id, model) / "meta.json").is_file()


# ----------------------------------------------------------------- saving
def _write_stems(folder: Path, stems: dict[str, np.ndarray]) -> float:
    """Write 24-bit FLAC. Returns the scale the audio was divided by.

    FLAC stores values between -1 and 1. Stems can overshoot slightly, so if
    needed every stem is divided by one shared factor, which is recorded and
    multiplied back in on load - the round trip never clips.
    """
    peak = max(float(np.abs(s).max()) for s in stems.values())
    scale = peak / 0.999 if peak > 0.999 else 1.0
    folder.mkdir(parents=True, exist_ok=True)

    def write(name: str) -> None:
        data = stems[name] if scale == 1.0 else stems[name] / scale
        sf.write(str(folder / f"{name}.flac"), data, SAMPLE_RATE, subtype="PCM_24", format="FLAC")

    with ThreadPoolExecutor(len(stems)) as pool:
        list(pool.map(write, stems))
    return scale


def _save_analysis(folder: Path, analysis: AnalysisResult) -> None:
    tmp = folder / "analysis.json.tmp"
    tmp.write_text(json.dumps(analysis.to_dict(include_grid=True)), encoding="utf-8")
    tmp.replace(folder / "analysis.json")


def save_entry(track_id: str, model: str, source: Path, stems: dict[str, np.ndarray],
               peaks: dict[str, Peaks], analysis: AnalysisResult, separation: dict) -> None:
    final = entry_dir(track_id, model)
    tmp = final.parent / f".tmp-{model}-{uuid.uuid4().hex[:8]}"
    try:
        scale = _write_stems(tmp / "stems", stems)
        arrays: dict[str, np.ndarray] = {}
        for name, pk in peaks.items():
            arrays.update(peaks_to_arrays(pk, name))
        np.savez(tmp / "peaks.npz", **arrays)
        _save_analysis(tmp, analysis)
        (tmp / "meta.json").write_text(json.dumps({
            "format": CACHE_FORMAT_VERSION,
            "track_id": track_id,
            "model": model,
            "title": source.stem,
            "source_name": source.name,
            "source_path": str(source),
            "n_frames": int(next(iter(stems.values())).shape[0]),
            "sample_rate": SAMPLE_RATE,
            "stem_scale": scale,
            "separation": separation,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2), encoding="utf-8")
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        tmp.replace(final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- loading
def load_entry(track_id: str, model: str, source_path: str = "",
               progress: StageProgress | None = None) -> Track:
    folder = entry_dir(track_id, model)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != CACHE_FORMAT_VERSION:
        raise ValueError("cache entry was written by an incompatible version")
    scale = float(meta.get("stem_scale", 1.0))

    def read(name: str) -> np.ndarray:
        data, rate = sf.read(str(folder / "stems" / f"{name}.flac"), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or data.shape[0] != meta["n_frames"]:
            raise ValueError(f"cached stem '{name}' does not match its metadata")
        if scale != 1.0:
            data *= scale
        return data

    if progress:
        progress("load", 0.1, "Loading stems from cache")
    with ThreadPoolExecutor(len(STEMS)) as pool:
        stems = dict(zip(STEMS, pool.map(read, STEMS)))

    with np.load(folder / "peaks.npz") as arrays:
        peaks = {name: peaks_from_arrays(arrays, name) for name in STEMS}

    analysis_data = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    if analysis_data.get("version") == ANALYSIS_VERSION:
        analysis = AnalysisResult.from_dict(analysis_data)
    else:
        # The analysis code has changed since this was cached: redo just that.
        if progress:
            progress("analyze", 0.5, "Updating BPM and key (analysis was improved)")
        analysis = analyze(stems)
        _save_analysis(folder, analysis)
    return Track(
        track_id=track_id, title=meta.get("title", ""),
        source_path=source_path or meta.get("source_path", ""), model=model,
        stems=stems, peaks=peaks, analysis=analysis,
        separation=meta.get("separation", {}), from_cache=True,
    )


def load_track(
    path: str | Path,
    model: str = DEFAULT_MODEL,
    separation_options: dict | None = None,
    progress: StageProgress | None = None,
    cancel: Callable[[], bool] | None = None,
    force: bool = False,
) -> Track:
    """Open a song: from the cache if we have seen it, otherwise do the work.

    ``separation_options`` is passed to ``stemdeck.separation.separate``
    (device, precision, batch_size ...).
    """
    path = Path(path)
    started = time.perf_counter()

    def report(stage: str, fraction: float, message: str) -> None:
        if cancel is not None and cancel():
            raise LoadCancelled()
        if progress is not None:
            progress(stage, fraction, message)

    report("hash", 0.0, "Identifying file")
    track_id = file_hash(path)

    if not force and is_cached(track_id, model):
        try:
            track = load_entry(track_id, model, str(path), progress)
            track.load_seconds = time.perf_counter() - started
            return track
        except Exception as exc:  # damaged entry: rebuild it rather than fail
            logger.warning("Cache entry for %s is unusable (%s); rebuilding", path.name, exc)

    report("decode", 0.0, "Reading audio")
    audio = load_audio(path)

    from .separation import SeparationCancelled, separate

    report("separate", 0.0, "Separating stems")
    try:
        stems, stats = separate(
            audio, model=model,
            progress=lambda f: report("separate", f, "Separating stems"),
            cancel=cancel,
            **(separation_options or {}),
        )
    except SeparationCancelled as exc:
        raise LoadCancelled() from exc

    report("analyze", 0.0, "Detecting BPM, key and beats")
    # Analysis runs on the CPU: it takes a few seconds per song there, and it
    # keeps the GPU code path limited to the one thing that needs it.
    analysis = analyze(stems, device="cpu")

    report("save", 0.0, "Saving to library")
    peaks = {name: compute_peaks(stems[name]) for name in STEMS}
    try:
        save_entry(track_id, model, path, stems, peaks, analysis, stats.to_dict())
    except OSError as exc:  # disk full etc.: the song still plays, just is not cached
        logger.warning("Could not cache %s: %s", path.name, exc)

    return Track(
        track_id=track_id, title=path.stem, source_path=str(path), model=model,
        stems=stems, peaks=peaks, analysis=analysis, separation=stats.to_dict(),
        from_cache=False, load_seconds=time.perf_counter() - started,
    )


def save_analysis(track: Track) -> None:
    """Store a changed analysis (e.g. tempo doubled by hand) for a cached song."""
    folder = entry_dir(track.track_id, track.model)
    if (folder / "meta.json").is_file():
        _save_analysis(folder, track.analysis)


def cache_size_bytes() -> int:
    return sum(p.stat().st_size for p in cache_dir().rglob("*") if p.is_file())


def clear_cache() -> None:
    shutil.rmtree(cache_dir(), ignore_errors=True)

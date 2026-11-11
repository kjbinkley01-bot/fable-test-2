"""Reading and writing audio files.

All audio inside StemDeck is float32, shape (frames, 2), at config.SAMPLE_RATE.
Converting once at the door keeps every other module simple.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import CHANNELS, SAMPLE_RATE

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif")

#: WAV flavours offered on export.
WAV_SUBTYPES = {
    "16-bit": "PCM_16",
    "24-bit": "PCM_24",
    "32-bit float": "FLOAT",
}


class AudioLoadError(RuntimeError):
    pass


def _not_audio_reason(path: Path) -> str | None:
    """Spot files that are plainly not audio, to give a useful message.

    The classic case is a failed download: the web server's error page
    ("Not Found", or some HTML) gets saved under the song's name, so the file
    looks like an MP3 in Explorer but holds a few bytes of text.
    """
    size = path.stat().st_size
    if size == 0:
        return "the file is empty (0 bytes). The download or copy probably did not finish."
    with open(path, "rb") as handle:
        head = handle.read(512)
    signatures = (b"RIFF", b"fLaC", b"OggS", b"ID3", b"FORM")
    looks_like_audio = (
        head.startswith(signatures)
        or head[4:8] == b"ftyp"                                   # m4a / mp4
        or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)   # bare MP3 frame
    )
    if looks_like_audio:
        return None
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
    if printable >= 0.95 * len(head):
        preview = head[:60].decode("ascii", errors="replace").strip().replace("\n", " ")
        return (f"it is not audio: the file is {size:,} bytes of text starting with "
                f"\"{preview}\". This usually means a download failed and the error "
                "message was saved under the song's name. Download the song again.")
    if size < 1024:
        return f"the file is only {size:,} bytes, far too small to be a song."
    return None


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        # Surround files: keep front left / right.
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return data
    import soxr

    return np.ascontiguousarray(
        soxr.resample(data, src_rate, dst_rate, quality="VHQ"), dtype=np.float32
    )


def _read_with_ffmpeg(path: Path) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError(
            f"Could not decode '{path.name}'. This format needs ffmpeg, which is not installed."
        )
    cmd = [
        ffmpeg, "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioLoadError(
            f"ffmpeg could not decode '{path.name}': {proc.stderr.decode(errors='replace')[:400]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, CHANNELS).copy()


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to float32 stereo at ``sample_rate``.

    Returns an array of shape (frames, 2).
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")
    reason = _not_audio_reason(path)
    if reason:
        raise AudioLoadError(f"Cannot open '{path.name}': {reason}")
    try:
        data, src_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile cannot read it (e.g. m4a) -> try ffmpeg
        try:
            data = _read_with_ffmpeg(path)
            src_rate = SAMPLE_RATE
        except AudioLoadError as ff_exc:
            raise AudioLoadError(
                f"Cannot open '{path.name}': the file is damaged or in a format StemDeck "
                f"cannot read (it reads MP3, WAV, FLAC, OGG and AIFF). Details: {ff_exc}"
            ) from exc
    if data.shape[0] == 0:
        raise AudioLoadError(f"'{path.name}' contains no audio.")
    data = _to_stereo(data)
    data = _resample(data, src_rate, sample_rate)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = "PCM_24",
) -> None:
    """Write (frames, channels) float audio as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype=subtype, format="WAV")


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return audio.shape[0] / float(sample_rate)

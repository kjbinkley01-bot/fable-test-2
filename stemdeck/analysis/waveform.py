"""Pre-computed waveform overviews for drawing.

Drawing a waveform means showing, for each pixel column, the lowest and
highest sample in the slice of audio that column covers. Scanning millions of
samples on every repaint would be far too slow, so once per song we store
min/max for blocks of 64 samples, then for blocks of 256, 1024 ... (a
"pyramid"). The view picks the level closest to the current zoom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASE_BLOCK = 64
LEVEL_FACTOR = 4
N_LEVELS = 6          # 64, 256, 1024, 4096, 16384, 65536 samples per block


@dataclass
class Peaks:
    """levels[i] has shape (n_blocks_i, 2): min and max of the mono signal."""
    levels: list[np.ndarray]
    n_frames: int

    def block_size(self, level: int) -> int:
        return BASE_BLOCK * LEVEL_FACTOR ** level

    def level_for(self, frames_per_pixel: float) -> int:
        """Finest level whose blocks are not larger than one pixel."""
        level = 0
        while level + 1 < len(self.levels) and self.block_size(level + 1) <= frames_per_pixel:
            level += 1
        return level

    def columns(self, start_frame: float, frames_per_pixel: float, n_pixels: int) -> np.ndarray:
        """(n_pixels, 2) min/max for pixel columns starting at ``start_frame``.

        Columns outside the audio are zero.
        """
        out = np.zeros((n_pixels, 2), dtype=np.float32)
        if n_pixels <= 0 or frames_per_pixel <= 0:
            return out
        level = self.level_for(frames_per_pixel)
        data = self.levels[level]
        block = self.block_size(level)
        edges = (start_frame + np.arange(n_pixels + 1) * frames_per_pixel) / block
        lo = np.floor(edges[:-1]).astype(np.int64)
        hi = np.maximum(np.ceil(edges[1:]).astype(np.int64), lo + 1)
        valid = (hi > 0) & (lo < len(data))
        if not valid.any():
            return out
        lo_c = np.clip(lo, 0, len(data) - 1)
        hi_c = np.clip(hi, 1, len(data))
        # A column covers blocks [lo, hi): take the min of mins and max of maxes.
        idx = np.flatnonzero(valid)
        starts = lo_c[idx]
        ends = hi_c[idx]
        width = int((ends - starts).max())
        gather = np.minimum(starts[:, None] + np.arange(width)[None, :], len(data) - 1)
        mask = (starts[:, None] + np.arange(width)[None, :]) < ends[:, None]
        mins = np.where(mask, data[gather, 0], np.inf).min(axis=1)
        maxs = np.where(mask, data[gather, 1], -np.inf).max(axis=1)
        out[idx, 0] = mins
        out[idx, 1] = maxs
        return out


def compute_peaks(audio: np.ndarray) -> Peaks:
    """Build the min/max pyramid for (frames, channels) audio."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    n = len(mono)
    n_blocks = max(1, int(np.ceil(n / BASE_BLOCK)))
    padded = np.zeros(n_blocks * BASE_BLOCK, dtype=np.float32)
    padded[:n] = mono
    blocks = padded.reshape(n_blocks, BASE_BLOCK)
    level = np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=1)
    levels = [level]
    for _ in range(N_LEVELS - 1):
        prev = levels[-1]
        m = int(np.ceil(len(prev) / LEVEL_FACTOR))
        pad = m * LEVEL_FACTOR - len(prev)
        if pad:
            prev = np.concatenate([prev, np.repeat(prev[-1:], pad, axis=0)])
        grouped = prev.reshape(m, LEVEL_FACTOR, 2)
        levels.append(np.stack([grouped[:, :, 0].min(axis=1), grouped[:, :, 1].max(axis=1)], axis=1))
    return Peaks([lv.astype(np.float32) for lv in levels], n)


def peaks_to_arrays(peaks: Peaks, prefix: str) -> dict[str, np.ndarray]:
    arrays = {f"{prefix}_n": np.asarray(peaks.n_frames)}
    for i, level in enumerate(peaks.levels):
        arrays[f"{prefix}_{i}"] = level.astype(np.float16)
    return arrays


def peaks_from_arrays(arrays, prefix: str) -> Peaks:
    levels = []
    i = 0
    while f"{prefix}_{i}" in arrays:
        levels.append(np.asarray(arrays[f"{prefix}_{i}"], dtype=np.float32))
        i += 1
    return Peaks(levels, int(arrays[f"{prefix}_n"]))

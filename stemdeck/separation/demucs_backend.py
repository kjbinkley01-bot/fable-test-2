"""Stem separation with (HT)Demucs on PyTorch.

Why we do not just call ``demucs.apply.apply_model``
----------------------------------------------------
Demucs cuts a song into ~7.8 second windows that overlap, runs the network on
each window and cross-fades the results. The reference implementation feeds
the windows to the network one at a time. A modern GPU is mostly idle that
way, so here the windows are stacked and processed several at once (a
"batch"). The maths is otherwise identical to the reference - same windows,
same padding, same cross-fade - and ``tests/test_separation.py`` checks that
the output matches the reference implementation to within rounding error.

Every batch has exactly the same shape (the last one is padded with silence).
GPU libraries, AMD's MIOpen in particular, pick and compile their fastest
routine per input shape the first time they see it, so constant shapes mean
that cost is paid once instead of in the middle of a song.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..config import DEFAULT_MODEL, SAMPLE_RATE, STEMS
from .devices import DeviceInfo

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float], None]
CancelFn = Callable[[], bool]


class SeparationCancelled(Exception):
    pass


class SeparationOutputError(RuntimeError):
    """The model ran but its output is unusable (NaN, or does not resemble the mix)."""


@dataclass
class SeparationStats:
    model: str
    backend: str               # rocm | cuda | cpu
    device_kind: str           # gpu | cpu
    device_name: str
    precision: str
    batch_size: int
    overlap: float
    audio_seconds: float
    load_seconds: float = 0.0
    separate_seconds: float = 0.0
    realtime_factor: float = 0.0       # seconds of music separated per second
    peak_vram_mb: float | None = None
    reconstruction_db: float | None = None
    fallback_reason: str | None = None  # set when the GPU was wanted but not used
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        where = "GPU" if self.device_kind == "gpu" else "CPU"
        text = (f"{self.audio_seconds:.1f}s of audio separated in {self.separate_seconds:.1f}s "
                f"on {where} ({self.device_name}, {self.backend}, {self.precision}, "
                f"batch {self.batch_size}) = {self.realtime_factor:.1f}x realtime")
        if self.fallback_reason:
            text += f"\n  !! GPU NOT USED: {self.fallback_reason}"
        return text


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def default_batch_size(device: DeviceInfo) -> int:
    """A safe starting batch size; the benchmark can find a better one."""
    if not device.is_gpu:
        return 1
    vram = device.vram_gb or 8.0
    if vram >= 15:
        return 4
    if vram >= 10:
        return 2
    return 1


class DemucsSeparator:
    """Loads a Demucs model once and separates songs with it."""

    def __init__(
        self,
        device: DeviceInfo,
        model_name: str = DEFAULT_MODEL,
        precision: str = "fp32",
        batch_size: int = 0,
        overlap: float = 0.25,
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be fp32 or fp16, not {precision!r}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if precision == "fp16" and not device.is_gpu:
            precision = "fp32"  # half precision only makes sense on a GPU
        self.device = device
        self.model_name = model_name
        self.precision = precision
        self.batch_size = batch_size or default_batch_size(device)
        self.overlap = overlap
        self.load_seconds = 0.0
        self._bag = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._bag is not None:
            return
        import torch
        from demucs.pretrained import get_model

        start = time.perf_counter()
        bag = get_model(self.model_name)
        bag.eval()
        for model in bag.models:
            model.to(self.device.torch_device)
            for param in model.parameters():
                param.requires_grad_(False)
        if bag.samplerate != SAMPLE_RATE:
            raise RuntimeError(
                f"Model {self.model_name} works at {bag.samplerate} Hz, StemDeck expects {SAMPLE_RATE}."
            )
        missing = [s for s in STEMS if s not in bag.sources]
        if missing:
            raise RuntimeError(f"Model {self.model_name} does not produce stems: {missing}")
        self._bag = bag
        if self.device.is_gpu:
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start

    def unload(self) -> None:
        self._bag = None
        if self.device.is_gpu:
            import torch

            torch.cuda.empty_cache()

    # -------------------------------------------------------------- separate
    def separate(
        self,
        audio: np.ndarray,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> tuple[dict[str, np.ndarray], SeparationStats]:
        """Split ``audio`` (frames, 2) float32 @ 44.1 kHz into stems.

        Returns ({stem name: (frames, 2) float32}, stats).
        """
        import torch

        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(f"expected audio of shape (frames, 2), got {audio.shape}")
        self.load()
        bag = self._bag
        dev = self.device.torch_device
        is_gpu = self.device.is_gpu
        if is_gpu:
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        mix = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))  # (2, L)
        length = mix.shape[-1]

        # Same input normalisation as the Demucs command line tool.
        ref = mix.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8
        mix = (mix - mean) / std

        n_sources = len(bag.sources)
        estimates = torch.zeros(n_sources, 2, length)
        totals = [0.0] * n_sources

        segments = [self._segment_length(m) for m in bag.models]
        plans = [self._plan(length, seg) for seg in segments]
        total_steps = sum(len(p) for p in plans)
        done_steps = 0

        for model, weights, seg_len, offsets in zip(bag.models, bag.weights, segments, plans):
            active = [k for k, w in enumerate(weights) if w != 0]
            out = torch.zeros(len(active), 2, length)
            sum_weight = torch.zeros(length)
            # Triangular cross-fade window, as in demucs.apply.apply_model.
            window = torch.cat([
                torch.arange(1, seg_len // 2 + 1),
                torch.arange(seg_len - seg_len // 2, 0, -1),
            ]).float()
            window = window / window.max()

            index = 0
            while index < len(offsets):
                if cancel is not None and cancel():
                    raise SeparationCancelled()
                batch_offsets = offsets[index:index + self.batch_size]
                try:
                    result = self._run_batch(model, mix, batch_offsets, seg_len, dev, active)
                except Exception as exc:
                    if _is_oom(exc) and self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        logger.warning("GPU ran out of memory; retrying with batch size %d",
                                       self.batch_size)
                        if is_gpu:
                            torch.cuda.empty_cache()
                        continue
                    raise
                for row, offset in enumerate(batch_offsets):
                    chunk_len = min(seg_len, length - offset)
                    delta = seg_len - chunk_len
                    piece = result[row, :, :, delta // 2: delta // 2 + chunk_len]
                    out[:, :, offset:offset + chunk_len] += window[:chunk_len] * piece
                    sum_weight[offset:offset + chunk_len] += window[:chunk_len]
                index += len(batch_offsets)
                done_steps += len(batch_offsets)
                if progress is not None:
                    progress(done_steps / total_steps)

            out /= sum_weight
            for row, k in enumerate(active):
                estimates[k] += weights[k] * out[row]
                totals[k] += weights[k]
            del out

        for k in range(n_sources):
            if totals[k] > 0:
                estimates[k] /= totals[k]
        estimates = estimates * std + mean

        if is_gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        stems: dict[str, np.ndarray] = {}
        for name in STEMS:
            k = bag.sources.index(name)
            stems[name] = np.ascontiguousarray(estimates[k].numpy().T)

        stats = SeparationStats(
            model=self.model_name,
            backend=self.device.backend,
            device_kind=self.device.kind,
            device_name=self.device.name,
            precision=self.precision,
            batch_size=self.batch_size,
            overlap=self.overlap,
            audio_seconds=length / SAMPLE_RATE,
            load_seconds=self.load_seconds,
            separate_seconds=elapsed,
            realtime_factor=(length / SAMPLE_RATE) / max(elapsed, 1e-9),
        )
        if is_gpu:
            stats.peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
        stats.reconstruction_db = check_output(audio, stems)
        return stems, stats

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _segment_length(model) -> int:
        return int(model.samplerate * model.segment)

    def _plan(self, length: int, segment_length: int) -> list[int]:
        stride = int((1 - self.overlap) * segment_length)
        return list(range(0, length, stride))

    def _run_batch(self, model, mix, offsets, seg_len, dev, active):
        """Run the network on the windows starting at ``offsets``.

        Returns a CPU float32 tensor (len(offsets), len(active), 2, seg_len).
        """
        import torch
        import torch.nn.functional as F

        length = mix.shape[-1]
        batch = torch.zeros(self.batch_size, 2, seg_len)
        for row, offset in enumerate(offsets):
            # Mirror demucs' TensorChunk.padded(): a window shorter than the
            # model's segment (only at the end of the song) is centred, taking
            # real audio as context where there is some and silence elsewhere.
            chunk_len = min(seg_len, length - offset)
            delta = seg_len - chunk_len
            begin = offset - delta // 2
            end = begin + seg_len
            lo, hi = max(0, begin), min(length, end)
            batch[row] = F.pad(mix[:, lo:hi], (lo - begin, end - hi))

        use_fp16 = self.precision == "fp16" and self.device.is_gpu
        with torch.inference_mode():
            x = batch.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda" if self.device.is_gpu else "cpu",
                                dtype=torch.float16 if use_fp16 else torch.bfloat16,
                                enabled=use_fp16):
                y = model(x)
            y = y[:len(offsets), active].float().cpu()
        return y


def check_output(mix: np.ndarray, stems: dict[str, np.ndarray]) -> float:
    """Sanity-check the stems and return how well they add back up to the song.

    The four stems of a healthy separation sum to something very close to the
    original. The result is the leftover error in dB relative to the song
    (more negative is better; a healthy run is typically around -20 dB or
    lower). A broken GPU code path tends to produce NaNs or noise instead,
    which this catches before bad stems are cached.
    """
    total = np.zeros_like(mix, dtype=np.float64)
    for name, stem in stems.items():
        if not np.isfinite(stem).all():
            raise SeparationOutputError(f"stem '{name}' contains NaN/Inf values")
        total += stem
    mix_power = float(np.mean(np.square(mix, dtype=np.float64)))
    if mix_power < 1e-12:
        return float("-inf")  # silence in, silence out
    err_power = float(np.mean(np.square(total - mix)))
    db = 10.0 * np.log10(max(err_power, 1e-20) / mix_power)
    if db > -6.0:
        raise SeparationOutputError(
            f"stems do not add back up to the song (error {db:.1f} dB); output is not trustworthy"
        )
    return db

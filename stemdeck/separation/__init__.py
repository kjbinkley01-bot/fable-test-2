"""Stem separation: pick a device, run the model, fall back to CPU loudly.

Public entry point: :func:`separate`.

Backends are small classes with ``load()`` and ``separate(audio, progress,
cancel)``. Today there is one (Demucs on PyTorch, which covers AMD ROCm,
NVIDIA CUDA and CPU). A different engine - for example ONNX Runtime with
DirectML - slots in by adding a class with the same two methods and a branch
in :func:`get_separator`.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from ..config import DEFAULT_MODEL
from .demucs_backend import (
    CancelFn,
    DemucsSeparator,
    ProgressFn,
    SeparationCancelled,
    SeparationOutputError,
    SeparationStats,
    check_output,
    default_batch_size,
)
from .devices import DeviceInfo, choose_device, cpu_device, environment_report, probe_gpu

logger = logging.getLogger(__name__)

__all__ = [
    "separate", "get_separator", "DemucsSeparator", "SeparationStats",
    "SeparationCancelled", "SeparationOutputError", "GpuRequiredError",
    "DeviceInfo", "choose_device", "cpu_device", "probe_gpu",
    "environment_report", "check_output", "default_batch_size",
]

_cache: dict[tuple, DemucsSeparator] = {}
_cache_lock = threading.Lock()


class GpuRequiredError(RuntimeError):
    """Raised instead of falling back when the caller insisted on the GPU."""


def get_separator(
    device: DeviceInfo,
    model: str = DEFAULT_MODEL,
    precision: str = "fp32",
    batch_size: int = 0,
    overlap: float = 0.25,
) -> DemucsSeparator:
    """Return a (cached) loaded-on-demand separator for this configuration."""
    key = (device.torch_device, model, precision, batch_size, overlap)
    with _cache_lock:
        separator = _cache.get(key)
        if separator is None:
            # Keep at most one model per device in memory.
            for old_key in [k for k in _cache if k[0] == device.torch_device]:
                _cache.pop(old_key).unload()
            separator = DemucsSeparator(device, model, precision, batch_size, overlap)
            _cache[key] = separator
        return separator


def separate(
    audio: np.ndarray,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    precision: str = "fp32",
    batch_size: int = 0,
    overlap: float = 0.25,
    require_gpu: bool = False,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> tuple[dict[str, np.ndarray], SeparationStats]:
    """Separate ``audio`` into stems on the best available device.

    Order of preference: GPU, then CPU. If the GPU cannot be used - not
    detected, fails its self-test, crashes mid-song or produces garbage - the
    song is separated on the CPU instead and ``stats.fallback_reason`` says
    exactly why. Nothing is downgraded silently. With ``require_gpu=True`` a
    GPU problem raises :class:`GpuRequiredError` instead.
    """
    chosen, fallback_reason = choose_device(device)
    if fallback_reason and require_gpu:
        raise GpuRequiredError(fallback_reason)
    if fallback_reason:
        logger.warning("GPU not available, using CPU: %s", fallback_reason)

    if chosen.is_gpu:
        try:
            separator = get_separator(chosen, model, precision, batch_size, overlap)
            return separator.separate(audio, progress, cancel)
        except SeparationCancelled:
            raise
        except Exception as exc:
            fallback_reason = f"GPU run failed ({type(exc).__name__}: {exc})"
            if require_gpu:
                raise GpuRequiredError(fallback_reason) from exc
            logger.warning("%s - retrying on CPU", fallback_reason, exc_info=True)
            with _cache_lock:
                for key in [k for k in _cache if k[0] == chosen.torch_device]:
                    _cache.pop(key).unload()
            if progress is not None:
                progress(0.0)

    separator = get_separator(cpu_device(), model, "fp32", 1, overlap)
    stems, stats = separator.separate(audio, progress, cancel)
    stats.fallback_reason = fallback_reason
    return stems, stats

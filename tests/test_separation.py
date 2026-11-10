"""Separation core: must match the reference Demucs implementation.

Run with:  python -m pytest tests/test_separation.py -s
These download the 'htdemucs' model (~80 MB) on first run.
"""
import numpy as np
import pytest
import torch

from stemdeck.config import SAMPLE_RATE, STEMS
from stemdeck.separation import DemucsSeparator, check_output, cpu_device, separate
from stemdeck.separation.demucs_backend import SeparationOutputError


def _test_signal(seconds: float) -> np.ndarray:
    """Synthetic but music-like: bass tone, chord, noise bursts as 'drums'."""
    rng = np.random.default_rng(0)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    chord = 0.1 * sum(np.sin(2 * np.pi * f * t) for f in (220, 277.2, 329.6))
    env = (np.modf(t * 2)[0] < 0.05).astype(float)
    drums = 0.3 * rng.standard_normal(n) * env
    left = bass + chord + drums
    right = bass + 0.8 * chord + drums
    return np.stack([left, right], axis=1).astype(np.float32)


def _reference(audio: np.ndarray, model_name: str) -> dict[str, np.ndarray]:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    wav = torch.from_numpy(audio.T.copy())
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(bag, ((wav - mean) / std)[None], shifts=0, split=True,
                          overlap=0.25, device="cpu")[0]
    out = out * std + mean
    return {name: out[bag.sources.index(name)].numpy().T for name in STEMS}


@pytest.mark.parametrize("seconds,batch", [(19.3, 1), (19.3, 3), (3.1, 2)])
def test_matches_reference_demucs(seconds, batch):
    audio = _test_signal(seconds)
    sep = DemucsSeparator(cpu_device(), "htdemucs", batch_size=batch)
    stems, stats = sep.separate(audio)
    reference = _reference(audio, "htdemucs")
    for name in STEMS:
        assert stems[name].shape == audio.shape
        err = np.abs(stems[name] - reference[name]).max()
        scale = np.abs(reference[name]).max() + 1e-9
        assert err / scale < 1e-3, f"{name}: relative error {err / scale:.2e}"
    assert stats.device_kind == "cpu"
    assert stats.reconstruction_db < -6


def test_progress_and_cancel():
    from stemdeck.separation import SeparationCancelled

    audio = _test_signal(12.0)
    sep = DemucsSeparator(cpu_device(), "htdemucs", batch_size=1)
    seen = []
    sep.separate(audio, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    with pytest.raises(SeparationCancelled):
        sep.separate(audio, cancel=lambda: True)


def test_silence_does_not_crash():
    audio = np.zeros((SAMPLE_RATE * 2, 2), dtype=np.float32)
    stems, _ = DemucsSeparator(cpu_device(), "htdemucs").separate(audio)
    assert all(np.isfinite(s).all() for s in stems.values())


def test_garbage_output_is_rejected():
    audio = _test_signal(1.0)
    bad = {name: np.random.default_rng(1).standard_normal(audio.shape).astype(np.float32)
           for name in STEMS}
    with pytest.raises(SeparationOutputError):
        check_output(audio, bad)
    bad["vocals"][10, 0] = np.nan
    with pytest.raises(SeparationOutputError):
        check_output(audio, bad)


def test_cpu_fallback_is_reported(monkeypatch):
    """When the GPU path blows up we still get stems, and are told why."""
    import stemdeck.separation as sepmod
    from stemdeck.separation.devices import DeviceInfo

    fake_gpu = DeviceInfo("gpu", "rocm", "cuda:0", "Fake Radeon", 16.0)
    monkeypatch.setattr(sepmod, "choose_device", lambda pref: (fake_gpu, None))
    audio = _test_signal(2.0)
    stems, stats = separate(audio, model="htdemucs")
    assert stats.device_kind == "cpu"
    assert stats.fallback_reason and "GPU run failed" in stats.fallback_reason
    with pytest.raises(sepmod.GpuRequiredError):
        separate(audio, model="htdemucs", require_gpu=True)

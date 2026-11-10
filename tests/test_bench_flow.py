"""Benchmark and GPU code paths, exercised without a GPU.

The machine that runs these tests has no GPU, so a pretend one is used: it
is labelled as a ROCm GPU but computes on the CPU. That cannot say anything
about GPU speed or GPU numerics, but it runs every line of the GPU branches
(sweep, half precision switch, memory stats, GPU-vs-CPU check, saving the
best configuration), which catches plain programming mistakes before they
reach the real card.
"""
import json

import pytest
import torch

import stemdeck.separation as sepmod
import stemdeck.separation.devices as devmod
from stemdeck.separation.devices import DeviceInfo


@pytest.fixture
def pretend_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("STEMDECK_HOME", str(tmp_path / "home"))
    fake = DeviceInfo("gpu", "rocm", "cpu", "Pretend Radeon", 16.0, ["test double"])
    monkeypatch.setattr(devmod, "probe_gpu", lambda run_smoke_test=True: (fake, None))
    monkeypatch.setattr(sepmod, "probe_gpu", lambda run_smoke_test=True: (fake, None))
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 3 * 2**30)
    return fake


def test_benchmark_sweep_runs_end_to_end(pretend_gpu, capsys, tmp_path):
    from stemdeck.bench import main
    from stemdeck.settings import load_settings

    code = main(["--seconds", "3", "--model", "htdemucs", "--sweep", "--runs", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Pretend Radeon" in out and "self-test passed" in out
    assert out.count("warm run 1") == 8                    # fp32/fp16 x batch 1,2,4,8
    assert "difference from fp32 result" in out
    assert "GPU vs CPU agreement" in out and "match" in out
    assert "Ran on:        GPU" in out
    # The pretend GPU computes at CPU speed, so either verdict is legitimate here.
    assert "RESULT: PASS" in out or "RESULT: GPU WORKS BUT IS SLOWER THAN TARGET" in out
    assert "Saved the best configuration" in out
    saved = load_settings()
    assert saved["model"] == "htdemucs" and saved["batch_size"] in (1, 2, 4, 8)
    results = sorted((tmp_path / "home" / "bench_results").glob("bench_*.json"))
    data = json.loads(results[-1].read_text())
    assert data["passed"] == ("RESULT: PASS" in out) and len(data["results"]) == 8
    assert all(r["ok"] for r in data["results"])


def test_gpu_out_of_memory_halves_the_batch(pretend_gpu, monkeypatch):
    import numpy as np

    from stemdeck.config import SAMPLE_RATE
    from stemdeck.separation import DemucsSeparator

    sep = DemucsSeparator(pretend_gpu, "htdemucs", batch_size=4)
    original = sep._run_batch
    calls = []

    def flaky(model, mix, offsets, seg_len, dev, active):
        calls.append(sep.batch_size)
        if sep.batch_size > 1:
            raise torch.OutOfMemoryError("pretend: out of memory")
        return original(model, mix, offsets, seg_len, dev, active)

    monkeypatch.setattr(sep, "_run_batch", flaky)
    audio = (0.1 * np.random.default_rng(0).standard_normal((SAMPLE_RATE * 2, 2))).astype(np.float32)
    stems, stats = sep.separate(audio)
    assert calls[:3] == [4, 2, 1] and stats.batch_size == 1
    assert all(np.isfinite(s).all() for s in stems.values())


def test_require_gpu_fails_loudly_without_one(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("STEMDECK_HOME", str(tmp_path / "home"))
    from stemdeck.bench import main

    code = main(["--seconds", "2", "--model", "htdemucs", "--require-gpu"])
    out = capsys.readouterr().out
    assert code == 2 and "GPU NOT USED" in out

"""Finding out what hardware we can run the separation model on.

On the target machine (AMD Radeon on Windows) PyTorch is AMD's ROCm build.
ROCm deliberately presents itself through PyTorch's ``torch.cuda`` interface,
so "cuda" in the code below means "the GPU", whichever vendor made it. We tell
ROCm and NVIDIA apart with ``torch.version.hip``, which only ROCm builds set.
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field


@dataclass
class DeviceInfo:
    kind: str                 # "gpu" or "cpu"
    backend: str              # "rocm" | "cuda" | "cpu"
    torch_device: str         # e.g. "cuda:0" or "cpu"
    name: str                 # human readable
    vram_gb: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        return self.kind == "gpu"

    def label(self) -> str:
        if self.is_gpu:
            vram = f", {self.vram_gb:.0f} GB" if self.vram_gb else ""
            return f"GPU: {self.name} ({self.backend.upper()}{vram})"
        return f"CPU: {self.name}"

    def to_dict(self) -> dict:
        return asdict(self)


def cpu_device() -> DeviceInfo:
    name = platform.processor() or platform.machine() or "CPU"
    return DeviceInfo(
        kind="cpu", backend="cpu", torch_device="cpu",
        name=f"{name} ({os.cpu_count()} threads)",
    )


def _gpu_smoke_test(device: str) -> str | None:
    """Run a tiny piece of the real workload on the GPU and compare it with the CPU.

    ``torch.cuda.is_available()`` only says a GPU was found. AMD's own notes warn
    that a build can install fine and still not work on a given card, so we
    exercise the operations the separation model depends on (convolution,
    FFT, attention) and check the numbers. Returns None if OK, else a reason.
    """
    import torch

    try:
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(2, 4, 4096, generator=gen)
        conv = torch.nn.Conv1d(4, 8, 8, stride=4, padding=2)
        attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        window = torch.hann_window(512)

        def run(dev: str):
            with torch.no_grad():
                xx = x.to(dev)
                y = conv.to(dev)(xx)
                a, _ = attn.to(dev)(y.transpose(1, 2), y.transpose(1, 2), y.transpose(1, 2))
                spec = torch.stft(xx.reshape(-1, 4096), 512, 128, window=window.to(dev),
                                  return_complex=True)
                back = torch.istft(spec, 512, 128, window=window.to(dev), length=4096)
                return y.cpu(), a.cpu(), back.cpu()

        ref = run("cpu")
        got = run(device)
        conv.to("cpu"), attn.to("cpu")
        for name, r, g in zip(("convolution", "attention", "FFT"), ref, got):
            if not torch.isfinite(g).all():
                return f"GPU {name} test produced NaN/Inf values"
            err = (r - g).abs().max().item() / (r.abs().max().item() + 1e-9)
            if err > 1e-2:
                return f"GPU {name} test disagrees with the CPU (relative error {err:.3g})"
    except Exception as exc:  # any failure here means: do not trust this GPU path
        return f"GPU self-test raised {type(exc).__name__}: {exc}"
    return None


def probe_gpu(run_smoke_test: bool = True) -> tuple[DeviceInfo | None, str | None]:
    """Return (gpu, None) if a working GPU is available, else (None, reason)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        return None, f"PyTorch could not be imported: {exc}"

    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        return None, f"GPU query failed: {type(exc).__name__}: {exc}"
    if not available:
        build = torch.__version__
        if "+cpu" in build:
            reason = (f"This PyTorch build ({build}) is CPU-only. Install AMD's ROCm build "
                      "of PyTorch (see README, step 3).")
        elif getattr(torch.version, "hip", None):
            reason = (f"PyTorch {build} has ROCm support but found no usable GPU. Update the "
                      "AMD Adrenalin driver and check `hipInfo.exe`.")
        else:
            reason = f"PyTorch {build} reports no GPU (torch.cuda.is_available() is False)."
        return None, reason

    try:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception as exc:
        return None, f"GPU found but could not be queried: {type(exc).__name__}: {exc}"

    hip = getattr(torch.version, "hip", None)
    backend = "rocm" if hip else "cuda"
    info = DeviceInfo(kind="gpu", backend=backend, torch_device="cuda:0", name=name, vram_gb=vram)
    if hip:
        info.notes.append(f"ROCm/HIP {hip}")
    if run_smoke_test:
        problem = _gpu_smoke_test(info.torch_device)
        if problem:
            return None, f"{name}: {problem}"
    return info, None


def choose_device(preference: str = "auto") -> tuple[DeviceInfo, str | None]:
    """Pick the device to run on.

    Returns (device, fallback_reason). ``fallback_reason`` is set when the GPU
    was wanted but is not usable, so callers can tell the user plainly.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return cpu_device(), None
    gpu, reason = probe_gpu()
    if gpu is not None:
        return gpu, None
    return cpu_device(), reason


def environment_report() -> dict:
    """Everything worth knowing when diagnosing a GPU problem."""
    import torch

    report = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "torch": torch.__version__,
        "torch_rocm_hip": getattr(torch.version, "hip", None),
        "torch_cuda_build": torch.version.cuda,
        "cpu_threads": os.cpu_count(),
    }
    gpu, reason = probe_gpu()
    report["gpu"] = gpu.to_dict() if gpu else None
    report["gpu_unavailable_reason"] = reason
    try:
        import onnxruntime as ort  # optional, only for the DirectML backup path

        report["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        report["onnxruntime_providers"] = None
    return report

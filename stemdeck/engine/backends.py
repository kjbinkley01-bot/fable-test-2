"""Audio output backends: where the mixer's audio goes.

``SoundDeviceBackend`` plays through the sound card (PortAudio). On Windows
it prefers WASAPI, the modern low-latency Windows audio system, and falls
back to the older ones if that cannot be opened.

``NullBackend`` plays to nowhere at real-time pace. It is used by the tests
and lets the app run on a machine with no sound card.

Both call ``callback(frames) -> float32 array (frames, 2)`` from an audio
thread, once per block.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ..config import SAMPLE_RATE

RenderFn = Callable[[int], np.ndarray]


@dataclass
class BackendInfo:
    backend: str            # "sounddevice" | "null"
    host_api: str = ""
    device: str = ""
    sample_rate: int = SAMPLE_RATE
    block_size: int = 0     # 0 = chosen by the sound card driver
    latency: float = 0.0    # seconds between rendering and hearing
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        if self.backend == "null":
            return "No audio device (silent)"
        return f"{self.device} via {self.host_api}, {self.latency * 1000:.0f} ms"


class NullBackend:
    """Calls the render callback at real-time pace and discards the audio."""

    def __init__(self, render: RenderFn, block_size: int = 512, note: str = "",
                 latency: float = 0.04) -> None:
        self._render = render
        self._block = block_size
        self._buffer = max(latency, block_size / SAMPLE_RATE)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.underflows = 0
        self.max_lateness = 0.0
        self.info = BackendInfo("null", block_size=block_size, latency=self._buffer, note=note)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="null-audio", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Models a sound card with ``latency`` seconds of buffer: a block may
        # be delivered late by up to that much before the buffer runs dry.
        period = self._block / SAMPLE_RATE
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self._render(self._block)
            deadline += period
            lateness = time.perf_counter() - deadline
            if lateness <= 0:
                time.sleep(-lateness)
                continue
            self.max_lateness = max(self.max_lateness, lateness)
            if lateness > self._buffer:
                self.underflows += 1            # the buffer ran dry: audible dropout
                deadline = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


class SoundDeviceBackend:
    """Real audio output through PortAudio (the ``sounddevice`` package)."""

    def __init__(self, render: RenderFn, device: int | str | None = None,
                 latency: float = 0.04, block_size: int = 0) -> None:
        import sounddevice as sd

        self._sd = sd
        self._render = render
        self.underflows = 0
        self._stream = None
        errors: list[str] = []
        for host_api, dev, extra in self._candidates(device):
            try:
                stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                    device=dev, blocksize=block_size, latency=latency,
                    callback=self._callback, extra_settings=extra,
                )
            except Exception as exc:
                errors.append(f"{host_api}: {exc}")
                continue
            self._stream = stream
            name = sd.query_devices(stream.device)["name"] if stream.device is not None else "default"
            self.info = BackendInfo(
                "sounddevice", host_api=host_api, device=str(name),
                sample_rate=int(stream.samplerate), block_size=block_size,
                latency=float(stream.latency),
                note="; ".join(errors),
            )
            break
        if self._stream is None:
            raise RuntimeError("could not open an audio output: " + "; ".join(errors or ["no devices"]))

    def _candidates(self, device):
        """(label, device, extra_settings) to try, best first."""
        sd = self._sd
        if device is not None:
            api = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
            extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None
            yield api, device, extra
        apis = sd.query_hostapis()
        for wanted in ("WASAPI", "DirectSound", "MME"):
            for api in apis:
                if wanted in api["name"] and api["default_output_device"] >= 0:
                    # auto_convert lets Windows resample if the device runs at
                    # e.g. 48 kHz; without it shared-mode WASAPI refuses 44.1 kHz.
                    extra = sd.WasapiSettings(auto_convert=True) if wanted == "WASAPI" else None
                    yield api["name"], api["default_output_device"], extra
        yield "system default", None, None

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status and status.output_underflow:
            self.underflows += 1
        try:
            outdata[:] = self._render(frames)
        except Exception:
            outdata.fill(0)  # never let an exception produce noise

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


def list_output_devices() -> list[dict]:
    """Output devices for a settings screen: [{index, name, host_api, default}]."""
    try:
        import sounddevice as sd

        apis = sd.query_hostapis()
        out = []
        for index, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] >= 2:
                api = apis[dev["hostapi"]]
                out.append({"index": index, "name": dev["name"], "host_api": api["name"],
                            "default": api["default_output_device"] == index})
        return out
    except Exception:
        return []


def open_backend(render: RenderFn, kind: str = "auto", device=None,
                 latency: float = 0.04, block_size: int = 0):
    """Open the requested backend; "auto" falls back to silent if no sound card."""
    if kind == "null":
        return NullBackend(render, block_size or 512, latency=latency)
    try:
        return SoundDeviceBackend(render, device, latency, block_size)
    except Exception as exc:
        if kind == "sounddevice":
            raise
        return NullBackend(render, block_size or 512, note=f"audio output unavailable: {exc}",
                           latency=latency)

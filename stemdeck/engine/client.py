"""UI-side handle on the audio engine process.

``EngineClient`` starts the engine process, sends it commands and reads its
status. Every method returns immediately; nothing here waits on audio.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np

from ..config import SAMPLE_RATE, STEMS
from . import process as proto


@dataclass
class DeckStatus:
    position: float = 0.0        # frames, as currently *heard* (latency compensated)
    playing: bool = False
    ended: bool = False
    n_frames: int = 0
    serial: int = 0
    stem_peaks: tuple = (0.0,) * len(STEMS)


@dataclass
class EngineStatus:
    running: bool = False
    sample_rate: int = SAMPLE_RATE
    latency: float = 0.0
    underflows: int = 0          # audible dropouts reported by the sound card
    late_blocks: int = 0         # blocks that used over 70% of their time budget
    max_render_ms: float = 0.0   # slowest block so far
    blocks: int = 0
    master_peak: float = 0.0
    clipped: int = 0
    decks: list = field(default_factory=list)


class EngineError(RuntimeError):
    pass


class EngineClient:
    def __init__(self, backend: str = "auto", device=None, latency: float = 0.04,
                 block_size: int = 0) -> None:
        self.options = {"backend": backend, "device": device, "latency": latency,
                        "block_size": block_size}
        self.info: dict = {}
        self._proc: mp.Process | None = None
        self._cmd = None
        self._events = None
        self._status_shm: shared_memory.SharedMemory | None = None
        self._status: np.ndarray | None = None
        self._stem_shm: dict[int, shared_memory.SharedMemory] = {}
        self._retired: list[tuple[float, shared_memory.SharedMemory]] = []
        self._serial = 0
        self._deck_serial: dict[int, int] = {}
        self.last_error: str | None = None
        atexit.register(self.stop)

    # -------------------------------------------------------------- lifecycle
    def start(self, timeout: float = 20.0) -> dict:
        """Start the engine process and wait until audio is running."""
        if self.is_alive():
            return self.info
        ctx = mp.get_context("spawn")       # same behaviour on Windows and Linux
        self._cmd = ctx.Queue()
        self._events = ctx.Queue()
        self._status_shm = shared_memory.SharedMemory(create=True, size=proto.STATUS_SIZE * 8)
        self._status = np.ndarray((proto.STATUS_SIZE,), dtype=np.float64, buffer=self._status_shm.buf)
        self._status[:] = 0.0
        self._proc = ctx.Process(
            target=proto.engine_main, name="stemdeck-audio",
            args=(self._cmd, self._events, self._status_shm.name, self.options),
            daemon=True,
        )
        self._proc.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                kind, payload = self._events.get(timeout=0.2)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise EngineError("audio engine process exited during start-up")
                continue
            if kind == "ready":
                self.info = payload
                return payload
            if kind == "error":
                self.stop()
                raise EngineError(payload)
        self.stop()
        raise EngineError("audio engine did not start in time")

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.is_alive():
            try:
                self._cmd.put(("quit",))
                proc.join(timeout=3)
            except Exception:
                pass
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        self._status = None
        for shm in [self._status_shm, *self._stem_shm.values(), *(s for _, s in self._retired)]:
            if shm is not None:
                self._release(shm)
        self._status_shm = None
        self._stem_shm.clear()
        self._retired.clear()

    @staticmethod
    def _release(shm: shared_memory.SharedMemory) -> None:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    def poll_events(self) -> list[tuple]:
        """Drain messages from the engine (errors). Call from a UI timer."""
        events = []
        if self._events is None:
            return events
        while True:
            try:
                event = self._events.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            if event[0] == "error":
                self.last_error = event[1]
            events.append(event)
        # Free stem memory of replaced songs once the engine has let go of it.
        now = time.monotonic()
        keep = []
        for when, shm in self._retired:
            if now - when > 2.0:
                self._release(shm)
            else:
                keep.append((when, shm))
        self._retired = keep
        return events

    # ---------------------------------------------------------------- commands
    def _send(self, *cmd) -> None:
        if self._cmd is not None and self.is_alive():
            self._cmd.put(cmd)

    def load(self, deck: int, stems: np.ndarray) -> int:
        """Give a deck a song. ``stems``: (n_stems, frames, 2) float32.

        Returns a serial number; ``status().decks[deck].serial`` equals it
        once the engine has switched to the new song.
        """
        if stems.ndim != 3 or stems.shape[0] != len(STEMS) or stems.shape[2] != 2:
            raise ValueError(f"expected stems of shape ({len(STEMS)}, frames, 2), got {stems.shape}")
        shm = shared_memory.SharedMemory(create=True, size=max(1, stems.nbytes))
        view = np.ndarray(stems.shape, dtype=np.float32, buffer=shm.buf)
        view[:] = stems
        del view
        self._serial += 1
        old = self._stem_shm.get(deck)
        self._stem_shm[deck] = shm
        self._deck_serial[deck] = self._serial
        self._send("load", deck, shm.name, int(stems.shape[1]), self._serial)
        if old is not None:
            self._retired.append((time.monotonic(), old))
        return self._serial

    def unload(self, deck: int) -> None:
        self._send("unload", deck)
        old = self._stem_shm.pop(deck, None)
        if old is not None:
            self._retired.append((time.monotonic(), old))

    def play(self, deck: int = 0) -> None:
        self._send("play", deck)

    def pause(self, deck: int = 0) -> None:
        self._send("pause", deck)

    def seek(self, deck: int, frame: int) -> None:
        self._send("seek", deck, int(frame))

    def set_fader(self, deck: int, stem: int, gain: float) -> None:
        self._send("fader", deck, stem, float(gain))

    def set_mute(self, deck: int, stem: int, muted: bool) -> None:
        self._send("mute", deck, stem, bool(muted))

    def set_solo(self, deck: int, stem: int, solo: bool) -> None:
        self._send("solo", deck, stem, bool(solo))

    def set_volume(self, deck: int, gain: float) -> None:
        self._send("volume", deck, float(gain))

    def set_master(self, gain: float) -> None:
        self._send("master", float(gain))

    def reset_stats(self) -> None:
        self._send("reset_stats")

    # ------------------------------------------------------------------ status
    def status(self) -> EngineStatus:
        st = self._status
        if st is None:
            return EngineStatus(decks=[DeckStatus() for _ in range(proto.N_DECKS)])
        snap = st.copy()
        now = time.perf_counter()
        latency = float(snap[proto.ST_LATENCY])
        out = EngineStatus(
            running=snap[proto.ST_STATE] == 1 and self.is_alive(),
            sample_rate=int(snap[proto.ST_SAMPLE_RATE] or SAMPLE_RATE),
            latency=latency,
            underflows=int(snap[proto.ST_UNDERFLOWS]),
            late_blocks=int(snap[proto.ST_LATE]),
            max_render_ms=float(snap[proto.ST_MAX_RENDER]) * 1000.0,
            blocks=int(snap[proto.ST_BLOCKS]),
            master_peak=float(snap[proto.ST_MASTER_PEAK]),
            clipped=int(snap[proto.ST_CLIPPED]),
        )
        for d in range(proto.N_DECKS):
            base = proto.ST_DECK0 + d * proto.DECK_FIELDS
            position = float(snap[base + proto.DK_POSITION])
            playing = snap[base + proto.DK_PLAYING] == 1.0
            n_frames = int(snap[base + proto.DK_FRAMES])
            if playing:
                # The engine wrote "next frame to render" at time STAMP. What is
                # coming out of the speakers now is that, advanced by the time
                # since, minus the output latency.
                since = min(max(now - float(snap[proto.ST_STAMP]), 0.0), 0.25)
                position += (since - latency) * SAMPLE_RATE
            position = min(max(position, 0.0), float(n_frames))
            out.decks.append(DeckStatus(
                position=position, playing=playing,
                ended=snap[base + proto.DK_ENDED] == 1.0, n_frames=n_frames,
                serial=int(snap[base + proto.DK_SERIAL]),
                stem_peaks=tuple(float(v) for v in snap[base + proto.DK_PEAKS:base + proto.DK_PEAKS + len(STEMS)]),
            ))
        return out

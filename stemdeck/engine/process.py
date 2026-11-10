"""The audio engine process.

Why a separate process
----------------------
Python runs one thread at a time inside a process. If audio shared a process
with the UI, a slow repaint (or the stem separation running next to it)
could hold audio up for a few milliseconds - and a late audio block is a
dropout you can hear. So the mixer and the sound card callback live in this
small dedicated process, which does nothing else, has its garbage collector
switched off (no pauses) and runs at raised priority. The UI talks to it by
sending short commands; a busy or even frozen UI cannot make it stutter.

Data flow
---------
* Stems: the UI puts a song's stems in a block of shared memory and sends
  its name. Both processes see the same bytes, nothing is copied or piped.
* Commands (UI -> engine): tuples on a queue, e.g. ("seek", deck, frame).
* Status (engine -> UI): playhead, levels, health counters in a small shared
  array that the engine overwrites every block and the UI reads when it
  repaints. No locks on the audio path.
* Events (engine -> UI): rare messages (ready / error) on a second queue.
"""
from __future__ import annotations

import gc
import os
import queue
import time
from multiprocessing import shared_memory

import numpy as np

from ..config import SAMPLE_RATE, STEMS
from .mixer import Mixer

N_DECKS = 2
N_STEMS = len(STEMS)

# ---- layout of the shared status array (float64) ---------------------------
ST_HEARTBEAT = 0        # +1 every audio block
ST_STATE = 1            # 0 starting, 1 running, -1 failed
ST_SAMPLE_RATE = 2
ST_LATENCY = 3          # output latency in seconds
ST_UNDERFLOWS = 4       # sound card ran out of audio (audible dropout)
ST_LATE = 5             # blocks that took over 70% of their time budget
ST_MAX_RENDER = 6       # slowest block so far, seconds
ST_LAST_BLOCK = 7       # frames in the most recent block
ST_MASTER_PEAK = 8
ST_CLIPPED = 9
ST_STAMP = 10           # time.perf_counter() when positions were written
ST_BLOCKS = 11          # total audio blocks rendered
ST_DECK0 = 16
DECK_FIELDS = 12
DK_POSITION = 0         # next frame to be rendered
DK_PLAYING = 1
DK_ENDED = 2
DK_FRAMES = 3
DK_SERIAL = 4           # serial of the load command that is active
DK_PEAKS = 5            # ... DK_PEAKS + N_STEMS
STATUS_SIZE = ST_DECK0 + N_DECKS * DECK_FIELDS


def _attach(name: str) -> shared_memory.SharedMemory:
    """Open a shared memory block created by the UI process.

    The UI process owns the block (creates and deletes it); here we only map
    it. The engine is started with the "spawn" method, so both processes
    share one resource tracker and attaching does not create a second owner.
    """
    return shared_memory.SharedMemory(name=name)


def _close_quietly(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        pass  # something still points into it; leaking beats crashing


def _raise_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes

            HIGH_PRIORITY_CLASS = 0x00000080
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
        else:
            os.nice(-5)
    except Exception:
        pass  # not allowed: carry on at normal priority


class Engine:
    def __init__(self, status: np.ndarray, options: dict) -> None:
        self.status = status
        self.mixer = Mixer(N_DECKS)
        self.options = options
        self.backend = None
        self._shm: list[shared_memory.SharedMemory | None] = [None] * N_DECKS
        self._serial = [0] * N_DECKS

    # ------------------------------------------------------------ audio side
    def render(self, frames: int) -> np.ndarray:
        """Called by the audio thread for every block."""
        started = time.perf_counter()
        out = self.mixer.render(frames)
        st = self.status
        for d, deck in enumerate(self.mixer.decks):
            base = ST_DECK0 + d * DECK_FIELDS
            st[base + DK_POSITION] = deck.position
            st[base + DK_PLAYING] = 1.0 if deck.playing else 0.0
            st[base + DK_ENDED] = 1.0 if deck.ended else 0.0
            st[base + DK_PEAKS:base + DK_PEAKS + N_STEMS] = deck.stem_peaks
        st[ST_MASTER_PEAK] = self.mixer.peak
        if self.mixer.clipped:
            st[ST_CLIPPED] += 1
        st[ST_LAST_BLOCK] = frames
        now = time.perf_counter()
        st[ST_STAMP] = now
        elapsed = now - started
        if elapsed > st[ST_MAX_RENDER]:
            st[ST_MAX_RENDER] = elapsed
        if elapsed > 0.7 * frames / SAMPLE_RATE:
            st[ST_LATE] += 1
        if self.backend is not None:
            st[ST_UNDERFLOWS] = self.backend.underflows
        st[ST_BLOCKS] += 1
        st[ST_HEARTBEAT] += 1
        return out

    def _wait_blocks(self, n: int = 2, timeout: float = 0.5) -> None:
        """Wait until the audio thread has finished ``n`` more blocks."""
        start = self.status[ST_HEARTBEAT]
        deadline = time.perf_counter() + timeout
        while self.status[ST_HEARTBEAT] < start + n and time.perf_counter() < deadline:
            time.sleep(0.002)

    # ---------------------------------------------------------- control side
    def handle(self, cmd: tuple) -> bool:
        """Apply one command. Returns False when asked to quit."""
        op = cmd[0]
        if op == "quit":
            return False
        if op == "master":
            self.mixer.master = float(cmd[1])
            return True
        if op == "reset_stats":
            self.status[ST_LATE] = 0
            self.status[ST_MAX_RENDER] = 0
            self.status[ST_CLIPPED] = 0
            return True
        deck_index = int(cmd[1])
        deck = self.mixer.decks[deck_index]
        base = ST_DECK0 + deck_index * DECK_FIELDS
        if op == "load":
            _, _, name, n_frames, serial = cmd
            shm = _attach(name)
            stems = np.ndarray((N_STEMS, int(n_frames), 2), dtype=np.float32, buffer=shm.buf)
            _ = float(stems[:, ::4096].sum())      # touch every page before audio needs it
            old = self._shm[deck_index]
            deck.load(stems)
            self._shm[deck_index] = shm
            self.status[base + DK_FRAMES] = n_frames
            self.status[base + DK_POSITION] = 0
            self.status[base + DK_SERIAL] = serial
            if old is not None:
                self._wait_blocks()                # audio thread may still be reading it
                _close_quietly(old)
        elif op == "unload":
            old = self._shm[deck_index]
            deck.load(None)
            self._shm[deck_index] = None
            self.status[base + DK_FRAMES] = 0
            self.status[base + DK_SERIAL] = 0
            if old is not None:
                self._wait_blocks()
                _close_quietly(old)
        elif op == "play":
            deck.play()
        elif op == "pause":
            deck.pause()
        elif op == "seek":
            deck.seek(int(cmd[2]))
        elif op == "fader":
            deck.fader[int(cmd[2])] = float(cmd[3])
        elif op == "mute":
            deck.mute[int(cmd[2])] = bool(cmd[3])
        elif op == "solo":
            deck.solo[int(cmd[2])] = bool(cmd[3])
        elif op == "volume":
            deck.volume = float(cmd[2])
        return True

    def close(self) -> None:
        if self.backend is not None:
            self.backend.stop()
        for deck in self.mixer.decks:
            deck.load(None)
        for shm in self._shm:
            if shm is not None:
                _close_quietly(shm)


def engine_main(cmd_queue, event_queue, status_name: str, options: dict) -> None:
    """Entry point of the engine process."""
    status_shm = _attach(status_name)
    status = np.ndarray((STATUS_SIZE,), dtype=np.float64, buffer=status_shm.buf)
    engine = Engine(status, options)
    try:
        from .backends import open_backend

        _raise_priority()
        backend = open_backend(
            engine.render,
            kind=options.get("backend", "auto"),
            device=options.get("device"),
            latency=float(options.get("latency", 0.04)),
            block_size=int(options.get("block_size", 0)),
        )
        engine.backend = backend
        status[ST_SAMPLE_RATE] = backend.info.sample_rate
        status[ST_LATENCY] = backend.info.latency
        # No garbage-collector pauses on the audio path: everything allocated
        # per block is freed immediately by reference counting anyway.
        gc.collect()
        gc.freeze()
        gc.disable()
        backend.start()
        status[ST_STATE] = 1
        event_queue.put(("ready", backend.info.to_dict()))
    except Exception as exc:
        status[ST_STATE] = -1
        event_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        status_shm.close()
        return

    import multiprocessing

    parent = multiprocessing.parent_process()

    def parent_alive() -> bool:
        return parent is None or parent.is_alive()

    try:
        running = True
        while running:
            try:
                cmd = cmd_queue.get(timeout=1.0)
            except queue.Empty:
                if not parent_alive():
                    break                      # the app is gone: do not linger
                continue
            except (EOFError, OSError):
                break
            try:
                running = engine.handle(cmd)
            except Exception as exc:
                event_queue.put(("error", f"command {cmd[0]!r} failed: {type(exc).__name__}: {exc}"))
    finally:
        engine.close()
        engine.status = None
        del status
        _close_quietly(status_shm)

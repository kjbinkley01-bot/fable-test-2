"""Background threads, so long jobs never freeze the window."""
from __future__ import annotations

import logging
import threading
import traceback

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class LoadWorker(QThread):
    """Loads a song through the library (cache, or separate + analyse)."""

    progress = Signal(str, float, str)      # stage, fraction, message
    loaded = Signal(object)                 # library.Track
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, path: str, model: str, separation_options: dict, parent=None) -> None:
        super().__init__(parent)
        self.path = path
        self.model = model
        self.separation_options = separation_options
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from ..library import LoadCancelled, load_track

        try:
            track = load_track(
                self.path, model=self.model, separation_options=self.separation_options,
                progress=lambda stage, frac, msg: self.progress.emit(stage, float(frac), msg),
                cancel=self._cancel.is_set,
            )
        except LoadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            logger.error("loading %s failed:\n%s", self.path, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.loaded.emit(track)


class ExportWorker(QThread):
    done = Signal(object)                   # list[ExportInfo]
    failed = Signal(str)

    def __init__(self, job, parent=None) -> None:
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        try:
            self.done.emit(self._job())
        except Exception as exc:
            logger.error("export failed:\n%s", traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")

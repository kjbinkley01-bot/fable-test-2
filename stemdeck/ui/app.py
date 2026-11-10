"""Application entry point:  python -m stemdeck app  [song]"""
from __future__ import annotations

import logging
import multiprocessing
import sys
import traceback

from ..config import data_dir


def _setup_logging() -> None:
    log_path = data_dir() / "stemdeck.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Started without a console (pythonw / StemDeck.bat): libraries that print
    # progress would crash on a missing stdout, so send their output to the log.
    if sys.stdout is None or sys.stderr is None:
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream


def main(files: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    _setup_logging()
    log = logging.getLogger("stemdeck")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv[:1])
    app.setApplicationName("StemDeck")

    def excepthook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("unhandled error:\n%s", text)
        QMessageBox.critical(None, "StemDeck", f"Something went wrong:\n\n{exc}\n\n"
                             f"Details were written to {data_dir() / 'stemdeck.log'}")

    sys.excepthook = excepthook

    from .main_window import MainWindow
    from .theme import STYLE

    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if files:
        QTimer.singleShot(100, lambda: window.load_file(files[0]))
    code = app.exec()
    window.engine.stop()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

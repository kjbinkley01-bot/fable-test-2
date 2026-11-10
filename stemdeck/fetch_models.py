"""Download every model the app needs, so first use works offline and fast.

    python -m stemdeck.fetch_models
"""
from __future__ import annotations

import sys

from .config import MODELS


def main() -> int:
    failures = 0
    from demucs.pretrained import get_model

    for name in sorted(set(MODELS.values())):
        print(f"  separation model {name} ...", flush=True)
        try:
            get_model(name)
        except Exception as exc:
            failures += 1
            print(f"    failed: {type(exc).__name__}: {exc}")
    print("  beat tracking model ...", flush=True)
    try:
        from .analysis.beats import load_beat_model

        load_beat_model("cpu")
    except Exception as exc:
        failures += 1
        print(f"    failed: {type(exc).__name__}: {exc}")
    print("  done." if not failures else f"  {failures} download(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

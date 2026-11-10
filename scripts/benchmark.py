"""Separation benchmark. See stemdeck/bench.py for what it measures.

    python scripts/benchmark.py [song] [--sweep] [--model htdemucs] ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stemdeck.bench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

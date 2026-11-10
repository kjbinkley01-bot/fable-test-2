"""Classic template-matching key detection, as a reference point.

    python tools/key_model/baseline.py FEATS.npz LABEL_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import key_index, load_features, load_labels, summarize, track_id  # noqa: E402

PROFILES = {
    "krumhansl": ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "edma": ([0.16519551, 0.04749026, 0.08293076, 0.06687112, 0.09994645, 0.09274123,
              0.05294487, 0.13159476, 0.05218986, 0.07443653, 0.06940723, 0.0642515],
             [0.17235348, 0.05336489, 0.0761009, 0.10043649, 0.05621498, 0.08527853,
              0.0497915, 0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106]),
}


def predict(semis: np.ndarray, profile: str, compress: bool) -> int:
    x = semis.mean(axis=0)
    if compress:
        x = np.log1p(10 * x / x.mean())
    chroma = x.reshape(-1, 12).sum(axis=0)  # MIDI 24 is a C, so bin 0 = C
    major, minor = (np.asarray(p) for p in PROFILES[profile])
    scores = [np.corrcoef(chroma, np.roll(major, k))[0, 1] for k in range(12)]
    scores += [np.corrcoef(chroma, np.roll(minor, k))[0, 1] for k in range(12)]
    return int(np.argmax(scores))


def main():
    feats = load_features(sys.argv[1])
    labels = load_labels(sys.argv[2])
    items = [(n, f) for n, f in feats.items() if track_id(n) in labels]
    true = [key_index(*labels[track_id(n)]) for n, _ in items]
    print(f"{len(items)} labelled tracks")
    for profile in PROFILES:
        for compress in (False, True):
            pred = [predict(f, profile, compress) for _, f in items]
            print(f"{profile:10s} log={compress!s:5s}", summarize(true, pred))


if __name__ == "__main__":
    main()

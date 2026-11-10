"""Shared helpers for the key-model tools: labels, scoring, feature loading."""
import re
from pathlib import Path

import numpy as np

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(text: str):
    """'F# minor aeolian | A major' -> (6, 'minor'). None if unusable."""
    first = text.split("|")[0].strip()
    m = re.match(r"^([A-Ga-g])([#b]?)\s+(major|minor)", first)
    if not m:
        return None
    pc = NOTE_TO_PC[m.group(1).upper()]
    pc += {"#": 1, "b": -1, "": 0}[m.group(2)]
    return pc % 12, m.group(3)


def key_index(pc: int, mode: str) -> int:
    """0-11 major keys (C..B), 12-23 minor keys."""
    return pc + (12 if mode == "minor" else 0)


def mirex_score(true_idx: int, pred_idx: int) -> tuple[str, float]:
    """MIREX key evaluation: exact 1.0, fifth 0.5, relative 0.3, parallel 0.2."""
    tpc, tminor = true_idx % 12, true_idx >= 12
    ppc, pminor = pred_idx % 12, pred_idx >= 12
    if true_idx == pred_idx:
        return "correct", 1.0
    if tminor == pminor and (ppc - tpc) % 12 in (7, 5):
        return "fifth", 0.5
    if tminor and not pminor and (ppc - tpc) % 12 == 3:
        return "relative", 0.3
    if not tminor and pminor and (ppc - tpc) % 12 == 9:
        return "relative", 0.3
    if tpc == ppc:
        return "parallel", 0.2
    return "other", 0.0


def summarize(true, pred) -> dict:
    cats = {"correct": 0, "fifth": 0, "relative": 0, "parallel": 0, "other": 0}
    total = 0.0
    for t, p in zip(true, pred):
        cat, score = mirex_score(int(t), int(p))
        cats[cat] += 1
        total += score
    n = max(1, len(true))
    out = {k: round(100.0 * v / n, 1) for k, v in cats.items()}
    out["weighted"] = round(100.0 * total / n, 1)
    out["n"] = len(true)
    return out


def load_features(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    return {str(name): data[f"f{i}"] for i, name in enumerate(data["names"])}


def load_labels(label_dir: str) -> dict[str, tuple[int, str]]:
    """Map leading numeric track id -> parsed key, from a folder of .txt files."""
    labels = {}
    for path in Path(label_dir).glob("*.txt"):
        if path.name.startswith("._"):
            continue
        key = parse_key(path.read_text(errors="replace").strip())
        if key is not None:
            labels[path.name.split(" ")[0].split(".")[0]] = key
    return labels


def track_id(filename: str) -> str:
    return filename.split(" ")[0].split(".")[0]

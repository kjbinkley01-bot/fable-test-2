"""Key names and the Camelot wheel.

The Camelot wheel renames the 24 keys as 1-12 plus a letter (A = minor,
B = major) so that keys that mix well are neighbours: same number, or one
step up or down with the same letter.
"""
from __future__ import annotations

MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def camelot_code(pitch_class: int, mode: str) -> str:
    """pitch_class 0 = C ... 11 = B; mode 'major' or 'minor'."""
    if mode == "minor":
        relative_major = (pitch_class + 3) % 12
        return f"{((relative_major * 7) % 12 + 7) % 12 + 1}A"
    return f"{((pitch_class * 7) % 12 + 7) % 12 + 1}B"


def key_name(pitch_class: int, mode: str) -> str:
    names = MINOR_NAMES if mode == "minor" else MAJOR_NAMES
    return f"{names[pitch_class % 12]} {mode}"


def display_key(pitch_class: int, mode: str) -> str:
    """e.g. 'A minor / 8A'."""
    return f"{key_name(pitch_class, mode)} / {camelot_code(pitch_class, mode)}"


def parse_key(text: str) -> tuple[int, str] | None:
    """Understand the ways tools write keys.

    Accepts 'A minor', 'Am', 'A min', 'F# maj', 'Gb', 'Bbm', 'D♭ major',
    Camelot codes ('8A', '11B') and Open Key codes ('1m', '6d').
    Returns (pitch_class, mode) or None.
    """
    import re

    t = text.strip().replace("♯", "#").replace("♭", "b")
    if not t:
        return None
    m = re.fullmatch(r"(\d{1,2})\s*([ABab])", t)
    if m and 1 <= int(m.group(1)) <= 12:
        number, letter = int(m.group(1)), m.group(2).upper()
        for pc in range(12):
            for mode in ("major", "minor"):
                if camelot_code(pc, mode) == f"{number}{letter}":
                    return pc, mode
    m = re.fullmatch(r"(\d{1,2})\s*([mdMD])", t)
    if m and 1 <= int(m.group(1)) <= 12:  # Open Key: 1d = C major = 8B
        number = (int(m.group(1)) + 6) % 12 + 1
        return parse_key(f"{number}{'A' if m.group(2).lower() == 'm' else 'B'}")
    m = re.fullmatch(r"([A-Ga-g])\s*([#b]?)\s*(.*)", t)
    if not m:
        return None
    pc = (_NOTE_TO_PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    rest = m.group(3).strip().lower()
    if rest in ("", "maj", "major", "dur", "ma"):
        return pc, "major"
    if rest in ("m", "min", "minor", "moll", "mi"):
        return pc, "minor"
    return None

"""Check StemDeck's BPM and key against reference values (Beatport / Mixed In Key).

    python scripts/check_analysis.py test_songs.csv
    python scripts/check_analysis.py song1.mp3 song2.flac      (just print what StemDeck finds)

CSV format (see test_songs.example.csv):  file,bpm,key
Songs go through the normal library, so stems are separated once and cached;
re-running the check is quick.

How a match is judged
  BPM  matches if it rounds to the reference, or is within 0.6 BPM of it.
       An exact double or half is reported separately as "half/double time".
  Key  matches if it is the same key (C# minor and Db minor are the same).
       Near misses are named, because tools often disagree in exactly these ways:
       relative (A minor vs C major: same notes), parallel (A minor vs A major),
       fifth (neighbouring keys on the Camelot wheel).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stemdeck.analysis import analyze_file, camelot_code, key_name, parse_key  # noqa: E402


def key_relation(ref: tuple[int, str], got: tuple[int, str]) -> str:
    (rpc, rmode), (gpc, gmode) = ref, got
    if ref == got:
        return "match"
    if rmode == gmode and (gpc - rpc) % 12 in (5, 7):
        return "fifth (adjacent on the Camelot wheel)"
    if rmode == "minor" and gmode == "major" and (gpc - rpc) % 12 == 3:
        return "relative major (same notes)"
    if rmode == "major" and gmode == "minor" and (gpc - rpc) % 12 == 9:
        return "relative minor (same notes)"
    if rpc == gpc:
        return "parallel (same root, major/minor swapped)"
    return "different"


def bpm_relation(ref: float, got: float) -> str:
    if abs(got - ref) <= 0.6 or round(got) == round(ref):
        return "match"
    for factor, label in ((2.0, "double time"), (0.5, "half time")):
        if abs(got - ref * factor) <= 0.6 * factor + 0.6:
            return label
    return "different"


def read_rows(path: Path) -> list[tuple[str, float | None, str | None]]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line in csv.reader(row for row in handle if row.strip() and not row.lstrip().startswith("#")):
            if not line or line[0].strip().lower() == "file":
                continue
            file = line[0].strip().strip('"')
            bpm = float(line[1]) if len(line) > 1 and line[1].strip() else None
            key = line[2].strip() if len(line) > 2 and line[2].strip() else None
            rows.append((file, bpm, key))
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if len(argv) == 1 and argv[0].lower().endswith(".csv"):
        rows = read_rows(Path(argv[0]))
    else:
        rows = [(arg, None, None) for arg in argv]

    bpm_ok = key_ok = bpm_n = key_n = both_ok = both_n = 0
    for file, ref_bpm, ref_key in rows:
        name = Path(file).name
        print(f"\n{name}")
        if not Path(file).is_file():
            print("  FILE NOT FOUND - check the path in the CSV")
            continue
        result = analyze_file(file)
        key = result.key
        got_key = (key.pitch_class, key.mode)
        line_ok = True

        text = f"  BPM  {result.bpm:7.2f}"
        if ref_bpm is not None:
            relation = bpm_relation(ref_bpm, result.bpm)
            bpm_n += 1
            bpm_ok += relation == "match"
            line_ok &= relation == "match"
            text += f"   reference {ref_bpm:g}  ->  {'OK' if relation == 'match' else relation.upper()}"
        if not result.grid.is_constant:
            text += "   (tempo varies in this track)"
        print(text)

        text = f"  Key  {key.display:<16} confidence {key.confidence:.0%}"
        if ref_key is not None:
            parsed = parse_key(ref_key)
            if parsed is None:
                text += f"   reference '{ref_key}' not understood"
                line_ok = False
            else:
                relation = key_relation(parsed, got_key)
                key_n += 1
                key_ok += relation == "match"
                line_ok &= relation == "match"
                ref_text = f"{key_name(*parsed)} / {camelot_code(*parsed)}"
                text += f"   reference {ref_text}  ->  {'OK' if relation == 'match' else 'MISMATCH: ' + relation}"
        print(text)
        if key.alternatives:
            alts = ", ".join(f"{a['name']} / {a['camelot']} ({a['probability']:.0%})" for a in key.alternatives[:2])
            print(f"       next best: {alts}")
        for note in result.grid.notes:
            print(f"       note: {note}")
        if ref_bpm is not None and ref_key is not None:
            both_n += 1
            both_ok += line_ok

    if bpm_n or key_n:
        print("\n" + "=" * 60)
        if bpm_n:
            print(f"BPM matches:  {bpm_ok} of {bpm_n}")
        if key_n:
            print(f"Key matches:  {key_ok} of {key_n}")
        if both_n:
            print(f"Both correct: {both_ok} of {both_n}")
            target = -(-4 * both_n // 5)          # 4 of 5, scaled
            verdict = "PASS" if bpm_ok >= target and key_ok >= target else "BELOW TARGET"
            print(f"Milestone target (at least 4 of 5 for BPM and for key): {verdict}")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

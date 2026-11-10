"""Score BPM detection against an annotated set (development tool).

    python tools/eval_tempo.py AUDIO_DIR ANNOTATIONS.tsv [OUT.json]

ANNOTATIONS.tsv: ID <tab> BPM <tab> KEY <tab> GENRE (GiantSteps tempo format).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from stemdeck.analysis.beats import detect_beats, fit_grid
    from stemdeck.audio_io import load_audio
    from stemdeck.config import SAMPLE_RATE

    audio_dir, ann = Path(sys.argv[1]), sys.argv[2]
    rows = [r for r in csv.reader(open(ann), delimiter="\t") if len(r) >= 4]
    results = []
    for i, (tid, bpm, _key, genre) in enumerate(rows):
        path = audio_dir / f"{tid}.LOFI.mp3"
        if not path.is_file():
            continue
        try:
            audio = load_audio(path)
            beats, downbeats = detect_beats(audio)
            grid = fit_grid(beats, downbeats, audio.shape[0] / SAMPLE_RATE, bpm_range=(1.0, 1000.0))
            results.append({"id": tid, "true": float(bpm), "genre": genre,
                            "raw": grid.bpm, "constant": grid.is_constant,
                            "confidence": grid.confidence, "n_beats": len(beats)})
        except Exception as exc:
            results.append({"id": tid, "true": float(bpm), "genre": genre, "error": repr(exc)})
        if (i + 1) % 20 == 0:
            print(i + 1, flush=True)
    out = sys.argv[3] if len(sys.argv) > 3 else "tempo_eval.json"
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out, len(results))


if __name__ == "__main__":
    main()

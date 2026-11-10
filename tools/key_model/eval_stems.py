"""Does removing the drums before key detection help? (development tool)

    python tools/key_model/eval_stems.py AUDIO_DIR LABEL_DIR OUT.json [--n 60] [--model htdemucs]

Separates a random sample of labelled tracks and runs the key detector on
the plain mix and on the mix without drums. Must be run with a key model
that was NOT trained on these tracks.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import key_index, load_labels, summarize, track_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("label_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    from stemdeck.analysis.key import detect_key
    from stemdeck.audio_io import load_audio
    from stemdeck.separation import DemucsSeparator, cpu_device

    labels = load_labels(args.label_dir)
    files = sorted(p for p in Path(args.audio_dir).glob("*.mp3") if track_id(p.name) in labels)
    random.Random(args.seed).shuffle(files)
    sep = DemucsSeparator(cpu_device(), args.model)
    results = []
    out = Path(args.out)
    if out.is_file():
        results = json.loads(out.read_text())
    done = {r["id"] for r in results}
    for path in files[:args.n]:
        tid = track_id(path.name)
        if tid in done:
            continue
        audio = load_audio(path)
        stems, _ = sep.separate(audio)
        variants = {
            "mix": audio,
            "no_drums": stems["vocals"] + stems["bass"] + stems["other"],
            "bass_other": stems["bass"] + stems["other"],
        }
        row = {"id": tid, "true": key_index(*labels[tid])}
        for name, sig in variants.items():
            k = detect_key(sig)
            row[name] = k.pitch_class + (12 if k.mode == "minor" else 0)
            row[name + "_conf"] = k.confidence
        results.append(row)
        out.write_text(json.dumps(results))
        if len(results) % 5 == 0:
            line = f"{len(results)} tracks:"
            for name in variants:
                s = summarize([r["true"] for r in results], [r[name] for r in results])
                line += f"  {name} {s['correct']}/{s['weighted']}"
            print(line, flush=True)


if __name__ == "__main__":
    main()

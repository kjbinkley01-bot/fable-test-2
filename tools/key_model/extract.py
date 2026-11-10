"""Compute and cache key-detection features for a folder of labelled audio.

    python tools/key_model/extract.py AUDIO_DIR OUT.npz [--jobs 4]

Each file becomes a (frames, 72) semitone spectrogram (see
stemdeck/analysis/keyfeatures.py). Development tool: only needed to retrain
or re-evaluate the key model, not to run the app.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def work(path: str):
    from stemdeck.analysis.keyfeatures import semitone_spectrogram
    from stemdeck.audio_io import load_audio

    try:
        audio = load_audio(path)
        semis, tuning = semitone_spectrogram(audio)
        return path, semis.astype(np.float32), tuning, None
    except Exception as exc:  # keep going; report at the end
        return path, None, 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(str(p) for p in Path(args.audio_dir).rglob("*")
                   if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg") and not p.name.startswith("._"))
    if args.limit:
        files = files[:args.limit]
    data, tunings, failed = {}, {}, []
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, (path, semis, tuning, err) in enumerate(pool.map(work, files, chunksize=4)):
            name = Path(path).name
            if err:
                failed.append((name, err))
            else:
                data[name] = semis
                tunings[name] = tuning
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)
    names = sorted(data)
    np.savez_compressed(args.out, names=np.array(names),
                        tunings=np.array([tunings[n] for n in names]),
                        **{f"f{i}": data[n] for i, n in enumerate(names)})
    print(f"saved {len(names)} tracks to {args.out}; {len(failed)} failed")
    for name, err in failed[:10]:
        print("  failed:", name, err)


if __name__ == "__main__":
    main()

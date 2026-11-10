"""Command line interface:  python -m stemdeck <command>

    devices                      show what hardware separation will run on
    separate SONG [-o DIR]       split a song into stems (WAV files)
    analyze SONG                 BPM, key and beat grid
    bench [SONG]                 separation benchmark (same as scripts/benchmark.py)
    app                          start the desktop app
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import DEFAULT_MODEL, STEMS


def _progress_printer(label: str):
    state = {"last": -1}

    def show(fraction: float) -> None:
        pct = int(fraction * 100)
        if pct != state["last"]:
            state["last"] = pct
            bar = "#" * (pct // 4)
            sys.stdout.write(f"\r{label} [{bar:<25}] {pct:3d}%")
            sys.stdout.flush()
            if pct >= 100:
                sys.stdout.write("\n")
    return show


def cmd_devices(args: argparse.Namespace) -> int:
    from .bench import _print_env
    from .separation import environment_report

    report = environment_report()
    _print_env(report)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0 if report["gpu"] else 1


def cmd_separate(args: argparse.Namespace) -> int:
    from .audio_io import load_audio, save_wav
    from .separation import GpuRequiredError, separate
    from .settings import load_settings

    settings = load_settings()
    song = Path(args.song)
    print(f"Loading {song.name} ...")
    audio = load_audio(song)
    try:
        stems, stats = separate(
            audio,
            model=args.model or settings["model"],
            device=args.device,
            precision=args.precision or settings["precision"],
            batch_size=args.batch_size or settings["batch_size"],
            require_gpu=args.require_gpu,
            progress=_progress_printer("Separating"),
        )
    except GpuRequiredError as exc:
        print(f"\nGPU required but not usable: {exc}")
        return 2
    print(stats.summary())
    out_dir = Path(args.output) if args.output else song.parent / f"{song.stem}_stems"
    for name in STEMS:
        save_wav(out_dir / f"{name}.wav", stems[name], subtype=args.subtype)
    print(f"Stems written to {out_dir}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze_file

    t0 = time.perf_counter()
    result = analyze_file(args.song, use_stems=not args.no_stems)
    if args.json:
        print(json.dumps(result.to_dict(include_grid=args.grid), indent=2))
    else:
        print(result.summary())
        print(f"(analysis took {time.perf_counter() - t0:.1f}s)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import main as bench_main

    return bench_main(args.rest)


def cmd_app(args: argparse.Namespace) -> int:
    from .ui.app import main as app_main

    return app_main(args.files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemdeck", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("devices", help="show the GPU/CPU that will be used")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("separate", help="split a song into stems")
    p.add_argument("song")
    p.add_argument("-o", "--output", help="output folder (default: <song>_stems next to the song)")
    p.add_argument("--model", help=f"htdemucs_ft (best quality) or htdemucs (fast); default {DEFAULT_MODEL}")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--precision", choices=["fp32", "fp16"])
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--subtype", default="PCM_24", choices=["PCM_16", "PCM_24", "FLOAT"])
    p.add_argument("--require-gpu", action="store_true", help="fail rather than fall back to CPU")
    p.set_defaults(func=cmd_separate)

    p = sub.add_parser("analyze", help="BPM, key and beat grid")
    p.add_argument("song")
    p.add_argument("--json", action="store_true")
    p.add_argument("--grid", action="store_true", help="include every beat time in --json output")
    p.add_argument("--no-stems", action="store_true",
                   help="analyse the plain mix without separating first (faster, less accurate key)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bench", help="separation benchmark", add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("app", help="start the desktop app")
    p.add_argument("files", nargs="*")
    p.set_defaults(func=cmd_app)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

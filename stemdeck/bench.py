"""Benchmark: how fast does this machine separate a song, and on what?

Usage (from the project folder):

    python scripts/benchmark.py path\\to\\song.mp3
    python scripts/benchmark.py                      (uses a generated 4:00 test signal)
    python scripts/benchmark.py song.mp3 --sweep     (try batch sizes / precision, save the best)

What it reports
  * which device actually ran the model (GPU or CPU) and why, if not the GPU
  * a "cold" run (includes one-off GPU kernel compilation) and "warm" runs
  * peak GPU memory, speed as a multiple of realtime
  * a correctness check of the GPU against the CPU on a short excerpt
  * PASS/FAIL against the milestone target (4:00 song in under 60 s on GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import DEFAULT_MODEL, SAMPLE_RATE, STEMS, data_dir

TARGET_SECONDS = 60.0
TARGET_SONG_SECONDS = 240.0


def synthetic_song(seconds: float = TARGET_SONG_SECONDS, seed: int = 0) -> np.ndarray:
    """A deterministic music-like test signal (drums, bass, chords, a lead).

    Speed does not depend on what the audio contains, so this is a fair
    stand-in when no song is given. Use a real song to judge stem quality.
    """
    rng = np.random.default_rng(seed)
    sr = SAMPLE_RATE
    n = int(seconds * sr)
    t = np.arange(n) / sr
    bpm = 124.0
    beat = 60.0 / bpm
    phase = np.modf(t / beat)[0]
    kick = np.sin(2 * np.pi * (50 + 60 * np.exp(-phase * 30)) * phase * beat) * np.exp(-phase * 12)
    hat_phase = np.modf(t / (beat / 2) + 0.5)[0]
    hat = rng.standard_normal(n) * np.exp(-hat_phase * 40) * 0.15
    notes = np.array([55.0, 55.0, 65.41, 49.0])
    bar = (t / (beat * 4)).astype(int) % len(notes)
    bass = 0.35 * np.sin(2 * np.pi * np.cumsum(notes[bar]) / sr)
    chord = sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate((220.0, 261.63, 329.63)))
    chord = 0.08 * chord * (0.6 + 0.4 * np.sin(2 * np.pi * t / (beat * 8)))
    lead = 0.12 * np.sin(2 * np.pi * 440 * t + 3 * np.sin(2 * np.pi * 5 * t))
    left = 0.6 * kick + hat + bass + chord + lead
    right = 0.6 * kick + 0.8 * hat + bass + 0.9 * chord + 0.8 * lead
    audio = np.stack([left, right], axis=1)
    return (0.8 * audio / np.abs(audio).max()).astype(np.float32)


def _snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = float(np.mean(np.square(reference.astype(np.float64) - test)))
    signal = float(np.mean(np.square(reference, dtype=np.float64)))
    if noise <= 0:
        return float("inf")
    return 10.0 * np.log10(max(signal, 1e-20) / noise)


def _print_env(report: dict) -> None:
    print("Environment")
    print(f"  Python {report['python']} on {report['os']}")
    hip = report.get("torch_rocm_hip")
    kind = f"AMD ROCm {hip}" if hip else (
        f"NVIDIA CUDA {report['torch_cuda_build']}" if report.get("torch_cuda_build") else "CPU-only build")
    print(f"  PyTorch {report['torch']} ({kind})")
    gpu = report.get("gpu")
    if gpu:
        print(f"  GPU: {gpu['name']} - {gpu['vram_gb']:.1f} GB VRAM - self-test passed")
    else:
        print(f"  GPU: NOT AVAILABLE - {report.get('gpu_unavailable_reason')}")
    print(f"  CPU threads: {report['cpu_threads']}")


def run_benchmark(args: argparse.Namespace) -> int:
    from .audio_io import load_audio
    from .separation import (DemucsSeparator, choose_device, cpu_device, default_batch_size,
                             environment_report)
    from .settings import load_settings, save_settings

    print("=" * 72)
    print("StemDeck separation benchmark")
    print("=" * 72)
    report = environment_report()
    _print_env(report)

    if args.song:
        print(f"\nLoading {args.song} ...")
        audio = load_audio(args.song)
        song_label = Path(args.song).name
    else:
        audio = synthetic_song(args.seconds)
        song_label = f"generated test signal ({args.seconds:.0f}s)"
    if args.max_seconds:
        audio = audio[: int(args.max_seconds * SAMPLE_RATE)]
    duration = audio.shape[0] / SAMPLE_RATE
    print(f"Audio: {song_label} - {int(duration // 60)}:{duration % 60:04.1f}")

    settings = load_settings()
    device, fallback_reason = choose_device(args.device)
    if fallback_reason:
        print(f"\n!! GPU NOT USED: {fallback_reason}")
        if args.require_gpu:
            print("   --require-gpu was given, stopping here.")
            return 2
        print("   Running on the CPU instead so you still get a number.")

    model = args.model or settings["model"]
    if device.is_gpu:
        if args.sweep:
            configs = [(p, b) for p in ("fp32", "fp16") for b in (1, 2, 4, 8)]
        else:
            configs = [(args.precision or settings["precision"],
                        args.batch_size or settings["batch_size"] or default_batch_size(device))]
    else:
        configs = [("fp32", 1)]

    results: list[dict] = []
    reference_stems = None
    for precision, batch in configs:
        print(f"\n--- {model} on {device.label()} | {precision} | batch {batch} ---")
        entry = {"model": model, "precision": precision, "batch_size": batch, "ok": False}
        try:
            sep = DemucsSeparator(device, model, precision, batch)
            sep.load()
            print(f"  model loaded in {sep.load_seconds:.1f}s")
            runs = []
            stems = None
            for i in range(args.runs + 1):
                label = "cold run (first use, includes one-off GPU setup)" if i == 0 else f"warm run {i}"
                t0 = time.perf_counter()
                stems, stats = sep.separate(audio)
                dt = time.perf_counter() - t0
                runs.append(dt)
                print(f"  {label}: {dt:6.1f}s  ({duration / dt:5.1f}x realtime)")
            warm = min(runs[1:]) if len(runs) > 1 else runs[0]
            entry.update(ok=True, cold_seconds=runs[0], warm_seconds=warm,
                         batch_size=sep.batch_size, peak_vram_mb=stats.peak_vram_mb,
                         reconstruction_db=stats.reconstruction_db,
                         device_kind=stats.device_kind, backend=stats.backend)
            if stats.peak_vram_mb:
                print(f"  peak GPU memory: {stats.peak_vram_mb / 1024:.1f} GB")
            print(f"  stems add back up to the song within {stats.reconstruction_db:.1f} dB")
            if sep.batch_size != batch:
                print(f"  note: batch size was reduced to {sep.batch_size} (out of GPU memory)")
            if precision == "fp32" and reference_stems is None:
                reference_stems = stems
            elif reference_stems is not None and stems is not None:
                snr = min(_snr_db(reference_stems[s], stems[s]) for s in STEMS)
                entry["snr_vs_fp32_db"] = snr
                print(f"  difference from fp32 result: {snr:.1f} dB SNR "
                      f"({'inaudible' if snr > 60 else 'CHECK BY EAR' if snr > 40 else 'TOO LARGE'})")
            sep.unload()
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {entry['error']}")
        results.append(entry)

    # ---- GPU vs CPU correctness check on a short excerpt -------------------
    verify = None
    if device.is_gpu and not args.no_verify and any(r["ok"] for r in results):
        best = min((r for r in results if r["ok"]), key=lambda r: r["warm_seconds"])
        excerpt_len = min(audio.shape[0], int(12 * SAMPLE_RATE))
        start = max(0, (audio.shape[0] - excerpt_len) // 2)
        excerpt = audio[start:start + excerpt_len]
        print(f"\nChecking GPU output against CPU on a {excerpt_len / SAMPLE_RATE:.0f}s excerpt ...")
        try:
            gpu_stems, _ = DemucsSeparator(device, model, best["precision"], 1).separate(excerpt)
            cpu_stems, _ = DemucsSeparator(cpu_device(), model).separate(excerpt)
            verify = min(_snr_db(cpu_stems[s], gpu_stems[s]) for s in STEMS)
            verdict = "match" if verify > 60 else ("close - check by ear" if verify > 40 else "MISMATCH")
            print(f"  GPU vs CPU agreement: {verify:.1f} dB SNR -> {verdict}")
        except Exception as exc:
            print(f"  verification failed: {type(exc).__name__}: {exc}")

    # ---- Verdict -----------------------------------------------------------
    print("\n" + "=" * 72)
    ok = [r for r in results if r["ok"]]
    if not ok:
        print("RESULT: every configuration failed. Please send this whole output back.")
        return 1
    usable = [r for r in ok if r.get("snr_vs_fp32_db", 999) > 40] or ok
    best = min(usable, key=lambda r: r["warm_seconds"])
    scaled = best["warm_seconds"] * TARGET_SONG_SECONDS / duration
    on_gpu = best.get("device_kind") == "gpu"
    print(f"Ran on:        {'GPU' if on_gpu else 'CPU'} - {device.name} [{best.get('backend')}]")
    print(f"Best config:   {best['model']} | {best['precision']} | batch {best['batch_size']}")
    print(f"Warm time:     {best['warm_seconds']:.1f}s for {duration:.0f}s of audio "
          f"({duration / best['warm_seconds']:.1f}x realtime)")
    print(f"Cold time:     {best['cold_seconds']:.1f}s (first run after install is the slow one)")
    print(f"Per 4:00 song: {scaled:.1f}s   (target: under {TARGET_SECONDS:.0f}s on GPU)")
    passed = on_gpu and scaled < TARGET_SECONDS
    if passed:
        print("RESULT: PASS - GPU separation meets the milestone target.")
    elif on_gpu:
        print("RESULT: GPU WORKS BUT IS SLOWER THAN TARGET. Try --sweep, then --model htdemucs.")
    else:
        print("RESULT: FAIL - separation ran on the CPU, not the GPU (reason above).")
    print("=" * 72)

    if on_gpu and (args.sweep or args.save):
        save_settings({"model": best["model"], "precision": best["precision"],
                       "batch_size": best["batch_size"]})
        print("Saved the best configuration; the app will use it from now on.")

    out_dir = data_dir() / "bench_results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / time.strftime("bench_%Y%m%d_%H%M%S.json")
    out_file.write_text(json.dumps({
        "environment": report, "audio": song_label, "audio_seconds": duration,
        "fallback_reason": fallback_reason, "results": results,
        "gpu_vs_cpu_snr_db": verify, "passed": passed,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Full results written to {out_file}")
    return 0 if passed or not args.require_gpu else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="StemDeck separation benchmark")
    p.add_argument("song", nargs="?", help="audio file to separate (default: generated 4:00 signal)")
    p.add_argument("--model", help=f"model name (default: saved setting, initially {DEFAULT_MODEL})")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--precision", choices=["fp32", "fp16"])
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--runs", type=int, default=1, help="warm runs after the cold run (default 1)")
    p.add_argument("--seconds", type=float, default=TARGET_SONG_SECONDS,
                   help="length of the generated signal when no song is given")
    p.add_argument("--max-seconds", type=float, default=0, help="only use the first N seconds")
    p.add_argument("--sweep", action="store_true",
                   help="try several batch sizes and precisions, save the fastest")
    p.add_argument("--save", action="store_true", help="save this configuration for the app")
    p.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to CPU")
    p.add_argument("--no-verify", action="store_true", help="skip the GPU-vs-CPU correctness check")
    return p


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

"""Export and the on-disk library (cache)."""
import numpy as np
import soundfile as sf

from stemdeck.config import SAMPLE_RATE, STEMS
from stemdeck.engine.export import export_mix, export_stems, mix_stems


def fake_stems(seconds=1.0, amp=0.2):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return {name: np.repeat((amp * np.sin(2 * np.pi * f * t))[:, None], 2, axis=1).astype(np.float32)
            for name, f in zip(STEMS, (440, 110, 55, 330))}


def test_mix_respects_fader_mute_solo():
    stems = fake_stems()
    assert np.allclose(mix_stems(stems), sum(stems.values()), atol=1e-6)
    assert np.allclose(mix_stems(stems, mutes={"drums": True}),
                       stems["vocals"] + stems["bass"] + stems["other"], atol=1e-6)
    assert np.allclose(mix_stems(stems, solos={"vocals": True}), stems["vocals"], atol=1e-6)
    assert np.allclose(mix_stems(stems, faders={"bass": 0.5}, solos={"bass": True}),
                       0.5 * stems["bass"], atol=1e-6)


def test_export_mix_roundtrip(tmp_path):
    stems = fake_stems()
    info = export_mix(stems, tmp_path / "mix.wav", solos={"vocals": True, "bass": True}, subtype="PCM_24")
    data, rate = sf.read(info.path, dtype="float32")
    assert rate == SAMPLE_RATE and data.shape == stems["vocals"].shape
    assert np.abs(data - (stems["vocals"] + stems["bass"])).max() < 1e-5
    assert info.attenuated_db == 0.0


def test_export_never_clips(tmp_path):
    stems = fake_stems(amp=0.6)                    # sum peaks well above 1.0
    info = export_mix(stems, tmp_path / "loud.wav", subtype="PCM_16")
    data, _ = sf.read(info.path, dtype="float32")
    assert np.abs(data).max() <= 1.0
    assert info.attenuated_db > 0
    # shape preserved: scaled copy of the true mix, no flattened peaks
    ref = sum(stems.values())
    scaled = ref * (np.abs(data).max() / np.abs(ref).max())
    assert np.abs(data - scaled).max() < 1e-3
    info = export_mix(stems, tmp_path / "loud_float.wav", subtype="FLOAT")
    data, _ = sf.read(info.path, dtype="float32")
    assert np.abs(data).max() > 1.0 and info.attenuated_db == 0.0   # float keeps it exactly


def test_export_individual_stems(tmp_path):
    stems = fake_stems()
    infos = export_stems(stems, tmp_path, "Song", ["vocals", "drums"])
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["Song - drums.wav", "Song - vocals.wav"]
    data, _ = sf.read(infos[0].path, dtype="float32")
    assert np.abs(data - stems["vocals"]).max() < 1e-5


def test_library_roundtrip_and_cache_hit(tmp_path, monkeypatch):
    """First load does the work; second load of the same bytes comes from disk."""
    monkeypatch.setenv("STEMDECK_HOME", str(tmp_path / "home"))
    import stemdeck.library as lib
    from stemdeck.analysis import AnalysisResult, BeatGrid, KeyResult
    from stemdeck.separation.demucs_backend import SeparationStats

    stems = fake_stems(2.0, amp=0.5)
    stems["vocals"] = stems["vocals"] * 3.0        # peak 1.5: exercises the no-clip scaling
    mix = sum(stems.values())
    song = tmp_path / "song.wav"
    sf.write(song, mix / np.abs(mix).max(), SAMPLE_RATE, subtype="FLOAT")

    calls = {"separate": 0, "analyze": 0}

    def fake_separate(audio, **kwargs):
        calls["separate"] += 1
        stats = SeparationStats("htdemucs", "cpu", "cpu", "test", "fp32", 1, 0.25, 2.0)
        return {k: v.copy() for k, v in stems.items()}, stats

    def fake_analyze(st, device="cpu", **kwargs):
        calls["analyze"] += 1
        grid = BeatGrid(bpm=120.0, beats=np.arange(0, 2, 0.5), downbeats=np.array([0.0]),
                        is_constant=True, confidence=1.0, detected_bpm=120.0)
        return AnalysisResult(120.0, KeyResult(9, "minor", 0.9), grid, 2.0, "stems")

    import stemdeck.separation as sepmod
    monkeypatch.setattr(sepmod, "separate", fake_separate)
    monkeypatch.setattr(lib, "analyze", fake_analyze)

    stages = []
    first = lib.load_track(song, model="htdemucs", progress=lambda s, f, m: stages.append(s))
    assert not first.from_cache and calls == {"separate": 1, "analyze": 1}
    assert "separate" in stages and "save" in stages

    second = lib.load_track(song, model="htdemucs")
    assert second.from_cache and calls == {"separate": 1, "analyze": 1}    # nothing recomputed
    for name in STEMS:
        assert np.abs(second.stems[name] - stems[name]).max() < 2e-6        # lossless (24-bit)
    assert second.analysis.bpm == 120.0 and second.analysis.key.camelot == "8A"
    assert second.peaks["vocals"].n_frames == first.n_frames
    assert second.stem_array().shape == (4, first.n_frames, 2)

    # same audio under a different name / location is still a cache hit
    copy = tmp_path / "renamed copy.wav"
    copy.write_bytes(song.read_bytes())
    third = lib.load_track(copy, model="htdemucs")
    assert third.from_cache and calls["separate"] == 1

    # a newer analysis version re-analyses without re-separating
    monkeypatch.setattr(lib, "ANALYSIS_VERSION", lib.ANALYSIS_VERSION + 1)
    fourth = lib.load_track(song, model="htdemucs")
    assert fourth.from_cache and calls == {"separate": 1, "analyze": 2}

    # a damaged entry is rebuilt instead of crashing
    monkeypatch.setattr(lib, "ANALYSIS_VERSION", lib.ANALYSIS_VERSION - 1)
    (lib.entry_dir(first.track_id, "htdemucs") / "stems" / "bass.flac").write_bytes(b"garbage")
    fifth = lib.load_track(song, model="htdemucs")
    assert not fifth.from_cache and calls["separate"] == 2

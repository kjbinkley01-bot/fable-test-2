"""Key naming (Camelot) and the key detector on synthetic chord progressions."""
import numpy as np
import pytest

from stemdeck.analysis.camelot import camelot_code, display_key, key_name, parse_key
from stemdeck.analysis.key import detect_key
from stemdeck.analysis.keyfeatures import estimate_tuning, semitone_spectrogram
from stemdeck.config import SAMPLE_RATE

CAMELOT = {
    ("minor", 8): "1A", ("minor", 3): "2A", ("minor", 10): "3A", ("minor", 5): "4A",
    ("minor", 0): "5A", ("minor", 7): "6A", ("minor", 2): "7A", ("minor", 9): "8A",
    ("minor", 4): "9A", ("minor", 11): "10A", ("minor", 6): "11A", ("minor", 1): "12A",
    ("major", 11): "1B", ("major", 6): "2B", ("major", 1): "3B", ("major", 8): "4B",
    ("major", 3): "5B", ("major", 10): "6B", ("major", 5): "7B", ("major", 0): "8B",
    ("major", 7): "9B", ("major", 2): "10B", ("major", 9): "11B", ("major", 4): "12B",
}


def test_camelot_wheel_is_complete_and_correct():
    assert len(set(CAMELOT.values())) == 24
    for (mode, pc), code in CAMELOT.items():
        assert camelot_code(pc, mode) == code
        assert parse_key(code) == (pc, mode)
        assert parse_key(key_name(pc, mode)) == (pc, mode)
    assert display_key(9, "minor") == "A minor / 8A"
    # neighbours on the wheel are a fifth apart; same number = relative major/minor
    assert camelot_code(4, "minor") == "9A" and camelot_code(0, "major") == "8B"


@pytest.mark.parametrize("text,expected", [
    ("A minor", (9, "minor")), ("Am", (9, "minor")), ("A min", (9, "minor")),
    ("F# maj", (6, "major")), ("Gb", (6, "major")), ("G♭ major", (6, "major")),
    ("D♭ min", (1, "minor")), ("C#m", (1, "minor")), ("Bbm", (10, "minor")),
    ("8A", (9, "minor")), ("11b", (9, "major")), ("1d", (0, "major")), ("1m", (9, "minor")),
])
def test_key_text_from_other_tools_is_understood(text, expected):
    assert parse_key(text) == expected


def test_unparseable_key_text():
    assert parse_key("") is None and parse_key("H minor") is None and parse_key("13A") is None


def tone(midi: float, seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 6))   # a few overtones
    return amp * wave * np.minimum(1.0, np.minimum(t, seconds - t) * 50)


def progression(root: int, minor: bool, cents: float = 0.0, bars: int = 24) -> np.ndarray:
    """An unambiguous cadence with bass notes: I-I-IV-V, or i-i-iv-V in minor
    (major chord on the fifth, as in most minor-key music). ``root`` = pitch class 0-11."""
    tonic_third = 3 if minor else 4
    plan = [(0, tonic_third), (0, tonic_third), (5, tonic_third), (7, 4)]
    chords = []
    for bar in range(bars):
        degree, third = plan[bar % 4]
        base = 48 + root + degree + cents / 100.0
        chord = tone(base, 2.0) + tone(base + third, 2.0) + tone(base + 7, 2.0)
        chord += tone(base - 12, 2.0, 0.3)                              # bass
        chords.append(chord)
    mono = np.concatenate(chords).astype(np.float32)
    return np.stack([mono, mono], axis=1)


@pytest.mark.parametrize("root,minor", [(0, False), (9, True), (5, True), (2, False), (10, True), (7, False)])
def test_detects_key_of_simple_progressions(root, minor):
    key = detect_key(progression(root, minor))
    assert (key.pitch_class, key.mode) == (root, "minor" if minor else "major"), key.display
    assert key.alternatives and 0 < key.confidence <= 1


def test_transposing_the_music_transposes_the_answer():
    found = [detect_key(progression(root, True)).pitch_class for root in range(12)]
    assert found == list(range(12))


def test_detuned_track_is_still_recognised():
    audio = progression(9, True, cents=35.0)
    semis, tuning = semitone_spectrogram(audio)
    assert abs(tuning * 100 - 35.0) < 8.0
    key = detect_key(audio)
    assert (key.pitch_class, key.mode) == (9, "minor")
    assert abs(key.tuning_cents - 35.0) < 8.0


def test_silence_gives_no_confident_key():
    key = detect_key(np.zeros((SAMPLE_RATE * 5, 2), dtype=np.float32))
    assert key.confidence == 0.0

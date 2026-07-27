# tests/test_pitch_correction.py
"""Unit tests for ``core.soulsx_singer._correct_pitch_from_f0``.

Verifies the post-ROSVOT pitch correction that fixes gross quantization
errors by re-deriving each note's MIDI number from the median of its
voiced f0 frames.

The oracle (test_correct_pitch_oracle_seg1) is the *ground-truth output
of the verbatim algorithm* on the user-supplied segment-1 fixture
(``tests/fixtures/pitch_correction_seg1.json``). It corrects 8 of 69
voiced notes; the residual max |Δsemi| after correction is < 1.5 and no
new ≥2-semitone errors are introduced.
"""
import copy
import json
import math
import os

import pytest

from core.soulsx_singer import _correct_pitch_from_f0


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "pitch_correction_seg1.json"
)


def _load_seg1_metadata() -> dict:
    """Load the segment-1 fixture as a fresh mutable metadata dict."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)[0]


# ---------------------------------------------------------------------------
# 1. Oracle: full-fixture integration of the verbatim algorithm
# ---------------------------------------------------------------------------


class TestOracleSeg1:
    """Ground-truth corrections produced by the algorithm on seg-1 fixture.

    Each tuple is (1-indexed_position, char, before, after). These values
    are the actual output of ``_correct_pitch_from_f0`` (which uses
    ``sorted(voiced)[len(voiced)//2]`` — the upper-middle element — as the
    median) on the fixture. Residual max |Δsemi| after correction is 1.408
    (< 1.5) and no new ≥2-semitone errors are introduced.
    """

    EXPECTED_CORRECTIONS = [
        (13, "兴", 65, 68),
        (22, "美", 64, 62),
        (44, "棘", 62, 60),
        (45, "上", 60, 62),
        (48, "地", 57, 59),
        (50, "花", 53, 57),
        (63, "兴", 64, 66),
        (69, "满", 60, 62),
    ]

    def test_correct_pitch_oracle_seg1(self):
        meta = _load_seg1_metadata()
        original = [int(x) for x in meta["note_pitch"].split()]
        texts = meta["text"].split()

        _correct_pitch_from_f0(meta)

        corrected = (
            [int(x) for x in meta["note_pitch"].split()]
            if isinstance(meta["note_pitch"], str)
            else list(meta["note_pitch"])
        )

        # Diff before/after to find every changed position.
        changed = []
        for i, (a, b) in enumerate(zip(original, corrected)):
            if a != b:
                changed.append((i + 1, texts[i], a, b))

        assert len(changed) == len(self.EXPECTED_CORRECTIONS), (
            f"expected {len(self.EXPECTED_CORRECTIONS)} corrections, "
            f"got {len(changed)}: {changed}"
        )
        for got, exp in zip(changed, self.EXPECTED_CORRECTIONS):
            assert got == exp, f"correction mismatch: got {got}, expected {exp}"

    def test_correct_pitch_count_is_eight(self):
        """Acceptance: exactly 8 corrections on seg-1 fixture."""
        meta = _load_seg1_metadata()
        before = [int(x) for x in meta["note_pitch"].split()]
        _correct_pitch_from_f0(meta)
        after = [int(x) for x in meta["note_pitch"].split()]
        n_changed = sum(1 for a, b in zip(before, after) if a != b)
        assert n_changed == 8

    def test_correct_pitch_no_new_gross_errors(self):
        """Acceptance: correction must not introduce any new |Δsemi| ≥ 2.

        After correction, recompute f0-median pitch for every voiced note
        and confirm none differ from note_pitch by ≥ 2 semitones.
        """
        from core.midi_format import FPS

        meta = _load_seg1_metadata()
        _correct_pitch_from_f0(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        durations = [float(x) for x in meta["duration"].split()]
        f0 = [float(x) for x in meta["f0"].split()]

        new_gross = []
        cum = 0
        for i in range(len(pitches)):
            n = int(round(durations[i] * FPS))
            start, end = cum, min(cum + n, len(f0))
            cum += n
            if pitches[i] == 0:
                continue
            voiced = [x for x in f0[start:end] if x > 0]
            if len(voiced) < 3:
                continue
            med = sorted(voiced)[len(voiced) // 2]
            f0_midi = round(69 + 12 * math.log2(med / 440.0))
            if abs(f0_midi - pitches[i]) >= 2:
                new_gross.append((i + 1, pitches[i], f0_midi))
        assert new_gross == [], f"new gross errors introduced: {new_gross}"


# ---------------------------------------------------------------------------
# 2. SP notes are skipped
# ---------------------------------------------------------------------------


class TestSkipSP:
    def test_sp_notes_are_not_touched(self):
        """note_pitch==0 (SP) entries must remain 0 even if f0 is voiced.

        Uses a mixed metadata: an SP note surrounded by grossly-wrong non-SP
        notes. The non-SP notes MUST be corrected (proving the function ran)
        while the SP entry stays 0.
        """
        # 3 notes, 0.30s each (15 frames). f0 = 440 Hz everywhere (MIDI 69).
        meta = {
            "item_name": "synthetic_sp_test",
            "note_pitch": "60 0 60",   # non-SP, SP, non-SP
            "duration": "0.30 0.30 0.30",
            "f0": " ".join(["440.0"] * 45),
        }
        _correct_pitch_from_f0(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        # Non-SP notes must be corrected 60 -> 69 (proves function ran).
        assert pitches[0] == 69, f"non-SP index 0 should correct to 69, got {pitches[0]}"
        assert pitches[2] == 69, f"non-SP index 2 should correct to 69, got {pitches[2]}"
        # SP entry must stay 0.
        assert pitches[1] == 0, f"SP index 1 must stay 0, got {pitches[1]}"


# ---------------------------------------------------------------------------
# 3. Notes with < 3 voiced frames are skipped
# ---------------------------------------------------------------------------


class TestSkipFewVoicedFrames:
    def test_skip_when_voiced_frames_below_three(self):
        """A note with only 1-2 voiced f0 frames is too unreliable; skip it.

        Construct two grossly-wrong notes (pitch 60, f0 ~440 -> MIDI 69):
        - index 0: < 3 voiced frames -> must NOT correct
        - index 1: many voiced frames -> MUST correct (proves function ran)
        """
        # index 0: 0.10s = 5 frames, only 2 voiced.
        # index 1: 0.30s = 15 frames, all voiced.
        frames = (
            ["0.0", "440.0", "440.0", "0.0", "0.0"]      # idx0: 2 voiced
            + ["440.0"] * 15                              # idx1: 15 voiced
        )
        meta = {
            "item_name": "few_voiced",
            "note_pitch": "60 60",
            "duration": "0.10 0.30",
            "f0": " ".join(frames),
        }
        _correct_pitch_from_f0(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        assert pitches[0] == 60, f"idx0 (<3 voiced) must stay 60, got {pitches[0]}"
        assert pitches[1] == 69, f"idx1 (many voiced) must correct to 69, got {pitches[1]}"


# ---------------------------------------------------------------------------
# 4. No correction when |Δsemi| < threshold (2)
# ---------------------------------------------------------------------------


class TestNoCorrectionUnderThreshold:
    def test_pitch_within_threshold_unchanged(self):
        """A note whose f0-median is within 1 semitone must be left alone,
        while a grossly-wrong note in the same metadata IS corrected."""
        # MIDI 60 ≈ 261.63 Hz. On-pitch note: f0=261.6 -> delta 0.
        # Gross note: f0=440.0 -> MIDI 69, delta 9 -> must correct.
        frames = ["261.6"] * 15 + ["440.0"] * 15
        meta = {
            "item_name": "mixed_threshold",
            "note_pitch": "60 60",
            "duration": "0.30 0.30",
            "f0": " ".join(frames),
        }
        _correct_pitch_from_f0(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        assert pitches[0] == 60, f"on-pitch idx0 must stay 60, got {pitches[0]}"
        assert pitches[1] == 69, f"gross idx1 must correct to 69, got {pitches[1]}"


# ---------------------------------------------------------------------------
# 5. String vs list input formats produce consistent results
# ---------------------------------------------------------------------------


class TestFormatConsistency:
    def test_string_and_list_inputs_match(self):
        """The function must accept both space-string and list note_pitch /
        duration and produce semantically identical corrections."""
        # Construct a grossly-wrong note: pitch 60 but f0 ~ 440 Hz (MIDI 69).
        frames = ["440.0"] * 15  # 0.30s window, all voiced
        base = {
            "item_name": "fmt_str",
            "note_pitch": "60",
            "duration": "0.30",
            "f0": " ".join(frames),
        }
        base_list = {
            "item_name": "fmt_list",
            "note_pitch": [60],
            "duration": [0.30],
            "f0": " ".join(frames),
        }
        _correct_pitch_from_f0(base)
        _correct_pitch_from_f0(base_list)

        # Both must correct 60 -> 69.
        assert base["note_pitch"] == "69"
        assert base_list["note_pitch"] == [69]


# ---------------------------------------------------------------------------
# 6. Missing / malformed fields do not raise
# ---------------------------------------------------------------------------


class TestRobustness:
    @pytest.mark.parametrize("meta", [
        {"item_name": "empty"},  # no relevant fields at all
        {"item_name": "no_pitch", "duration": "0.3", "f0": "261.6"},
        {"item_name": "no_dur", "note_pitch": "60", "f0": "261.6"},
        {"item_name": "no_f0", "note_pitch": "60", "duration": "0.3"},
        {"item_name": "empty_f0", "note_pitch": "60", "duration": "0.3", "f0": ""},
        # Length mismatch between note_pitch and duration -> bail out silently.
        {"item_name": "len_mismatch",
         "note_pitch": "60 62", "duration": "0.3", "f0": "261.6 261.6 261.6"},
    ])
    def test_missing_or_malformed_fields_no_raise(self, meta):
        """Function must be a no-op (never raise) on incomplete metadata."""
        before = copy.deepcopy(meta)
        _correct_pitch_from_f0(meta)  # must not raise
        assert meta == before, f"metadata mutated unexpectedly: {meta}"

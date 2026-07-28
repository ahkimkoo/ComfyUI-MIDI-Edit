# tests/test_subnote_split.py
"""Unit tests for ``core.soulsx_singer._split_notes_by_contour``.

Verifies the score-mode sub-note splitting that encodes intra-note f0
contour into multiple note_pitch values (staircase approximation).
"""
import copy
import json
import math
import os

import pytest

from core.soulsx_singer import _split_notes_by_contour, _fix_zero_pitch_notes


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "pitch_correction_seg1.json"
)


def _load_fixture():
    with open(FIXTURE_PATH) as f:
        d = json.load(f)
    return d[0] if isinstance(d, list) else d


def _hz_to_midi(h):
    return 69 + 12 * math.log2(h / 440.0) if h > 0 else None


# ---------------------------------------------------------------------------
# Oracle: fixture data
# ---------------------------------------------------------------------------


class TestOracleFixture:
    """Verify splitting on the real user-supplied segment-1 fixture."""

    def test_split_count_is_33(self):
        meta = _load_fixture()
        orig_len = len(meta["note_pitch"].split())
        _split_notes_by_contour(meta)
        new_len = len(meta["note_pitch"].split())
        # 28 notes split (per-phrase limit of 8 trims 2 low-priority splits)
        assert new_len == orig_len + 28, f"expected {orig_len + 28} tokens, got {new_len}"

    def test_ya_pos25_split_62_64(self):
        """User-reported '涯' (pos 25): single pitch 62 → [62, 64]."""
        meta = _load_fixture()
        _split_notes_by_contour(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        texts = meta["text"].split()
        # Find the two sub-notes for 涯
        ya_indices = [i for i, t in enumerate(texts) if t == "涯"]
        assert len(ya_indices) == 2, f"expected 2 涯 tokens, got {len(ya_indices)}"
        p1, p2 = pitches[ya_indices[0]], pitches[ya_indices[1]]
        assert (p1, p2) == (62, 64), f"涯 pitches: expected (62, 64), got ({p1}, {p2})"

    def test_xing_pos13_split_69_65(self):
        """'兴' (pos 13): falling contour, single pitch 65 → [69, 65]."""
        meta = _load_fixture()
        _split_notes_by_contour(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        texts = meta["text"].split()
        types = [int(x) for x in meta["note_type"].split()]
        xing_indices = [i for i, t in enumerate(texts) if t == "兴"]
        assert len(xing_indices) >= 2
        p1, p2 = pitches[xing_indices[0]], pitches[xing_indices[1]]
        assert (p1, p2) == (69, 65), f"兴 pitches: expected (69, 65), got ({p1}, {p2})"
        # Second sub-note should be type 3 (continuation)
        assert types[xing_indices[1]] == 3

    def test_sp_not_split(self):
        """SP tokens (pitch=0) must not be split."""
        meta = _load_fixture()
        orig_texts = meta["text"].split()
        orig_sp_count = sum(1 for t in orig_texts if t == "<SP>")
        _split_notes_by_contour(meta)
        new_texts = meta["text"].split()
        new_sp_count = sum(1 for t in new_texts if t == "<SP>")
        assert new_sp_count == orig_sp_count, "SP count changed after split"

    def test_duration_preserved(self):
        """Total duration must be unchanged after splitting."""
        meta = _load_fixture()
        orig_total = sum(float(x) for x in meta["duration"].split())
        _split_notes_by_contour(meta)
        new_total = sum(float(x) for x in meta["duration"].split())
        assert abs(new_total - orig_total) < 0.01, f"duration changed: {orig_total:.2f} → {new_total:.2f}"


# ---------------------------------------------------------------------------
# Edge cases (synthetic metadata)
# ---------------------------------------------------------------------------


def _make_meta(pitches, durations, f0_frames, texts=None, types=None, phons=None):
    """Build a minimal metadata dict for testing."""
    n = len(pitches)
    return {
        "note_pitch": " ".join(str(p) for p in pitches),
        "duration": " ".join(f"{d:.2f}" for d in durations),
        "text": " ".join(texts or [f"c{i}" for i in range(n)]),
        "note_type": " ".join(str(t) for t in (types or [2] * n)),
        "phoneme": " ".join(phons or [f"zh_c{i}" for i in range(n)]),
        "f0": " ".join(f"{f:.1f}" for f in f0_frames),
    }


class TestEdgeCases:
    def test_no_split_for_sp(self):
        """SP (pitch=0) is never split, even with large f0 spread."""
        # 1 SP note, 1s, f0 varies wildly
        f0 = [200.0 + i * 10 for i in range(50)]  # rising f0
        meta = _make_meta([0], [1.0], f0, texts=["<SP>"], types=[1], phons=["<SP>"])
        _split_notes_by_contour(meta)
        assert meta["note_pitch"] == "0"
        assert len(meta["text"].split()) == 1

    def test_no_split_for_short_note(self):
        """Notes shorter than 0.2s are not split."""
        f0 = [200.0, 210.0, 220.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0]
        meta = _make_meta([60], [0.10], f0)  # 0.10s < 0.2s
        _split_notes_by_contour(meta)
        assert len(meta["note_pitch"].split()) == 1

    def test_no_split_for_flat_contour(self):
        """Notes with f0 spread < threshold are not split."""
        # All f0 values very close (spread < 2 semitones)
        f0 = [261.0 + i * 0.5 for i in range(50)]  # ~261-285 Hz, spread ~1.5 semi
        meta = _make_meta([60], [1.0], f0)
        _split_notes_by_contour(meta)
        assert len(meta["note_pitch"].split()) == 1

    def test_no_split_when_halves_close(self):
        """If the two halves' medians differ by < min_half_diff, don't split."""
        # f0 rises then falls — overall spread is large but halves are similar
        f0 = [300.0] * 25 + [300.0] * 25  # flat, spread = 0
        meta = _make_meta([60], [1.0], f0)
        _split_notes_by_contour(meta)
        assert len(meta["note_pitch"].split()) == 1

    def test_split_rising_contour(self):
        """A clearly rising note should be split into [low, high]."""
        # 50 frames: first half ~220 Hz (A3, MIDI 57), second half ~330 Hz (E4, MIDI 64)
        f0 = [220.0] * 25 + [330.0] * 25
        meta = _make_meta([60], [1.0], f0)
        _split_notes_by_contour(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        assert len(pitches) == 2, f"expected 2 sub-notes, got {len(pitches)}"
        assert pitches[0] < pitches[1], f"rising contour should give [low, high], got {pitches}"

    def test_split_falling_contour(self):
        """A clearly falling note should be split into [high, low]."""
        f0 = [330.0] * 25 + [220.0] * 25
        meta = _make_meta([60], [1.0], f0)
        _split_notes_by_contour(meta)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        assert len(pitches) == 2
        assert pitches[0] > pitches[1], f"falling contour should give [high, low], got {pitches}"

    def test_second_subnote_has_type_3(self):
        """The second sub-note must have note_type=3 (continuation)."""
        f0 = [220.0] * 25 + [330.0] * 25
        meta = _make_meta([60], [1.0], f0, types=[2])
        _split_notes_by_contour(meta)
        types = [int(x) for x in meta["note_type"].split()]
        assert len(types) == 2
        assert types[0] == 2, "first sub-note keeps original type"
        assert types[1] == 3, "second sub-note must be type 3 (continuation)"

    def test_string_and_list_formats(self):
        """Both string and list metadata formats produce identical results."""
        f0 = [220.0] * 25 + [330.0] * 25
        # String format
        meta_str = _make_meta([60], [1.0], f0)
        _split_notes_by_contour(meta_str)
        # List format
        meta_list = {
            "note_pitch": [60],
            "duration": [1.0],
            "text": ["c0"],
            "note_type": [2],
            "phoneme": ["zh_c0"],
            "f0": " ".join(f"{f:.1f}" for f in f0),
        }
        _split_notes_by_contour(meta_list)
        assert meta_str["note_pitch"] == " ".join(str(x) for x in meta_list["note_pitch"])

    def test_missing_fields_no_crash(self):
        """Missing or malformed fields should not raise."""
        _split_notes_by_contour({})  # empty dict
        _split_notes_by_contour({"note_pitch": "60 62"})  # missing other fields
        _split_notes_by_contour({"note_pitch": "60", "duration": "1.0"})  # missing f0
        # No exception = pass

    def test_mismatched_lengths_no_crash(self):
        """Mismatched note_pitch/duration lengths should bail out silently."""
        meta = {
            "note_pitch": "60 62 64",
            "duration": "1.0 1.0",  # 2 vs 3
            "text": "a b c",
            "note_type": "2 2 2",
            "phoneme": "zh_a zh_b zh_c",
            "f0": " ".join(["220.0"] * 50),
        }
        _split_notes_by_contour(meta)
        assert meta["note_pitch"] == "60 62 64"  # unchanged


# ---------------------------------------------------------------------------
# _fix_zero_pitch_notes tests
# ---------------------------------------------------------------------------


class TestFixZeroPitchNotes:
    def test_fixes_broken_token(self):
        """Non-SP token with pitch=0 gets f0 median as pitch."""
        # 1 note, dur=1.0s, f0 ~261 Hz (MIDI 60), but pitch=0 (broken)
        f0 = [261.0] * 50
        meta = _make_meta([0], [1.0], f0, texts=["那"], types=[2], phons=["zh_na4"])
        _fix_zero_pitch_notes(meta)
        pitch = int(meta["note_pitch"].split()[0])
        assert pitch == 60, f"expected 60, got {pitch}"

    def test_sp_not_fixed(self):
        """SP token (phoneme=<SP>) with pitch=0 is left as-is."""
        f0 = [261.0] * 50
        meta = _make_meta([0], [1.0], f0, texts=["<SP>"], types=[1], phons=["<SP>"])
        _fix_zero_pitch_notes(meta)
        assert meta["note_pitch"] == "0"

    def test_nonzero_pitch_untouched(self):
        """Tokens with pitch > 0 are not modified."""
        f0 = [261.0] * 50
        meta = _make_meta([65], [1.0], f0)
        _fix_zero_pitch_notes(meta)
        assert meta["note_pitch"] == "65"

    def test_insufficient_voiced_frames(self):
        """If < 3 voiced frames, don't fix (not enough data)."""
        f0 = [0.0] * 48 + [261.0, 262.0]  # only 2 voiced frames
        meta = _make_meta([0], [1.0], f0, texts=["那"], types=[2], phons=["zh_na4"])
        _fix_zero_pitch_notes(meta)
        assert meta["note_pitch"] == "0"  # unchanged


# ---------------------------------------------------------------------------
# Per-phrase split limit tests
# ---------------------------------------------------------------------------


class TestPerPhraseLimit:
    def test_limit_respected(self):
        """No phrase should have more than max_splits_per_phrase splits."""
        # Build a phrase with 12 notes, all with large f0 spread
        # (each note: first half 220 Hz, second half 440 Hz → ~12 semi spread)
        n_notes = 12
        f0_per_note = [220.0] * 25 + [440.0] * 25  # 50 frames per note
        f0 = []
        for _ in range(n_notes):
            f0.extend(f0_per_note)
        pitches = [60] * n_notes
        durations = [1.0] * n_notes
        meta = _make_meta(pitches, durations, f0)

        _split_notes_by_contour(meta, max_splits_per_phrase=5)
        new_len = len(meta["note_pitch"].split())
        # 5 splits → 5 extra tokens → 12 + 5 = 17
        assert new_len == n_notes + 5, f"expected {n_notes + 5}, got {new_len}"

    def test_prioritizes_largest_spread(self):
        """When limiting, the notes with the largest f0 spread are kept."""
        # 3 notes: spreads of ~2, ~6, ~12 semitones
        # With max_splits_per_phrase=1, only the ~12 semi note should be split
        f0_small = [260.0] * 25 + [280.0] * 25   # ~1.3 semi spread (won't qualify)
        f0_med = [220.0] * 25 + [330.0] * 25      # ~7 semi spread
        f0_large = [220.0] * 25 + [440.0] * 25    # ~12 semi spread
        f0 = f0_small + f0_med + f0_large
        meta = _make_meta([60, 60, 60], [1.0, 1.0, 1.0], f0)

        _split_notes_by_contour(meta, max_splits_per_phrase=1)
        pitches = [int(x) for x in meta["note_pitch"].split()]
        # Only 1 split (the largest spread note)
        assert len(pitches) == 4, f"expected 4 tokens (3 + 1 split), got {len(pitches)}"
        # The split note should be the last one (largest spread)
        # Its sub-pitches should be ~57 and ~69
        assert pitches[2] < pitches[3], "split note should have rising sub-pitches"

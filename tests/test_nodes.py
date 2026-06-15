"""Tests for ComfyUI-MIDI-Edit nodes.py module."""

import json
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(text, phoneme, duration, note_pitch, note_type, f0="0.0"):
    """Build a minimal track dict suitable for replace_lyrics."""
    return {
        "text": text,
        "phoneme": phoneme,
        "duration": duration,
        "note_pitch": note_pitch,
        "note_type": note_type,
        "f0": f0,
    }


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from nodes import (
    replace_lyrics,
    extract_lyrics,
    merge_repeated_chars,
    clean_lyrics,
    char_to_phoneme,
    is_chinese_char,
    is_english_word,
    _split_lyrics_to_sentences,
    _count_total_sections,
    _get_section_sizes,
    _split_by_section_sizes,
    _build_collapsed_slots,
    _split_token,
    _apply_char,
    _process_section,
    _split_into_segments,
    _smart_split_sentences,
    _split_at_punctuation,
    _first_punct_cut,
    _compute_expected_char_counts,
    _get_section_durations,
    _apply_speed,
    _fmt_dur,
    _fmt_f0,
    _code_mix_split_words,
    _split_to_mini_sentence,
    _TokenIDConverter,
    _CTTransformerPunc,
    _PUNC_LIST_NORMALIZED,
    _SPLIT_SIZE,
)


# ===================================================================
# char_to_phoneme / char helpers
# ===================================================================


class TestCharToPhoneme:
    def test_sp_returns_sp(self):
        assert char_to_phoneme("<SP>") == "<SP>"

    def test_chinese_char(self):
        result = char_to_phoneme("你")
        assert result.startswith("zh_")
        assert result != "zh_"  # should have actual pinyin

    def test_single_english_letter(self):
        assert char_to_phoneme("A") == "en_a"
        assert char_to_phoneme("b") == "en_b"

    def test_unknown_char(self):
        assert char_to_phoneme("123") == "<SP>"
        assert char_to_phoneme("!") == "<SP>"


class TestIsChineseChar:
    def test_chinese_true(self):
        assert is_chinese_char("你")
        assert is_chinese_char("的")

    def test_chinese_false(self):
        assert not is_chinese_char("A")
        assert not is_chinese_char("1")
        assert not is_chinese_char("")


class TestIsEnglishWord:
    def test_english_true(self):
        assert is_english_word("hello")
        assert is_english_word("don't")

    def test_english_false(self):
        assert not is_english_word("")
        assert not is_english_word("123")
        assert not is_english_word("你")


# ===================================================================
# clean_lyrics / merge_repeated_chars
# ===================================================================


class TestCleanLyrics:
    def test_removes_punctuation(self):
        assert clean_lyrics("你好，世界！") == "你好世界"

    def test_removes_spaces_and_newlines(self):
        assert clean_lyrics("hello world\nnew line") == "helloworldnewline"

    def test_keeps_chinese_and_english(self):
        assert clean_lyrics("A你好B") == "A你好B"


class TestMergeRepeatedChars:
    def test_consecutive_duplicates(self):
        assert merge_repeated_chars("向向往") == "向往"
        assert merge_repeated_chars("天天马") == "天马"

    def test_non_consecutive(self):
        assert merge_repeated_chars("好世界好") == "好世界好"

    def test_empty(self):
        assert merge_repeated_chars("") == ""

    def test_single_char(self):
        assert merge_repeated_chars("A") == "A"


# ===================================================================
# Sentence splitting
# ===================================================================


class TestSplitLyricsToSentences:
    def test_newline_split(self):
        result = _split_lyrics_to_sentences("你好\n世界")
        assert result == ["你好", "世界"]

    def test_punctuation_split(self):
        result = _split_lyrics_to_sentences("你好，世界！再见")
        assert result == ["你好", "世界", "再见"]

    def test_empty_returns_empty(self):
        assert _split_lyrics_to_sentences("") == []
        assert _split_lyrics_to_sentences("，。！") == []


# ===================================================================
# Section size helpers
# ===================================================================


class TestSectionHelpers:
    def test_count_total_sections(self):
        midi = [_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            "0.5 0.3 0.3 0.5 0.3 0.3 0.3 0.5",
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )]
        assert _count_total_sections(midi) == 2

    def test_get_section_sizes(self):
        midi = [_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            "0.5 0.3 0.3 0.5 0.3 0.3 0.3 0.5",
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )]
        assert _get_section_sizes(midi) == [2, 3]

    def test_split_by_section_sizes(self):
        assert _split_by_section_sizes("ABCDE", [2, 3]) == ["AB", "CDE"]
        assert _split_by_section_sizes("ABC", [2, 3]) == ["AB", "C"]
        assert _split_by_section_sizes("A", [3]) == ["A"]


# ===================================================================
# _build_collapsed_slots
# ===================================================================


class TestBuildCollapsedSlots:
    def test_no_duplicates(self):
        tokens = [
            {"text": "A", "phoneme": "en_a", "duration": 0.3, "note_pitch": 60, "note_type": 2},
            {"text": "B", "phoneme": "en_b", "duration": 0.3, "note_pitch": 62, "note_type": 2},
        ]
        slots = _build_collapsed_slots(tokens)
        assert len(slots) == 2
        assert slots[0] == ("A", 1, [0])
        assert slots[1] == ("B", 1, [1])

    def test_with_duplicates(self):
        tokens = [
            {"text": "A", "phoneme": "en_a", "duration": 0.3, "note_pitch": 60, "note_type": 2},
            {"text": "A", "phoneme": "en_a", "duration": 0.3, "note_pitch": 60, "note_type": 2},
            {"text": "B", "phoneme": "en_b", "duration": 0.3, "note_pitch": 62, "note_type": 2},
        ]
        slots = _build_collapsed_slots(tokens)
        assert len(slots) == 2
        assert slots[0] == ("A", 2, [0, 1])
        assert slots[1] == ("B", 1, [2])

    def test_empty(self):
        assert _build_collapsed_slots([]) == []


# ===================================================================
# _split_token
# ===================================================================


class TestSplitToken:
    def test_split_halves_duration(self):
        tokens = [
            {"text": "A", "phoneme": "en_a", "duration": 1.0, "note_pitch": 60, "note_type": 2},
        ]
        _split_token(tokens, 0)
        assert len(tokens) == 2
        assert tokens[0]["duration"] == pytest.approx(0.5)
        assert tokens[1]["duration"] == pytest.approx(0.5)
        assert tokens[1]["note_pitch"] == 60  # pitch preserved

    def test_split_preserves_internal_tags(self):
        tokens = [
            {"text": "A", "phoneme": "en_a", "duration": 1.0,
             "note_pitch": 60, "note_type": 2, "_sec_id": 5},
        ]
        _split_token(tokens, 0)
        assert tokens[1]["_sec_id"] == 5


# ===================================================================
# _process_section
# ===================================================================


class TestProcessSection:
    def _make_tokens(self, texts, durations):
        return [
            {"text": t, "phoneme": f"en_{t.lower()}", "duration": d,
             "note_pitch": 60, "note_type": 2}
            for t, d in zip(texts, durations)
        ]

    def test_collapse_mode_right_aligns(self):
        """Bug 1 fix: Collapse mode should right-align, mapping last char to last slot."""
        # 3 slots, 2 chars → skip first slot, map to last 2
        tokens = self._make_tokens(["A", "B", "C"], [0.3, 0.3, 1.0])
        _process_section(tokens, "XY", False, 79)
        # First slot should be empty, last 2 filled
        assert tokens[0]["text"] == ""
        assert tokens[1]["text"] == "X"
        assert tokens[2]["text"] == "Y"

    def test_collapse_mode_right_align_preserves_long_note(self):
        """The longest note (last slot) should retain its text in collapse mode."""
        tokens = self._make_tokens(["A", "B", "C", "D"], [0.3, 0.3, 0.3, 1.08])
        # 3 chars, 4 slots → skip first slot
        _process_section(tokens, "XYZ", False, 79)
        assert tokens[0]["text"] == ""
        assert tokens[1]["text"] == "X"
        assert tokens[2]["text"] == "Y"
        assert tokens[3]["text"] == "Z"
        # Last slot (Z) keeps the long duration
        assert tokens[3]["duration"] == pytest.approx(1.08)

    def test_collapse_equal_slots_no_skip(self):
        """When N == S, no slots are skipped (skip_count = 0)."""
        tokens = self._make_tokens(["A", "B", "C"], [0.3, 0.3, 0.3])
        _process_section(tokens, "XYZ", False, 79)
        assert tokens[0]["text"] == "X"
        assert tokens[1]["text"] == "Y"
        assert tokens[2]["text"] == "Z"

    def test_expand_mode_splits_tokens(self):
        """Expand mode should split longest tokens until count matches."""
        tokens = self._make_tokens(["A", "B"], [0.6, 0.4])
        _process_section(tokens, "WXYZ", False, 79)
        assert len(tokens) == 4
        assert tokens[0]["text"] == "W"
        assert tokens[1]["text"] == "X"
        assert tokens[2]["text"] == "Y"
        assert tokens[3]["text"] == "Z"

    def test_empty_sentence_empties_all(self):
        """Empty sentence should empty all token text/phoneme."""
        tokens = self._make_tokens(["A", "B"], [0.3, 0.3])
        _process_section(tokens, "", False, 79)
        assert all(t["text"] == "" for t in tokens)

    def test_empty_tokens_returns_empty(self):
        """Empty token list should return as-is."""
        result = _process_section([], "ABC", False, 79)
        assert result == []

    def test_collapse_distribute_mode(self):
        """Collapse+Distribute: multi-count slots distribute chars to individual tokens."""
        # "A A B C" → slots: A(×2, [0,1]), B(×1, [2]), C(×1, [3])
        # S=3, M=4. Sentence "WXYZ" → N=4. S < N <= M → Collapse+Distribute
        tokens = self._make_tokens(["A", "A", "B", "C"], [0.29, 0.29, 0.40, 1.73])
        _process_section(tokens, "WXYZ", False, 79)
        assert tokens[0]["text"] == "W"
        assert tokens[1]["text"] == "X"
        assert tokens[2]["text"] == "Y"
        assert tokens[3]["text"] == "Z"

    def test_collapse_distribute_partial_fill(self):
        """Collapse+Distribute: when N < total capacity, trailing slot tokens are emptied."""
        # "A A A B" → slots: A(×3, [0,1,2]), B(×1, [3])
        # S=2, M=4. Sentence "WXYZ" → N=4. S < N <= M → Collapse+Distribute
        # A slot gets W,X,Y; B slot gets Z. No empties since N=M=4.
        # For partial fill, use N=3: S < 3 < M=4
        tokens = self._make_tokens(["A", "A", "A", "B"], [0.29, 0.29, 0.40, 1.73])
        _process_section(tokens, "WXY", False, 79)
        # A slot(×3): gets "W","X","Y"; B slot(×1): empty
        assert tokens[0]["text"] == "W"
        assert tokens[1]["text"] == "X"
        assert tokens[2]["text"] == "Y"
        assert tokens[3]["text"] == ""

    def test_collapse_distribute_tiantian_case(self):
        """天天马行空的生涯 (S=7,M=8) + 把它扔给你的烦恼 (N=8) → Collapse+Distribute."""
        tokens = [
            {"text": "天", "phoneme": "zh_tian1", "duration": 0.29,
             "note_pitch": 52, "note_type": 2},
            {"text": "天", "phoneme": "zh_tian1", "duration": 0.29,
             "note_pitch": 53, "note_type": 3},
            {"text": "马", "phoneme": "zh_ma3", "duration": 0.40,
             "note_pitch": 46, "note_type": 2},
            {"text": "行", "phoneme": "zh_xing2", "duration": 0.30,
             "note_pitch": 53, "note_type": 2},
            {"text": "空", "phoneme": "zh_kong1", "duration": 0.34,
             "note_pitch": 53, "note_type": 2},
            {"text": "的", "phoneme": "zh_de5", "duration": 0.26,
             "note_pitch": 54, "note_type": 2},
            {"text": "生", "phoneme": "zh_sheng1", "duration": 0.32,
             "note_pitch": 53, "note_type": 2},
            {"text": "涯", "phoneme": "zh_ya2", "duration": 1.73,
             "note_pitch": 51, "note_type": 2},
        ]
        sentence = "把它扔给你的烦恼"
        _process_section(tokens, sentence, False, 79)
        text_out = [t["text"] for t in tokens]
        assert text_out == ["把", "它", "扔", "给", "你", "的", "烦", "恼"]


# ===================================================================
# replace_lyrics — integration tests
# ===================================================================


class TestReplaceLyrics:
    def test_simple_replacement(self):
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.3 0.3 0.5",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY"))
        tokens = result[0]["text"].split(" ")
        assert tokens == ["<SP>", "X", "Y", "<SP>"]

    def test_collapse_right_align_integration(self):
        """Integration: collapse mode should right-align across full pipeline."""
        orig = json.dumps([_make_track(
            "<SP> A B C <SP>",
            "<SP> en_a en_b en_c <SP>",
            "0.5 0.3 0.3 1.0 0.5",
            "0 60 62 64 0",
            "1 2 2 2 1",
        )])
        # 2 chars → 3 slots, should right-align (skip first slot)
        result = json.loads(replace_lyrics(orig, "XY"))
        tokens = result[0]["text"].split(" ")
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        # The last non-SP token should be 'Y' with the long duration ~1.0
        non_sp = [(t, d) for t, d in zip(tokens, durations) if t != "<SP>"]
        assert non_sp[-1][0] == "Y"
        assert non_sp[-1][1] > 0.8  # long duration preserved

    def test_min_duration_enforcement(self):
        """Integration: tokens below 0.30s should be boosted by borrowing from longest."""
        orig = json.dumps([_make_track(
            "<SP> A B C D E F G <SP>",
            "<SP> en_a en_b en_c en_d en_e en_f en_g <SP>",
            "0.5 0.29 0.29 0.40 0.30 0.34 0.26 0.32 1.73 0.15",
            "0 52 53 46 53 53 54 53 51 0",
            "1 2 3 2 2 2 2 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "ABCDEFG"))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        # No non-SP token should be below 0.30s
        for t, d in zip(text, durations):
            if t != "<SP>":
                assert d >= 0.30, f"Token '{t}' has duration {d:.4f} < 0.30"

    def test_duration_preserved_total(self):
        """Total duration should be preserved after replacement."""
        orig_dur = "0.5 0.3 0.3 0.3 0.5"
        orig = json.dumps([_make_track(
            "<SP> A B C <SP>",
            "<SP> en_a en_b en_c <SP>",
            orig_dur,
            "0 60 62 64 0",
            "1 2 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XYZ"))
        result_dur = sum(float(x) for x in result[0]["duration"].split(" "))
        orig_total = sum(float(x) for x in orig_dur.split(" "))
        assert result_dur == pytest.approx(orig_total)

    def test_multi_sentence(self):
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            "0.5 0.3 0.3 0.5 0.3 0.3 0.3 0.5",
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY\nABC"))
        tokens = result[0]["text"].split(" ")
        # First section: XY, second section: ABC
        non_sp = [t for t in tokens if t != "<SP>"]
        assert non_sp[:2] == ["X", "Y"]
        assert non_sp[2:] == ["A", "B", "C"]


# ===================================================================
# extract_lyrics
# ===================================================================


class TestExtractLyrics:
    def test_basic_extraction(self):
        midi = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.3 0.3 0.5",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = extract_lyrics(midi)
        assert "A" in result
        assert "B" in result
        assert "<SP>" not in result

    def test_invalid_json(self):
        assert extract_lyrics("not json") == ""
        assert extract_lyrics("123") == ""

    def test_merge_repeated(self):
        midi = json.dumps([_make_track(
            "A A B <SP>",
            "en_a en_a en_b <SP>",
            "0.3 0.3 0.3 0.5",
            "60 60 62 0",
            "2 2 2 1",
        )])
        result = extract_lyrics(midi, merge_repeated=True)
        assert "AA" not in result
        assert "A" in result


# ===================================================================
# CT-Transformer copied utility functions
# ===================================================================


class TestCodeMixSplitWords:
    def test_chinese_chars(self):
        result = _code_mix_split_words("如果你觉得")
        assert result == ["如", "果", "你", "觉", "得"]

    def test_english_words(self):
        result = _code_mix_split_words("hello world")
        assert result == ["hello", "world"]

    def test_mixed(self):
        result = _code_mix_split_words("hello你好world")
        assert result == ["hello", "你", "好", "world"]


class TestSplitToMiniSentence:
    def test_short_no_split(self):
        words = list("ABCDEFGHIJ")  # 10 words
        result = _split_to_mini_sentence(words, 20)
        assert len(result) == 1
        assert result[0] == words

    def test_long_split(self):
        words = list("ABCDEFGHIJKLMNO")  # 15 words, limit 5
        result = _split_to_mini_sentence(words, 5)
        assert len(result) == 3
        assert result[0] == list("ABCDE")
        assert result[1] == list("FGHIJ")
        assert result[2] == list("KLMNO")

    def test_remainder(self):
        words = list("ABCDEFGH")  # 8 words, limit 5
        result = _split_to_mini_sentence(words, 5)
        assert len(result) == 2
        assert result[0] == list("ABCDE")
        assert result[1] == list("FGH")


class TestTokenIDConverter:
    def test_basic_conversion(self):
        tokens = ["A", "B", "C", "<UNK>"]
        conv = _TokenIDConverter(tokens)
        assert conv.tokens2ids(["A", "B", "C"]) == [0, 1, 2]
        assert conv.tokens2ids(["A", "X"]) == [0, 3]  # unknown -> UNK id
        assert conv.unk_id == 3
        assert conv.unk_symbol == "<UNK>"


# ===================================================================
# Smart split helpers
# ===================================================================


class TestSplitAtPunctuation:
    def test_comma_split(self):
        result = _split_at_punctuation("你好，世界")
        assert result == ["你好", "世界"]

    def test_period_split(self):
        result = _split_at_punctuation("春天到了。花儿开了")
        assert result == ["春天到了", "花儿开了"]

    def test_question_mark_split(self):
        result = _split_at_punctuation("你好吗？我很好")
        assert result == ["你好吗", "我很好"]

    def test_no_punctuation(self):
        result = _split_at_punctuation("你好世界")
        assert result == ["你好世界"]

    def test_single_char(self):
        result = _split_at_punctuation("你")
        assert result == ["你"]

    def test_empty(self):
        result = _split_at_punctuation("")
        assert result == [""]

    def test_trailing_punctuation_stripped(self):
        # Trailing punctuation with no text after it: punctuation is removed
        result = _split_at_punctuation("你好。")
        assert result == ["你好"]


# ===================================================================
# _smart_split_sentences — unit tests with mocked model
# ===================================================================


class TestSmartSplitSentences:
    """Test the smart split algorithm with mocked punctuation restoration."""

    @staticmethod
    def _make_midi_data(section_token_lists):
        """Build midi_data from a list of section token lists.

        Each section becomes a consecutive run of tokens; sections are
        separated by <SP>. All sections go into a single track.
        """
        parts = []
        for tokens in section_token_lists:
            parts.append("<SP>")
            parts.extend(tokens)
        parts.append("<SP>")
        text = " ".join(parts)
        n = len(parts)
        phoneme = " ".join("zh_x" for _ in range(n))
        duration = " ".join("0.3" for _ in range(n))
        pitch = " ".join("60" for _ in range(n))
        ntype = " ".join("1" for _ in range(n))
        return [{"text": text, "phoneme": phoneme,
                 "duration": duration, "note_pitch": pitch, "note_type": ntype}]

    def _patch_restore(self, sentences_map):
        """Monkey-patch _restore_punctuation to return predetermined results."""
        import nodes
        original_restore = nodes._restore_punctuation

        def mock_restore(text):
            return sentences_map.get(text, text)

        nodes._restore_punctuation = mock_restore
        return original_restore

    def _restore(self, original):
        import nodes
        nodes._restore_punctuation = original

    # --- Unit tests for helpers ---

    def test_compute_expected_simple(self):
        """Proportional allocation with last-section remainder."""
        # 2 sections, slot counts [3, 1], total 12 chars → 9 + 3
        result = _compute_expected_char_counts([3, 1], 12)
        assert result == [9, 3]
        assert sum(result) == 12

    def test_compute_expected_rounding(self):
        """Rounding with remainder goes to last section."""
        # 3 sections, slot counts [2, 2, 2], total 10 chars → 3 + 3 + 4
        result = _compute_expected_char_counts([2, 2, 2], 10)
        assert sum(result) == 10

    def test_compute_expected_single_section(self):
        """Single section gets all chars."""
        result = _compute_expected_char_counts([5], 20)
        assert result == [20]

    # --- Integration tests for _smart_split_sentences ---

    def test_split_proportional_hard_cut(self):
        """When AI can't punctuate, hard-cut at expected positions."""
        orig = self._patch_restore({
            "ABCDEFGHIJ": "ABCDEFGHIJ",  # no punctuation
        })
        try:
            # 2 sections: slot counts [3, 2] → 10 chars → expected [6, 4]
            # No AI punctuation → hard-cut at 6
            midi = self._make_midi_data([["A", "B", "C"], ["D", "E"]])
            result = _smart_split_sentences("ABCDEFGHIJ", midi)
            assert len(result) == 2
            assert result[0] == "ABCDEF"
            assert result[1] == "GHIJ"
        finally:
            self._restore(orig)

    def test_split_ai_cut_within_tolerance(self):
        """AI cut within ±30% tolerance → use AI result."""
        orig = self._patch_restore({
            # 36 chars, section slot counts [6,5,5,5] → expected [10,9,9,8]
            # First cut: AI puts comma at char 11 → within 30% of 10 (tolerance=3)
            "如果你觉得有点累送你个小炸弹把它扔给你的烦恼把烦恼都炸飞拉上你的老闺蜜呀":
                "如果你觉得有点累送你个小炸弹，把它扔给你的烦恼把烦恼都炸飞拉上你的老闺蜜呀",
            # Remaining 25 chars for 3 sections, expected [9,9,8]
            "把它扔给你的烦恼把烦恼都炸飞拉上你的老闺蜜呀":
                "把它扔给你的烦恼把烦恼都炸飞，拉上你的老闺蜜呀",
            # Remaining 12 chars for 2 sections, expected [9,8] → 12 chars, AI cut
            "把烦恼都炸飞拉上你的老闺蜜呀":
                "把烦恼都炸飞，拉上你的老闺蜜呀",
        })
        try:
            # slot counts: 没有什么能够阻挡=6 unique, 你对自由的向向往=6 unique(向×2→1),
            # 天天马行空的生涯=6 unique(天×2→1), 你的心了无牵挂=6 unique
            midi = self._make_midi_data([
                ["没", "有", "什", "么", "能", "够", "阻", "挡"],
                ["你", "对", "自", "由", "的", "向", "向", "往"],
                ["天", "天", "马", "行", "空", "的", "生", "涯"],
                ["你", "的", "心", "了", "无", "牵", "挂"],
            ])
            result = _smart_split_sentences(
                "如果你觉得有点累送你个小炸弹把它扔给你的烦恼把烦恼都炸飞拉上你的老闺蜜呀",
                midi,
            )
            assert len(result) == 4
            assert sum(len(s) for s in result) == 36
        finally:
            self._restore(orig)

    def test_last_section_gets_remainder(self):
        """Last section gets all remaining chars."""
        orig = self._patch_restore({
            "ABCDEFGH": "ABCD，EFGH",
        })
        try:
            # 2 sections, slot counts [1, 1] → 8 chars → expected [4, 4]
            # AI cut at 4 → within tolerance
            midi = self._make_midi_data([["A"], ["B"]])
            result = _smart_split_sentences("ABCDEFGH", midi)
            assert len(result) == 2
            assert result[0] == "ABCD"
            assert result[1] == "EFGH"
        finally:
            self._restore(orig)

    def test_empty_input(self):
        """Empty lyrics → all sections get empty strings."""
        midi = self._make_midi_data([["A", "B"], ["C"]])
        result = _smart_split_sentences("", midi)
        assert result == ["", ""]

    def test_first_punct_cut(self):
        """_first_punct_cut splits at first punctuation mark."""
        result = _first_punct_cut("如果你觉得，有点儿累", 5)
        assert result == ["如果你觉得", "有点儿累"]

    def test_first_punct_cut_no_punct(self):
        """_first_punct_cut returns None when no punctuation."""
        result = _first_punct_cut("你好世界", 3)
        assert result is None


# ===================================================================
# CT-Transformer inference tests (requires model download)
# ===================================================================


class TestCTTransformerPunc:
    """Tests for the actual CT-Transformer model inference.

    These tests require the model to be downloaded, so they are marked
    with a slow marker. Run with: pytest -v -m slow
    """

    @pytest.fixture(scope="class")
    def punc_model(self):
        """Download and load the CT-Transformer model."""
        import nodes
        model_dir = nodes._ensure_punc_model()
        return nodes._CTTransformerPunc(model_dir)

    @pytest.mark.slow
    def test_add_punctuation_basic(self, punc_model):
        """Model should add punctuation to Chinese text."""
        result = punc_model.add_punctuation("如果你觉得有点儿累")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain at least one punctuation mark
        has_punc = any(c in result for c in "，。？！、")
        assert has_punc, f"Expected punctuation in result: {result}"

    @pytest.mark.slow
    def test_add_punctuation_empty(self, punc_model):
        """Empty string should return empty."""
        result = punc_model.add_punctuation("")
        assert result == ""

    @pytest.mark.slow
    def test_add_punctuation_single_char(self, punc_model):
        """Single char should work without error."""
        result = punc_model.add_punctuation("你")
        assert isinstance(result, str)

    @pytest.mark.slow
    def test_add_punctuation_long_sentence(self, punc_model):
        """Long sentence should get multiple punctuation marks."""
        text = "今天天气真好我想出去走走看看花草树木感受阳光的温暖和微风的轻拂"
        result = punc_model.add_punctuation(text)
        assert isinstance(result, str)
        # Long sentence should have at least 2 punctuation marks
        punc_count = sum(1 for c in result if c in "，。？！、")
        assert punc_count >= 2, f"Expected >= 2 punctuation marks in: {result}"

    @pytest.mark.slow
    def test_add_punctuation_ends_with_period(self, punc_model):
        """Output should always end with 。 or ？"""
        result = punc_model.add_punctuation("春天来了花开了")
        assert result.endswith("。") or result.endswith("？")

    @pytest.mark.slow
    def test_token_id_converter_roundtrip(self, punc_model):
        """TokenIDConverter should handle unknown tokens gracefully."""
        conv = punc_model.converter
        ids = conv.tokens2ids(["你", "好", "UNKNOWNWORD123"])
        assert len(ids) == 3
        assert isinstance(ids[0], int)
        assert isinstance(ids[1], int)
        # Unknown token should map to unk_id
        assert ids[2] == conv.unk_id


# ===================================================================
# Flexible pause mode tests
# ===================================================================


class TestFlexiblePause:
    """Tests for fixed_pause=False (flexible pause) mode."""

    def test_fixed_pause_default_unchanged(self):
        """With fixed_pause=True (default), SP duration should not change."""
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "2.0 0.3 0.3 2.0",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY", fixed_pause=True))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        # SP durations should remain unchanged
        assert durations[0] == pytest.approx(2.0)
        assert durations[-1] == pytest.approx(2.0)

    def test_flexible_pause_redistributes_long_sp(self):
        """When SP >= 2x avg token dur, flexible mode should redistribute."""
        # SP=2.0, tokens=0.3+0.3=0.6, avg=0.3, sp_ratio=2.0/0.3=6.67 > 2
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "2.0 0.3 0.3 2.0",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        # Total must be preserved
        assert sum(durations) == pytest.approx(2.0 + 0.3 + 0.3 + 2.0)
        # First SP should be reduced
        assert durations[0] < 2.0
        # Tokens should be longer than original
        tok_durs = [d for t, d in zip(text, durations) if t != "<SP>"]
        assert all(d > 0.3 for d in tok_durs)

    def test_flexible_pause_triggered_by_short_tokens(self):
        """When avg token < 0.30s, flexible mode should trigger regardless of SP ratio."""
        # SP=0.4, tokens=0.15+0.15=0.3, avg=0.15 < 0.30
        # sp_ratio=0.4/0.15=2.67 > 2 actually, but the avg_tok < 0.30 should also trigger
        orig = json.dumps([_make_track(
            "<SP> A B C D E F G <SP>",
            "<SP> en_a en_b en_c en_d en_e en_f en_g <SP>",
            "0.5 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.50",
            "0 52 53 46 53 53 54 53 0",
            "1 2 3 2 2 2 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "ABCDEFG", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        tok_durs = [d for t, d in zip(text, durations) if t != "<SP>"]
        # Tokens should be boosted above original 0.20
        assert all(d > 0.20 for d in tok_durs)
        # Total preserved
        orig_total = 0.5 + 7 * 0.20 + 0.5
        assert sum(durations) == pytest.approx(orig_total)

    def test_flexible_pause_no_trigger_when_balanced(self):
        """When SP ≈ avg token duration, no redistribution should happen."""
        # SP=0.5, tokens=0.5+0.5=1.0, avg=0.5, sp_ratio=1.0 < 2, avg>=0.30
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.5 0.5 0.5",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        # Should remain unchanged — no trigger condition met
        assert durations == [pytest.approx(0.5)] * 4

    def test_flexible_pause_proportional_distribution(self):
        """Redistributed time should be proportional to existing durations.
        
        Note: min-duration enforcement (0.30s) runs BEFORE flexible pause,
        so tokens below 0.30s get boosted first. We use tokens above 0.30s
        to test pure proportional distribution.
        """
        # SP=3.0, tokens=0.6+1.2=1.8, avg=0.9, sp_ratio=3.0/0.9=3.33 > 2
        # freed = 3.0 - 0.9 = 2.1
        # Token A gets: 0.6 + 2.1*(0.6/1.8) = 0.6 + 0.7 = 1.3
        # Token B gets: 1.2 + 2.1*(1.2/1.8) = 1.2 + 1.4 = 2.6
        # Ratio should stay: B/A = 2:1
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "3.0 0.6 1.2 2.0",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        tok_durs = [d for t, d in zip(text, durations) if t != "<SP>"]
        # B should still be ~2x A
        assert tok_durs[1] / tok_durs[0] == pytest.approx(2.0, abs=0.05)
        # Total preserved
        assert sum(durations) == pytest.approx(3.0 + 0.6 + 1.2 + 2.0)

    def test_flexible_pause_total_group_preserved(self):
        """Group (section + trailing SP) total duration must be preserved."""
        # Section tokens + trailing SP total = 0.3+0.3+2.0 = 2.6
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D <SP>",
            "<SP> en_a en_b <SP> en_c en_d <SP>",
            "0.5 0.3 0.3 2.0 0.4 0.4 1.5",
            "0 60 62 0 64 65 0",
            "1 2 2 1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY\nCD", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        # Group 1: SP + A + B + SP = 0.5+0.3+0.3+2.0 = 3.1
        # Group 2: SP + C + D + SP — but first SP of group2 is actually emitted
        #   as the trailing SP of group 1
        # Total should be preserved
        orig_total = 0.5 + 0.3 + 0.3 + 2.0 + 0.4 + 0.4 + 1.5
        assert sum(durations) == pytest.approx(orig_total)

    def test_flexible_pause_multi_section(self):
        """Each section's SP should be independently evaluated."""
        # Section 1: SP=2.0, tokens avg=0.3 → triggers
        # Section 2: SP=0.5, tokens avg=0.5 → no trigger
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D <SP>",
            "<SP> en_a en_b <SP> en_c en_d <SP>",
            "2.0 0.3 0.3 0.5 0.5 0.5 0.5",
            "0 60 62 0 64 65 0",
            "1 2 2 1 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "XY\nCD", fixed_pause=False))
        durations = [float(x) for x in result[0]["duration"].split(" ")]
        text = result[0]["text"].split(" ")
        non_sp = [(t, d) for t, d in zip(text, durations) if t != "<SP>"]
        sp_durs = [d for t, d in zip(text, durations) if t == "<SP>"]
        # First SP should be reduced (was 2.0)
        assert sp_durs[0] < 2.0
        # Section 1 tokens should be longer than original 0.3
        assert non_sp[0][1] > 0.3
        assert non_sp[1][1] > 0.3
        # Second SP should remain 0.5 (no trigger: ratio=1.0 < 2, avg=0.5 >= 0.30)
        assert sp_durs[1] == pytest.approx(0.5)
        # Total preserved
        orig_total = 2.0 + 0.3 + 0.3 + 0.5 + 0.5 + 0.5 + 0.5
        assert sum(durations) == pytest.approx(orig_total)


# ===================================================================
# Duration-based split mode tests
# ===================================================================


class TestGetSectionDurations:
    """Tests for _get_section_durations helper."""

    def test_single_section(self):
        midi = [{"text": "<SP> A B <SP>", "duration": "0.5 0.3 0.4 0.5"}]
        durs = _get_section_durations(midi)
        assert durs == [pytest.approx(0.7)]

    def test_multiple_sections(self):
        midi = [{"text": "<SP> A B <SP> C D E <SP>",
                 "duration": "0.5 0.3 0.4 0.5 0.6 0.2 0.3 0.5"}]
        durs = _get_section_durations(midi)
        assert durs == [pytest.approx(0.7), pytest.approx(1.1)]

    def test_multi_track(self):
        midi = [
            {"text": "<SP> A <SP>", "duration": "0.5 0.3 0.5"},
            {"text": "<SP> B C <SP>", "duration": "0.5 0.2 0.4 0.5"},
        ]
        durs = _get_section_durations(midi)
        assert durs == [pytest.approx(0.3), pytest.approx(0.6)]

    def test_no_track_text(self):
        midi = [{"other": "data"}]
        durs = _get_section_durations(midi)
        assert durs == []


class TestDurationBasedSplit:
    """Tests for split_mode='duration' in replace_lyrics."""

    def test_duration_split_more_chars_for_longer_sections(self):
        """Duration mode should give more chars to longer-timed sections."""
        # Section 1: 2 tokens, dur=0.3+0.3=0.6
        # Section 2: 3 tokens, dur=1.0+1.0+1.0=3.0
        # Total dur = 3.6, section 2 is ~83% of total
        # With 10 chars, duration mode: sec1≈2, sec2≈8
        # Token mode: sec1=2/5*10=4, sec2=3/5*10=6
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            "0.5 0.3 0.3 0.5 1.0 1.0 1.0 0.5",
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )])
        result_tok = json.loads(replace_lyrics(orig, "一二三四五六七八九十",
                                                split_mode="token"))
        result_dur = json.loads(replace_lyrics(orig, "一二三四五六七八九十",
                                                split_mode="duration"))
        tok_text = result_tok[0]["text"].split(" ")
        dur_text = result_dur[0]["text"].split(" ")

        # Count non-SP tokens in first section (between 1st and 2nd SP)
        tok_sps = [i for i, t in enumerate(tok_text) if t == "<SP>"]
        dur_sps = [i for i, t in enumerate(dur_text) if t == "<SP>"]

        sec1_tok = [t for t in tok_text[tok_sps[0]+1:tok_sps[1]]
                     if t and t != "<SP>"]
        sec1_dur = [t for t in dur_text[dur_sps[0]+1:dur_sps[1]]
                     if t and t != "<SP>"]

        # Duration mode should allocate fewer chars to section 1
        # (which has short duration) compared to token mode
        assert len(sec1_dur) < len(sec1_tok)
        assert len(sec1_dur) == 2  # 0.6/3.6 * 10 ≈ 2
        assert len(sec1_tok) == 4  # 2/5 * 10 = 4

    def test_duration_split_total_preserved(self):
        """Total duration must be preserved with duration split mode."""
        orig_dur = "0.5 0.3 0.3 0.5 1.0 1.0 1.0 0.5"
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            orig_dur,
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )])
        result = json.loads(replace_lyrics(orig, "一二三四五六七八九十",
                                            split_mode="duration"))
        result_dur = sum(float(x) for x in result[0]["duration"].split(" "))
        orig_total = sum(float(x) for x in orig_dur.split(" "))
        assert result_dur == pytest.approx(orig_total)

    def test_token_mode_unchanged(self):
        """Token mode (default) should produce same results as before."""
        orig = json.dumps([_make_track(
            "<SP> A B <SP> C D E <SP>",
            "<SP> en_a en_b <SP> en_c en_d en_e <SP>",
            "0.5 0.3 0.3 0.5 0.3 0.3 0.3 0.5",
            "0 60 62 0 64 65 67 0",
            "1 2 2 1 2 2 2 1",
        )])
        result_default = json.loads(replace_lyrics(orig, "XY\nCDE"))
        result_token = json.loads(replace_lyrics(orig, "XY\nCDE",
                                                  split_mode="token"))
        assert result_default[0]["text"] == result_token[0]["text"]


# ===================================================================
# Speed adjustment tests
# ===================================================================


class TestFmtDur:
    """Tests for _fmt_dur helper."""

    def test_clean_float(self):
        assert _fmt_dur(0.3) == "0.3"
        assert _fmt_dur(0.30000000000000004) == "0.3"

    def test_precision(self):
        assert _fmt_dur(0.12345) == "0.1235"

    def test_zero(self):
        assert _fmt_dur(0.0) == "0"


class TestFmtF0:
    """Tests for _fmt_f0 helper."""

    def test_zero(self):
        assert _fmt_f0(0.0) == "0.0"

    def test_normal_value(self):
        assert _fmt_f0(267.2) == "267.2"

    def test_float_artifact(self):
        assert _fmt_f0(267.1999999) == "267.2"


class TestApplySpeed:
    """Tests for _apply_speed function."""

    def test_speed_1_no_change(self):
        """Speed 1.0 should not modify anything."""
        midi = [{"duration": "0.3 0.5 0.2", "f0": "0.0 267.2 0.0"}]
        result = _apply_speed(midi, 1.0)
        assert result[0]["duration"] == "0.3 0.5 0.2"
        assert result[0]["f0"] == "0.0 267.2 0.0"

    def test_duration_scaled(self):
        """Speed 2.0 (faster) → durations halved."""
        midi = [{"duration": "0.3 0.5 0.2", "f0": "0.0 267.2 0.0"}]
        result = _apply_speed(midi, 2.0)
        durs = [float(x) for x in result[0]["duration"].split(" ")]
        assert durs == [pytest.approx(0.15), pytest.approx(0.25), pytest.approx(0.1)]

    def test_duration_slower(self):
        """Speed 0.5 (slower) → durations doubled."""
        midi = [{"duration": "0.4 0.6", "f0": "0.0 267.2"}]
        result = _apply_speed(midi, 0.5)
        durs = [float(x) for x in result[0]["duration"].split(" ")]
        assert durs == [pytest.approx(0.8), pytest.approx(1.2)]

    def test_f0_stretch(self):
        """Speed 0.5 (slower) → f0 gets more frames (stretched)."""
        # 4 f0 frames → speed 0.5 → 8 frames (linear interpolation)
        f0_in = "100.0 200.0 300.0 400.0"
        midi = [{"duration": "1.0", "f0": f0_in}]
        result = _apply_speed(midi, 0.5)
        f0_out = [float(x) for x in result[0]["f0"].split(" ")]
        assert len(f0_out) == 8
        # First and last should be preserved
        assert f0_out[0] == pytest.approx(100.0)
        assert f0_out[-1] == pytest.approx(400.0)
        # Middle should be interpolated
        assert f0_out[3] == pytest.approx(228.6, abs=1.0)

    def test_f0_shrink(self):
        """Speed 2.0 (faster) → f0 gets fewer frames (shrunk)."""
        f0_in = "100.0 150.0 200.0 250.0 300.0 350.0 400.0 450.0"
        midi = [{"duration": "1.0", "f0": f0_in}]
        result = _apply_speed(midi, 2.0)
        f0_out = [float(x) for x in result[0]["f0"].split(" ")]
        assert len(f0_out) == 4
        assert f0_out[0] == pytest.approx(100.0)
        assert f0_out[-1] == pytest.approx(450.0)

    def test_f0_zeros_preserved(self):
        """Zero f0 values (silence) should stay zero after resampling."""
        f0_in = "0.0 0.0 0.0 0.0 267.2 267.2 0.0 0.0"
        midi = [{"duration": "1.0", "f0": f0_in}]
        result = _apply_speed(midi, 0.5)
        f0_out = [float(x) for x in result[0]["f0"].split(" ")]
        assert len(f0_out) == 16
        # Early frames (silence) should be ~0
        assert f0_out[0] == pytest.approx(0.0, abs=0.5)
        # Late frames (silence) should be ~0
        assert f0_out[-1] == pytest.approx(0.0, abs=0.5)

    def test_f0_empty_string(self):
        """Empty f0 should not crash."""
        midi = [{"duration": "0.5", "f0": ""}]
        result = _apply_speed(midi, 1.5)
        assert result[0]["f0"] == ""

    def test_no_f0_field(self):
        """Track without f0 field should not crash."""
        midi = [{"duration": "0.5"}]
        result = _apply_speed(midi, 1.5)
        assert "f0" not in result[0]
        durs = [float(x) for x in result[0]["duration"].split(" ")]
        # speed 1.5 → duration / 1.5 = 0.3333
        assert durs[0] == pytest.approx(0.3333, abs=0.001)

    def test_multi_track(self):
        """Multiple tracks should all be processed."""
        midi = [
            {"duration": "0.5", "f0": "100.0 200.0"},
            {"duration": "1.0", "f0": "300.0 400.0 500.0 600.0"},
        ]
        result = _apply_speed(midi, 2.0)
        # speed 2.0 → durations halved, f0 halved
        assert float(result[0]["duration"]) == pytest.approx(0.25)
        assert len(result[0]["f0"].split(" ")) == 1
        assert float(result[1]["duration"]) == pytest.approx(0.5)
        assert len(result[1]["f0"].split(" ")) == 2

    def test_f0_frame_ratio_preserved(self):
        """Frame count should be orig / speed."""
        # 100 frames at speed 2.0 → 50 frames (faster = fewer frames)
        f0_in = " ".join(str(float(i)) for i in range(100))
        midi = [{"duration": "2.0", "f0": f0_in}]
        result = _apply_speed(midi, 2.0)
        f0_out = result[0]["f0"].split(" ")
        assert len(f0_out) == 50

    def test_time_field_scaled(self):
        """time field [start, end] should shrink when speed > 1 (faster)."""
        midi = [{"duration": "0.5", "f0": "100.0 200.0", "time": [0, 1000]}]
        result = _apply_speed(midi, 2.0)
        assert result[0]["time"] == [0, 500]

    def test_time_field_slower(self):
        """time field should grow when speed < 1 (slower)."""
        midi = [{"duration": "0.5", "f0": "100.0 200.0", "time": [0, 15000]}]
        result = _apply_speed(midi, 0.5)
        assert result[0]["time"] == [0, 30000]

    def test_time_field_missing(self):
        """Missing time field should not crash."""
        midi = [{"duration": "0.5", "f0": "100.0"}]
        result = _apply_speed(midi, 2.0)
        assert "time" not in result[0]

    def test_time_field_not_list(self):
        """time field that is not a list should be left alone."""
        midi = [{"duration": "0.5", "f0": "100.0", "time": "0,1000"}]
        result = _apply_speed(midi, 2.0)
        assert result[0]["time"] == "0,1000"


class TestSpeedIntegration:
    """Integration tests for speed in replace_lyrics."""

    def test_speed_via_replace_lyrics(self):
        """replace_lyrics with speed=2.0 (faster) should halve durations and f0 frames."""
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.3 0.3 0.5",
            "0 60 62 0",
            "1 2 2 1",
            "0.0 0.0 267.2 267.2 0.0 0.0 0.0 0.0",
        )])
        result = json.loads(replace_lyrics(orig, "XY", speed=2.0))
        durs = [float(x) for x in result[0]["duration"].split(" ")]
        # Total duration should halve (speed up = shorter)
        orig_total = 0.5 + 0.3 + 0.3 + 0.5
        new_total = sum(durs)
        assert new_total == pytest.approx(orig_total / 2.0)
        # f0 should have fewer frames
        orig_f0_count = len("0.0 0.0 267.2 267.2 0.0 0.0 0.0 0.0".split(" "))
        new_f0_count = len(result[0]["f0"].split(" "))
        assert new_f0_count == round(orig_f0_count / 2.0)

    def test_speed_default_unchanged(self):
        """Default speed=1.0 should not affect output."""
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.3 0.3 0.5",
            "0 60 62 0",
            "1 2 2 1",
        )])
        result = replace_lyrics(orig, "XY")
        result_default = replace_lyrics(orig, "XY", speed=1.0)
        assert result == result_default

    def test_speed_slower_total_duration(self):
        """Speed 0.5 (slower) should double total duration."""
        orig = json.dumps([_make_track(
            "<SP> A B <SP>",
            "<SP> en_a en_b <SP>",
            "0.5 0.3 0.3 0.5",
            "0 60 62 0",
            "1 2 2 1",
            "0.0 0.0 267.2 267.2 0.0 0.0 0.0 0.0",
        )])
        result = json.loads(replace_lyrics(orig, "XY", speed=0.5))
        durs = [float(x) for x in result[0]["duration"].split(" ")]
        orig_total = 0.5 + 0.3 + 0.3 + 0.5
        new_total = sum(durs)
        assert new_total == pytest.approx(orig_total / 0.5)
        f0_count = len(result[0]["f0"].split(" "))
        assert f0_count == 16  # 8 / 0.5 = 16


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
        # _split_at_punctuation doesn't strip trailing — it finds first split point
        result = _split_at_punctuation("你好。")
        assert result == ["你好。"]  # no second part after period


# ===================================================================
# _smart_split_sentences — unit tests with mocked model
# ===================================================================


class TestSmartSplitSentences:
    """Test the smart split algorithm with mocked punctuation restoration."""

    def _patch_restore(self, sentences_map):
        """Monkey-patch _restore_punctuation to return predetermined results.

        sentences_map: dict mapping sentence text -> punctuated text.
        """
        import nodes
        original_restore = nodes._restore_punctuation

        def mock_restore(text):
            return sentences_map.get(text, text)

        nodes._restore_punctuation = mock_restore
        return original_restore

    def _restore(self, original):
        import nodes
        nodes._restore_punctuation = original

    def test_exact_match_no_split(self):
        """When sentence count matches target, return as-is."""
        result = _smart_split_sentences(["ABC", "DEF"], 2)
        assert result == ["ABC", "DEF"]

    def test_more_than_target_no_split(self):
        """When sentence count exceeds target, return as-is."""
        result = _smart_split_sentences(["ABC", "DEF", "GHI"], 2)
        assert result == ["ABC", "DEF", "GHI"]

    def test_split_one_sentence_to_two(self):
        """One sentence needs to become two via AI punctuation."""
        orig = self._patch_restore({
            "如果你觉得有点儿累": "如果你觉得，有点儿累",
        })
        try:
            result = _smart_split_sentences(["如果你觉得有点儿累", "短句"], 3)
            assert len(result) == 3
            assert "如果你觉得" in result
            assert "有点儿累" in result
            assert "短句" in result
        finally:
            self._restore(orig)

    def test_iterative_split_longest_first(self):
        """Should iteratively split the longest sentence until count matches."""
        orig = self._patch_restore({
            "一二三四五六七八": "一二三四，五六七八",
            "一二三四": "一二，三四",
            "五六七八": "五六，七八",
        })
        try:
            result = _smart_split_sentences(
                ["一二三四五六七八", "短"], 4
            )
            assert len(result) == 4
        finally:
            self._restore(orig)

    def test_unsplitable_fallback_to_next(self):
        """If a sentence can't be split, try the next longest."""
        orig = self._patch_restore({
            "AB": "AB",  # model returns no punctuation
            "CDEFGHIJKL": "CDEFG，HIJKL",
        })
        try:
            result = _smart_split_sentences(["AB", "CDEFGHIJKL"], 3)
            assert len(result) == 3
            # AB should remain unsplit, CDEFGHIJKL should be split
            assert "AB" in result
        finally:
            self._restore(orig)

    def test_all_unsplitable_hard_split(self):
        """If all sentences are unsplitable by AI, hard-split the longest."""
        orig = self._patch_restore({
            "ABCDEF": "ABCDEF",  # no punctuation
            "GH": "GH",
        })
        try:
            result = _smart_split_sentences(["ABCDEF", "GH"], 3)
            assert len(result) == 3
            # ABCDEF should be hard-split into two parts
            abc_parts = [s for s in result if any(c in s for c in "ABCDEF")]
            assert len(abc_parts) == 2
        finally:
            self._restore(orig)

    def test_single_char_padding(self):
        """Single-char sentences that can't be split → pad with empty strings."""
        orig = self._patch_restore({
            "A": "A",  # can't split single char
        })
        try:
            result = _smart_split_sentences(["A"], 3)
            assert len(result) == 3
            assert result.count("") >= 1  # padded with empties
        finally:
            self._restore(orig)


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

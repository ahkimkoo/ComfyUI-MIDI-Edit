# tests/test_alignment.py
"""MidiLyricsAlignment 测试套件."""
import pytest
from alignment.models import Token, Unit, AlignmentOp, AlignmentPath, CostWeights, Track


class TestModels:
    def test_token_is_sp_true(self):
        t = Token("<SP>", "<SP>", 0.3, 0, 1, 0)
        assert t.is_sp is True

    def test_token_is_sp_false(self):
        t = Token("你", "zh_ni3", 0.4, 60, 2, 1)
        assert t.is_sp is False

    def test_unit_zh_defaults(self):
        u = Unit("你", "zh_ni3", "zh", 1)
        assert u.source == "lyric"

    def test_cost_weights_defaults(self):
        w = CostWeights()
        assert w.w_pitch == 0.5
        assert w.min_duration == 0.30
        assert w.max_word_occupy == 4

    def test_alignment_op_frozen(self):
        op = AlignmentOp("REPLACE", None, (0,), 0.0)
        with pytest.raises(Exception):
            op.kind = "DROP"  # frozen


import json
from alignment.parser import parse_tracks, serialize_track, serialize_tracks


class TestParser:
    TRACK_JSON = json.dumps([{
        "index": "vocal_0_15000",
        "language": "Mandarin",
        "time": [0, 15000],
        "text": "<SP> 你 好 <SP>",
        "phoneme": "<SP> zh_ni3 zh_hao3 <SP>",
        "duration": "0.30 0.40 0.40 0.30",
        "note_pitch": "0 60 62 0",
        "note_type": "1 2 2 1",
        "f0": "0.0 0.0 261.6 0.0",
    }])

    def test_parse_single_track(self):
        tracks = parse_tracks(self.TRACK_JSON)
        assert len(tracks) == 1
        t = tracks[0]
        assert len(t.tokens) == 4
        assert t.tokens[0].is_sp
        assert t.tokens[1].text == "你"
        assert t.tokens[1].phoneme == "zh_ni3"
        assert t.tokens[1].duration == 0.40
        assert t.tokens[1].note_pitch == 60
        assert t.meta["language"] == "Mandarin"
        assert t.f0 == "0.0 0.0 261.6 0.0"

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_tracks("")

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_tracks("{not json")

    def test_parse_missing_field_raises(self):
        bad = json.dumps([{"text": "你"}])  # 缺其他字段
        with pytest.raises(ValueError, match="missing field"):
            parse_tracks(bad)

    def test_serialize_roundtrip(self):
        tracks = parse_tracks(self.TRACK_JSON)
        s = serialize_tracks(tracks)
        again = parse_tracks(s)
        assert len(again[0].tokens) == 4
        assert again[0].tokens[1].text == "你"


from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)


class TestCost:
    def setup_method(self):
        self.w = CostWeights()
        self.token = Token("你", "zh_ni3", 0.4, 60, 2, 1)
        self.unit_zh = Unit("好", "zh_hao3", "zh", 1)
        self.unit_en = Unit("love", "en_L-AH1-V", "en", 3)
        self.unit_sp = Unit("<SP>", "<SP>", "sp", 1, "punct")

    def test_replace_cost_zero(self):
        assert replace_cost(self.token, self.unit_zh, self.w) == 0.0

    def test_word_span_balanced_low_cost(self):
        span = [self.token, self.token, self.token]  # k=3 = ideal
        c = word_span_cost(span, self.unit_en, self.w)
        assert c == 0.0  # k==ideal → imbalance=0

    def test_word_span_imbalanced(self):
        span = [self.token]  # k=1, ideal=3
        c = word_span_cost(span, self.unit_en, self.w)
        assert c > 0.0

    def test_split_cost_below_min_duration(self):
        # host.duration=0.4, 共享后 est=0.2 < 0.30 → 惩罚
        c = split_cost(self.token, self.unit_zh, self.w, current_share_count=0)
        assert c > 0.0

    def test_split_cost_above_min_duration(self):
        long_token = Token("啊", "zh_a1", 1.0, 60, 2, 0)
        c = split_cost(long_token, self.unit_zh, self.w, current_share_count=0)
        assert c == 0.0  # 1.0/2=0.5 > 0.30

    def test_drop_cost_pitch_loss(self):
        tokens = [
            Token("<SP>", "<SP>", 0.3, 0, 1, 0),
            self.token,  # idx=1, pitch=60
            Token("好", "zh_hao3", 0.4, 62, 2, 2),  # idx=2
        ]
        c = drop_cost(tokens[1], tokens, 1, self.w)
        # nearest = idx=2 pitch=62, loss = |60-62| = 2
        assert c == self.w.w_pitch * 2

    def test_sp_align_at_orig_position_zero_structure(self):
        sp_token = Token("<SP>", "<SP>", 0.3, 0, 1, 5)
        c = sp_align_cost(sp_token, self.unit_sp, 5, [5], self.w)
        # min_dist=0, is_sp → P=0
        assert c == 0.0

    def test_sp_align_moved(self):
        lyric_token = Token("你", "zh_ni3", 0.4, 60, 2, 3)
        c = sp_align_cost(lyric_token, self.unit_sp, 3, [7], self.w)
        # min_dist = |3-7| = 4, P = 60
        assert c == self.w.w_structure * 4 + self.w.w_pitch * 60


from alignment.preprocess import normalize_lyrics


class TestNormalizer:
    def test_basic_chinese(self):
        text, sp = normalize_lyrics("你好世界", sp_target=1)
        assert text == "你好世界"
        assert len(sp) == 1

    def test_strong_punct_used_first(self):
        text, sp = normalize_lyrics("你好。世界！", sp_target=2)
        assert len(sp) == 2

    def test_median_punct_fallback(self):
        text, sp = normalize_lyrics("你好，世界", sp_target=2)
        assert len(sp) == 2

    def test_digit_normalization(self):
        text, sp = normalize_lyrics("123", sp_target=0, normalize_digits=True)
        assert text == "一二三"

    def test_digit_normalization_disabled(self):
        text, sp = normalize_lyrics("123", sp_target=0, normalize_digits=False)
        assert "1" in text

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_lyrics("", sp_target=0)

    def test_delete_quotes(self):
        text, sp = normalize_lyrics('"你好"世界', sp_target=0)
        assert '"' not in text
        assert text == "你好世界"

    def test_uniform_fill_distribution(self):
        text, sp = normalize_lyrics("一二三四五六七八九十", sp_target=3)
        assert len(sp) == 3
        diffs = [sp[i+1] - sp[i] for i in range(len(sp)-1)]
        assert max(diffs) - min(diffs) <= 2


from alignment.preprocess import tokenize_units


class TestTokenizer:
    def setup_method(self):
        self.w = CostWeights()

    def test_pure_chinese(self):
        units = tokenize_units("你好", [], self.w)
        assert len(units) == 2
        assert all(u.kind == "zh" for u in units)
        assert units[0].text == "你"
        assert units[0].phoneme == "zh_ni3"
        assert units[0].max_occupy == 1

    def test_english_word(self):
        units = tokenize_units("love", [], self.w)
        assert len(units) == 1
        assert units[0].kind == "en"
        assert units[0].text == "love"
        assert units[0].phoneme.startswith("en_")
        assert units[0].max_occupy == min(4, 4)

    def test_mixed(self):
        units = tokenize_units("你love好", [], self.w)
        assert len(units) == 3
        assert units[0].kind == "zh"
        assert units[1].kind == "en"
        assert units[2].kind == "zh"

    def test_sp_insertion(self):
        units = tokenize_units("你好", [1], self.w)
        assert len(units) == 3
        assert units[0].text == "你"
        assert units[1].kind == "sp"
        assert units[2].text == "好"

    def test_long_english_word_max_occupy_capped(self):
        units = tokenize_units("extraordinarily", [], self.w)
        assert units[0].max_occupy == 4  # capped

    def test_spaces_ignored(self):
        units = tokenize_units("hello world", [], self.w)
        assert len(units) == 2
        assert units[0].text == "hello"
        assert units[1].text == "world"

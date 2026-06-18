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


from alignment.dp import solve_alignment


def _make_tokens(specs):
    """快捷构造 token 列表。specs = [(text, pitch, type, dur), ...]"""
    return [Token(t, "<SP>" if t == "<SP>" else f"ph_{t}",
                  d, p, nt, i) for i, (t, p, nt, d) in enumerate(specs)]


class TestDP:
    def setup_method(self):
        self.w = CostWeights()

    def test_perfect_match_zero_cost(self):
        """字数等长 + SP 对齐 → 全 REPLACE，代价 0."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("呀", "zh_ya1", "zh", 1),
            Unit("哎", "zh_ai1", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        assert path.total_cost == 0.0
        assert len(path.ops) == 4
        assert all(o.kind == "REPLACE" or o.kind == "SP_ALIGN"
                   for o in path.ops)

    def test_split_triggered_when_more_units(self):
        """字数多于 token → SPLIT 触发."""
        tokens = _make_tokens([("<SP>", 0, 1, 0.3), ("啊", 60, 2, 1.0), ("<SP>", 0, 1, 0.3)])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("天", "zh_tian1", "zh", 1),
            Unit("气", "zh_qi4", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "SPLIT" in kinds

    def test_drop_triggered_when_fewer_units(self):
        """字数少于 token → DROP 触发."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("哎", "zh_ai1", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "DROP" in kinds

    def test_sp_count_conserved(self):
        """SP_ALIGN 次数 = SP 单元数."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("呀", "zh_ya1", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        assert len(path.sp_placements) == 2

    def test_word_span_for_english(self):
        """英文词占多 token → WORD_SPAN."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3),
            ("la", 60, 2, 0.3), ("la", 62, 2, 0.3), ("la", 64, 2, 0.3),
            ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("love", "en_L-AH1-V", "en", 3),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "WORD_SPAN" in kinds


from alignment.rebuild import rebuild_tokens, _find_sections


class TestRebuilder:
    def setup_method(self):
        self.w = CostWeights()
        self.tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])

    def test_replace_keeps_pitch_type(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        ops = [
            AlignmentOp("SP_ALIGN", Unit("<SP>", "<SP>", "sp", 1), (0,), 0.0),
            AlignmentOp("REPLACE", Unit("呀", "zh_ya1", "zh", 1), (1,), 0.0),
            AlignmentOp("REPLACE", Unit("哎", "zh_ai1", "zh", 1), (2,), 0.0),
            AlignmentOp("SP_ALIGN", Unit("<SP>", "<SP>", "sp", 1), (3,), 0.0),
        ]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, self.tokens, self.w)
        assert len(result) == 4
        assert result[0].text == "<SP>"
        assert result[1].text == "呀"
        assert result[1].note_pitch == 60
        assert result[1].note_type == 2

    def test_word_span_sets_note_type(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        tokens = _make_tokens([
            ("la", 60, 2, 0.3), ("la", 62, 2, 0.3), ("la", 64, 2, 0.3),
        ])
        ops = [AlignmentOp("WORD_SPAN", Unit("love", "en_L-AH1-V", "en", 3), (0, 1, 2), 0.0)]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, tokens, self.w)
        assert len(result) == 3
        assert all(r.text == "love" for r in result)
        assert result[0].note_type == 2
        assert result[1].note_type == 3
        assert result[2].note_type == 3

    def test_drop_produces_nothing(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        ops = [AlignmentOp("DROP", None, (1,), 0.0)]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, self.tokens, self.w)
        assert len(result) == 0

    def test_find_sections(self):
        sections = _find_sections(self.tokens)
        assert sections == [(1, 3)]


from alignment.rebuild import allocate_durations
from alignment.models import AlignmentOp, AlignmentPath, Unit


class TestDurationAllocator:
    def setup_method(self):
        self.w = CostWeights()

    def test_total_duration_conserved(self):
        orig = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.5), ("<SP>", 0, 1, 0.3),
        ])
        new = [
            Token("<SP>", "<SP>", 0.3, 0, 1, 0),
            Token("呀", "zh_ya1", 0.5, 60, 2, 1),
            Token("<SP>", "<SP>", 0.3, 0, 1, 2),
        ]
        path = AlignmentPath(ops=[
            AlignmentOp("SP_ALIGN", None, (0,), 0.0),
            AlignmentOp("REPLACE", None, (1,), 0.0),
            AlignmentOp("SP_ALIGN", None, (2,), 0.0),
        ], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        orig_sum = sum(t.duration for t in orig)
        new_sum = sum(t.duration for t in result)
        assert abs(orig_sum - new_sum) < 0.01

    def test_min_duration_enforced(self):
        orig = _make_tokens([("长", 60, 2, 1.0), ("短", 62, 2, 0.1)])
        new = [
            Token("长", "zh_chang2", 1.0, 60, 2, 0),
            Token("短", "zh_duan3", 0.1, 62, 2, 1),
        ]
        path = AlignmentPath(ops=[
            AlignmentOp("REPLACE", None, (0,), 0.0),
            AlignmentOp("REPLACE", None, (1,), 0.0),
        ], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        non_sp = [t for t in result if not t.is_sp]
        assert all(t.duration >= 0.30 - 0.01 for t in non_sp)

    def test_float_cleanup(self):
        orig = _make_tokens([("你", 60, 2, 0.333333)])
        new = [Token("呀", "zh_ya1", 0.333333, 60, 2, 0)]
        path = AlignmentPath(ops=[AlignmentOp("REPLACE", None, (0,), 0.0)], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        assert result[0].duration == 0.33

    def test_split_shares_host_duration(self):
        """SPLIT 场景：2 个 unit 共享 1 个 host token，duration 均分."""
        # host token "啊" 时长 1.0s，被 2 个字共享（REPLACE + 1 SPLIT）
        orig = _make_tokens([("<SP>", 0, 1, 0.3), ("啊", 60, 2, 1.0), ("<SP>", 0, 1, 0.3)])
        # rebuild_tokens 已经给两个新 token 都填了 host.duration=1.0（未均分）
        new = [
            Token("<SP>", "<SP>", 0.3, 0, 1, 0),
            Token("天", "zh_tian1", 1.0, 60, 2, 1),   # REPLACE 消费 host_idx=1
            Token("气", "zh_qi4", 1.0, 60, 2, 2),    # SPLIT 消费 host_idx=1
            Token("<SP>", "<SP>", 0.3, 0, 1, 3),
        ]
        path = AlignmentPath(ops=[
            AlignmentOp("SP_ALIGN", None, (0,), 0.0),
            AlignmentOp("REPLACE", None, (1,), 0.0),
            AlignmentOp("SPLIT", None, (1,), 0.0),
            AlignmentOp("SP_ALIGN", None, (2,), 0.0),
        ], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        # 两个非 SP token 应均分 1.0s → 各 0.5s
        non_sp = [t for t in result if not t.is_sp]
        assert len(non_sp) == 2
        assert abs(non_sp[0].duration - 0.5) < 0.01
        assert abs(non_sp[1].duration - 0.5) < 0.01
        # 总时长守恒
        orig_sum = sum(t.duration for t in orig)
        new_sum = sum(t.duration for t in result)
        assert abs(orig_sum - new_sum) < 0.01

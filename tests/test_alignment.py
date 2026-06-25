# tests/test_alignment.py
"""MidiLyricsAlignment 测试套件 (v3: 顺序映射 + 贪心压缩)."""
import json
import os

import pytest

from alignment.models import Token, Unit, AlignmentOp, AlignmentPath, CostWeights, Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokens(specs):
    """快捷构造 token 列表。specs = [(text, pitch, type, dur), ...]"""
    return [Token(t, "<SP>" if t == "<SP>" else f"ph_{t}",
                  d, p, nt, i) for i, (t, p, nt, d) in enumerate(specs)]


def _make_track(specs, f0="", meta=None):
    """快捷构造 Track。specs 同 _make_tokens。"""
    return Track(tokens=_make_tokens(specs), meta=meta or {}, f0=f0)


# ---------------------------------------------------------------------------
# 数据结构 (models.py 不变)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 解析 (parser.py 不变)
# ---------------------------------------------------------------------------


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

    def test_serialize_spd_duration_format(self):
        """SP duration 是 SPD(如 0.49)，serialize_track 用 :.2f 格式化正确。"""
        tokens = [
            Token("<SP>", "<SP>", 0.491, 0, 1, 0),
            Token("你", "zh_ni3", 0.40, 60, 2, 1),
        ]
        track = Track(tokens=tokens, meta={}, f0="")
        d = serialize_track(track)
        assert d["duration"].split()[0] == "0.49"


# ---------------------------------------------------------------------------
# 断句 segment_sentences
# ---------------------------------------------------------------------------


from alignment.align import segment_sentences, calculate_spd, align_track


class TestSegmentSentences:
    def test_split_by_punctuation(self):
        assert segment_sentences("你好。世界", 0) == ["你好", "世界"]

    def test_split_by_newline(self):
        assert segment_sentences("你好\n世界", 0) == ["你好", "世界"]

    def test_no_split_when_target_zero(self):
        assert segment_sentences("你好世界", 0) == ["你好世界"]

    def test_split_to_meet_target(self):
        """超过 10 字无标点、无 CT-Transformer 时保持原句（只断一次需 punctuate_fn）。"""
        result = segment_sentences("我是一只小小鸟想要飞呀飞不过却怎么也飞不高")
        # 无 punctuate_fn → 无法切，原句保留
        assert len(result) == 1
        assert result[0] == "我是一只小小鸟想要飞呀飞不过却怎么也飞不高"

    def test_no_split_when_too_short(self):
        """<= 10 字不切。"""
        result = segment_sentences("你好")
        assert len(result) == 1
        assert result == ["你好"]

    def test_split_preserves_all_chars(self):
        result = segment_sentences("天空海洋世界大地")
        assert "".join(result) == "天空海洋世界大地"

    def test_empty_lyrics(self):
        assert segment_sentences("") == []


# ---------------------------------------------------------------------------
# SPD calculate_spd
# ---------------------------------------------------------------------------


class TestCalculateSpd:
    def test_basic_formula(self):
        # AVG=0.3, M=4, N=8 → 0.3 * 0.5 = 0.15
        assert abs(calculate_spd([0.3, 0.3], 4, 8) - 0.15) < 1e-9

    def test_clamped_to_max(self):
        # AVG=0.5, M/N 很大 → 应被 MAX=0.6 截断
        spd = calculate_spd([0.4, 0.6], 100, 10)
        assert spd == 0.6

    def test_clamped_to_min(self):
        # ratio 极小 → 应被 0.1 兜底
        spd = calculate_spd([0.3, 0.3], 1, 100)
        assert spd == 0.1

    def test_no_orig_sp_uses_default(self):
        # 无原始 SP → AVG=MAX=0.3
        spd = calculate_spd([], 4, 8)
        assert abs(spd - 0.3 * 0.5) < 1e-9

    def test_no_orig_sp_clamped(self):
        spd = calculate_spd([], 1, 100)
        assert spd == 0.1


# ---------------------------------------------------------------------------
# 核心对齐 align_track
# ---------------------------------------------------------------------------


class TestAlign:
    def setup_method(self):
        self.weights = CostWeights()
        # 2 个非 SP token(你 pitch60 dur0.4, 好 pitch62 dur0.4)，2 个 SP(dur0.3)
        self.track = _make_track(
            [("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
             ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3)],
            f0="0.0 " * 70,
            meta={"language": "Mandarin", "time": [0, 1400]},
        )

    def test_basic_replacement(self):
        """干净 1:1 映射：2 字 → 2 个非 SP token。"""
        new_track, warns = align_track(
            self.track, "天空", self.weights, True, False,
        )
        texts = [t.text for t in new_track.tokens]
        # [SP] 天 空 [SP]
        assert texts == ["<SP>", "天", "空", "<SP>"]
        assert warns == []

    def test_pitch_inherited_from_source_token(self):
        """字继承源 token 的 pitch。"""
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        # 天→你(pitch60), 空→好(pitch62)
        assert new_track.tokens[1].note_pitch == 60
        assert new_track.tokens[2].note_pitch == 62

    def test_sp_duration_is_spd(self):
        """新 SP 的 duration 是 SPD 计算值。"""
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        spd = calculate_spd([0.3, 0.3], 4, 4)  # M=4, N=4 → 0.3
        sp_tokens = [t for t in new_track.tokens if t.is_sp]
        assert all(abs(t.duration - spd) < 1e-9 for t in sp_tokens)

    def test_sp_note_type_is_1(self):
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        sp_tokens = [t for t in new_track.tokens if t.is_sp]
        assert all(t.note_type == 1 for t in sp_tokens)
        assert all(t.note_pitch == 0 for t in sp_tokens)

    def test_char_note_type_is_2(self):
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        assert all(t.note_type == 2 for t in char_tokens)

    def test_repeat_char_note_type_3(self):
        """连续相同字(非叠词)→ type=3。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.8),
             ("啊", 62, 2, 0.8), ("<SP>", 0, 1, 0.3)],
        )
        # "我我" 非叠词表词 → 第二个 type=3
        new_track, _ = align_track(track, "我我", self.weights, True, False)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        assert char_tokens[0].note_type == 2
        assert char_tokens[1].note_type == 3

    def test_reduplication_not_type_3(self):
        """叠词(哥哥)两字都独立演唱 → 都 type=2。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.8),
             ("啊", 62, 2, 0.8), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "哥哥", self.weights, True, False)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        assert all(t.note_type == 2 for t in char_tokens)
        assert all(t.text == "哥" for t in char_tokens)

    def test_split_keeps_word_on_longest_token(self):
        """字数 > 非 SP token：duration 按比例分配。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.8),
             ("啊", 62, 2, 0.6), ("<SP>", 0, 1, 0.3)],
        )
        # "天空世界" 4 字，2 个非 SP token，每 token 分 2 字
        new_track, warns = align_track(track, "天空世界", self.weights, True, False)
        texts = [t.text for t in new_track.tokens]
        # 1 句 → [SP] 天 空 世 界 [SP]
        assert texts == ["<SP>", "天", "空", "世", "界", "<SP>"]
        # token0 (0.8s) 分 2 字 各 0.4, token1 (0.6s) 分 2 字 各 0.3
        non_sp = [t for t in new_track.tokens if not t.is_sp]
        assert len(non_sp) == 4
        assert non_sp[0].note_pitch == 60
        assert non_sp[2].note_pitch == 62
        assert non_sp[2].note_pitch == 62

    def test_more_tokens_than_chars_drops_extras(self):
        """字数 < 非 SP token：多余 token 丢弃(Case 1)。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.4),
             ("啊", 62, 2, 0.4), ("啊", 64, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "天空", self.weights, True, False)
        texts = [t.text for t in new_track.tokens]
        # 2 字映射到前 2 个非 SP token，第 3 个丢弃
        assert texts == ["<SP>", "天", "空", "<SP>"]

    def test_f0_sp_inserts_zeros(self):
        """新 SP 处 f0 插全 0(round(SPD*50) 帧)。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
             ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3)],
            f0="261.6 " * 70,
        )
        new_track, _ = align_track(track, "天空", self.weights, True, False)
        f0_vals = [float(x) for x in new_track.f0.split()]
        # 前 round(0.3*50)=15 帧应是 0(SP)
        assert all(v == 0.0 for v in f0_vals[:15])

    def test_f0_split_slices_segment(self):
        """SPLIT 时 f0 段按字数切片(不插值)。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 1.0), ("<SP>", 0, 1, 0.3)],
            f0="100.0 " * 50 + "200.0 " * 50,
        )
        # "天空" 2 字压到 1 个 token(1.0s → 50 帧)
        new_track, _ = align_track(track, "天空", self.weights, True, False)
        f0_vals = [float(x) for x in new_track.f0.split()]
        # 字的非零 f0 应来自原 token 段(100/200)，不插值
        non_sp_f0 = [v for v in f0_vals if v != 0.0]
        assert all(v in (100.0, 200.0) for v in non_sp_f0)

    def test_digit_normalization(self):
        """normalize_digits=True 时数字转中文。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.4),
             ("啊", 62, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "12", self.weights, True, False)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        # "12" → "一二"
        assert [t.text for t in char_tokens] == ["一", "二"]

    def test_digit_normalization_disabled(self):
        """normalize_digits=False：英文词作为整词单元。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.4),
             ("啊", 62, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "ab", self.weights, False, False)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        # "ab" 是一个英文词单元
        assert len(char_tokens) == 1
        assert char_tokens[0].text == "ab"

    def test_force_tone4_applied(self):
        """force_tone4 把高音中文音素改四声。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 90, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "天", self.weights, True, True)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        # pitch=90 >= 79 → 末位声调改 4
        assert char_tokens[0].phoneme[-1] == "4"

    def test_force_tone4_not_applied_when_disabled(self):
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 90, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, _ = align_track(track, "天", self.weights, True, False)
        char_tokens = [t for t in new_track.tokens if not t.is_sp]
        # 默认 tone 1 不改
        assert char_tokens[0].phoneme == "zh_tian1"

    def test_empty_lyrics_raises(self):
        with pytest.raises(ValueError, match="empty"):
            align_track(self.track, "", self.weights, True, False)

    def test_time_field_updated(self):
        """time 字段更新为新总 duration(毫秒)。"""
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        new_total = sum(t.duration for t in new_track.tokens)
        assert new_track.meta["time"] == [0, round(new_total * 1000)]

    def test_min_split_duration_floor(self):
        """SPLIT 后单字 duration < 0.1 时抬到 0.1(从同 token 其他字匀)。"""
        # token 0.3s 分给 5 字 → 每 0.06 < 0.1，需借匀
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.3),
             ("啊", 62, 2, 0.3), ("<SP>", 0, 1, 0.3)],
        )
        # 10 个不同字塞进 2 个 token，每个 token 5 字
        new_track, warns = align_track(
            track, "甲乙丙丁戊己庚辛壬癸", self.weights, True, False,
        )
        non_sp = [t for t in new_track.tokens if not t.is_sp]
        # 每字至少 0.1(总 token dur 0.3，5 字需 0.5 > 0.3 → 无法全满足，触发告警)
        # 但能抬多少抬多少
        assert any("MIN_DURATION" in w or "HIGH_SPLIT" in w for w in warns)

    def test_warns_on_high_split_ratio(self):
        """字数远超 token 数 → HIGH_SPLIT_RATIO 告警。"""
        track = _make_track(
            [("<SP>", 0, 1, 0.3), ("啊", 60, 2, 0.4),
             ("啊", 62, 2, 0.4), ("<SP>", 0, 1, 0.3)],
        )
        new_track, warns = align_track(track, "我" * 50, self.weights, True, False)
        assert any("HIGH_SPLIT" in w for w in warns)

    def test_returns_track_type(self):
        new_track, _ = align_track(
            self.track, "天空", self.weights, True, False,
        )
        assert isinstance(new_track, Track)
        assert new_track.f0  # f0 非空


# ---------------------------------------------------------------------------
# 节点级集成 (MidiLyricsAlignment)
# ---------------------------------------------------------------------------


class TestNodeIntegration:
    TRACK_JSON = json.dumps([{
        "index": "vocal_0_3000",
        "language": "Mandarin",
        "time": [0, 3000],
        "text": "<SP> 你 好 <SP>",
        "phoneme": "<SP> zh_ni3 zh_hao3 <SP>",
        "duration": "0.30 0.40 0.40 0.30",
        "note_pitch": "0 60 62 0",
        "note_type": "1 2 2 1",
        "f0": "0.0 0.0 261.6 293.7",
    }])

    def _node(self):
        from nodes import MidiLyricsAlignment
        return MidiLyricsAlignment()

    def test_output_is_valid_json(self):
        node = self._node()
        result = node.align_lyrics(self.TRACK_JSON, "天空")
        out = result[0]
        assert not out.startswith("Error"), f"Node error: {out[:100]}"
        parsed = json.loads(out)
        assert len(parsed) == 1
        assert "text" in parsed[0]
        assert "duration" in parsed[0]
        assert "f0" in parsed[0]

    def test_returns_two_tuple(self):
        """RF-2: 返回 (midi_json, warnings) 二元组。"""
        node = self._node()
        result = node.align_lyrics(self.TRACK_JSON, "天空")
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_sp_count_equals_sentences_plus_one(self):
        """v3: SP 数 = 句数 + 1(不再守恒原 SP 数)。"""
        node = self._node()
        result = node.align_lyrics(self.TRACK_JSON, "天空\n大海")
        parsed = json.loads(result[0])
        new_tokens = parsed[0]["text"].split()
        new_sp = sum(1 for t in new_tokens if t == "<SP>")
        # "天空\n大海" → 2 句 → 3 SP
        assert new_sp == 3

    def test_clean_alignment_no_warnings(self):
        """干净对齐时 warnings 为空字符串。"""
        node = self._node()
        result = node.align_lyrics(self.TRACK_JSON, "天空")
        assert result[1] == ""

    def test_extreme_split_emits_warning(self):
        """200 字塞进 2 token → 触发质量告警。"""
        node = self._node()
        result = node.align_lyrics(self.TRACK_JSON, "我" * 200)
        assert "Error" not in result[0]
        assert result[1], "expected non-empty warnings for 200-char in 2 slots"
        assert (
            "MIN_DURATION_UNRESOLVED" in result[1]
            or "HIGH_SPLIT" in result[1]
        ), f"unexpected warnings: {result[1]!r}"

    def test_multi_track_lyrics_partitioned(self):
        """多 track：歌词按 duration 比例分配，不重复。"""
        multi_track_json = json.dumps([
            {"text": "<SP> 啊 啊 <SP>",
             "phoneme": "<SP> zh_a1 zh_a1 <SP>",
             "duration": "0.3 0.5 0.5 0.3",
             "note_pitch": "0 60 62 0",
             "note_type": "1 2 2 1"},
            {"text": "<SP> 啊 <SP>",
             "phoneme": "<SP> zh_a1 <SP>",
             "duration": "0.3 0.4 0.3",
             "note_pitch": "0 60 0",
             "note_type": "1 2 1"},
        ])
        node = self._node()
        result = node.align_lyrics(multi_track_json, "天空\n海洋")
        out = result[0]
        assert not out.startswith("Error"), f"unexpected error: {out}"
        parsed = json.loads(out)
        assert len(parsed) == 2, "both tracks must survive"

        def non_sp_chars(track_text):
            return [c for c in track_text.split() if c != "<SP>"]

        track0_chars = non_sp_chars(parsed[0]["text"])
        track1_chars = non_sp_chars(parsed[1]["text"])
        # 两 track 都应有字
        assert len(track0_chars) > 0
        assert len(track1_chars) > 0

    def test_invalid_json_returns_error(self):
        node = self._node()
        result = node.align_lyrics("{bad json", "天空")
        assert result[0].startswith("Error")

    def test_none_inputs_handled(self):
        """ComfyUI 可能传 None。"""
        node = self._node()
        result = node.align_lyrics(None, None)
        assert result[0].startswith("Error")  # 空 midi_json 报错

    def test_speed_change_applied(self):
        """speed != 1 时 duration 缩放。"""
        node = self._node()
        result_normal = node.align_lyrics(self.TRACK_JSON, "天空", speed=1.0)
        result_fast = node.align_lyrics(self.TRACK_JSON, "天空", speed=2.0)
        normal_dur = sum(float(x) for x in
                         json.loads(result_normal[0])[0]["duration"].split())
        fast_dur = sum(float(x) for x in
                       json.loads(result_fast[0])[0]["duration"].split())
        # 2x speed → duration 减半
        assert abs(fast_dur - normal_dur / 2) < 0.05


# ---------------------------------------------------------------------------
# force_tone4 helper (nodes.py 保留)
# ---------------------------------------------------------------------------


class TestForceTone4:
    def test_high_pitch_zh_forced_to_tone4(self):
        from nodes import _apply_force_tone4
        tokens = [Token("你", "zh_ni3", 0.4, 80, 2, 0)]
        result = _apply_force_tone4(tokens, threshold=79)
        assert result[0].phoneme == "zh_ni4"

    def test_low_pitch_unchanged(self):
        from nodes import _apply_force_tone4
        tokens = [Token("你", "zh_ni3", 0.4, 60, 2, 0)]
        result = _apply_force_tone4(tokens, threshold=79)
        assert result[0].phoneme == "zh_ni3"

    def test_sp_unchanged(self):
        from nodes import _apply_force_tone4
        tokens = [Token("<SP>", "<SP>", 0.3, 80, 1, 0)]
        result = _apply_force_tone4(tokens, threshold=79)
        assert result[0].phoneme == "<SP>"

    def test_boundary_threshold_inclusive(self):
        """pitch == threshold 也改(>= 语义)。"""
        from nodes import _apply_force_tone4
        tokens = [Token("啊", "zh_a1", 0.4, 79, 2, 0)]
        result = _apply_force_tone4(tokens, threshold=79)
        assert result[0].phoneme == "zh_a4"


# ---------------------------------------------------------------------------
# 变速 (speed.py 保留)
# ---------------------------------------------------------------------------


from alignment.speed import apply_speed_change


class TestSpeedAdapter:
    def test_no_change_when_speed_1(self):
        tokens = _make_tokens([("你", 60, 2, 0.4), ("好", 62, 2, 0.4)])
        track = Track(tokens=tokens, meta={}, f0="261.6 293.7")
        result = apply_speed_change([track], 1.0)
        assert result[0].tokens[0].duration == 0.4

    def test_speedup_halves_duration(self):
        tokens = _make_tokens([("你", 60, 2, 0.8)])
        track = Track(tokens=tokens, meta={}, f0="261.6 261.6 261.6 261.6")
        result = apply_speed_change([track], 2.0)
        assert abs(result[0].tokens[0].duration - 0.4) < 0.01


# ---------------------------------------------------------------------------
# 回归: 真实人声 track (v3 不变量)
# ---------------------------------------------------------------------------


class TestRegression:
    FIXTURE_PATH = "tests/fixtures/vocal_sample.json"

    def setup_method(self):
        if not os.path.exists(self.FIXTURE_PATH):
            self.FIXTURE_PATH = os.path.join(
                os.path.dirname(__file__), "fixtures", "vocal_sample.json"
            )

    def test_real_track_aligns_without_error(self):
        """真实 42-token track: 对齐不报错，输出字段完整。"""
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        from nodes import MidiLyricsAlignment
        node = MidiLyricsAlignment()
        result = node.align_lyrics(track_json, "我是一只小小鸟想要飞呀飞")
        out = result[0]
        assert not out.startswith("Error"), f"Node error: {out[:100]}"
        parsed = json.loads(out)
        track = parsed[0]
        # 必需字段齐全
        for field in ("text", "phoneme", "duration", "note_pitch", "note_type", "f0"):
            assert field in track, f"missing field: {field}"

    def test_real_track_sp_at_least_two(self):
        """v3: 至少有首尾 SP(>=2)。"""
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        from nodes import MidiLyricsAlignment
        node = MidiLyricsAlignment()
        result = node.align_lyrics(track_json, "我是一只小小鸟")
        parsed = json.loads(result[0])
        new_tokens = parsed[0]["text"].split()
        new_sp = sum(1 for t in new_tokens if t == "<SP>")
        assert new_sp >= 2, f"expected >=2 SP, got {new_sp}"

    def test_real_track_f0_valid_floats(self):
        """f0 是空格分隔的浮点数。"""
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        from nodes import MidiLyricsAlignment
        node = MidiLyricsAlignment()
        result = node.align_lyrics(track_json, "我是一只小小鸟")
        parsed = json.loads(result[0])
        f0_str = parsed[0]["f0"]
        f0_vals = f0_str.split()
        # 全部可解析为 float
        for v in f0_vals:
            float(v)
        # f0 非空
        assert len(f0_vals) > 0

    def test_real_track_note_types_valid(self):
        """note_type 全部在 {1,2,3}。"""
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        from nodes import MidiLyricsAlignment
        node = MidiLyricsAlignment()
        result = node.align_lyrics(track_json, "我是一只小小鸟")
        parsed = json.loads(result[0])
        types = [int(x) for x in parsed[0]["note_type"].split()]
        assert all(t in (1, 2, 3) for t in types)

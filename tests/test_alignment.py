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

# alignment/__init__.py
"""MIDI 歌词统一对齐算法子包."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.parser import parse_tracks, serialize_track, serialize_tracks
from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)
__all__ = [
    "Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights",
    "parse_tracks", "serialize_track", "serialize_tracks",
    "replace_cost", "word_span_cost", "split_cost", "drop_cost", "sp_align_cost",
]

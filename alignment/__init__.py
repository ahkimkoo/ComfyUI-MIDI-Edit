# alignment/__init__.py
"""MIDI 歌词统一对齐算法子包."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.parser import parse_tracks, serialize_track, serialize_tracks
from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)
from alignment.preprocess import normalize_lyrics, tokenize_units
from alignment.dp import solve_alignment
from alignment.rebuild import rebuild_tokens, allocate_durations
from alignment.speed import apply_speed_change
__all__ = [
    "Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights",
    "parse_tracks", "serialize_track", "serialize_tracks",
    "replace_cost", "word_span_cost", "split_cost", "drop_cost", "sp_align_cost",
    "normalize_lyrics", "tokenize_units",
    "solve_alignment",
    "rebuild_tokens", "allocate_durations",
    "apply_speed_change",
]

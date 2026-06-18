# alignment/__init__.py
"""MIDI 歌词统一对齐算法子包."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.parser import parse_tracks, serialize_track, serialize_tracks
__all__ = [
    "Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights",
    "parse_tracks", "serialize_track", "serialize_tracks",
]

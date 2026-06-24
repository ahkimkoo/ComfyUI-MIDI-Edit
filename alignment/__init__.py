# alignment/__init__.py
"""MIDI 歌词统一对齐算法子包 (v3: 顺序映射 + 贪心压缩)."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.parser import parse_tracks, serialize_track, serialize_tracks
from alignment.phoneme import char_to_phoneme, word_to_phoneme
from alignment.align import align_track, segment_sentences, calculate_spd
from alignment.speed import apply_speed_change
__all__ = [
    "Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights",
    "parse_tracks", "serialize_track", "serialize_tracks",
    "char_to_phoneme", "word_to_phoneme",
    "align_track", "segment_sentences", "calculate_spd",
    "apply_speed_change",
]

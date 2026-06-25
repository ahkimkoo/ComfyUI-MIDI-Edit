# core/__init__.py
"""ComfyUI-MIDI-Edit core package: modular MIDI lyrics editing & alignment.

Public API surface re-exported here for convenience. Individual modules may
also be imported directly (e.g. ``from core.g2p import char_to_phoneme``).
"""
from core.g2p import (
    char_to_phoneme,
    is_chinese_char,
    is_english_word,
    normalize_digits,
    word_to_phoneme,
    ZH_FLAG,
    EN_FLAG,
)
from core.text_utils import (
    clean_lyrics,
    is_reduplication,
    split_lyrics_to_sentences,
)
from core.midi_format import (
    Token,
    Track,
    parse_tracks,
    serialize_track,
    serialize_tracks,
    FPS,
)
from core.ct_transformer import (
    get_ct_transformer,
    restore_punctuation,
)
from core.speed import (
    apply_speed,
    apply_speed_change,
    format_durations,
    format_f0,
)
from core.edit_algorithm import (
    replace_lyrics,
    extract_lyrics,
    merge_repeated_chars,
)
from core.align_algorithm import (
    align_track,
    segment_sentences,
    calculate_spd,
    CostWeights,
    Unit,
    AlignmentOp,
    AlignmentPath,
)

__all__ = [
    # g2p
    "char_to_phoneme", "is_chinese_char", "is_english_word",
    "normalize_digits", "word_to_phoneme", "ZH_FLAG", "EN_FLAG",
    # text_utils
    "clean_lyrics", "is_reduplication", "split_lyrics_to_sentences",
    # midi_format
    "Token", "Track", "parse_tracks", "serialize_track",
    "serialize_tracks", "FPS",
    # ct_transformer
    "get_ct_transformer", "restore_punctuation",
    # speed
    "apply_speed", "apply_speed_change", "format_durations", "format_f0",
    # edit_algorithm
    "replace_lyrics", "extract_lyrics", "merge_repeated_chars",
    # align_algorithm
    "align_track", "segment_sentences", "calculate_spd",
    "CostWeights", "Unit", "AlignmentOp", "AlignmentPath",
]

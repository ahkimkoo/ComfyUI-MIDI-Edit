"""
ComfyUI-MIDI-Edit: Custom nodes for replacing lyrics in MIDI JSON data
and auto-generating corresponding phonemes.

The heavy lifting lives in the :mod:`core` package. This module is the
ComfyUI-facing surface: it wires the node classes to the public ``core`` API
and re-exports the historical symbol names (including underscore-prefixed
aliases) so existing tests and downstream code keep importing from ``nodes``.
"""

import json
import os
import sys

# Ensure the package directory is importable under ComfyUI's loader.
# ComfyUI imports this module dynamically; the extension dir may not be on
# sys.path, so `from core import ...` below would fail without this.
_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

# ---------------------------------------------------------------------------
# Re-export the core public API under historical names.
# Tests import many of these symbols directly from ``nodes``; keeping the
# aliases here means the import paths stay stable across the refactor.
# ---------------------------------------------------------------------------

from core.g2p import (  # noqa: F401
    char_to_phoneme,
    is_chinese_char,
    is_english_word,
    normalize_digits as _normalize_digits,
    word_to_phoneme as _word_to_phoneme,
    ZH_FLAG,
    EN_FLAG,
)
from core.text_utils import (  # noqa: F401
    clean_lyrics,
    is_reduplication,
    split_lyrics_to_sentences as _split_lyrics_to_sentences,
)
from core.midi_format import (  # noqa: F401
    Token,
    Track,
    parse_tracks,
    serialize_track,
    serialize_tracks,
    FPS,
)
from core.ct_transformer import (  # noqa: F401
    _code_mix_split_words,
    _split_to_mini_sentence,
    _TokenIDConverter,
    _CTTransformerPunc,
    _ensure_punc_model,
    get_ct_transformer as _get_ct_transformer,
    restore_punctuation as _restore_punctuation,
    _MODELSCOPE_MODEL_ID,
    _PUNC_MODEL_FILES,
    _PUNC_LIST,
    _PUNC_LIST_NORMALIZED,
    _SPLIT_SIZE,
)
from core.speed import (  # noqa: F401
    apply_speed as _apply_speed,
    apply_speed_change,
    _fmt_dur,
    format_durations as _fmt_durs,
    format_f0 as _fmt_f0,
)
from core.edit_algorithm import (  # noqa: F401
    _build_units,
    _count_total_sections,
    _split_by_section_sizes,
    _get_section_sizes,
    _get_section_durations,
    _build_collapsed_slots,
    _split_token,
    _apply_char,
    _process_section,
    _split_into_segments,
    _split_at_punctuation,
    _first_punct_cut,
    _compute_expected_char_counts,
    _distribute_units_proportional,
    replace_lyrics,
    extract_lyrics,
    merge_repeated_chars,
    _distribute_lyrics,
)
from core.align_algorithm import (  # noqa: F401
    align_track,
    segment_sentences,
    calculate_spd,
    _apply_force_tone4,
    CostWeights,
    Unit,
    AlignmentOp,
    AlignmentPath,
)


# ``_smart_split_sentences`` resolves its punctuation function from THIS module's
# namespace at call time, so tests that monkey-patch ``nodes._restore_punctuation``
# take effect. The real algorithm (with an injectable punctuation fn) lives in
# core.edit_algorithm; this thin wrapper just injects the patchable name.
from core.edit_algorithm import _smart_split_sentences as _core_smart_split_sentences


def _smart_split_sentences(plain_lyrics, midi_data, split_mode="token"):
    """Delegate to core.edit_algorithm, injecting the (monkey-patchable)
    punctuation resolver from the nodes namespace."""
    return _core_smart_split_sentences(
        plain_lyrics, midi_data, split_mode=split_mode,
        restore_punc_fn=_restore_punctuation,
    )


# --- ComfyUI Nodes ---


class MIDIEditLyrics:
    """ComfyUI node that replaces lyrics in MIDI JSON and auto-generates phonemes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "new_lyrics": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "force_tone4": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                 "high_pitch_threshold": ("INT", {"default": 79, "min": 0, "max": 127, "step": 1}),
                "fixed_pause": ("BOOLEAN", {"default": True, "label_on": "Fixed", "label_off": "Flexible"}),
                "split_mode": (["token", "duration"], {"default": "token"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_json",)
    FUNCTION = "edit_lyrics"
    CATEGORY = "MIDI-Edit"
    DESCRIPTION = (
        "Replace lyrics in MIDI JSON with new text and auto-generate phonemes. "
        "Smart matching: splits lyrics by newlines/punctuation into sentences, "
        "each mapped to a MIDI section. If new sentence is shorter than original, "
        "extra characters keep original text. If longer, automatically splits "
        "the longest-duration note (pitch unchanged, duration halved) to create "
        "more positions. Consecutive repeated characters in original are grouped "
        "(e.g. '向向' → slot with count 2) and expanded accordingly. "
        "Chinese chars → zh_ pinyin; English → en_ phonemes. "
        "Force Tone 4: forces Chinese phonemes at pitch >= threshold to tone 4. "
        "Fixed Pause (default ON): keep SP durations unchanged. "
        "Flexible Pause: when tokens are crowded or SP is overly long, "
        "redistribute SP time proportionally to tokens. "
        "Split Mode: 'token' = allocate chars by original token count proportion; "
        "'duration' = allocate chars by original section duration proportion. "
        "Speed: adjust playback speed (1.0 = normal, >1 faster, <1 slower). "
        "Duration and f0 are scaled proportionally."
    )

    def edit_lyrics(self, midi_json: str, new_lyrics: str,
                    force_tone4: bool, high_pitch_threshold: int,
                    fixed_pause: bool, split_mode: str,
                    speed: float) -> tuple:
        # Guard against None inputs from ComfyUI (empty/disconnected nodes)
        midi_json = midi_json or ""
        new_lyrics = new_lyrics or ""
        if not midi_json.strip():
            return (midi_json,)
        try:
            return (replace_lyrics(midi_json, new_lyrics,
                                    force_tone4_high_pitch=force_tone4,
                                    high_pitch_threshold=high_pitch_threshold,
                                    fixed_pause=fixed_pause,
                                    split_mode=split_mode,
                                    speed=speed),)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid MIDI JSON input: {e}") from e
        except ValueError as e:
            raise ValueError(f"Lyrics replacement error: {e}") from e


class MIDIExtractLyrics:
    """ComfyUI node that extracts lyrics text from MIDI JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "merge_repeated": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "extract"
    CATEGORY = "MIDI-Edit"
    DESCRIPTION = (
        "Extract lyrics from MIDI JSON. Concatenates text from all tracks, "
        "removes spaces, and replaces <SP> markers with newlines. "
        "When Merge Repeated is ON, consecutive duplicate characters are collapsed into one."
    )

    def extract(self, midi_json: str, merge_repeated: bool) -> tuple:
        try:
            return (extract_lyrics(midi_json, merge_repeated=merge_repeated),)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid MIDI JSON input: {e}") from e
        except ValueError as e:
            raise ValueError(f"Lyrics extraction error: {e}") from e


class MIDIMergeRepeatedChars:
    """ComfyUI node that merges consecutive repeated characters, keeping one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "merge"
    CATEGORY = "MIDI-Edit"
    DESCRIPTION = (
        "Merge consecutive repeated characters in text, keeping only one. "
        'E.g. "向向往" → "向往". Useful for cleaning duplicated lyrics characters.'
    )

    def merge(self, text: str) -> tuple:
        return (merge_repeated_chars(text),)


# --- Mappings ---

NODE_CLASS_MAPPINGS = {
    "MIDIEditLyrics": MIDIEditLyrics,
    "MIDIExtractLyrics": MIDIExtractLyrics,
    "MIDIMergeRepeatedChars": MIDIMergeRepeatedChars,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIEditLyrics": "MIDI Edit Lyrics",
    "MIDIExtractLyrics": "MIDI Extract Lyrics",
    "MIDIMergeRepeatedChars": "MIDI Merge Repeated Chars",
}


# --- Unified DP-based alignment node ---


class MidiLyricsAlignment:
    """统一对齐算法节点（基于顺序映射 + 贪心压缩）.

    替代 MIDIEditLyrics 的场景分支式处理，用单一算法求全局最优对齐。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True}),
                "lyrics": ("STRING", {"multiline": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "normalize_digits": ("BOOLEAN", {"default": True}),
                "force_tone4": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("midi_json", "warnings")
    FUNCTION = "align_lyrics"
    CATEGORY = "MIDI"

    def align_lyrics(self, midi_json, lyrics, speed=1.0,
                     normalize_digits=True, force_tone4=False):
        midi_json = "" if midi_json is None else str(midi_json)
        lyrics = "" if lyrics is None else str(lyrics)

        try:
            tracks = parse_tracks(midi_json)
        except ValueError as e:
            return (f"Error: {e}", "")

        weights = CostWeights()

        # 多 track：按非 SP duration 比例分配歌词（避免小 track 被塞全文）。
        lyrics_per_track = _distribute_lyrics(lyrics, tracks)

        result_tracks = []
        warnings_list = []
        for track_idx, track in enumerate(tracks):
            track_lyrics = (
                lyrics_per_track[track_idx]
                if track_idx < len(lyrics_per_track) else ""
            )

            # 空 track / 无歌词分配 -> 原样保留。
            if not track_lyrics or not track_lyrics.strip():
                result_tracks.append(track)
                continue

            try:
                new_track, warns = align_track(
                    track, track_lyrics, weights,
                    normalize_digits, force_tone4,
                    punctuate_fn=_restore_punctuation,
                )
            except ValueError as e:
                return (f"Error: {e}", "")

            # 给 warning 打上 track 索引前缀，便于定位。
            for w in warns:
                warnings_list.append(f"{w}(t{track_idx})")
            result_tracks.append(new_track)

        if speed != 1.0:
            result_tracks = apply_speed_change(result_tracks, speed)

        output_json = serialize_tracks(result_tracks)

        # warnings 通过 RETURN_TYPES 的第二个输出返回，供 ComfyUI
        # Web UI 反馈。替代原来的 print() 静默输出。
        warnings_str = "; ".join(warnings_list) if warnings_list else ""
        return (output_json, warnings_str)

"""
ComfyUI-MIDI-Edit: Custom node for replacing lyrics in MIDI JSON data
and auto-generating corresponding phonemes.
"""

import json
import os
import re
import warnings

# --- NLTK data directory ---
# Use a local models/nltk directory under the project so data stays
# self-contained and does not pollute the user's home directory.
_NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nltk")
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", _NLTK_DATA_DIR)

# Suppress numpy runtime warnings from g2pM model inference
warnings.filterwarnings("ignore", category=RuntimeWarning)

from g2pM import G2pM
from g2p_en import G2p as G2pE

# --- G2P setup ---

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

ZH_FLAG = "zh_"
EN_FLAG = "en_"

_g2p_zh = None
_g2p_en = None

_NLTK_PACKAGES = [
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "punkt_tab",
]


def _ensure_nltk_data():
    """Download required NLTK data to the local models/nltk directory if missing."""
    import nltk

    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(pkg)
        except LookupError:
            print(f"[MIDI-Edit] Downloading NLTK data: {pkg}")
            nltk.download(pkg, download_dir=_NLTK_DATA_DIR, quiet=True)


def _get_g2p_zh():
    """Lazy-initialise the Chinese G2P model (downloads on first use)."""
    global _g2p_zh
    if _g2p_zh is None:
        _g2p_zh = G2pM()
    return _g2p_zh


def _get_g2p_en():
    """Lazy-initialise the English G2P model."""
    global _g2p_en
    if _g2p_en is None:
        _ensure_nltk_data()
        _g2p_en = G2pE()
    return _g2p_en


# --- Helpers ---


def is_chinese_char(char: str) -> bool:
    return len(char) == 1 and bool(_ZH_CHAR_RE.fullmatch(char))


def is_english_word(word: str) -> bool:
    return bool(word) and bool(_EN_WORD_RE.fullmatch(word))


def char_to_phoneme(char: str, lang: str = "Mandarin") -> str:
    """Convert a single character to its phoneme representation.

    Chinese chars  -> zh_<pinyin_with_tone>
    English letters -> en_<lowercase_letter>  (single letter passthrough)
    English words  -> en_<phonemes_joined_by_dash>
    Unknown / SP   -> <SP>
    """
    if char == "<SP>":
        return "<SP>"
    if is_chinese_char(char):
        g2p = _get_g2p_zh()
        result = g2p(char, tone=True, char_split=False)
        return ZH_FLAG + result[0]
    if len(char) == 1 and char.isascii() and char.isalpha():
        # Single English letter — use directly as phoneme
        return EN_FLAG + char.lower()
    if is_english_word(char):
        g2p = _get_g2p_en()
        result = g2p(char.lower())
        return EN_FLAG + "-".join(result)
    return "<SP>"


def clean_lyrics(text: str) -> str:
    """Remove punctuation, spaces, and newlines. Keep only Chinese chars and English letters."""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", text)


# --- Core logic ---


def replace_lyrics(midi_json_str: str, new_lyrics: str,
                    force_tone4_high_pitch: bool = False,
                     high_pitch_threshold: int = 79) -> str:
    """Replace lyrics in MIDI JSON and regenerate phonemes.

    Parameters
    ----------
    midi_json_str : str
        JSON string of the MIDI data array.
    new_lyrics : str
        The new lyrics text to substitute.
    force_tone4_high_pitch : bool
        When True, force all Chinese phonemes in slots whose note_pitch
        >= *high_pitch_threshold* to use tone 4 (去声).
        Off by default for backward compat.
    high_pitch_threshold : int
        MIDI note value threshold for high-pitch detection (0–127).
         Defaults to 79 (G5).  Only effective when *force_tone4_high_pitch*
        is True.

    Returns
    -------
    str
        Modified MIDI JSON string.
    """
    midi_data = json.loads(midi_json_str)

    if not isinstance(midi_data, list):
        raise ValueError("MIDI JSON must be a list (array) of track objects")

    # Split new lyrics by lines, clean each line independently.
    # This preserves section boundaries: each line maps to one <SP>-delimited
    # section in the original MIDI text.  Backward compatible: a single-line input
    # (no newlines) behaves identically to the old flat-mapping approach.
    lyrics_lines = [clean_lyrics(line) for line in new_lyrics.split("\n")]
    while lyrics_lines and not lyrics_lines[-1]:
        lyrics_lines.pop()

    result = []
    for track in midi_data:
        if "text" not in track or "phoneme" not in track:
            result.append(track)
            continue

        original_text_tokens = track["text"].split(" ")
        original_phoneme_tokens = track["phoneme"].split(" ")

        # Parse note_pitch values (if available) for high-pitch detection.
        note_pitch_tokens = []
        if force_tone4_high_pitch and "note_pitch" in track:
            note_pitch_tokens = track["note_pitch"].split(" ")

        # Build slot list: group consecutive identical non-SP tokens into one slot.
        # Each slot is (char_or_SP, repeat_count, start_token_index, section_idx).
        # SP tokens are always ("SP", 1, i, -1).
        # Consecutive identical non-SP tokens merge: e.g. "兄 兄" → ("兄", 2).
        # A different char or <SP> breaks the group.
        # section_idx tracks which <SP>-delimited section each slot belongs to.
        slots = []
        section_idx = -1
        i = 0
        while i < len(original_text_tokens):
            token = original_text_tokens[i]
            if token == "<SP>":
                slots.append(("SP", 1, i, -1))
                section_idx += 1
                i += 1
            else:
                char = token
                start = i
                count = 0
                while i < len(original_text_tokens) and original_text_tokens[i] == char:
                    count += 1
                    i += 1
                slots.append((char, count, start, section_idx))

        # Walk slots and produce new text/phoneme tokens.
        # Total output token count must equal original token count.
        # Characters are allocated per-section: each section gets chars from its
        # corresponding lyrics line independently.
        new_text_tokens = []
        new_phoneme_tokens = []
        section_char_pos = {}  # section_idx -> char position in that section's lyrics line

        # Backward compat: single-line input shares one char pool across all sections.
        single_line = len(lyrics_lines) == 1

        for slot_char, slot_count, slot_start, slot_section in slots:
            if slot_char == "SP":
                new_text_tokens.append("<SP>")
                if slot_start < len(original_phoneme_tokens):
                    new_phoneme_tokens.append(original_phoneme_tokens[slot_start])
                else:
                    new_phoneme_tokens.append("<SP>")
            else:
                # Get the lyrics line for this section.
                # Single-line input: all sections share the same line (flat mapping).
                if single_line:
                    line = lyrics_lines[0]
                    pos = section_char_pos.get(0, 0)
                elif slot_section < len(lyrics_lines):
                    line = lyrics_lines[slot_section]
                    pos = section_char_pos.get(slot_section, 0)
                else:
                    line = ""
                    pos = section_char_pos.get(slot_section, 0)

                if pos < len(line):
                    replacement_char = line[pos]
                    phoneme = char_to_phoneme(replacement_char)

                    # Check if this slot spans any note_pitch >= high_pitch_threshold.
                    # Only applies to zh_ prefixed phonemes; skip en_ and <SP>.
                    if (force_tone4_high_pitch
                            and phoneme.startswith(ZH_FLAG)
                            and note_pitch_tokens):
                        is_high_pitch = False
                        for pi in range(slot_start, slot_start + slot_count):
                            if pi < len(note_pitch_tokens):
                                try:
                                    pval = int(note_pitch_tokens[pi])
                                    if pval >= high_pitch_threshold:
                                        is_high_pitch = True
                                        break
                                except ValueError:
                                    pass
                        if is_high_pitch:
                            phoneme = re.sub(r"(\d)$", "4", phoneme)

                    for _ in range(slot_count):
                        new_text_tokens.append(replacement_char)
                        new_phoneme_tokens.append(phoneme)
                    section_char_pos[0 if single_line else slot_section] = pos + 1
                else:
                    # Not enough chars in this section's line — keep original
                    for j in range(slot_count):
                        token_idx = slot_start + j
                        new_text_tokens.append(original_text_tokens[token_idx])
                        if token_idx < len(original_phoneme_tokens):
                            new_phoneme_tokens.append(original_phoneme_tokens[token_idx])
                        else:
                            new_phoneme_tokens.append("<SP>")

        new_track = dict(track)
        new_track["text"] = " ".join(new_text_tokens)
        new_track["phoneme"] = " ".join(new_phoneme_tokens)
        result.append(new_track)

    return json.dumps(result, ensure_ascii=False, indent=2)


def extract_lyrics(midi_json_str: str, merge_repeated: bool = False) -> str:
    """Extract and concatenate lyrics text from MIDI JSON.

    Iterates all tracks, collects the ``text`` field, strips spaces,
    and replaces ``<SP>`` markers with newlines.

    Parameters
    ----------
    midi_json_str : str
        JSON string of the MIDI data array.

    Returns
    -------
    str
        Concatenated lyrics with ``<SP>`` replaced by ``\\n`` and
        all spaces removed.  Empty string on invalid / empty input.
    """
    try:
        midi_data = json.loads(midi_json_str)
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(midi_data, list):
        return ""

    # Concatenate text from every track that has one
    all_text = ""
    for track in midi_data:
        text = track.get("text") if isinstance(track, dict) else None
        if text:
            all_text += text

    if not all_text:
        return ""

    # Remove all plain spaces, then convert <SP> → newline
    all_text = all_text.replace(" ", "")
    all_text = all_text.replace("<SP>", "\n")

    result = all_text.strip()
    if merge_repeated:
        result = merge_repeated_chars(result)
    return result


def merge_repeated_chars(text: str) -> str:
    """Remove consecutive duplicate characters from *text*, keeping one.

    Example::

        "向向往"  → "向往"
        "天天马"  → "天马"
        "好世界好" → "好世界好"  (not consecutive)
    """
    if not text:
        return text
    result = [text[0]]
    for ch in text[1:]:
        if ch != result[-1]:
            result.append(ch)
    return "".join(result)


# --- ComfyUI Node ---


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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_json",)
    FUNCTION = "edit_lyrics"
    CATEGORY = "MIDI-Edit"
    DESCRIPTION = (
        "Replace lyrics in MIDI JSON with new text and auto-generate phonemes. "
        "Chinese characters are converted to zh_ prefixed pinyin; "
        "English words to en_ prefixed phonemes. "
        "<SP> markers and non-lyric fields are preserved. "
        "Force Tone 4 (Smart Pitch): when ON, forces Chinese phonemes at "
        "pitch >= threshold to tone 4 (去声). Only effective when the toggle is ON."
    )

    def edit_lyrics(self, midi_json: str, new_lyrics: str, force_tone4: bool, high_pitch_threshold: int) -> tuple:
        try:
            return (replace_lyrics(midi_json, new_lyrics,
                                    force_tone4_high_pitch=force_tone4,
                                    high_pitch_threshold=high_pitch_threshold),)
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

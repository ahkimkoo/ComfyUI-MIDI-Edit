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


def replace_lyrics(midi_json_str: str, new_lyrics: str) -> str:
    """Replace lyrics in MIDI JSON and regenerate phonemes.

    Parameters
    ----------
    midi_json_str : str
        JSON string of the MIDI data array.
    new_lyrics : str
        The new lyrics text to substitute.

    Returns
    -------
    str
        Modified MIDI JSON string.
    """
    midi_data = json.loads(midi_json_str)

    if not isinstance(midi_data, list):
        raise ValueError("MIDI JSON must be a list (array) of track objects")

    cleaned = clean_lyrics(new_lyrics)

    result = []
    for track in midi_data:
        if "text" not in track or "phoneme" not in track:
            result.append(track)
            continue

        original_text_tokens = track["text"].split(" ")
        original_phoneme_tokens = track["phoneme"].split(" ")

        # Build new text and phoneme — replace by position, extras ignored, shortage keeps original
        new_text_tokens = []
        new_phoneme_tokens = []
        char_idx = 0
        for i, token in enumerate(original_text_tokens):
            if token == "<SP>":
                new_text_tokens.append("<SP>")
                if i < len(original_phoneme_tokens):
                    new_phoneme_tokens.append(original_phoneme_tokens[i])
                else:
                    new_phoneme_tokens.append("<SP>")
            elif char_idx < len(cleaned):
                new_text_tokens.append(cleaned[char_idx])
                new_phoneme_tokens.append(char_to_phoneme(cleaned[char_idx]))
                char_idx += 1
            else:
                # User provided fewer chars than original — keep original text & phoneme
                new_text_tokens.append(token)
                if i < len(original_phoneme_tokens):
                    new_phoneme_tokens.append(original_phoneme_tokens[i])
                else:
                    new_phoneme_tokens.append("<SP>")

        new_track = dict(track)
        new_track["text"] = " ".join(new_text_tokens)
        new_track["phoneme"] = " ".join(new_phoneme_tokens)
        result.append(new_track)

    return json.dumps(result, ensure_ascii=False, indent=2)


def extract_lyrics(midi_json_str: str) -> str:
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

    return all_text.strip()


# --- ComfyUI Node ---


class MIDIEditLyrics:
    """ComfyUI node that replaces lyrics in MIDI JSON and auto-generates phonemes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "new_lyrics": ("STRING", {"multiline": True, "dynamicPrompts": False}),
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
        "<SP> markers and non-lyric fields are preserved."
    )

    def edit_lyrics(self, midi_json: str, new_lyrics: str) -> tuple:
        try:
            return (replace_lyrics(midi_json, new_lyrics),)
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "extract"
    CATEGORY = "MIDI-Edit"
    DESCRIPTION = (
        "Extract lyrics from MIDI JSON. Concatenates text from all tracks, "
        "removes spaces, and replaces <SP> markers with newlines."
    )

    def extract(self, midi_json: str) -> tuple:
        try:
            return (extract_lyrics(midi_json),)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid MIDI JSON input: {e}") from e
        except ValueError as e:
            raise ValueError(f"Lyrics extraction error: {e}") from e


# --- Mappings ---

NODE_CLASS_MAPPINGS = {
    "MIDIEditLyrics": MIDIEditLyrics,
    "MIDIExtractLyrics": MIDIExtractLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIEditLyrics": "MIDI Edit Lyrics",
    "MIDIExtractLyrics": "MIDI Extract Lyrics",
}

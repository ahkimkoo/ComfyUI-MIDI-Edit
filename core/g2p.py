# core/g2p.py
"""G2P model loading and phoneme inference (g2pM + g2p_en).

Single source of truth for character/word → phoneme conversion and the
related digit/charset helpers. Both G2P models are imported lazily inside
``_get_g2p_zh`` / ``_get_g2p_en`` so that importing this module (and hence
ComfyUI startup) is not blocked by the heavy g2pM / g2p_en dependencies.
"""
from __future__ import annotations

import os
import re
import warnings

# Suppress numpy runtime warnings from g2pM model inference.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- NLTK data directory ---
# Resolve to <repo>/models/nltk regardless of import site. This file lives at
# <repo>/core/g2p.py, so two dirname() calls reach the repository root.
_NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nltk"
)
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", _NLTK_DATA_DIR)

# --- Constants ---

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

ZH_FLAG = "zh_"
EN_FLAG = "en_"

# --- G2P singletons (lazy) ---

_g2p_zh = None
_g2p_en = None

_NLTK_PACKAGES = [
    "taggers/averaged_perceptron_tagger_eng",
    "taggers/averaged_perceptron_tagger",
    "tokenizers/punkt_tab",
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
        # Lazy import: g2pM is heavy and must not block module import.
        from g2pM import G2pM

        _g2p_zh = G2pM()
    return _g2p_zh


def _get_g2p_en():
    """Lazy-initialise the English G2P model."""
    global _g2p_en
    if _g2p_en is None:
        # Lazy import: g2p_en pulls in nltk + model data on construction.
        from g2p_en import G2p as G2pE

        _ensure_nltk_data()
        _g2p_en = G2pE()
    return _g2p_en


# --- Helpers ---


def is_chinese_char(char: str) -> bool:
    return len(char) == 1 and bool(_ZH_CHAR_RE.fullmatch(char))


def is_english_word(word: str) -> bool:
    return bool(word) and bool(_EN_WORD_RE.fullmatch(word))


# Arabic digits → Chinese number chars (g2pM can pronounce these)
_DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")


def normalize_digits(text: str) -> str:
    """Convert Arabic digits to Chinese number chars so they get proper phonemes."""
    if not text:
        return text
    return text.translate(_DIGIT_TO_ZH)


# Single English letter → single ARPAbet phoneme.
# Letter-name pronunciation (e.g., g2p_en("s") → "EH1-S") produces multi-phoneme
# tokens that don't fit in short MIDI notes (< 0.1s). Instead, map each letter to
# its most common consonant/vowel sound so each note has exactly 1 phoneme.
_EN_LETTER_TO_PHONEME: dict[str, str] = {
    # Vowels — most common sound (schwa/short vowel)
    "a": "AH0", "e": "EH0", "i": "IH0", "o": "OW0", "u": "AH0",
    # Consonants — direct ARPAbet mapping
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
    "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}


def char_to_phoneme(char: str, lang: str = "Mandarin") -> str:
    """Convert a single character to its phoneme representation.

    Chinese chars   -> zh_<pinyin_with_tone>
    English letters -> en_<single ARPAbet phoneme>  (consonant/vowel, not letter name)
    English words   -> en_<ARPAbet_phonemes_joined_by_dash>  (via g2p_en)
    Unknown / SP    -> <SP>
    """
    if char == "<SP>":
        return "<SP>"
    if is_chinese_char(char):
        g2p = _get_g2p_zh()
        result = g2p(char, tone=True, char_split=False)
        return ZH_FLAG + result[0]
    if len(char) == 1 and char.isascii() and char.isalpha():
        # Single English letter — use single ARPAbet phoneme (not letter name)
        return EN_FLAG + _EN_LETTER_TO_PHONEME.get(char.lower(), "AH0")
    if is_english_word(char):
        g2p = _get_g2p_en()
        result = g2p(char.lower())
        return EN_FLAG + "-".join(result)
    return "<SP>"


def word_to_phoneme(word: str) -> str:
    """Convert an English word to its ARPAbet phoneme string (en_X-Y-Z)."""
    g2p = _get_g2p_en()
    result = g2p(word.lower())
    return EN_FLAG + "-".join(result)

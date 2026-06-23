# alignment/phoneme.py
"""Phoneme conversion utilities (self-contained, no dependency on external nodes.py).

Extracted from nodes.py to avoid ComfyUI's ``nodes.py`` naming conflict.
Under ComfyUI, ``sys.modules['nodes']`` resolves to ComfyUI core's
``nodes.py`` which has none of these helpers, so the previous
``from nodes import char_to_phoneme`` raised ``ImportError`` at runtime.

This module is the single source of truth for the alignment subpackage.
``nodes.py`` keeps its own copies for the legacy ``MIDIEditLyrics`` node
(unchanged); the two are intentionally decoupled.
"""
from __future__ import annotations

import os
import re
import warnings

# Suppress numpy runtime warnings from g2pM model inference (mirrors nodes.py).
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- NLTK data directory ---
# Resolve to <repo>/models/nltk regardless of whether this file is imported
# from within alignment/ or copied elsewhere. __file__ is alignment/phoneme.py,
# so two dirname() calls reach the repository root.
_NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nltk"
)
os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", _NLTK_DATA_DIR)

from g2pM import G2pM  # noqa: E402
from g2p_en import G2p as G2pE  # noqa: E402

# --- Constants ---

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

ZH_FLAG = "zh_"
EN_FLAG = "en_"

# --- G2P singletons ---

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


# Arabic digits -> Chinese number chars (g2pM can pronounce these)
_DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")


def _normalize_digits(text: str) -> str:
    """Convert Arabic digits to Chinese number chars so they get proper phonemes."""
    if not text:
        return text
    return text.translate(_DIGIT_TO_ZH)


# Single English letter -> single ARPAbet phoneme.
# Letter-name pronunciation (e.g., g2p_en("s") -> "EH1-S") produces multi-phoneme
# tokens that don't fit in short MIDI notes (< 0.1s). Instead, map each letter to
# its most common consonant/vowel sound so each note has exactly 1 phoneme.
_EN_LETTER_TO_PHONEME: dict[str, str] = {
    # Vowels -- most common sound (schwa/short vowel)
    "a": "AH0", "e": "EH0", "i": "IH0", "o": "OW0", "u": "AH0",
    # Consonants -- direct ARPAbet mapping
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G",
    "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "p": "P", "q": "K", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}


# --- Main API ---


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
        # Single English letter -- use single ARPAbet phoneme (not letter name)
        return EN_FLAG + _EN_LETTER_TO_PHONEME.get(char.lower(), "AH0")
    if is_english_word(char):
        g2p = _get_g2p_en()
        result = g2p(char.lower())
        return EN_FLAG + "-".join(result)
    return "<SP>"


def word_to_phoneme(word: str) -> str:
    """Convert an English word to its ARPAbet phoneme string (en_X-Y-Z).

    Public alias of the former ``nodes._word_to_phoneme``.
    """
    g2p = _get_g2p_en()
    result = g2p(word.lower())
    return EN_FLAG + "-".join(result)


# --- 叠词判断 ---
# 叠词（reduplication）是中文里两字相同但构成独立词汇的词，
# 如"哥哥""慢慢"。它们的每个字都应独立演唱（type=2），
# 不应标 type=3（延续音，SoulX-Singer 不独立唱第二个字）。
#
# 与之相对的是"节奏重复"——如"完完""了了"，第二个字是第一个的
# 延续（type=3），原 MIDI 数据里这类重复的第二个字确实是 type=3。
_REDUP_WORDS = frozenset({
    # 家人/称呼
    "爸爸", "妈妈", "爷爷", "奶奶", "哥哥", "弟弟", "姐姐", "妹妹",
    "叔叔", "伯伯", "舅舅", "姑姑", "婆婆", "公公", "外婆", "外公",
    "阿姨", "嫂嫂", "婶婶", "太太", "爹爹", "宝宝", "娃娃",
    # 常用叠词
    "星星", "慢慢", "轻轻", "悄悄", "渐渐", "常常", "刚刚",
    "处处", "人人", "天天", "年年", "步步", "声声", "点点",
    "阵阵", "朵朵", "条条", "闪闪", "洋洋", "匆匆", "纷纷",
    "茫茫", "圆圆", "方方", "长长", "短短", "高高", "低低",
    "大大", "小小", "多多", "少少", "好好", "坏坏", "快快",
    "暖暖", "凉凉", "红红", "白白", "黑黑", "黄黄", "绿绿",
    "甜甜", "苦苦", "酸酸", "辣辣", "香香", "臭臭",
    "嘻嘻", "哈哈", "呵呵", "嗯嗯", "哎哎",
})


def is_reduplication(char: str, prev_char: str) -> bool:
    """判断两个字是否构成叠词（独立词汇，不是节奏重复）。

    Returns True 如果 prev_char+char 在叠词表中（如"哥哥"），
    表示两个字都应独立演唱（type=2），不应标 type=3。
    """
    if len(char) != 1 or len(prev_char) != 1 or char != prev_char:
        return False
    return (prev_char + char) in _REDUP_WORDS

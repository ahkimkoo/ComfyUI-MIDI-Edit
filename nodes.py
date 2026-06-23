"""
ComfyUI-MIDI-Edit: Custom node for replacing lyrics in MIDI JSON data
and auto-generating corresponding phonemes.
"""

import json
import math
import os
import re
import sys
import warnings

# Ensure the alignment/ subpackage is importable under ComfyUI's loader
# (ComfyUI imports this module dynamically; the extension dir may not be
# on sys.path, so `from alignment import ...` in MidiLyricsAlignment would fail).
_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

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


# Arabic digits → Chinese number chars (g2pM can pronounce these)
_DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")


def _normalize_digits(text: str) -> str:
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


def _word_to_phoneme(word: str) -> str:
    """Convert an English word to its ARPAbet phoneme string (en_X-Y-Z)."""
    g2p = _get_g2p_en()
    result = g2p(word.lower())
    return EN_FLAG + "-".join(result)


def _build_units(sentence: str) -> list[dict]:
    """Parse a sentence into a list of units for slot mapping.

    Each unit is a dict:
      - Chinese char: {"text": "你", "phoneme": "zh_ni3", "is_word": False}
      - English word: {"text": "love", "phoneme": "en_L-AH1-V", "is_word": True}

    English words stay as single units (matching SoulX-Singer's format where
    one word = one token). Spaces serve as word delimiters and are excluded.
    """
    units: list[dict] = []
    if not sentence:
        return units
    sentence = _normalize_digits(sentence)
    parts = sentence.split()
    for part in parts:
        if not part:
            continue
        i = 0
        while i < len(part):
            if is_chinese_char(part[i]):
                ph = char_to_phoneme(part[i])
                units.append({"text": part[i], "phoneme": ph, "is_word": False})
                i += 1
            elif part[i].isascii() and part[i].isalpha():
                j = i
                while j < len(part) and part[j].isascii() and part[j].isalpha():
                    j += 1
                word = part[i:j]
                ph = _word_to_phoneme(word)
                units.append({"text": word, "phoneme": ph, "is_word": True})
                i = j
            else:
                i += 1  # skip punctuation
    return units


def clean_lyrics(text: str) -> str:
    """Remove punctuation and newlines. Keep Chinese chars, English letters, and spaces.
    Arabic digits are converted to Chinese number chars."""
    if not text:
        return ""
    text = _normalize_digits(text)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z ]", "", text)


# --- Lyrics sentence splitting ---


_SENTENCE_DELIM_RE = re.compile(r"[\n，。！？；：、,.\!\?;:]+")


def _split_lyrics_to_sentences(new_lyrics: str) -> list[str]:
    """Split user lyrics into sentences by newlines and punctuation.

    Returns a list of cleaned (non-empty) sentence strings.
    Preserves spaces within sentences (needed for English word boundaries).
    """
    if not new_lyrics:
        return []
    raw = _SENTENCE_DELIM_RE.split(new_lyrics)
    result = []
    for s in raw:
        # Strip leading/trailing whitespace and collapse multiple spaces
        cleaned = " ".join(s.split())
        if cleaned:
            result.append(cleaned)
    return result


def _count_total_sections(midi_data: list) -> int:
    """Count total non-SP sections across all tracks."""
    count = 0
    for track in midi_data:
        if "text" not in track:
            continue
        tokens = track["text"].split(" ")
        i = 0
        while i < len(tokens):
            if tokens[i] != "<SP>":
                count += 1
                while i < len(tokens) and tokens[i] != "<SP>":
                    i += 1
            else:
                i += 1
    return count


def _split_by_section_sizes(plain_text: str, section_sizes: list[int]) -> list[str]:
    """Split plain text into chunks matching original section sizes.

    Walks through section_sizes, slicing plain_text at each boundary.
    For English text (with spaces), cuts at word boundaries when possible.
    The last chunk gets whatever remains (may be shorter or empty).
    """
    result = []
    pos = 0
    for size in section_sizes:
        if pos >= len(plain_text):
            result.append("")
        else:
            end = min(pos + size, len(plain_text))
            # If in the middle of a word, extend to word boundary
            if end < len(plain_text) and plain_text[end - 1] != " " and plain_text[end] != " ":
                word_end = plain_text.find(" ", end)
                if word_end != -1 and word_end < end + 20:  # don't extend too far
                    end = word_end
            chunk = plain_text[pos:end].strip()
            result.append(chunk)
            pos = end
    return result


def _get_section_sizes(midi_data: list) -> list[int]:
    """Get the token count of each non-SP section across all tracks."""
    sizes = []
    for track in midi_data:
        if "text" not in track:
            continue
        tokens = track["text"].split(" ")
        i = 0
        while i < len(tokens):
            if tokens[i] != "<SP>":
                start = i
                while i < len(tokens) and tokens[i] != "<SP>":
                    i += 1
                sizes.append(i - start)
            else:
                i += 1
    return sizes


def _get_section_durations(midi_data: list) -> list[float]:
    """Get the total duration of each non-SP section across all tracks."""
    durations = []
    for track in midi_data:
        if "text" not in track or "duration" not in track:
            continue
        tokens = track["text"].split(" ")
        dur_tokens = track["duration"].split(" ")
        i = 0
        while i < len(tokens):
            if tokens[i] != "<SP>":
                sec_dur = 0.0
                while i < len(tokens) and tokens[i] != "<SP>":
                    sec_dur += float(dur_tokens[i])
                    i += 1
                durations.append(sec_dur)
            else:
                i += 1
    return durations


# --- CT-Transformer Punctuation Model (for smart sentence splitting) ---

_MODELSCOPE_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
_PUNC_MODEL_FILES = {"model_quant.onnx", "tokens.json"}
_PUNC_LIST = ["<unk>", "_", ",", "。", "?", "、"]
_PUNC_LIST_NORMALIZED = ["<unk>", "_", "，", "。", "？", "、"]
_SPLIT_SIZE = 20

_ct_transformer = None  # module-level singleton


def _get_models_base_dir():
    """Return the base models directory (ComfyUI models_dir or local fallback)."""
    try:
        import folder_paths
        return folder_paths.models_dir
    except ImportError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _ensure_punc_model():
    """Download CT-Transformer model from ModelScope if not present.

    Stores model files under {models_base}/ct-transformer-punc/.
    """
    model_dir = os.path.join(_get_models_base_dir(), "ct-transformer-punc")
    if all(os.path.exists(os.path.join(model_dir, f)) for f in _PUNC_MODEL_FILES):
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    print(f"[MIDI-Edit] Downloading CT-Transformer punctuation model from ModelScope...")

    from modelscope import snapshot_download
    downloaded = snapshot_download(
        _MODELSCOPE_MODEL_ID,
        cache_dir=model_dir,
    )

    # snapshot_download may create a subdirectory; copy needed files to model_dir
    if downloaded != model_dir:
        for fname in _PUNC_MODEL_FILES:
            src = os.path.join(downloaded, fname)
            dst = os.path.join(model_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

    # Verify files exist after download
    missing = [f for f in _PUNC_MODEL_FILES
               if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"[MIDI-Edit] Failed to download CT-Transformer model. "
            f"Missing files: {missing}"
        )

    print(f"[MIDI-Edit] CT-Transformer punctuation model ready at {model_dir}")
    return model_dir


def _load_token_list(tokens_path: str) -> list[str]:
    """Load the token list from a JSON file."""
    with open(tokens_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Copied utility functions from cttPunctuator (adapted, no typeguard dep) ---


def _code_mix_split_words(text: str) -> list[str]:
    """Split text into words: each Chinese char is a word, consecutive ASCII is one word."""
    words = []
    segs = text.split()
    for seg in segs:
        current_word = ""
        for c in seg:
            if len(c.encode()) == 1:
                current_word += c
            else:
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = ""
                words.append(c)
        if len(current_word) > 0:
            words.append(current_word)
    return words


def _split_to_mini_sentence(words: list, word_limit: int = _SPLIT_SIZE) -> list[list]:
    """Split word list into chunks of at most word_limit."""
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit: (i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit:])
    return sentences


class _TokenIDConverter:
    """Bidirectional converter between token strings and integer IDs."""

    def __init__(self, token_list: list[str]):
        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def tokens2ids(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]


class _CTTransformerPunc:
    """Lightweight CT-Transformer punctuation restorer.

    Directly loads the ModelScope ONNX model and runs inference,
    without the lovemefan/cttPunctuator dependency.
    """

    def __init__(self, model_dir: str):
        import numpy as np
        import onnxruntime

        tokens_path = os.path.join(model_dir, "tokens.json")
        model_path = os.path.join(model_dir, "model_quant.onnx")

        token_list = _load_token_list(tokens_path)
        self.converter = _TokenIDConverter(token_list)

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3  # suppress warnings
        self.session = onnxruntime.InferenceSession(model_path, sess_options=opts)

        # Determine period index in punc list
        self.punc_list = list(_PUNC_LIST_NORMALIZED)
        self.period_idx = _PUNC_LIST_NORMALIZED.index("。")

    def add_punctuation(self, text: str) -> str:
        """Run CT-Transformer inference to add punctuation to text.

        Returns the text with punctuation inserted.
        """
        import numpy as np

        split_text = _code_mix_split_words(text)
        if not split_text:
            return text

        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = _split_to_mini_sentence(split_text)
        mini_sentences_id = _split_to_mini_sentence(split_text_id)

        cache_sent = []
        cache_sent_id = []
        result_parts = []

        for mini_i in range(len(mini_sentences)):
            mini_sentence = cache_sent + mini_sentences[mini_i]
            mini_sentence_id = cache_sent_id + mini_sentences_id[mini_i]
            # ModelScope quantized model uses int32 input named "inputs"
            mini_sentence_id = np.array(mini_sentence_id, dtype="int32")
            text_lengths = np.array([len(mini_sentence)], dtype="int32")

            mapped_feed = {
                "inputs": mini_sentence_id[None, :],
                "text_lengths": text_lengths,
            }

            outputs = self.session.run(None, mapped_feed)
            punctuations = np.argmax(outputs[0], axis=-1)[0]

            # Search for last period/question mark as cache boundary
            if mini_i < len(mini_sentences) - 1:
                sentence_end = -1
                last_comma_idx = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    p = self.punc_list[punctuations[i]]
                    if p == "。" or p == "？":
                        sentence_end = i
                        break
                    if last_comma_idx < 0 and p == "，":
                        last_comma_idx = i

                if sentence_end < 0 and len(mini_sentence) > 200 and last_comma_idx >= 0:
                    sentence_end = last_comma_idx
                    punctuations[sentence_end] = self.period_idx

                cache_sent = mini_sentence[sentence_end + 1:]
                cache_sent_id = mini_sentence_id[sentence_end + 1:].tolist()
                mini_sentence = mini_sentence[:sentence_end + 1]
                punctuations = punctuations[:sentence_end + 1]

            # Build words with punctuation
            for i in range(len(mini_sentence)):
                if i > 0:
                    # Add space between two consecutive ASCII words
                    if (len(mini_sentence[i][0].encode()) == 1
                            and len(mini_sentence[i - 1][0].encode()) == 1):
                        result_parts.append(" ")
                result_parts.append(mini_sentence[i])
                p = self.punc_list[punctuations[i]]
                if p != "_":
                    result_parts.append(p)

        # Force sentence to end with period
        result = "".join(result_parts)
        if result and result[-1] in ("，", "、"):
            result = result[:-1] + "。"
        elif result and result[-1] not in ("。", "？"):
            result = result + "。"

        return result


def _get_ct_transformer() -> _CTTransformerPunc:
    """Lazy-initialise the CT-Transformer punctuation model (downloads on first use)."""
    global _ct_transformer
    if _ct_transformer is None:
        model_dir = _ensure_punc_model()
        _ct_transformer = _CTTransformerPunc(model_dir)
    return _ct_transformer


# --- Smart sentence splitting ---


_PUNC_SPLIT_RE = re.compile(r'[，。？！；、]')


def _restore_punctuation(text: str) -> str:
    """Run CT-Transformer to add punctuation to a sentence."""
    model = _get_ct_transformer()
    return model.add_punctuation(text)


# Punctuation priority for splitting: period > question > exclamation > comma > enumeration
_PERIOD_RE = re.compile(r'[。.？?！!]')
_COMMA_RE = re.compile(r'[，,、；;：:]')


def _split_at_punctuation(text: str) -> list[str]:
    """Split punctuated text into exactly 2 clean pieces at the best cut point.

    Cut point selection (priority order):
    1. Period/question/exclamation mark closest to the middle
    2. Comma/enumeration mark closest to the middle

    Punctuation marks are stripped from the output.
    Returns [left, right] if a cut is found, otherwise [original_text_no_punc].
    """
    text = text.strip()
    if len(text) <= 1:
        return [text]

    mid = len(text) / 2.0

    # Try period-family first, then comma-family
    for punc_re in (_PERIOD_RE, _COMMA_RE):
        best_pos = None
        best_dist = float('inf')
        for m in punc_re.finditer(text):
            dist = abs(m.start() - mid)
            if dist < best_dist:
                best_dist = dist
                best_pos = m.start()
        if best_pos is not None:
            left = text[:best_pos].strip()
            right = text[best_pos + 1:].strip()
            if left and right:
                # Strip ALL remaining punctuation from both halves
                left = re.sub(r'[，。！？；：、,.!?;:]', '', left).strip()
                right = re.sub(r'[，。！？；：、,.!?;:]', '', right).strip()
                if left and right:
                    return [left, right]

    # No usable punctuation found — return cleaned text as single piece
    clean = re.sub(r'[，。！？；：、,.!?;:]', '', text).strip()
    return [clean] if clean else [text]


def _compute_expected_char_counts(
    section_sizes: list[int], total_new_chars: int
) -> list[int]:
    """Compute expected char count per section, proportional to original token counts.

    Uses round-half-up for each section except the last, which gets the remainder
    to ensure the total equals total_new_chars exactly.
    """
    total_tokens = sum(section_sizes)
    if total_tokens == 0:
        return [0] * len(section_sizes)

    expected = []
    for i, sc in enumerate(section_sizes):
        if i < len(section_sizes) - 1:
            expected.append(round(sc * total_new_chars / total_tokens))
        else:
            # Last section gets the remainder
            expected.append(total_new_chars - sum(expected))
    return expected


def _first_punct_cut(text: str, target_len: int) -> list[str] | None:
    """Find the first punctuation mark in AI-punctuated text and split there.

    Returns [left, right] (cleaned of all punctuation) if a suitable cut is found,
    or None if no usable punctuation exists.
    """
    # Find the first punctuation mark position
    for m in re.finditer(r'[，。！？；：、,.!?;:]', text):
        pos = m.start()
        left = text[:pos].strip()
        right = text[pos + 1:].strip()
        if left and right:
            # Strip ALL remaining punctuation from both halves
            left = re.sub(r'[，。！？；：、,.!?;:]', '', left).strip()
            right = re.sub(r'[，。！？；：、,.!?;:]', '', right).strip()
            if left and right:
                return [left, right]
    return None


def _smart_split_sentences(
    plain_lyrics: str,
    midi_data: list,
    split_mode: str = "token",
) -> list[str]:
    """Split plain lyrics into sentences matching original MIDI section structure.

    Algorithm (triggered when initial punctuation/newline split yields a
    different sentence count than original MIDI sections):

    1. Compute section weights (token counts or durations depending on split_mode)
    2. Compute expected char count per section (proportional, rounded)
    3. For each section, from left to right:
       a. Run CT-Transformer on remaining lyrics to add punctuation
       b. Find first punctuation mark, check if left part's char count is
          within ±15% (rounded up) of expected
       c. If within tolerance → use AI cut point
       d. If not → hard-cut at expected char count
       e. Remaining lyrics continue to next section
    4. Return list of sentences (one per section)

    Args:
        plain_lyrics: Clean lyrics string (no punctuation, no spaces).
        midi_data: Original MIDI data list.
        split_mode: "token" = proportional to token count (default),
                    "duration" = proportional to section duration.

    Returns:
        List of sentence strings, one per original section.
    """
    if split_mode == "duration":
        section_weights = _get_section_durations(midi_data)
    else:
        section_weights = _get_section_sizes(midi_data)

    num_sections = len(section_weights)
    total_chars = len(plain_lyrics)

    sentences = []
    remaining = plain_lyrics

    for i in range(num_sections):
        if not remaining:
            sentences.append("")
            continue

        # Remaining sections to fill
        remaining_sections = num_sections - i
        if remaining_sections == 1:
            # Last section gets everything left
            sentences.append(remaining)
            break

        # Re-compute expected for this section based on remaining chars
        # and remaining weights (keeps proportions balanced)
        remaining_weights = section_weights[i:]
        remaining_total = sum(remaining_weights)
        exp = round(remaining_weights[0] * len(remaining) / remaining_total)
        # Clamp: must leave at least 1 char per remaining section
        max_exp = len(remaining) - (remaining_sections - 1)
        exp = max(1, min(exp, max_exp))
        tolerance = math.ceil(exp * 0.15)

        # Try AI punctuation cut
        try:
            punctuated = _restore_punctuation(remaining)
            cut = _first_punct_cut(punctuated, exp)
        except Exception:
            cut = None

        if cut is not None:
            left_len = len(cut[0])
            if abs(left_len - exp) <= tolerance:
                # AI cut is within tolerance — use it
                sentences.append(cut[0])
                remaining = cut[1]
                continue

        # Hard-cut at expected char count
        sentences.append(remaining[:exp])
        remaining = remaining[exp:]

    return sentences


# --- Segment / slot helpers ---


def _split_into_segments(
    text_tokens: list[str],
    phoneme_tokens: list[str],
    duration_tokens: list[str],
    note_pitch_tokens: list[str],
    note_type_tokens: list[str],
        f0_tokens: list[str],
) -> list[tuple[str, object]]:
    """Split token arrays into segments: ('section', [token_dicts]) or ('sp', sp_dict).

    Each non-SP token is represented as a dict with token-level MIDI fields.
    **f0 is NOT included** because it is frame-level data (not 1:1 with tokens).
    SP tokens are kept as individual sp dicts.
    """
    segments: list[tuple[str, object]] = []
    current: list[dict] = []

    for i, text in enumerate(text_tokens):
        token = {
            "text": text,
            "phoneme": phoneme_tokens[i] if i < len(phoneme_tokens) else "<SP>",
            "duration": float(duration_tokens[i]) if i < len(duration_tokens) else 0.0,
            "note_pitch": int(note_pitch_tokens[i]) if i < len(note_pitch_tokens) else 0,
            "note_type": int(note_type_tokens[i]) if i < len(note_type_tokens) else 0,
        }

        if text == "<SP>":
            if current:
                segments.append(("section", current))
                current = []
            segments.append(("sp", token))
        else:
            current.append(token)

    if current:
        segments.append(("section", current))
    return segments


def _build_collapsed_slots(tokens: list[dict]) -> list[tuple[str, int, list[int]]]:
    """Group consecutive identical text tokens into slots.

    Returns list of (char, count, [token_indices]).
    """
    if not tokens:
        return []
    slots: list[tuple[str, int, list[int]]] = []
    cur_char = tokens[0]["text"]
    cur_count = 1
    cur_indices = [0]
    for i in range(1, len(tokens)):
        if tokens[i]["text"] == cur_char:
            cur_count += 1
            cur_indices.append(i)
        else:
            slots.append((cur_char, cur_count, cur_indices))
            cur_char = tokens[i]["text"]
            cur_count = 1
            cur_indices = [i]
    slots.append((cur_char, cur_count, cur_indices))
    return slots


def _split_token(tokens: list[dict], idx: int) -> None:
    """Split token at *idx* into two halves with half duration. Pitch unchanged."""
    token = tokens[idx]
    half_dur = token["duration"] / 2.0
    token["duration"] = half_dur
    new_token = {
        "text": token["text"],
        "phoneme": token["phoneme"],
        "duration": half_dur,
        "note_pitch": token["note_pitch"],
        "note_type": token["note_type"],
    }
    # Preserve internal tags (e.g. _sec_id used by flat mode regrouping)
    for key in token:
        if key.startswith("_"):
            new_token[key] = token[key]
    tokens.insert(idx + 1, new_token)


def _apply_char(tokens: list[dict], idx: int, char: str,
                force_tone4: bool, threshold: int,
                preset_phoneme: str | None = None,
                is_continuation: bool = False) -> None:
    """Replace token text with *char*, regenerate phoneme, optional high-pitch adjustment.

    If *preset_phoneme* is given (English word-level), use it instead of calling
    char_to_phoneme. If *is_continuation* is True, set note_type to 3 (SoulX-Singer
    continuation/tie marker for multi-note words).
    """
    if preset_phoneme is not None:
        phoneme = preset_phoneme
    else:
        phoneme = char_to_phoneme(char)
        if force_tone4 and phoneme.startswith(ZH_FLAG):
            try:
                if int(tokens[idx].get("note_pitch", 0)) >= threshold:
                    phoneme = re.sub(r"(\d)$", "4", phoneme)
            except (ValueError, TypeError):
                pass
    tokens[idx]["text"] = char
    tokens[idx]["phoneme"] = phoneme
    if is_continuation:
        tokens[idx]["note_type"] = 3
    else:
        # 连续重复字检测：当前字与前一个非空非SP token 相同 → type=3
        # 但排除叠词（哥哥/妹妹等独立词汇，两字都独立演唱）
        is_repeat = False
        if char and char != "<SP>":
            for prev_idx in range(idx - 1, -1, -1):
                prev_text = tokens[prev_idx].get("text", "")
                if prev_text == "<SP>" or not prev_text:
                    continue
                if prev_text == char:
                    # 检查是否叠词
                    from alignment.phoneme import is_reduplication
                    is_repeat = not is_reduplication(char, prev_text)
                break
        if is_repeat:
            tokens[idx]["note_type"] = 3
        elif tokens[idx].get("note_type") == 3:
            tokens[idx]["note_type"] = 2


def _process_section(
    tokens: list[dict], sentence: str,
    force_tone4_high_pitch: bool, high_pitch_threshold: int,
) -> list[dict]:
    """Apply the slot-mapping algorithm to a single section.

    Supports Chinese (char-by-char) and English (word-level) and mixed.
    Builds a unit list where each Chinese char or English word is one unit,
    then distributes units across note slots.

    Distribution modes:
    - Collapse (N <= S): right-aligned, skip leading slots. For English words
      when N < S, extra slots become continuation notes (type=3).
    - Collapse+Distribute (S < N <= M): multi-count slots accept multiple units.
    - Expand (N > M): split longest tokens, then 1:1.
    """
    if not tokens:
        return tokens

    if not sentence:
        for tok in tokens:
            tok["text"] = ""
            tok["phoneme"] = ""
        return tokens

    units = _build_units(sentence)
    N = len(units)
    M = len(tokens)
    slots = _build_collapsed_slots(tokens)
    S = len(slots)

    has_english = any(u["is_word"] for u in units)

    # --- English/mixed with fewer units than slots: proportional distribution ---
    if has_english and N < S:
        return _distribute_units_proportional(
            tokens, units, slots, force_tone4_high_pitch, high_pitch_threshold
        )

    # --- Standard collapse / distribute / expand (works for Chinese and English) ---
    if N <= S:
        skip_count = S - N
        for slot_idx, (_orig_char, _count, token_indices) in enumerate(slots):
            if slot_idx < skip_count:
                for ti in token_indices:
                    tokens[ti]["text"] = ""
                    tokens[ti]["phoneme"] = ""
            else:
                unit_idx = slot_idx - skip_count
                unit = units[unit_idx]
                for ti in token_indices:
                    _apply_char(tokens, ti, unit["text"],
                                force_tone4_high_pitch, high_pitch_threshold,
                                preset_phoneme=unit["phoneme"],
                                is_continuation=False)
        return tokens

    if N <= M:
        remaining = list(units)
        for _orig_char, count, token_indices in slots:
            if not remaining:
                for ti in token_indices:
                    tokens[ti]["text"] = ""
                    tokens[ti]["phoneme"] = ""
            elif count == 1:
                unit = remaining.pop(0)
                _apply_char(tokens, token_indices[0], unit["text"],
                            force_tone4_high_pitch, high_pitch_threshold,
                            preset_phoneme=unit["phoneme"])
            else:
                n_assign = min(count, len(remaining))
                for j in range(n_assign):
                    unit = remaining.pop(0)
                    _apply_char(tokens, token_indices[j], unit["text"],
                                force_tone4_high_pitch, high_pitch_threshold,
                                preset_phoneme=unit["phoneme"])
                for j in range(n_assign, count):
                    tokens[token_indices[j]]["text"] = ""
                    tokens[token_indices[j]]["phoneme"] = ""
        return tokens

    # Expand mode
    while len(tokens) < N:
        longest_idx = max(range(len(tokens)), key=lambda i: tokens[i]["duration"])
        _split_token(tokens, longest_idx)

    for i in range(min(N, len(tokens))):
        unit = units[i]
        _apply_char(tokens, i, unit["text"],
                    force_tone4_high_pitch, high_pitch_threshold,
                    preset_phoneme=unit["phoneme"])

    for i in range(N, len(tokens)):
        tokens[i]["text"] = ""
        tokens[i]["phoneme"] = ""

    return tokens


def _distribute_units_proportional(
    tokens: list[dict], units: list[dict], slots: list,
    force_tone4: bool, threshold: int,
) -> list[dict]:
    """Distribute units across slots when there are fewer units than slots.

    English words get multiple consecutive slots (first = type 2, rest = type 3
    continuation). Chinese chars get exactly 1 slot each.

    Allocation: each unit starts with 1 slot. Extra slots are distributed to
    English words proportionally by word length. Chinese chars never expand.
    """
    N = len(units)
    S = len(slots)

    # Collect ALL token indices in order
    all_token_indices: list[int] = []
    for _, _, ti_list in slots:
        all_token_indices.extend(ti_list)

    total_tokens = len(all_token_indices)

    # Base allocation: 1 per unit
    allocation = [1] * N
    extra = total_tokens - N

    if extra > 0:
        # Only English words can absorb extra slots
        en_indices = [i for i, u in enumerate(units) if u["is_word"]]
        if en_indices:
            en_lens = [len(units[i]["text"]) for i in en_indices]
            total_en_len = sum(en_lens)
            for j, idx in enumerate(en_indices):
                allocation[idx] += round(extra * en_lens[j] / total_en_len)

            # Fix rounding
            while sum(allocation) > total_tokens:
                # Remove from largest English allocation
                idx = max(en_indices, key=lambda i: allocation[i])
                if allocation[idx] > 1:
                    allocation[idx] -= 1
                else:
                    break
            while sum(allocation) < total_tokens:
                # Add to any English word
                idx = min(en_indices, key=lambda i: allocation[i])
                allocation[idx] += 1

    # Assign units to consecutive token ranges
    tok_pos = 0
    for unit_idx in range(N):
        unit = units[unit_idx]
        count = allocation[unit_idx]
        for j in range(count):
            if tok_pos < total_tokens:
                ti = all_token_indices[tok_pos]
                _apply_char(tokens, ti, unit["text"],
                            force_tone4, threshold,
                            preset_phoneme=unit["phoneme"],
                            is_continuation=(j > 0 and unit["is_word"]))
                tok_pos += 1

    return tokens


# --- Core logic ---


def replace_lyrics(midi_json_str: str, new_lyrics: str,
                    force_tone4_high_pitch: bool = False,
                     high_pitch_threshold: int = 79,
                     fixed_pause: bool = True,
                     split_mode: str = "token",
                     speed: float = 1.0) -> str:
    """Replace lyrics in MIDI JSON with smart 3-mode algorithm.

    Splits user lyrics by newlines/punctuation into sentences, each mapped to
    a MIDI section (SP-delimited).  Three modes per section:

    - **Collapse** (N <= S): chars map to slots (deduplicated consecutive
      chars), right-aligned so the last char maps to the last slot.
    - **Collapse+Distribute** (S < N <= M): chars assigned to collapsed slots;
      multi-count slots accept multiple chars (one per token in the slot).
    - **Expand** (N > M): longest-duration tokens are split (duration halved,
      pitch unchanged) until token count matches sentence length, then 1:1.

    Parameters
    ----------
    midi_json_str : str
        JSON string of the MIDI data array.
    new_lyrics : str
        The new lyrics text to substitute.
    force_tone4_high_pitch : bool
        When True, force Chinese phonemes at note_pitch >= threshold to tone 4.
    high_pitch_threshold : int
        MIDI note value threshold (0-127). Defaults to 79 (G5).

    Returns
    -------
    str
        Modified MIDI JSON string.
    """
    midi_data = json.loads(midi_json_str)

    if not isinstance(midi_data, list):
        raise ValueError("MIDI JSON must be a list (array) of track objects")

    # Guard against None / empty lyrics input
    if not new_lyrics or not new_lyrics.strip():
        return midi_json_str

    sentences = _split_lyrics_to_sentences(new_lyrics)

    # Smart split: when sentence count != original section count, use
    # proportional expected-char-count algorithm with CT-Transformer AI cuts.
    total_sections = _count_total_sections(midi_data)
    if len(sentences) != total_sections:
        try:
            plain = clean_lyrics(new_lyrics)
            sentences = _smart_split_sentences(plain, midi_data,
                                               split_mode=split_mode)
        except Exception as e:
            # If smart split fails (model download error, etc.),
            # fall back to crude positional split.
            print(f"[MIDI-Edit] Smart split failed ({e}), falling back to positional split")
            section_sizes = _get_section_sizes(midi_data)
            plain = clean_lyrics(new_lyrics)
            sentences = _split_by_section_sizes(plain, section_sizes)

    # Global sentence counter — persists across tracks so that the second
    # track continues where the first track left off.
    global_sec_idx = 0

    result = []
    for track in midi_data:
        if not isinstance(track, dict):
            result.append(track)
            continue
        track_text = track.get("text") or ""
        track_phoneme = track.get("phoneme") or ""
        if not track_text or not track_phoneme:
            result.append(track)
            continue

        # Parse all token arrays from the track
        text_tokens = track_text.split(" ")
        phoneme_tokens = track_phoneme.split(" ")
        duration_raw = track.get("duration", "")
        duration_tokens = duration_raw.split(" ") if duration_raw else []
        pitch_raw = track.get("note_pitch", "")
        note_pitch_tokens = pitch_raw.split(" ") if pitch_raw else []
        type_raw = track.get("note_type", "")
        note_type_tokens = type_raw.split(" ") if type_raw else []
        f0_raw = track.get("f0", "")
        f0_tokens = f0_raw.split(" ") if f0_raw else []

        # Split into segments (sections + SP markers)
        segments = _split_into_segments(
            text_tokens, phoneme_tokens, duration_tokens,
            note_pitch_tokens, note_type_tokens, f0_tokens,
        )

        # Flat mode: single sentence without punctuation maps across ALL sections
        # by treating all non-SP tokens as one combined section, processing, then
        # re-inserting SP markers at their original positions.
        # Multi-sentence mode: each section maps to its own sentence independently.
        flat_mode = len(sentences) <= 1

        if flat_mode:
            # Collect all non-SP tokens, tagging each with its section index
            # so we can regroup after processing (which may add tokens via expand).
            all_tokens: list[dict] = []
            sp_markers: list[tuple[int, dict]] = []  # (position_in_all_tokens, sp_dict)
            sec_counter = 0

            for seg_type, seg_data in segments:
                if seg_type == "sp":
                    sp_markers.append((len(all_tokens), seg_data))
                else:
                    for tok in seg_data:  # type: ignore[iteration]
                        tok["_sec_id"] = sec_counter
                        all_tokens.append(tok)
                    sec_counter += 1

            sentence = sentences[0] if sentences else ""
            _process_section(
                all_tokens, sentence, force_tone4_high_pitch, high_pitch_threshold,
            )

            # Regroup tokens by _sec_id (new tokens from split inherit from parent)
            section_groups: dict[int, list[dict]] = {}
            for tok in all_tokens:
                sid = tok.get("_sec_id", 0)
                section_groups.setdefault(sid, []).append(tok)
                # Clean up internal tag before output
                tok.pop("_sec_id", None)

            new_segments: list[tuple[str, object, int, bool]] = []
            seg_id_counter = 0
            marker_idx = 0
            for orig_seg_type, _orig_seg_data in segments:
                if orig_seg_type == "sp":
                    new_segments.append(("sp", sp_markers[marker_idx][1], 0, False))
                    marker_idx += 1
                else:
                    orig_count = len(_orig_seg_data)
                    sec_group = section_groups.get(seg_id_counter, [])
                    was_expanded = len(sec_group) > orig_count
                    new_segments.append(("section", sec_group, orig_count, was_expanded))
                    seg_id_counter += 1
        else:
            new_segments: list[tuple[str, object, int, bool]] = []
            for seg_type, seg_data in segments:
                if seg_type == "sp":
                    new_segments.append(("sp", seg_data, 0, False))
                else:
                    orig_token_count = len(seg_data)
                    sentence = sentences[global_sec_idx] if global_sec_idx < len(sentences) else ""
                    global_sec_idx += 1
                    processed = _process_section(
                        seg_data, sentence, force_tone4_high_pitch, high_pitch_threshold,
                    )
                    was_expanded = len(processed) > orig_token_count
                    new_segments.append(("section", processed, orig_token_count, was_expanded))

        # Reconstruct track: emit tokens, removing empties, merging SPs.
        # Empty tokens' duration is either:
        # (a) Redistributed to filled tokens in the same section (when the section
        #     has at least one filled token), OR
        # (b) Absorbed into the nearest <SP> (when the section is completely empty).
        new_track = dict(track)
        all_text: list[str] = []
        all_phoneme: list[str] = []
        all_duration: list[float] = []
        all_pitch: list[int] = []
        all_type: list[int] = []

        pending_sp: dict | None = None
        pending_empty_dur: float = 0.0

        for seg_type, seg_data, orig_token_count, was_expanded in new_segments:
            if seg_type == "sp":
                sp = dict(seg_data)
                if pending_sp is not None:
                    # Consecutive SPs (or SP after an empty section) → merge
                    pending_sp["duration"] += pending_empty_dur + sp["duration"]
                    pending_empty_dur = 0.0
                else:
                    pending_sp = sp
                    pending_sp["duration"] += pending_empty_dur
                    pending_empty_dur = 0.0
            else:
                filled = [t for t in seg_data if t["text"]]
                empty_tokens = [t for t in seg_data if not t["text"]]

                if filled and empty_tokens:
                    # Redistribute empty duration evenly to filled tokens.
                    total_empty = sum(t["duration"] for t in empty_tokens)
                    n = len(filled)
                    base_per = total_empty / n
                    for i in range(n - 1):
                        filled[i]["duration"] += base_per
                    filled[-1]["duration"] += total_empty - base_per * (n - 1)
                    empty_dur = 0.0
                else:
                    empty_dur = sum(t["duration"] for t in seg_data if not t["text"])

                # Enforce minimum duration: boost tokens below 0.30s by borrowing
                # from the longest token in the same section.
                # Only applies when Expand occurred (token was split), because
                # Collapse mode must preserve original durations exactly.
                MIN_DUR = 0.30
                if filled and was_expanded:
                    needs_boost = [
                        (i, MIN_DUR - t["duration"])
                        for i, t in enumerate(filled)
                        if t["duration"] < MIN_DUR
                    ]
                    if needs_boost:
                        longest_idx = max(range(len(filled)),
                                          key=lambda i: filled[i]["duration"])
                        for i, deficit in needs_boost:
                            if (longest_idx != i
                                    and filled[longest_idx]["duration"]
                                    > deficit + MIN_DUR):
                                filled[i]["duration"] = MIN_DUR
                                filled[longest_idx]["duration"] -= deficit

                # --- Non-fixed pause mode: redistribute SP time to tokens ---
                if (not fixed_pause and filled and pending_sp is not None):
                    sec_dur = sum(t["duration"] for t in filled)
                    n_tok = len(filled)
                    avg_tok = sec_dur / n_tok if n_tok > 0 else 0
                    sp_dur = pending_sp["duration"]
                    sp_ratio = sp_dur / avg_tok if avg_tok > 0 else 0
                    # Trigger if SP is >= 2x avg token duration
                    #   OR any token is below minimum duration
                    if sp_ratio >= 2 or avg_tok < 0.30:
                        # Target SP = one average token's duration
                        target_sp = avg_tok
                        if target_sp < sp_dur:
                            freed = sp_dur - target_sp
                            # Distribute freed time proportionally by duration
                            for t in filled:
                                if sec_dur > 0:
                                    t["duration"] += freed * (t["duration"] / sec_dur)
                            pending_sp["duration"] = target_sp

                # Emit pending SP
                if filled and pending_sp is not None:
                    all_text.append(pending_sp["text"])
                    all_phoneme.append(pending_sp["phoneme"])
                    all_duration.append(pending_sp["duration"])
                    all_pitch.append(pending_sp["note_pitch"])
                    all_type.append(pending_sp["note_type"])
                    pending_sp = None
                elif pending_sp is None and filled:
                    pass

                # Completely empty section → absorb into SP
                if not filled:
                    pending_empty_dur += empty_dur

                # Emit filled tokens
                for token in filled:
                    all_text.append(token["text"])
                    all_phoneme.append(token["phoneme"])
                    all_duration.append(token["duration"])
                    all_pitch.append(token["note_pitch"])
                    all_type.append(token["note_type"])

        # Emit trailing SP (or new SP for trailing empty duration)
        if pending_sp is not None:
            pending_sp["duration"] += pending_empty_dur
            all_text.append(pending_sp["text"])
            all_phoneme.append(pending_sp["phoneme"])
            all_duration.append(pending_sp["duration"])
            all_pitch.append(pending_sp["note_pitch"])
            all_type.append(pending_sp["note_type"])
        elif pending_empty_dur > 0:
            # Trailing empty duration with no SP — create one
            all_text.append("<SP>")
            all_phoneme.append("<SP>")
            all_duration.append(pending_empty_dur)
            all_pitch.append(0)
            all_type.append(1)

        new_track["text"] = " ".join(all_text)
        new_track["phoneme"] = " ".join(all_phoneme)
        new_track["duration"] = " ".join(_fmt_durs(all_duration))
        new_track["note_pitch"] = " ".join(str(p) for p in all_pitch)
        new_track["note_type"] = " ".join(str(t) for t in all_type)
        # f0 is NOT rebuilt — it is frame-level data, not token-level.
        # dict(track) already preserves the original f0 from the input.
        result.append(new_track)

    # --- Speed adjustment ---
    if speed != 1.0:
        result = _apply_speed(result, speed)

    return json.dumps(result, ensure_ascii=False, indent=2)


def _apply_speed(midi_data: list, speed: float) -> list:
    """Apply speed change to all tracks: scale durations and resample f0.

    Args:
        midi_data: List of track dicts (already processed by replace_lyrics).
        speed: Speed multiplier (e.g. 1.5 = 150% speed = faster = shorter duration,
               0.5 = 50% speed = slower = longer duration).
               duration_new = duration_orig / speed

    Returns:
        Modified midi_data with scaled durations and resampled f0.
    """
    import numpy as _np

    ratio = 1.0 / speed  # duration scale factor (speed up → ratio < 1 → shorter)

    for track in midi_data:
        # Scale durations
        if "duration" in track:
            dur_vals = [float(x) * ratio for x in track["duration"].split(" ")]
            track["duration"] = " ".join(_fmt_durs(dur_vals))

        # Scale time range (used by downstream to preallocate audio buffer)
        if "time" in track and isinstance(track["time"], list) and len(track["time"]) == 2:
            track["time"] = [round(track["time"][0] * ratio), round(track["time"][1] * ratio)]

        # Resample f0 (frame-level data at ~50fps)
        if "f0" in track and track["f0"].strip():
            f0_vals = [float(x) for x in track["f0"].split(" ")]
            orig_len = len(f0_vals)
            new_len = max(1, round(orig_len * ratio))

            if new_len == orig_len:
                # No change needed
                track["f0"] = " ".join(_fmt_f0(v) for v in f0_vals)
            elif new_len > orig_len:
                # Stretch: linear interpolation
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(_fmt_f0(v) for v in resampled)
            else:
                # Shrink: linear interpolation then take fewer samples
                old_indices = _np.linspace(0, orig_len - 1, orig_len)
                new_indices = _np.linspace(0, orig_len - 1, new_len)
                resampled = _np.interp(new_indices, old_indices, f0_vals)
                track["f0"] = " ".join(_fmt_f0(v) for v in resampled)

    return midi_data


def _fmt_dur(v: float) -> str:
    """Format a duration value, cleaning up float artifacts. Keeps 2 decimal places."""
    return f"{v:.2f}"


def _fmt_durs(durations: list[float]) -> list[str]:
    """Format a list of durations to 2 decimal places, adjusting the last
    element so the rounded total matches the true total."""
    if not durations:
        return []
    true_total = sum(durations)
    rounded = [round(d, 2) for d in durations]
    rounded_total = sum(rounded)
    diff = round(true_total - rounded_total, 2)
    if diff != 0 and rounded:
        rounded[-1] = round(rounded[-1] + diff, 2)
    return [f"{d:.2f}" for d in rounded]


def _fmt_f0(v) -> str:
    """Format an f0 value, cleaning up float artifacts."""
    f = float(v)
    if f == 0.0:
        return "0.0"
    s = f"{f:.1f}".rstrip("0").rstrip(".")
    return s if s else "0"


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


# --- Unified DP-based alignment node (Task 10) ---


def _distribute_lyrics(lyrics: str, tracks: list) -> list[str]:
    """Split a full lyric block across multiple tracks by duration proportion.

    Real-world inputs (e.g. a long vocal track plus a short echo track) feed
    a single lyric string into the alignment node. Feeding that full string
    into every track independently causes catastrophic SPLITs on the small
    track (see P1 in CHANGELOG / plan). This helper partitions the lyric
    along line boundaries so each track receives a share proportional to
    its non-SP token duration.

    Args:
        lyrics: Full lyric string (may contain newlines for sentence splits).
        tracks: Parsed ``Track`` objects (only ``tokens[*].duration`` and
            ``tokens[*].is_sp`` are read).

    Returns:
        A list of per-track lyric substrings (same order as ``tracks``).
        When ``tracks`` has only one entry, the original lyric is returned
        unchanged. When a track ends up with no share (degenerate split),
        its entry is an empty string — callers must preserve such tracks
        verbatim instead of trying to align empty lyrics.
    """
    if len(tracks) <= 1:
        return [lyrics]

    # Capacity = total non-SP duration. Duration (not token count) tracks
    # real acoustic capacity: a 0.4s slot can hold ~2x as many chars as a
    # 0.2s slot regardless of how the source MIDI quantised them.
    capacities = [
        sum(t.duration for t in track.tokens if not t.is_sp)
        for track in tracks
    ]
    total_cap = sum(capacities)
    if total_cap <= 0:
        # Degenerate: no acoustic capacity anywhere. Fall back to giving
        # every track the full lyric so the existing per-track alignment
        # decides what to do (back-compat with all-zero-duration inputs).
        return [lyrics] * len(tracks)

    # Split on newlines but keep non-empty lines. Line boundaries are the
    # natural place to partition — mid-sentence cuts would corrupt meaning
    # and the normalizer relies on line breaks as soft SP placement hints.
    lines = [ln for ln in lyrics.split("\n") if ln.strip()]
    if not lines:
        lines = [lyrics]

    total_chars = sum(len(line) for line in lines)
    result: list[str] = []
    line_idx = 0

    for i in range(len(tracks)):
        if i == len(tracks) - 1:
            # Last track absorbs whatever remains so the entire lyric is
            # always covered (no char is silently dropped).
            result.append("\n".join(lines[line_idx:]))
            break

        # Proportional char budget for this track, snapped to int.
        target_chars = int(total_chars * capacities[i] / total_cap)

        # Accumulate whole lines until the budget is met. A track with
        # capacity 0 (e.g. all-SP) gets target_chars=0 and naturally
        # receives an empty share — ``align_lyrics`` then preserves it
        # verbatim, which is the right behaviour for a no-content track.
        accumulated = 0
        end_idx = line_idx
        while end_idx < len(lines) and accumulated < target_chars:
            accumulated += len(lines[end_idx])
            end_idx += 1

        result.append("\n".join(lines[line_idx:end_idx]))
        line_idx = end_idx

    # Defensive padding: if there were fewer lines than tracks, later
    # tracks get "" and callers preserve them unchanged.
    while len(result) < len(tracks):
        result.append("")
    return result


def _apply_force_tone4(tokens: list, threshold: int = 79) -> list:
    """高音中文音素强制改四声（复用现有 force_tone4 逻辑）.

    RF-3: 当 token 的 note_pitch >= threshold（默认 79=G5）且 phoneme
    以 ``zh_`` 开头、末位为声调数字时，把末位改 4。SP 与非中文音素
    不受影响。threshold 与 nodes.py:803-806 的 ``_apply_char`` 内联逻辑
    保持一致。
    """
    from alignment.models import Token
    result = []
    for t in tokens:
        if (not t.is_sp
                and t.note_pitch >= threshold
                and t.phoneme.startswith(ZH_FLAG)
                and t.phoneme[-1].isdigit()):
            new_phoneme = re.sub(r"(\d)$", "4", t.phoneme)
            result.append(Token(t.text, new_phoneme, t.duration,
                                t.note_pitch, t.note_type, t.index))
        else:
            result.append(t)
    return result


def _split_by_sp(tokens: list) -> list[tuple[list, list]]:
    """把 token 序列按 SP 切分成 segments.

    返回 [(sp_tokens, content_tokens), ...] 列表。
    sp_tokens 是该 segment 的前置 SP token（0~多个）。
    content_tokens 是非 SP token（DP 对齐目标）。
    末尾的 SP token 归入最后一个 segment 的 sp_tokens。
    """
    segments: list[tuple[list, list]] = []
    current_sp: list = []
    current_content: list = []

    for t in tokens:
        if t.is_sp:
            if current_content:
                segments.append((current_sp, current_content))
                current_sp = []
                current_content = []
            current_sp.append(t)
        else:
            current_content.append(t)

    if current_content or current_sp:
        segments.append((current_sp, current_content))
    return segments


def _distribute_lyrics_to_segments(lyrics: str,
                                   segments: list[tuple[list, list]]
                                   ) -> list[str]:
    """把歌词按 section 容量比例分配（jieba 分词，不拆词）。

    用 jieba 分词，按词数比例分配。多字词不跨 section 拆分。
    """
    import jieba

    flat = lyrics.replace("\n", "").replace(" ", "").strip()
    if not flat:
        return [""] * len(segments)

    # jieba 分词
    words = [w for w in jieba.cut(flat) if w.strip()]
    total_words = len(words)

    capacities = [len(content) for _, content in segments]
    total_cap = sum(capacities)
    if total_cap == 0:
        return [""] * len(segments)

    result: list[str] = []
    word_idx = 0

    for i, (_, content) in enumerate(segments):
        cap = len(content)
        if cap == 0:
            result.append("")
            continue

        if i == len(segments) - 1:
            result.append("".join(words[word_idx:]))
            break

        # 按比例分配词数，上限 2×token（SPLIT 限制）
        target = round(total_words * cap / total_cap)
        target = min(target, cap * 2, total_words - word_idx)
        end_idx = min(word_idx + target, len(words))
        result.append("".join(words[word_idx:end_idx]))
        word_idx = end_idx

    while len(result) < len(segments):
        result.append("")
    return result


class MidiLyricsAlignment:
    """统一对齐算法节点（基于联合 DP）.

    替代 MIDIEditLyrics 的场景分支式处理，用单一 DP 求全局最优对齐。
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
            },
            "optional": {
                "w_pitch": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "w_duration": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "w_structure": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("midi_json", "warnings")
    FUNCTION = "align_lyrics"
    CATEGORY = "MIDI"

    def align_lyrics(self, midi_json, lyrics, speed=1.0,
                     normalize_digits=True, force_tone4=False,
                     w_pitch=0.5, w_duration=0.3, w_structure=0.2):
        from alignment import (
            parse_tracks, serialize_tracks, normalize_lyrics, tokenize_units,
            solve_alignment, rebuild_tokens, allocate_durations,
            apply_speed_change, CostWeights,
        )
        from alignment.models import Track

        # 输入校验（内联 str() 替代 plan 原文的 _safe_string — 该 helper
        # 在 nodes.py 中无定义；这里直接处理 None 输入）
        if midi_json is None:
            midi_json = ""
        else:
            midi_json = str(midi_json)
        if lyrics is None:
            lyrics = ""
        else:
            lyrics = str(lyrics)

        try:
            tracks = parse_tracks(midi_json)
        except ValueError as e:
            return (f"Error: {e}", "")

        weights = CostWeights(
            w_pitch=w_pitch, w_duration=w_duration, w_structure=w_structure,
        )

        # P1: partition the full lyric across tracks by duration proportion.
        # Without this, every track is force-fed the entire lyric and small
        # tracks undergo catastrophic SPLIT storms.
        lyrics_per_track = _distribute_lyrics(lyrics, tracks)

        result_tracks = []
        warnings_list = []
        for track_idx, track in enumerate(tracks):
            track_lyrics = (
                lyrics_per_track[track_idx]
                if track_idx < len(lyrics_per_track) else ""
            )

            if not track_lyrics or not track_lyrics.strip():
                result_tracks.append(track)
                continue

            # SP 硬保留：按 SP token 切分成 segments，每段独立 DP。
            # SP token 原样保留（f0/pitch/duration 不动），DP 只在
            # 非 SP token 之间对齐字。这样旋律（f0+note_pitch）与
            # 原 track 完全对应，不会错位。
            segments = _split_by_sp(track.tokens)
            lyrics_per_seg = _distribute_lyrics_to_segments(
                track_lyrics, segments)

            all_new_tokens = []
            for seg_idx, ((sp_tokens, content_tokens), seg_lyrics) in enumerate(
                    zip(segments, lyrics_per_seg)):
                # SP token 原样保留
                all_new_tokens.extend(sp_tokens)

                if not content_tokens or not seg_lyrics.strip():
                    all_new_tokens.extend(content_tokens)
                    continue

                # section 内 DP 对齐（不含 SP unit → sp_target=0）
                try:
                    norm_text, _ = normalize_lyrics(seg_lyrics, 0, normalize_digits)
                    seg_units = tokenize_units(norm_text, [], weights)
                except ValueError as e:
                    return (f"Error: {e}", "")

                try:
                    seg_path = solve_alignment(content_tokens, seg_units, weights)
                except ValueError as e:
                    return (f"Error: {e}", "")

                seg_new = rebuild_tokens(seg_path, content_tokens, weights)
                seg_new = allocate_durations(
                    seg_new, content_tokens, seg_path, weights)

                if force_tone4:
                    seg_new = _apply_force_tone4(seg_new, threshold=79)

                short_count = sum(
                    1 for t in seg_new
                    if not t.is_sp and t.duration < weights.min_duration
                )
                if short_count > 0:
                    warnings_list.append(
                        f"MIN_DURATION_UNRESOLVED(t{track_idx}s{seg_idx}:{short_count})"
                    )

                split_count = sum(1 for o in seg_path.ops if o.kind == "SPLIT")
                drop_count = sum(1 for o in seg_path.ops if o.kind == "DROP")
                if split_count > 0.4 * len(seg_units):
                    warnings_list.append(f"HIGH_SPLIT_RATIO(t{track_idx}s{seg_idx})")
                if drop_count > 0.3 * len(content_tokens):
                    warnings_list.append(f"HIGH_DROP_RATIO(t{track_idx}s{seg_idx})")

                all_new_tokens.extend(seg_new)

            # f0 原样保留（SP 硬保留 → token 位置/顺序与原 track 对应 → f0 对应不变）
            result_track = Track(tokens=all_new_tokens, meta=dict(track.meta),
                                 f0=track.f0)
            result_tracks.append(result_track)

        if speed != 1.0:
            result_tracks = apply_speed_change(result_tracks, speed)

        output_json = serialize_tracks(result_tracks)

        # RF-2: warnings 通过 RETURN_TYPES 的第二个输出返回，供 ComfyUI
        # Web UI 反馈（spec §7.1）。替代原来的 print() 静默输出。
        warnings_str = "; ".join(warnings_list) if warnings_list else ""
        return (output_json, warnings_str)

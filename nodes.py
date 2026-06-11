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


# --- Lyrics sentence splitting ---


_SENTENCE_DELIM_RE = re.compile(r"[\n，。！？；：、,.\!\?;:]+")


def _split_lyrics_to_sentences(new_lyrics: str) -> list[str]:
    """Split user lyrics into sentences by newlines and punctuation.

    Returns a list of cleaned (non-empty) sentence strings.
    """
    raw = _SENTENCE_DELIM_RE.split(new_lyrics)
    return [clean_lyrics(s) for s in raw if clean_lyrics(s)]


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
    """Split plain text (no punctuation) into chunks matching original section sizes.

    Walks through section_sizes, slicing plain_text at each boundary.
    The last chunk gets whatever remains (may be shorter or empty).
    """
    result = []
    pos = 0
    for size in section_sizes:
        if pos >= len(plain_text):
            result.append("")
        else:
            end = min(pos + size, len(plain_text))
            result.append(plain_text[pos:end])
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
                force_tone4: bool, threshold: int) -> None:
    """Replace token text with *char*, regenerate phoneme, optional high-pitch adjustment."""
    phoneme = char_to_phoneme(char)
    if force_tone4 and phoneme.startswith(ZH_FLAG):
        try:
            if int(tokens[idx].get("note_pitch", 0)) >= threshold:
                phoneme = re.sub(r"(\d)$", "4", phoneme)
        except (ValueError, TypeError):
            pass
    tokens[idx]["text"] = char
    tokens[idx]["phoneme"] = phoneme


def _process_section(
    tokens: list[dict], sentence: str,
    force_tone4_high_pitch: bool, high_pitch_threshold: int,
) -> list[dict]:
    """Apply the 3-mode algorithm to a single section.

    Modes:
    - Collapse (N <= S): map chars to slots, expand by repeat count.
    - Token (S < N <= M): 1:1 token mapping.
    - Expand (N > M): split longest-duration tokens until count matches, then 1:1.
    """
    if not tokens:
        return tokens

    if not sentence:
        # No sentence available for this section — empty all text/phoneme
        # (keep token count for timing/alignment).
        for tok in tokens:
            tok["text"] = ""
            tok["phoneme"] = ""
        return tokens

    N = len(sentence)
    M = len(tokens)
    slots = _build_collapsed_slots(tokens)
    S = len(slots)

    if N <= S:
        # ===== Collapse mode =====
        for slot_idx, (_orig_char, _count, token_indices) in enumerate(slots):
            if slot_idx < N:
                new_char = sentence[slot_idx]
                for ti in token_indices:
                    _apply_char(tokens, ti, new_char,
                                force_tone4_high_pitch, high_pitch_threshold)
            else:
                # Remaining slots: empty text/phoneme (don't keep original lyrics),
                # but keep token count to preserve timing and section alignment.
                for ti in token_indices:
                    tokens[ti]["text"] = ""
                    tokens[ti]["phoneme"] = ""
        return tokens

    # ===== Token / Expand mode =====
    # Split longest tokens if needed (only when N > M)
    while len(tokens) < N:
        longest_idx = max(range(len(tokens)), key=lambda i: tokens[i]["duration"])
        _split_token(tokens, longest_idx)

    # 1:1 mapping
    for i in range(min(N, len(tokens))):
        _apply_char(tokens, i, sentence[i],
                    force_tone4_high_pitch, high_pitch_threshold)

    # Empty remaining tokens (don't keep original lyrics)
    for i in range(N, len(tokens)):
        tokens[i]["text"] = ""
        tokens[i]["phoneme"] = ""

    return tokens


# --- Core logic ---


def replace_lyrics(midi_json_str: str, new_lyrics: str,
                    force_tone4_high_pitch: bool = False,
                     high_pitch_threshold: int = 79) -> str:
    """Replace lyrics in MIDI JSON with smart 3-mode algorithm.

    Splits user lyrics by newlines/punctuation into sentences, each mapped to
    a MIDI section (SP-delimited).  Three modes per section:

    - **Collapse** (N <= S): chars map to slots (deduplicated consecutive
      chars), expanded by original repeat count.
    - **Token** (S < N <= M): 1:1 token mapping.
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

    sentences = _split_lyrics_to_sentences(new_lyrics)

    # Fallback: if punctuation/newline split yields fewer sentences than
    # original sections, re-split by matching the original section sizes.
    total_sections = _count_total_sections(midi_data)
    if len(sentences) < total_sections:
        section_sizes = _get_section_sizes(midi_data)
        plain = clean_lyrics(new_lyrics)
        sentences = _split_by_section_sizes(plain, section_sizes)

    # Global sentence counter — persists across tracks so that the second
    # track continues where the first track left off.
    global_sec_idx = 0

    result = []
    for track in midi_data:
        if "text" not in track or "phoneme" not in track:
            result.append(track)
            continue

        # Parse all token arrays from the track
        text_tokens = track["text"].split(" ")
        phoneme_tokens = track["phoneme"].split(" ")
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

            new_segments: list[tuple[str, object]] = []
            seg_id_counter = 0
            marker_idx = 0
            for orig_seg_type, _orig_seg_data in segments:
                if orig_seg_type == "sp":
                    new_segments.append(("sp", sp_markers[marker_idx][1]))
                    marker_idx += 1
                else:
                    new_segments.append(("section", section_groups.get(seg_id_counter, [])))
                    seg_id_counter += 1
        else:
            new_segments: list[tuple[str, object]] = []
            for seg_type, seg_data in segments:
                if seg_type == "sp":
                    new_segments.append(("sp", seg_data))
                else:
                    sentence = sentences[global_sec_idx] if global_sec_idx < len(sentences) else ""
                    global_sec_idx += 1
                    processed = _process_section(
                        seg_data, sentence, force_tone4_high_pitch, high_pitch_threshold,
                    )
                    new_segments.append(("section", processed))

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

        for seg_type, seg_data in new_segments:
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
                    # base_per = total_empty / n_filled for each, remainder to last.
                    total_empty = sum(t["duration"] for t in empty_tokens)
                    n = len(filled)
                    base_per = total_empty / n
                    for i in range(n - 1):
                        filled[i]["duration"] += base_per
                    filled[-1]["duration"] += total_empty - base_per * (n - 1)
                    empty_dur = 0.0  # already redistributed
                else:
                    empty_dur = sum(t["duration"] for t in seg_data if not t["text"])

                # Emit pending SP ONLY if we have filled tokens to follow.
                # If the section is completely empty, the SP stays pending and
                # will merge with the next SP or be emitted at the end.
                if filled and pending_sp is not None:
                    all_text.append(pending_sp["text"])
                    all_phoneme.append(pending_sp["phoneme"])
                    all_duration.append(pending_sp["duration"])
                    all_pitch.append(pending_sp["note_pitch"])
                    all_type.append(pending_sp["note_type"])
                    pending_sp = None
                elif pending_sp is None and filled:
                    pass  # No preceding SP — filled tokens at the start

                # Completely empty section → absorb into SP
                if not filled:
                    pending_empty_dur += empty_dur

                # Emit filled tokens (duration already adjusted if redistributed)
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
        new_track["duration"] = " ".join(str(d) for d in all_duration)
        new_track["note_pitch"] = " ".join(str(p) for p in all_pitch)
        new_track["note_type"] = " ".join(str(t) for t in all_type)
        # f0 is NOT rebuilt — it is frame-level data, not token-level.
        # dict(track) already preserves the original f0 from the input.
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
        "Smart matching: splits lyrics by newlines/punctuation into sentences, "
        "each mapped to a MIDI section. If new sentence is shorter than original, "
        "extra characters keep original text. If longer, automatically splits "
        "the longest-duration note (pitch unchanged, duration halved) to create "
        "more positions. Consecutive repeated characters in original are grouped "
        "(e.g. '向向' → slot with count 2) and expanded accordingly. "
        "Chinese chars → zh_ pinyin; English → en_ phonemes. "
        "Force Tone 4: forces Chinese phonemes at pitch >= threshold to tone 4."
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

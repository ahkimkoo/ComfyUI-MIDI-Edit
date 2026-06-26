# core/edit_algorithm.py
"""MIDIEditLyrics core algorithm: slot-mapping lyrics replacement.

Implements the three-mode (Collapse / Collapse+Distribute / Expand) lyrics
replacement used by the ``MIDIEditLyrics`` ComfyUI node, plus smart sentence
splitting driven by the CT-Transformer punctuation model.
"""
from __future__ import annotations

import json
import math
import re

from core.g2p import (
    char_to_phoneme,
    is_chinese_char,
    normalize_digits,
    word_to_phoneme,
    ZH_FLAG,
)
from core.text_utils import (
    clean_lyrics,
    is_reduplication,
    split_lyrics_to_sentences,
)
from core.ct_transformer import restore_punctuation
from core.speed import apply_speed, format_durations


# --- Unit construction ---


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
    sentence = normalize_digits(sentence)
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
                ph = word_to_phoneme(word)
                units.append({"text": word, "phoneme": ph, "is_word": True})
                i = j
            else:
                i += 1  # skip punctuation
    return units


# --- Section helpers ---


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


# --- Smart sentence splitting (CT-Transformer) ---


_PUNC_SPLIT_RE = re.compile(r'[，。？！；、]')

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
    restore_punc_fn=None,
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

    # Punctuation function is injectable so the nodes.py wrapper (and tests
    # that monkey-patch nodes._restore_punctuation) can substitute it.
    if restore_punc_fn is None:
        restore_punc_fn = restore_punctuation

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
            punctuated = restore_punc_fn(remaining)
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

    sentences = split_lyrics_to_sentences(new_lyrics)

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
            new_segments = []
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
        new_track["duration"] = " ".join(format_durations(all_duration))
        new_track["note_pitch"] = " ".join(str(p) for p in all_pitch)
        new_track["note_type"] = " ".join(str(t) for t in all_type)
        # f0 is NOT rebuilt — it is frame-level data, not token-level.
        # dict(track) already preserves the original f0 from the input.
        result.append(new_track)

    # --- Speed adjustment ---
    if speed != 1.0:
        result = apply_speed(result, speed)

    return json.dumps(result, ensure_ascii=False, indent=2)


def _concat_track_text(midi_json_str: str) -> str:
    """Concatenate the ``text`` field of every track in the MIDI JSON.

    Shared helper for :func:`extract_lyrics`. Returns ``""`` on invalid input
    (bad JSON, non-list payload) or when no track carries text.
    """
    try:
        midi_data = json.loads(midi_json_str)
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(midi_data, list):
        return ""

    all_text = ""
    for track in midi_data:
        text = track.get("text") if isinstance(track, dict) else None
        if text:
            all_text += text
    return all_text


def extract_lyrics(
    midi_json_str: str,
    merge_repeated: bool = False,
    resegment: bool = False,
) -> str:
    """Extract lyrics text from MIDI JSON.

    Default behaviour (*resegment* = False): iterate every track, collect the
    ``text`` field, strip spaces and convert ``<SP>`` markers to newlines.
    When *merge_repeated* is True, consecutive duplicate characters are
    collapsed into one.

    Re-segment pipeline (*resegment* = True): ignore the original ``<SP>``
    phrasing and rebuild natural sentence boundaries. "One sentence per line"
    here means *one punctuation-delimited fragment per line* (every comma /
    period / question-mark etc. starts a new line):

    1. Concatenate text from all tracks (via :func:`_concat_track_text`).
    2. Normalise Arabic digits to Chinese number chars, drop ``<SP>`` markers,
       then keep only Chinese chars + ASCII letters (every space, newline and
       punctuation is removed).
    3. Merge consecutive repeated characters (:func:`merge_repeated_chars`).
    4. Run CT-Transformer (:func:`restore_punctuation`) to add punctuation
       back. The model handles arbitrarily long text internally via
       mini-sentence splitting + a cache, so the whole song can be fed in one
       call.
    5. Split by punctuation (:func:`split_lyrics_to_sentences`) into fragments
       and emit one fragment per line.

    Note: when *resegment* is True the merge step is already part of the
    pipeline (step 3), so *merge_repeated* has no additional effect — it is
    intentionally covered to avoid double processing.
    """
    all_text = _concat_track_text(midi_json_str)
    if not all_text:
        return ""

    if resegment:
        # 2. Normalise digits, drop <SP>, keep only CJK + ASCII letters.
        cleaned = normalize_digits(all_text)
        cleaned = cleaned.replace("<SP>", "")
        cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", cleaned)
        if not cleaned:
            return ""
        # 3. Merge consecutive repeated characters.
        cleaned = merge_repeated_chars(cleaned)
        # 4. CT-Transformer adds punctuation back (，。？…).
        punctuated = restore_punctuation(cleaned)
        # 5-6. Split by every punctuation mark and emit one fragment per line.
        sentences = split_lyrics_to_sentences(punctuated)
        return "\n".join(sentences)

    # Original behaviour: strip spaces, then <SP> → newline.
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

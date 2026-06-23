# alignment/preprocess.py
"""新歌词预处理：归一化 + 单元切分."""
from __future__ import annotations

import re

from alignment.models import Unit, CostWeights
# Self-contained phoneme helpers (g2pM + g2p_en). Imported from the alignment
# subpackage itself rather than the top-level nodes.py to avoid the ComfyUI
# ``sys.modules['nodes']`` naming conflict (ComfyUI core ships its own
# nodes.py without these functions).
from alignment.phoneme import (
    _normalize_digits,
    char_to_phoneme,
    word_to_phoneme as _word_to_phoneme,
)


# 标点强度分类
_STRONG_PUNCT = set("\n。.！!？?")     # 强：换行、句号类
_MEDIAN_PUNCT = set("，,；;：:，")      # 中：逗号类
_DELETE_PUNCT = set("\"'“”‘’（）()[]【】{}〈〉《》«»")  # 删除：引号括号


def normalize_lyrics(text: str, sp_target: int,
                     normalize_digits: bool = True) -> tuple[str, list[int]]:
    """归一化新歌词，提取 SP 候选位置.

    Args:
        text: 用户原始歌词（中英混合，可能含标点/换行）
        sp_target: 目标 SP 数量（= 原 SP 数）
        normalize_digits: 是否把阿拉伯数字转中文

    Returns:
        (normalized_text, sp_candidate_positions)
        sp_candidate_positions: 在 normalized_text 中的字符索引列表（按顺序）
    """
    if not text or not text.strip():
        raise ValueError("lyrics text is empty")

    # 收集标点位置（在原 text 中的位置 + 强度）
    strong_pos: list[int] = []
    median_pos: list[int] = []
    cleaned_chars: list[str] = []
    stripped_offset = 0

    for raw_ch in text:
        if raw_ch in _DELETE_PUNCT:
            continue
        if raw_ch == " " or raw_ch == "\t":
            cleaned_chars.append(" ")
            stripped_offset += 1
            continue
        if raw_ch in _STRONG_PUNCT:
            strong_pos.append(stripped_offset)
            continue
        if raw_ch in _MEDIAN_PUNCT:
            median_pos.append(stripped_offset)
            continue
        cleaned_chars.append(raw_ch)
        stripped_offset += 1

    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if normalize_digits:
        cleaned = _normalize_digits(cleaned)

    sp_positions = _select_sp_candidates(
        strong_pos, median_pos, sp_target, cleaned
    )

    return cleaned, sp_positions


def _english_word_interiors(text: str) -> "set[int]":
    """Return the set of SP-candidate positions that fall INSIDE an English word.

    A position ``p`` is "inside" if it would split a maximal run of ASCII
    letters — i.e. ``p`` is in ``(word_start, word_end]`` for some word
    spanning ``[word_start, word_end]`` (word_end is the inclusive index of
    the last letter). Positions at word boundaries (``p == word_start`` or
    ``p == word_end + 1``) are VALID and not returned.

    Rationale: the tokenizer's en-branch scans a full maximal run of ASCII
    letters as one Unit, advancing ``char_offset`` past every interior
    index. Any SP candidate placed on an interior index is silently
    skipped → SP conservation breaks. Pre-filtering these positions in the
    normalizer avoids the loss.
    """
    invalid: "set[int]" = set()
    i = 0
    n = len(text)
    while i < n:
        if text[i].isascii() and text[i].isalpha():
            word_start = i
            while i < n and text[i].isascii() and text[i].isalpha():
                i += 1
            word_end = i - 1  # inclusive last-letter index
            # Interior positions: (word_start, word_end] = word_start+1 .. word_end
            for p in range(word_start + 1, word_end + 1):
                invalid.add(p)
        else:
            i += 1
    return invalid


def _select_sp_candidates(strong: "list[int]", median: "list[int]",
                          target: int, text: str) -> "list[int]":
    """按强度筛选 SP 候选到 target 个。不足则均匀补充，避免位置冲突.

    优先级：强标点 > 中标点 > 均匀填充。使用 set 去重，并通过
    ``_uniform_sp_fill`` 的 exclude 参数防止填充位置与已有候选重合导致
    最终数量不足 target。

    额外排除：``text`` 中落在英文词内部的位置（见
    ``_english_word_interiors``），防止 tokenizer 的 en-分支扫描整个词
    时静默吞掉 SP 候选。

    RF-8: 该过滤同时应用于 strong/median/uniform 三类候选。场景：
    ``"hello\\nworld"`` 归一化后为 ``"helloworld"``，\\n 的位置 5 落在
    合并词内部，若不过滤会被 tokenizer 的 en-分支吞掉。过滤后若候选
    不足 target，上层 SP_COUNT_REDUCED 警告会触发（比静默丢失好）。
    """
    if target <= 0:
        return []
    # RF-8: 过滤掉落在英文词内部的位置（三类候选统一过滤）。
    invalid = _english_word_interiors(text)
    valid_strong = [p for p in strong if p not in invalid]
    valid_median = [p for p in median if p not in invalid]
    candidates: "set[int]" = set()
    for p in sorted(valid_strong):
        if len(candidates) >= target:
            break
        candidates.add(p)
    for p in sorted(valid_median):
        if len(candidates) >= target:
            break
        candidates.add(p)
    if len(candidates) < target:
        # 合并已有候选 + 英文词内部位置作为排除集，
        # 防止均匀填充与候选重合或落在词内部。
        exclude = invalid | candidates
        candidates.update(
            _uniform_sp_fill(len(text), target - len(candidates), exclude)
        )
    return sorted(candidates)[:target]


def _uniform_sp_fill(text_len: int, count: int,
                     exclude: "set[int] | None" = None) -> list[int]:
    """在文本中均匀分布 count 个 SP 位置，避开 exclude 中已占用的位置.

    当 count > 可用槽数（= text_len + 1 - len(exclude)）时（文本太短），
    填充所有可用间隙（物理上限），由调用方决定是否触发 SP_COUNT_REDUCED
    警告。返回的位置绝不与 exclude 重合，且彼此不重复。

    旧实现的 bug：当 count >= text_len 时，``step = text_len / (count + 1)``
    会让多个 i 映射到同一 base，碰撞搜索在槽位耗尽后回退到 base 自身，
    产生重复位置；上层 ``sorted(set(candidates))`` 再去重，导致最终 SP
    数量少于 target 且无任何信号 —— SP 守恒被悄悄打破。
    """
    if count <= 0 or text_len < 0:
        return []
    exclude = set(exclude) if exclude else set()
    # 可用位置范围 [0, text_len]，扣除 exclude
    all_positions = [p for p in range(text_len + 1) if p not in exclude]
    if len(all_positions) <= count:
        # 槽位不够：返回全部可用位置（物理上限）。
        # 调用方据此判断是否需要 SP_COUNT_REDUCED 警告。
        return all_positions
    # 槽位充足：在 all_positions 上均匀采样，索引严格递增故无重复
    step = len(all_positions) / (count + 1)
    return [all_positions[int(step * (i + 1))] for i in range(count)]


# CJK Unicode 范围（中日韩统一表意文字 + 扩展 A）
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
)


def _is_cjk(ch: str) -> bool:
    """字符是否属于 CJK 统一表意文字范围."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_ascii_letter(ch: str) -> bool:
    """字符是否为 ASCII 字母（用于英文单词边界判定）."""
    return ch.isascii() and ch.isalpha()


def tokenize_units(text: str, sp_positions: list[int],
                   weights: CostWeights) -> list[Unit]:
    """把归一化文本切分为 Unit 列表（jieba 分词）.

    中文用 jieba 分词：多字词（如"全程"）作为一个 Unit，
    text=完整词、phoneme=组合拼音（zh_quan2-cheng2）。
    单字仍为独立 Unit。这样当词数 ≤ token 数时每词占一 token（自然），
    词数 > token 数时 SPLIT 只发生在单字词上。

    英文连续字母一词，SP 位置插入 sp 单元。
    """
    import jieba

    units: list[Unit] = []
    sp_set = set(sp_positions)
    char_offset = 0

    while char_offset < len(text):
        while char_offset in sp_set:
            units.append(Unit(
                text="<SP>", phoneme="<SP>", kind="sp",
                max_occupy=1, source="punct",
            ))
            sp_set.discard(char_offset)

        if char_offset >= len(text):
            break

        ch = text[char_offset]
        if ch == " ":
            char_offset += 1
            continue

        if _is_cjk(ch):
            # 收集连续 CJK 字符段，jieba 整体分词
            cjk_start = char_offset
            while char_offset < len(text) and _is_cjk(text[char_offset]) and char_offset not in sp_set:
                char_offset += 1
            cjk_text = text[cjk_start:char_offset]

            for word in jieba.cut(cjk_text):
                if not word or word.strip() == "":
                    continue
                if len(word) == 1:
                    units.append(Unit(
                        text=word, phoneme=char_to_phoneme(word),
                        kind="zh", max_occupy=1,
                    ))
                else:
                    # 多字词：组合拼音，压缩到一 token
                    parts = []
                    for c in word:
                        p = char_to_phoneme(c)
                        parts.append(p.replace("zh_", "") if p.startswith("zh_") else p)
                    combined = "zh_" + "-".join(parts)
                    units.append(Unit(
                        text=word, phoneme=combined,
                        kind="zh", max_occupy=1,
                    ))
        elif _is_ascii_letter(ch):
            word_start = char_offset
            while char_offset < len(text) and _is_ascii_letter(text[char_offset]):
                char_offset += 1
            word = text[word_start:char_offset]
            ph = _word_to_phoneme(word)
            max_occ = max(1, min(len(word), weights.max_word_occupy))
            units.append(Unit(
                text=word, phoneme=ph, kind="en", max_occupy=max_occ,
            ))
        else:
            char_offset += 1

    while char_offset in sp_set:
        units.append(Unit(
            text="<SP>", phoneme="<SP>", kind="sp",
            max_occupy=1, source="punct",
        ))
        char_offset += 1

    return units

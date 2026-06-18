# alignment/preprocess.py
"""新歌词预处理：归一化 + 单元切分."""
from __future__ import annotations

import re

from alignment.models import Unit, CostWeights

try:
    # 复用 nodes.py 的 _normalize_digits（数字转中文）。
    # 注意：nodes.py 顶层 import g2pM，未安装该依赖时会 ModuleNotFoundError，
    # 此时回退到内联实现（_normalize_digits 是纯函数，仅依赖 _DIGIT_TO_ZH）。
    from nodes import _normalize_digits
except ImportError:
    # g2pM 未安装时回退到内联实现（_normalize_digits 是纯函数）。
    # 内联自 nodes.py:84-91，保持映射完全一致。
    _DIGIT_TO_ZH = str.maketrans("0123456789", "零一二三四五六七八九")

    def _normalize_digits(text: str) -> str:
        """Convert Arabic digits to Chinese number chars so they get proper phonemes."""
        if not text:
            return text
        return text.translate(_DIGIT_TO_ZH)


# tokenize_units 依赖 nodes.py 的拼音/英文音素函数（g2pM + g2p_en）。
# 这两个函数无法内联（依赖模型权重），必须在 comfyui 环境运行。
from nodes import char_to_phoneme, _word_to_phoneme


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
        strong_pos, median_pos, sp_target, len(cleaned)
    )

    return cleaned, sp_positions


def _select_sp_candidates(strong: list[int], median: list[int],
                          target: int, text_len: int) -> list[int]:
    """按强度筛选 SP 候选到 target 个。不足则均匀补充，避免位置冲突.

    优先级：强标点 > 中标点 > 均匀填充。使用 set 去重，并通过 _uniform_sp_fill
    的 exclude 参数防止填充位置与已有候选重合导致最终数量不足 target。
    """
    if target <= 0:
        return []
    candidates: set[int] = set()
    for p in sorted(strong):
        if len(candidates) >= target:
            break
        candidates.add(p)
    for p in sorted(median):
        if len(candidates) >= target:
            break
        candidates.add(p)
    if len(candidates) < target:
        candidates.update(
            _uniform_sp_fill(text_len, target - len(candidates), candidates)
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
    """把归一化文本切分为 Unit 列表.

    中文一字一 zh 单元，英文连续字母一 en 单元，SP 位置插入 sp 单元。
    保持文本顺序，SP 单元穿插在对应字符位置。

    Args:
        text: normalize_lyrics 返回的归一化文本
        sp_positions: SP 候选位置（归一化文本中的字符索引）
        weights: 代价权重（用 weights.max_word_occupy 截断英文词长度）

    Returns:
        按文本顺序排列的 Unit 列表（zh / en / sp 穿插）
    """
    units: list[Unit] = []
    sp_set = set(sp_positions)
    char_offset = 0

    while char_offset < len(text):
        # 当前位置需要插入 SP：先消费所有连续 SP（按 sp_positions 顺序）
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
            units.append(Unit(
                text=ch, phoneme=char_to_phoneme(ch), kind="zh", max_occupy=1,
            ))
            char_offset += 1
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
            # 其他字符（理论上 normalize_lyrics 已过滤，保险起见跳过）
            char_offset += 1

    # 文本末尾的 SP（sp_positions 可包含 == len(text) 的位置）
    while char_offset in sp_set:
        units.append(Unit(
            text="<SP>", phoneme="<SP>", kind="sp",
            max_occupy=1, source="punct",
        ))
        char_offset += 1

    return units

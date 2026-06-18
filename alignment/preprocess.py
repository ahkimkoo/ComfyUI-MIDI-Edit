# alignment/preprocess.py
"""新歌词预处理：归一化 + 单元切分."""
from __future__ import annotations

import re

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

    当均匀分布点与 exclude 冲突时，向两侧寻找最近可用位置，保证返回
    恰好 count 个不重复位置（只要 text_len + 1 个槽位足够）。
    """
    if count <= 0 or text_len <= 0:
        return []
    exclude = set(exclude) if exclude else set()
    result: list[int] = []
    step = text_len / (count + 1)
    for i in range(count):
        base = int(step * (i + 1))
        pos = base
        if pos in exclude or pos in result:
            for delta in range(text_len + 1):
                picked = None
                for cand in (base + delta, base - delta):
                    if 0 <= cand <= text_len and cand not in exclude and cand not in result:
                        picked = cand
                        break
                if picked is not None:
                    pos = picked
                    break
        result.append(pos)
        exclude.add(pos)
    return result

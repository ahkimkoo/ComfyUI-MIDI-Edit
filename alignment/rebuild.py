# alignment/rebuild.py
"""对齐路径 → 新 token 序列 + duration 分配."""
from __future__ import annotations
from alignment.models import Token, AlignmentPath, CostWeights


def rebuild_tokens(path: AlignmentPath, original_tokens: list[Token],
                   weights: CostWeights) -> list[Token]:
    """把对齐路径翻译为新 token 序列.

    依据 spec §6.3 的操作→token 映射表。

    ``weights`` 当前供 Task 8 的 ``allocate_durations`` 使用，此处保留入参
    以保证 rebuild.py 的公开 API 稳定。
    """
    new_tokens: list[Token] = []
    next_index = 0

    for op in path.ops:
        if op.kind == "REPLACE":
            orig = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=orig.duration,
                note_pitch=orig.note_pitch,
                note_type=orig.note_type,
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "WORD_SPAN":
            # 一个英文词横跨多个 token：首 token 词首(type=2)，其余词内(type=3)
            for k_pos, tidx in enumerate(op.token_indices):
                orig = original_tokens[tidx]
                new_tokens.append(Token(
                    text=op.unit.text,
                    phoneme=op.unit.phoneme,
                    duration=orig.duration,
                    note_pitch=orig.note_pitch,
                    note_type=2 if k_pos == 0 else 3,
                    index=next_index,
                ))
                next_index += 1

        elif op.kind == "SPLIT":
            # 一个原 token 容纳多个字：每个字复用宿主 token 的音高/类型
            host = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=host.duration,
                note_pitch=host.note_pitch,
                note_type=host.note_type,
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "DROP":
            # 多余的原 token 被丢弃，不产生任何新 token
            pass

        elif op.kind == "SP_ALIGN":
            # 休止对齐：生成 <SP> token，音高 0、类型 1（段尾）
            orig = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text="<SP>",
                phoneme="<SP>",
                duration=orig.duration,
                note_pitch=0,
                note_type=1,
                index=next_index,
            ))
            next_index += 1

    return new_tokens


def _find_sections(tokens: list[Token]) -> list[tuple[int, int]]:
    """识别 section（相邻 SP 之间的 token 区间）.

    SP token 本身不计入任何 section；连续 SP 之间的非空 token 区间
    以 ``[start, end)`` 半开区间返回。首尾的 SP 不会产生空 section。
    """
    sections = []
    start = 0
    for i, t in enumerate(tokens):
        if t.is_sp:
            if i > start:
                sections.append((start, i))
            start = i + 1
    if start < len(tokens):
        sections.append((start, len(tokens)))
    return sections

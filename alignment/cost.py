# alignment/cost.py
"""DP 操作代价的纯函数（P/D/S 三项）.

每个函数返回 op_cost = w_p*P + w_d*D + w_s*S。
依据 spec §5.4 的代价映射表。
"""
from __future__ import annotations
from alignment.models import Token, Unit, CostWeights


def replace_cost(token: Token, unit: Unit, w: CostWeights) -> float:
    """REPLACE: 1 zh 单元 ↔ 1 token。pitch/duration 全继承，代价 0."""
    return 0.0


def word_span_cost(span: list[Token], unit: Unit, w: CostWeights) -> float:
    """WORD_SPAN: 1 en 单元 ↔ k token。各 token 继承，轻微失衡惩罚."""
    k = len(span)
    # 英文词的理想占用 ≈ 音素数；偏离则有轻微结构惩罚
    ideal = unit.max_occupy
    imbalance = abs(k - ideal) / max(ideal, 1)
    S = imbalance * 0.1  # 轻微惩罚
    return w.w_structure * S


def split_cost(host: Token, unit: Unit, w: CostWeights,
               current_share_count: int = 0) -> float:
    """SPLIT: zh 单元共享宿主 token。duration 被切细 → min_duration 惩罚.

    宿主已被前序操作（REPLACE/WORD_SPAN/SP_ALIGN）消费，含 1 个初始消费者。
    current_share_count: 宿主上 prior SPLIT 次数（不含初始消费者）。
    本次 SPLIT 后总 unit 数 = current_share_count + 2。
    """
    q = current_share_count + 2  # 1 初始消费者 + current_share_count prior + 1 本次
    est_dur = host.duration / q
    D = w.lambda_min_dur * max(0.0, w.min_duration - est_dur)
    return w.w_duration * D


def drop_cost(token: Token, all_tokens: list[Token], idx: int,
              w: CostWeights) -> float:
    """DROP: 原 token 被丢弃。pitch 损失 = 与最近保留 token 的 pitch 差."""
    # 找最近的非丢弃 token（启发式：用相邻 token）
    nearest_pitch = 0
    for offset in (1, -1, 2, -2):
        ni = idx + offset
        if 0 <= ni < len(all_tokens) and not all_tokens[ni].is_sp:
            nearest_pitch = all_tokens[ni].note_pitch
            break
    P = abs(token.note_pitch - nearest_pitch) if not token.is_sp else 0.0
    return w.w_pitch * P


def sp_align_cost(token: Token, unit: Unit, new_pos: int,
                  orig_sp_positions: list[int], w: CostWeights) -> float:
    """SP_ALIGN: 把 token 变成 SP。位置移动惩罚 + 覆盖原歌词的 pitch 损失."""
    # S: 位置移动量（找最近的原 SP 位置）
    if orig_sp_positions:
        min_dist = min(abs(new_pos - p) for p in orig_sp_positions)
    else:
        min_dist = 0
    S = float(min_dist)

    # P: 若 token 原本非 SP，其 pitch 信息丢失
    P = float(token.note_pitch) if not token.is_sp else 0.0

    return w.w_structure * S + w.w_pitch * P

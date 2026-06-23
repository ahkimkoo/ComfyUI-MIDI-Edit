# alignment/dp.py
"""联合动态规划对齐求解器.

状态 f(i, j, s, c) = 已处理 T[:i] 与 U[:j]、SP_ALIGN s 次、
                     c∈{0,1} 表示 t_{i-1} 是否可被 SPLIT 共享 的最小代价。

5 种操作：REPLACE / WORD_SPAN / SPLIT / DROP / SP_ALIGN
依据 spec §5。
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from alignment.models import (
    Token, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)


@dataclass
class _DPCell:
    """DP 表的一个条目."""
    cost: float = math.inf
    back: tuple | None = None  # (prev_i, prev_j, prev_s, prev_c, op)


def _effective_k(unit: Unit, remaining_tokens: int, w: CostWeights) -> int:
    """英文词的有效占用上限（动态放宽，spec §8.1）."""
    word_len = len(unit.text)
    return max(1, min(word_len, remaining_tokens, w.max_word_occupy))


def solve_alignment(tokens: list[Token], units: list[Unit],
                    weights: CostWeights) -> AlignmentPath:
    """求解最优对齐路径.

    Args:
        tokens: 原 token 序列
        units: 新单元序列（含 SP 候选）
        weights: 代价权重

    Returns:
        AlignmentPath: 最优对齐路径

    Raises:
        ValueError: 无可行路径
    """
    m, n = len(tokens), len(units)
    sp_target = sum(1 for u in units if u.kind == "sp")
    orig_sp_positions = [i for i, t in enumerate(tokens) if t.is_sp]

    # DP 表：dp[i][j][s][c]
    # c ∈ {0, 1, ..., MAX_SPLIT+1}：
    #   0 = 初始或刚 DROP（禁止 SPLIT）
    #   k ≥ 1 = 当前 token 已被 k-1 次 SPLIT + 1 次 REPLACE 消费（共 k 字）
    #   SPLIT cost 随 k 递增（per-char duration 越来越短）
    MAX_SPLIT = 8  # 单 token 最多容纳 8 字（DP 表上限）
    dp = [[[[_DPCell() for _ in range(MAX_SPLIT + 2)]
            for _ in range(sp_target + 1)]
           for _ in range(n + 1)]
          for _ in range(m + 1)]
    dp[0][0][0][0].cost = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            for s in range(sp_target + 1):
                for c in range(MAX_SPLIT + 2):
                    cur = dp[i][j][s][c]
                    if cur.cost == math.inf:
                        continue
                    C = cur.cost
                    u = units[j] if j < n else None

                    # --- DROP ---
                    if i < m:
                        cost_d = drop_cost(tokens[i], tokens, i, weights)
                        _relax(dp, i + 1, j, s, 0, C + cost_d,
                               (i, j, s, c),
                               AlignmentOp("DROP", None, (i,), cost_d))

                    if u is None:
                        continue

                    # --- REPLACE (zh 占 1 token) ---
                    if u.kind == "zh" and i < m:
                        prev_tok = tokens[i - 1] if i > 0 else None
                        cost_r = replace_cost(tokens[i], u, weights, prev_tok)
                        _relax(dp, i + 1, j + 1, s, 1, C + cost_r,
                               (i, j, s, c),
                               AlignmentOp("REPLACE", u, (i,), cost_r))

                    # --- WORD_SPAN (en 占 k token) ---
                    if u.kind == "en" and i < m:
                        k_max = _effective_k(u, m - i, weights)
                        for k in range(1, k_max + 1):
                            span = tokens[i:i + k]
                            if any(t.is_sp for t in span):
                                break
                            cost_w = word_span_cost(span, u, weights)
                            _relax(dp, i + k, j + 1, s, 1, C + cost_w,
                                   (i, j, s, c),
                                   AlignmentOp("WORD_SPAN", u,
                                               tuple(range(i, i + k)), cost_w))

                    # --- SPLIT (zh 共享 t_{i-1}) ---
                    # c=0 禁止（刚 DROP），c ≥ 1 允许，c 越大 cost 越高
                    if u.kind == "zh" and c >= 1 and i >= 1 and c <= MAX_SPLIT:
                        cost_sp = split_cost(tokens[i - 1], u, weights,
                                             current_share_count=c - 1)
                        _relax(dp, i, j + 1, s, c + 1, C + cost_sp,
                               (i, j, s, c),
                               AlignmentOp("SPLIT", u, (i - 1,), cost_sp))

                    # --- SP_ALIGN (sp 占 1 token，计数 +1) ---
                    # SP 硬保留：只在原 SP token 位置对齐。
                    # 不允许把非 SP token 变成 SP——否则字会落在原 SP 位置
                    # （f0=0），SoulX-Singer 不唱 → 漏唱。
                    if u.kind == "sp" and i < m and s < sp_target and tokens[i].is_sp:
                        cost_sa = sp_align_cost(
                            tokens[i], u, i, orig_sp_positions, weights
                        )
                        _relax(dp, i + 1, j + 1, s + 1, 1, C + cost_sa,
                               (i, j, s, c),
                               AlignmentOp("SP_ALIGN", u, (i,), cost_sa))

    # --- 找终态 ---
    best_c = min(range(MAX_SPLIT + 2), key=lambda c: dp[m][n][sp_target][c].cost)
    final = dp[m][n][sp_target][best_c]
    if final.cost == math.inf:
        raise ValueError(
            f"No valid alignment path (m={m}, n={n}, sp_target={sp_target})"
        )

    return _reconstruct(dp, m, n, sp_target, best_c)


def _relax(dp, i, j, s, c, new_cost, prev_state, op):
    """松弛操作：若 new_cost 更小则更新."""
    cell = dp[i][j][s][c]
    if new_cost < cell.cost:
        cell.cost = new_cost
        cell.back = (*prev_state, op)


def _reconstruct(dp, m, n, sp_target, best_c) -> AlignmentPath:
    """回溯重建对齐路径."""
    ops: list[AlignmentOp] = []
    i, j, s, c = m, n, sp_target, best_c
    while (i, j, s, c) != (0, 0, 0, 0):
        cell = dp[i][j][s][c]
        if cell.back is None:
            break
        pi, pj, ps, pc, op = cell.back
        ops.append(op)
        i, j, s, c = pi, pj, ps, pc
    ops.reverse()

    sp_placements = [op.token_indices[0] for op in ops if op.kind == "SP_ALIGN"]
    total = dp[m][n][sp_target][best_c].cost
    return AlignmentPath(ops=ops, total_cost=total, sp_placements=sp_placements)

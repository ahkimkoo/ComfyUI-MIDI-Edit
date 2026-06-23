# alignment/rebuild.py
"""对齐路径 → 新 token 序列 + duration 分配."""
from __future__ import annotations
from alignment.models import Token, Unit, AlignmentPath, CostWeights
from alignment.phoneme import is_reduplication


def _note_type(unit: Unit, is_continuation: bool = False) -> int:
    """根据单元在乐句与词中的角色，决定 note_type。

    定义：
      1 = 段落/乐句收尾音（SP 位置或长音结尾）
      2 = 普通音符 / 词首音符
      3 = 延续音符（英文词跨多音符时的后续，共享同一音素）

    判断依据是 token 的**实际角色**，不是继承原值也不是写死：
    - SP 单元 → 1（段尾标记）
    - 英文词 + 非首音符 → 3（延续，与前序音符共享同一词的音素）
    - 其他（中文字、英文词首）→ 2（独立普通音符）

    中文字即使是 SPLIT（共享宿主 token），各字 phoneme 不同，
    不满足 type=3 的"共享同一音素"前提，故为 2。
    """
    if unit.kind == "sp":
        return 1
    if unit.kind == "en" and is_continuation:
        return 3
    return 2


def rebuild_tokens(path: AlignmentPath, original_tokens: list[Token],
                   weights: CostWeights) -> list[Token]:
    """把对齐路径翻译为新 token 序列.

    依据 spec §6.3 的操作→token 映射表。

    ``weights`` 当前供 Task 8 的 ``allocate_durations`` 使用，此处保留入参
    以保证 rebuild.py 的公开 API 稳定。
    """
    new_tokens: list[Token] = []
    next_index = 0

    def _prev_non_sp_text() -> str:
        """返回前一个非 SP 输出 token 的 text（用于重复字检测）。"""
        if new_tokens and not new_tokens[-1].is_sp:
            return new_tokens[-1].text
        return ""

    def _is_repeat_continuation(unit_text: str) -> bool:
        """连续相同字且非叠词 → type=3（延续音）。

        叠词（哥哥/妹妹等）不算延续，两字都独立演唱（type=2）。
        """
        prev = _prev_non_sp_text()
        return (prev == unit_text and not is_reduplication(unit_text, prev))

    for op in path.ops:
        if op.kind == "REPLACE":
            orig = original_tokens[op.token_indices[0]]
            # 连续相同字的第二个 → type=3（共享同一字/音素，是延续音）
            is_repeat = _is_repeat_continuation(op.unit.text)
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=orig.duration,
                note_pitch=orig.note_pitch,
                note_type=_note_type(op.unit, is_continuation=is_repeat),
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "WORD_SPAN":
            # 英文词横跨 k 个 token：首音符 = 词首(2)，后续 = 延续(3)
            for k_pos, tidx in enumerate(op.token_indices):
                orig = original_tokens[tidx]
                new_tokens.append(Token(
                    text=op.unit.text,
                    phoneme=op.unit.phoneme,
                    duration=orig.duration,
                    note_pitch=orig.note_pitch,
                    note_type=_note_type(op.unit, is_continuation=(k_pos > 0)),
                    index=next_index,
                ))
                next_index += 1

        elif op.kind == "SPLIT":
            # 一个原 token 容纳多个字：每个字复用宿主 token 的音高
            host = original_tokens[op.token_indices[0]]
            is_repeat = _is_repeat_continuation(op.unit.text)
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=host.duration,
                note_pitch=host.note_pitch,
                note_type=_note_type(op.unit, is_continuation=is_repeat),
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "DROP":
            # 多余的原 token 被丢弃，不产生任何新 token
            pass

        elif op.kind == "SP_ALIGN":
            # 休止对齐：生成 <SP> token
            orig = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text="<SP>",
                phoneme="<SP>",
                duration=orig.duration,
                note_pitch=0,
                note_type=_note_type(op.unit),
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


# ---------------------------------------------------------------------------
# DurationAllocator（spec §6.4 四阶段）
# ---------------------------------------------------------------------------


def allocate_durations(new_tokens: list[Token], original_tokens: list[Token],
                       path: AlignmentPath, weights: CostWeights) -> list[Token]:
    """分配 duration（spec §6.4 四阶段）.

    约束：① 总时长守恒；② 非 SP ≥ min_duration；③ 浮点清理。
    返回更新了 duration 的新 token 列表（不修改原对象）。

    四阶段：
      1. SPLIT 区间内 duration 均分（初始消费者 + N 个 SPLIT 共享同一 host）
      2. DROP 的 duration 按比例转移给同 section 已填 token
      3. min_duration 下限保护（从同 section 最长 token 借时长）
      4. 浮点清理（保留 2 位小数，尾项兜底保证总和不变）
    """
    # 阶段 1 预处理：统计 SPLIT 共享 + 收集 DROP duration
    split_groups = _group_splits(path, original_tokens)
    drop_durations = _collect_drops(path, original_tokens)

    # 浅拷贝 new_tokens（Token 是 frozen dataclass，后续整体替换而非原地修改）
    result = [Token(t.text, t.phoneme, t.duration, t.note_pitch,
                    t.note_type, t.index) for t in new_tokens]

    # 阶段 1 细化：SPLIT 区间内 duration 均分
    # 实现：遍历 path.ops，构建 host_idx → [result_idx, ...] 映射。
    # DROP 不产出 token（跳过），WORD_SPAN 每个位置都消费一个独立 token
    # （使用各自原 duration，不参与 host 共享），仅 REPLACE/SP_ALIGN/SPLIT
    # 共享 host_idx[0] 的 duration。
    if split_groups:
        host_to_result_idxs: dict[int, list[int]] = {}
        ridx = 0
        for op in path.ops:
            if op.kind == "DROP":
                continue
            elif op.kind in ("REPLACE", "SP_ALIGN", "SPLIT"):
                host_idx = op.token_indices[0]
                host_to_result_idxs.setdefault(host_idx, []).append(ridx)
                ridx += 1
            elif op.kind == "WORD_SPAN":
                # 词内每个 token 使用各自原 duration，不参与 host 共享
                ridx += len(op.token_indices)

        for host_idx, share_count in split_groups.items():
            result_indices = host_to_result_idxs.get(host_idx, [])
            if not result_indices:
                continue
            host_dur = original_tokens[host_idx].duration
            # 合法 DP path 中：len(result_indices) == share_count + 1
            # （初始消费者 REPLACE/SP_ALIGN + N 个 SPLIT）。用实际长度更稳妥。
            per_share = host_dur / len(result_indices)
            for ri in result_indices:
                t = result[ri]
                result[ri] = Token(t.text, t.phoneme, per_share,
                                   t.note_pitch, t.note_type, t.index)

    # 阶段 2：DROP duration 转移给同 section 已填 token（原地修改 result）
    _redistribute_drops(result, drop_durations, weights)

    # 阶段 3：min_duration 下限保护（从同 section 最长 token 借）
    _enforce_min_duration(result, weights)

    # 阶段 4：浮点清理（保留 2 位，尾项兜底，保证总和不变）
    durs = [t.duration for t in result]
    formatted = _fmt_durs_inline(durs)
    for i, t in enumerate(result):
        result[i] = Token(t.text, t.phoneme, float(formatted[i]),
                          t.note_pitch, t.note_type, t.index)

    return result


def _group_splits(path: AlignmentPath, original_tokens: list[Token]) -> dict[int, int]:
    """统计每个宿主 token 被 SPLIT 共享的次数."""
    counts: dict[int, int] = {}
    for op in path.ops:
        if op.kind == "SPLIT":
            host_idx = op.token_indices[0]
            counts[host_idx] = counts.get(host_idx, 0) + 1
    return counts


def _collect_drops(path: AlignmentPath, original_tokens: list[Token]) -> list[float]:
    """收集所有 DROP 的 duration."""
    return [original_tokens[op.token_indices[0]].duration
            for op in path.ops if op.kind == "DROP"]


def _redistribute_drops(tokens: list[Token], drop_durations: list[float],
                        weights: CostWeights) -> None:
    """把 DROP 的 duration 均匀分配给所有已填 token（原地修改）.

    均匀分配：每个非 SP token 分到 ``total_drop / 字数``。
    保证总 duration 守恒，且避免"富者愈富"（按比例分配时原 duration
    长的 token 分到更多，短的仍短）。
    """
    if not drop_durations or not tokens:
        return
    total_drop = sum(drop_durations)
    filled = [(i, tokens[i]) for i in range(len(tokens)) if not tokens[i].is_sp]
    if not filled:
        return
    per = total_drop / len(filled)
    for i, t in filled:
        tokens[i] = Token(t.text, t.phoneme, t.duration + per,
                          t.note_pitch, t.note_type, t.index)


def _enforce_min_duration(tokens: list[Token], weights: CostWeights) -> None:
    """非 SP token 不低于 min_duration，从同 section 最长 token 借（原地修改）.

    迭代式补偿：每轮找到第一个过短 token 和最长可借 token，搬移差额。
    借出方不会被压到 min_duration 以下。最多迭代 10 轮防死循环。
    """
    sections = _find_sections(tokens)
    for start, end in sections:
        for _ in range(10):
            short = [(i, tokens[i]) for i in range(start, end)
                     if not tokens[i].is_sp and tokens[i].duration < weights.min_duration]
            if not short:
                break
            candidates = [(i, tokens[i]) for i in range(start, end)
                          if not tokens[i].is_sp
                          and tokens[i].duration > weights.min_duration + 0.01]
            if not candidates:
                break
            longest_idx, longest = max(candidates, key=lambda x: x[1].duration)
            si, st = short[0]
            need = weights.min_duration - st.duration
            longest_new = longest.duration - need
            if longest_new < weights.min_duration:
                # 不能让借出方也跌破下限：clamp
                need = longest.duration - weights.min_duration
                longest_new = weights.min_duration
            tokens[si] = Token(st.text, st.phoneme,
                               st.duration + need,
                               st.note_pitch, st.note_type, st.index)
            tokens[longest_idx] = Token(longest.text, longest.phoneme,
                                        longest_new,
                                        longest.note_pitch, longest.note_type,
                                        longest.index)


def _fmt_durs_inline(durations: list[float]) -> list[str]:
    """内联的 duration 格式化（避免循环导入 nodes._fmt_durs）.

    保留 2 位小数；累计舍入误差由末项吸收，保证 round 后总和 == 真实总和。
    """
    if not durations:
        return []
    true_total = sum(durations)
    rounded = [round(d, 2) for d in durations]
    rounded_total = sum(rounded)
    diff = round(true_total - rounded_total, 2)
    if diff != 0 and rounded:
        rounded[-1] = round(rounded[-1] + diff, 2)
    return [f"{d:.2f}" for d in rounded]


def rebuild_f0(new_tokens: list[Token], orig_f0_frame_count: int,
               orig_total_duration: float) -> str:
    """根据新 token 的 note_pitch 重建 f0 序列。

    当 DP 重排了 token（REPLACE 继承不同原 token 的 pitch，DROP/SPLIT 改变
    duration 分配），原 f0 序列与新 token 的 note_pitch 不再对应。如果原样
    保留 f0，SoulX-Singer 合成时 pitch 与 f0 冲突 → 唱不出来。

    重建策略：每个新 token 的 f0 段为其 note_pitch 对应的平直频率
    （MIDI → Hz）。帧率与原 f0 一致，总帧数 ≈ 原 f0 帧数（总时长守恒）。

    这丢失了原 f0 的颤音/滑音等表现力，但保证 pitch 与 f0 一致。
    后续可用原 f0 重采样方案恢复表现力。
    """
    if orig_f0_frame_count <= 0 or orig_total_duration <= 0:
        return ""

    fps = orig_f0_frame_count / orig_total_duration
    new_total_dur = sum(t.duration for t in new_tokens)
    target_frames = max(1, round(new_total_dur * fps))

    f0_vals: list[float] = []
    for t in new_tokens:
        n = max(1, round(t.duration * fps))
        if t.note_pitch > 0:
            freq = 440.0 * (2.0 ** ((t.note_pitch - 69) / 12.0))
        else:
            freq = 0.0
        f0_vals.extend([round(freq, 1)] * n)

    # 末尾校正：确保总帧数与 target 一致（round 误差补偿）
    while len(f0_vals) < target_frames:
        f0_vals.append(f0_vals[-1] if f0_vals else 0.0)
    while len(f0_vals) > target_frames:
        f0_vals.pop()

    return " ".join(f"{v:.1f}" for v in f0_vals)

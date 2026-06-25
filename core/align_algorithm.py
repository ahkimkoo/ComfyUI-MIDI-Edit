# core/align_algorithm.py
"""MidiLyricsAlignment v3 core algorithm: sequential mapping + greedy packing.

Design: ``docs/superpowers/specs/2026-06-24-alignment-v3-design.md``.

Core ideas:
  1. Original SP tokens are all removed; rebuild ``[SP] sent1 [SP] ... [SP]``
     from the new lyrics' sentence boundaries.
  2. New chars map onto the original non-SP tokens (the melodic source).
  3. chars <= non-SP tokens -> 1:1 sequential mapping, surplus tokens dropped.
  4. chars > non-SP tokens -> greedy packing, multi-char words (jieba) share
     the longest tokens.
  5. f0 is sliced per 50fps segment, SP fills zeros, SPLIT slices by char count.
  6. SPD formula decides the new SP duration (total duration is not conserved).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from core.midi_format import Token, FPS
from core.g2p import (
    char_to_phoneme,
    is_chinese_char,
    normalize_digits,
    word_to_phoneme,
    ZH_FLAG,
)
from core.text_utils import is_reduplication

# SPLIT 后单字最低 duration 保障(spec §3.5 / §5)。
MIN_SPLIT_DUR = 0.1
# 无原始 SP 时的默认 AVG/MAX(spec §5)。
DEFAULT_SP_DUR = 0.3
# force_tone4 触发阈值。
TONE4_THRESHOLD = 79

# 断句标点。
_SENTENCE_PUNCT_RE = re.compile(r"[。！？，、；：\n]")


# ---------------------------------------------------------------------------
# Data structures (alignment-specific)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """预处理后的对齐单元."""

    text: str
    phoneme: str
    kind: Literal["zh", "en", "sp"]
    max_occupy: int                                       # zh=1, en≤K, sp=1
    source: Literal["lyric", "punct", "orig_sp"] = "lyric"


@dataclass(frozen=True)
class AlignmentOp:
    """DP 转移的原子操作."""

    kind: Literal["REPLACE", "WORD_SPAN", "SPLIT", "DROP", "SP_ALIGN"]
    unit: Unit | None
    token_indices: tuple[int, ...]
    op_cost: float


@dataclass
class AlignmentPath:
    """完整的对齐路径."""

    ops: list[AlignmentOp]
    total_cost: float
    sp_placements: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class CostWeights:
    """代价函数权重配置."""

    w_pitch: float = 0.5
    w_duration: float = 0.3
    w_structure: float = 0.2
    min_duration: float = 0.30
    lambda_min_dur: float = 5.0
    mu_word_boundary: float = 10.0
    max_word_occupy: int = 4
    w_pitch_cont: float = 0.05  # pitch 连贯性：相邻字 pitch 差异惩罚


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def segment_sentences(lyrics: str, target_count: int = 0,
                      punctuate_fn=None) -> list[str]:
    """智能断句：先按标点，超过 10 字的用 CT-Transformer 标点模型找切分点。

    Args:
        punctuate_fn: 可选，接收纯文本返回带标点的文本（CT-Transformer）。
                      若为 None 则退化为在中间硬切。
    """
    parts = _SENTENCE_PUNCT_RE.split(lyrics or "")
    sentences = [s.strip() for s in parts if s.strip()]
    if not sentences:
        return [lyrics.strip()] if (lyrics and lyrics.strip()) else []

    MAX_SENTENCE_LEN = 10
    result = []
    for s in sentences:
        if len(s) <= MAX_SENTENCE_LEN:
            result.append(s)
        else:
            result.extend(_split_long_sentence(s, MAX_SENTENCE_LEN, punctuate_fn))
    return result


def _split_long_sentence(text: str, max_len: int,
                         punctuate_fn=None) -> list[str]:
    """对超过 max_len 的句子用 CT-Transformer 加标点后按标点切分。

    CT-Transformer 一次输出多个标点，直接全切，不需要递归。
    """
    if len(text) <= max_len:
        return [text]

    if punctuate_fn is not None:
        try:
            punctuated = punctuate_fn(text)
            # 按所有标点切分
            pieces = re.split(r'[，。！？；：、]', punctuated)
            pieces = [p.strip() for p in pieces if p.strip()]
            if len(pieces) > 1:
                return pieces
        except Exception:
            pass

    # CT-Transformer 不可用或没有标点 → 原句保留
    return [text]


def _cut_at_punctuation(text: str) -> list[str] | None:
    """在标点处切成两段，优先句号 > 逗号。"""
    mid = len(text) / 2.0
    for pattern in (re.compile(r'[。！？]'), re.compile(r'[，、；]')):
        best_pos = None
        best_dist = float('inf')
        for m in pattern.finditer(text):
            d = abs(m.start() - mid)
            if d < best_dist:
                best_dist = d
                best_pos = m.start()
        if best_pos is not None:
            left = re.sub(r'[，。！？；：、,.!?;:]', '', text[:best_pos]).strip()
            right = re.sub(r'[，。！？；：、,.!?;:]', '', text[best_pos+1:]).strip()
            if left and right:
                return [left, right]
    return None


def calculate_spd(orig_sp_durations: list[float], orig_total: int,
                  new_total: int) -> float:
    """SPD = AVG(orig_sp_durations) * (orig_total / new_total)。

    限制 0.1 <= SPD <= MAX(orig_sp_durations)；无原始 SP 时用默认 0.3。
    """
    if orig_sp_durations:
        avg = sum(orig_sp_durations) / len(orig_sp_durations)
        mx = max(orig_sp_durations)
    else:
        avg = DEFAULT_SP_DUR
        mx = DEFAULT_SP_DUR
    ratio = orig_total / new_total if new_total > 0 else 1.0
    spd = avg * ratio
    return max(0.1, min(spd, mx))


def align_track(track, lyrics_text: str, weights, normalize_digits_flag: bool,
                force_tone4: bool, punctuate_fn=None) -> tuple:
    """主入口：把新歌词对齐到单个 track。

    Args:
        track: core.midi_format.Track。
        lyrics_text: 该 track 分到的新歌词(可能含多句/标点/换行)。
        weights: CostWeights(v3 基本不用，保留入参以稳定 API)。
        normalize_digits_flag: 是否把阿拉伯数字转中文数字字。
        force_tone4: 是否对高音中文音素强制改四声。
        punctuate_fn: 可选的 CT-Transformer 标点函数，用于智能断句。

    Returns:
        (新 Track, warnings: list[str])。
    """
    warnings: list[str] = []

    # ---- 1. 分离原 SP / 非 SP token ----
    orig_tokens = track.tokens
    orig_sp_tokens = [t for t in orig_tokens if t.is_sp]
    orig_nonsl_tokens = [t for t in orig_tokens if not t.is_sp]
    M = len(orig_tokens)

    # ---- 2. 归一化 + 断句 ----
    text = lyrics_text or ""
    if normalize_digits_flag:
        text = normalize_digits(text)
    sentences = segment_sentences(text, punctuate_fn=punctuate_fn)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        raise ValueError("empty lyrics after normalization")

    # ---- 3. 构建新单元结构 [SP] 句1 [SP] 句2 ... [SP] ----
    char_units: list[dict] = []
    new_units: list[dict] = []
    new_units.append({"kind": "sp"})
    for sent in sentences:
        for u in _build_units(sent):
            char_units.append(u)
            new_units.append({"kind": "char", "unit": u})
        new_units.append({"kind": "sp"})

    num_new_sp = len(sentences) + 1
    C = len(char_units)
    N = num_new_sp + C

    # ---- 4. SPD ----
    orig_sp_durations = [t.duration for t in orig_sp_tokens]
    spd = calculate_spd(orig_sp_durations, M, N)

    # ---- 5. 切分原 f0(含原 SP 段) ----
    f0_vals = _parse_f0(track.f0)
    all_segments = _segment_f0(f0_vals, orig_tokens, FPS)
    # 非 SP token 对应的 f0 段(旋律来源)；丢弃原 SP 段。
    nonsl_segments = [
        all_segments[i] for i, t in enumerate(orig_tokens) if not t.is_sp
    ]
    nonsl_count = len(orig_nonsl_tokens)

    # ---- 6. 计算贪心压缩方案 pack[token_idx] = 该 token 承载的字数 ----
    pack = _compute_pack(char_units, orig_nonsl_tokens)
    # char_assignment[char_idx] = (token_idx, pos_within_token)
    char_assignment: list[tuple[int, int]] = []
    for tok_idx, cnt in enumerate(pack):
        for pos in range(cnt):
            char_assignment.append((tok_idx, pos))
    # 兜底：若 pack 总和 < C(异常)，把剩余字挂到最后一个 token。
    if len(char_assignment) < C and nonsl_count > 0:
        last = nonsl_count - 1
        while len(char_assignment) < C:
            char_assignment.append((last, pack[last]))
            pack[last] += 1

    # ---- 7. 每组 SPLIT 的 duration(含 0.1 下限保障) ----
    group_durations: dict[int, list[float]] = {}
    min_unresolved = False
    for tok_idx, cnt in enumerate(pack):
        if cnt <= 0:
            continue
        src = orig_nonsl_tokens[tok_idx]
        per = [src.duration / cnt] * cnt
        per = _enforce_min_split(per)
        if any(v < MIN_SPLIT_DUR - 1e-9 for v in per):
            min_unresolved = True
        group_durations[tok_idx] = per

    # 高压缩告警
    if nonsl_count > 0 and C > nonsl_count and (C - nonsl_count) > 0.4 * nonsl_count:
        warnings.append("HIGH_SPLIT_RATIO(chars=%d,slots=%d)" % (C, nonsl_count))
    if min_unresolved:
        warnings.append("MIN_DURATION_UNRESOLVED")

    # ---- 8. 构建新 token + f0 ----
    new_tokens: list[Token] = []
    new_f0: list[float] = []
    char_ptr = 0
    prev_non_sp_text = ""

    for unit in new_units:
        if unit["kind"] == "sp":
            new_tokens.append(Token(
                text="<SP>", phoneme="<SP>", duration=spd,
                note_pitch=0, note_type=1, index=len(new_tokens),
            ))
            new_f0.extend([0.0] * max(0, round(spd * FPS)))
        else:
            cu = char_units[char_ptr]
            tok_idx, pos = char_assignment[char_ptr]
            src = orig_nonsl_tokens[tok_idx]
            durs = group_durations.get(tok_idx, [src.duration])
            dur = durs[pos] if pos < len(durs) else src.duration
            seg = nonsl_segments[tok_idx] if tok_idx < len(nonsl_segments) else []
            cnt = pack[tok_idx]
            if cnt <= 1:
                char_f0 = list(seg)
            else:
                char_f0 = _slice_segment(seg, cnt, pos)
            nt = _note_type(cu, prev_non_sp_text)
            new_tokens.append(Token(
                text=cu["text"], phoneme=cu["phoneme"], duration=dur,
                note_pitch=src.note_pitch, note_type=nt, index=len(new_tokens),
            ))
            new_f0.extend(char_f0)
            prev_non_sp_text = cu["text"]
            char_ptr += 1

    # ---- 9. force_tone4 后处理 ----
    if force_tone4:
        new_tokens = _apply_force_tone4(new_tokens, TONE4_THRESHOLD)

    # ---- 10. 组装 Track(更新 time) ----
    # time 必须与 f0 帧数精确对齐：SoulX-Singer 按 time 预分配音频缓冲区，
    # 按帧数生成音频。两者不匹配会导致 "could not broadcast" 错误。
    # 用实际 f0 帧数反算 time，消除 sum(round(d*50)) ≠ round(sum(d)*50) 的累积误差。
    meta = dict(track.meta)
    meta["time"] = [0, round(len(new_f0) / FPS * 1000)]
    result_track = type(track)(
        tokens=new_tokens, meta=meta, f0=_format_f0(new_f0),
    )
    return result_track, warnings


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _build_units(sentence: str) -> list[dict]:
    """把句子解析为单元列表：中文字各一单元，英文词各一单元。

    与 core.edit_algorithm._build_units 行为一致，但用 core.g2p 的函数。
    """
    units: list[dict] = []
    if not sentence:
        return units
    i = 0
    n = len(sentence)
    while i < n:
        c = sentence[i]
        if is_chinese_char(c):
            units.append({
                "text": c, "phoneme": char_to_phoneme(c), "is_word": False,
            })
            i += 1
        elif c.isascii() and c.isalpha():
            j = i
            while j < n and sentence[j].isascii() and sentence[j].isalpha():
                j += 1
            word = sentence[i:j]
            units.append({
                "text": word, "phoneme": word_to_phoneme(word), "is_word": True,
            })
            i = j
        else:
            # 空格/标点等跳过(断句已处理主要标点)。
            i += 1
    return units


def _unit_spans(units: list[dict]) -> list[tuple[int, int]]:
    """返回 [(unit_start_idx, length_in_units)]。

    连续中文单元按 jieba 分词成组；英文单元各自长度 1。
    用于贪心压缩时识别"多字词"。
    """
    import jieba  # lazy: avoids loading jieba's dictionary at package import

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        if units[i]["is_word"]:
            spans.append((i, 1))
            i += 1
            continue
        # 收集连续中文单元。
        j = i
        while j < n and not units[j]["is_word"]:
            j += 1
        run = "".join(units[k]["text"] for k in range(i, j))
        words = [w for w in jieba.cut(run) if w]
        pos = i
        for w in words:
            wl = len(w)
            spans.append((pos, wl))
            pos += wl
        i = j
    return spans


def _word_len_at(spans: list[tuple[int, int]], idx: int) -> int:
    """返回从 idx 开始的词长度(单位：单元)。无词从此开始则返回 1。"""
    for start, length in spans:
        if start == idx:
            return max(1, length)
    return 1


def _compute_pack(char_units: list[dict],
                  nonsl_tokens: list) -> list[int]:
    """决定每个非 SP token 承载多少字。pack[t] = 字数，sum(pack) = len(char_units)。

    分配原则（效果优先）：
    - 字数 <= 非 SP token 数：前 C 个 token 各 1 字，其余丢弃。
    - 字数 > 非 SP token 数：按 duration 比例分配。
      每 token 承载 round(D / target_per_char) 字，其中 target = ΣD / C。
      长 token 多分（它们分得起），短 token 少分（不低于 1 字）。
      这样每字至少 ~target 秒，不会出现 0.03s 的灾难。
    """
    nonsl = len(nonsl_tokens)
    C = len(char_units)
    if C <= nonsl:
        return [1] * C + [0] * (nonsl - C)

    total_dur = sum(t.duration for t in nonsl_tokens)
    target = total_dur / C  # 每字目标 duration

    # 初始分配：每 token round(D/target) 字，至少 1
    pack = []
    for t in nonsl_tokens:
        n = max(1, round(t.duration / target))
        pack.append(n)

    # 调整：sum(pack) 可能 != C，需要增减
    diff = C - sum(pack)
    if diff > 0:
        # 需要增加：给 duration 最长的 token 加（它们分得起）
        order = sorted(range(nonsl),
                       key=lambda k: nonsl_tokens[k].duration / pack[k],
                       reverse=True)
        for k in range(diff):
            pack[order[k % nonsl]] += 1
    elif diff < 0:
        # 需要减少：从 duration 最短的 token 减（保持至少 1）
        order = sorted(range(nonsl),
                       key=lambda k: nonsl_tokens[k].duration / pack[k])
        remaining = -diff
        for k in range(nonsl):
            if remaining <= 0:
                break
            reduce = min(pack[order[k]] - 1, remaining)
            pack[order[k]] -= reduce
            remaining -= reduce

    return pack


def _parse_f0(f0_str: str) -> list[float]:
    if not f0_str or not f0_str.strip():
        return []
    out = []
    for tok in f0_str.split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(0.0)
    return out


def _segment_f0(f0_vals: list[float], orig_tokens: list, fps: int) -> list[list[float]]:
    """按 orig_tokens 的 duration * fps 把 f0 切段，每段对应一个原 token。"""
    segments: list[list[float]] = []
    pos = 0
    total = len(f0_vals)
    for t in orig_tokens:
        nframes = max(0, round(t.duration * fps))
        end = min(pos + nframes, total)
        segments.append(f0_vals[pos:end])
        pos = end
    # 舍入误差产生的尾部帧并入最后一段。
    if pos < total and segments:
        segments[-1] = segments[-1] + f0_vals[pos:]
    return segments


def _slice_segment(seg: list[float], cnt: int, pos: int) -> list[float]:
    """把 f0 段尽可能均分成 cnt 份，取第 pos 份(直接切，不插值)。"""
    n = len(seg)
    if cnt <= 1 or n == 0:
        return list(seg)
    base = n // cnt
    rem = n % cnt
    # 前 rem 份各 base+1 帧，其余各 base 帧。
    start = pos * base + min(pos, rem)
    end = start + base + (1 if pos < rem else 0)
    return seg[start:end]


def _note_type(unit: dict, prev_text: str) -> int:
    """v3 note_type: SP=1(由调用方处理)；重复字(非叠词)=3；其余=2。"""
    text = unit["text"]
    # 仅单字参与重复检测(英文词整词独立)。
    if len(text) == 1 and prev_text == text and not is_reduplication(text, prev_text):
        return 3
    return 2


def _enforce_min_split(durations: list[float],
                       min_dur: float = MIN_SPLIT_DUR) -> list[float]:
    """SPLIT 组内：把 < min_dur 的字抬到 min_dur，从同组更宽裕的字匀出。"""
    d = list(durations)
    if len(d) <= 1:
        return d
    for _ in range(20):
        short = [i for i, v in enumerate(d) if v < min_dur - 1e-9]
        if not short:
            break
        progressed = False
        for si in short:
            need = min_dur - d[si]
            if need <= 1e-9:
                continue
            donors = [i for i, v in enumerate(d)
                      if i != si and v > min_dur + 1e-9]
            if not donors:
                continue
            donor = max(donors, key=lambda i: d[i])
            give = min(need, d[donor] - min_dur)
            if give <= 1e-9:
                continue
            d[si] += give
            d[donor] -= give
            progressed = True
        if not progressed:
            break
    return d


def _apply_force_tone4(tokens: list, threshold: int = TONE4_THRESHOLD) -> list:
    """高音(note_pitch >= threshold)中文音素末位声调改 4。SP/低音/非中文不动。"""
    result = []
    for t in tokens:
        if (not t.is_sp and t.note_pitch >= threshold
                and t.phoneme.startswith(ZH_FLAG)
                and t.phoneme[-1:].isdigit()):
            new_ph = re.sub(r"(\d)$", "4", t.phoneme)
            result.append(Token(
                t.text, new_ph, t.duration, t.note_pitch, t.note_type, t.index,
            ))
        else:
            result.append(t)
    return result


def _format_f0(vals: list[float]) -> str:
    """格式化 f0 帧：0.0 用 "0.0"，其余去尾零(与 core.speed.format_f0 一致)。"""
    out = []
    for v in vals:
        f = float(v)
        if f == 0.0:
            out.append("0.0")
        else:
            s = f"{f:.1f}".rstrip("0").rstrip(".")
            out.append(s if s else "0")
    return " ".join(out)

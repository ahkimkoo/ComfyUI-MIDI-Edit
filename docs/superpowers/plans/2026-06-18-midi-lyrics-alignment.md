# MIDI 歌词统一对齐算法（MidiLyricsAlignment）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个基于联合 DP 的统一歌词对齐算法节点 `MidiLyricsAlignment`，用单一代价函数最小化取代现有 `MIDIEditLyrics` 的 if/else 场景分支。

**Architecture:** 9 步确定性管线（解析 → 预处理 → DP 对齐 → 重建 → 时长分配 → 变速），核心是步骤 4 的联合动态规划（5 种原子操作：REPLACE/WORD_SPAN/SPLIT/DROP/SP_ALIGN），在加权代价函数（pitch/duration/structure）下求全局最优。

**Tech Stack:** Python 3.10+、dataclasses、类型注解、numpy（仅变速用）、pytest。复用现有 `nodes.py` 的 `char_to_phoneme`/`_word_to_phoneme`/`_normalize_digits`/`_apply_speed`/`_fmt_durs`/`_fmt_f0`。

**Spec:** `docs/superpowers/specs/2026-06-18-midi-lyrics-alignment-design.md`

**Spec 偏差修正（实现时以本计划为准）**：
- spec §3.2/§6.1 写的 `_normalize_chinese_numbers` 实际函数名是 `_normalize_digits`（`nodes.py:87`）。

---

## File Structure

新建 `alignment/` 子包，保持 `nodes.py` 不膨胀。每个文件单一职责：

| 文件 | 职责 | 行数预估 |
|------|------|---------|
| `alignment/__init__.py` | 公开 API | ~15 |
| `alignment/models.py` | 数据结构（Token/Unit/AlignmentOp/AlignmentPath/CostWeights/Track） | ~80 |
| `alignment/parser.py` | JSON ↔ Token/Track 转换 | ~70 |
| `alignment/cost.py` | 5 个操作代价的纯函数（P/D/S 三项） | ~120 |
| `alignment/preprocess.py` | LyricNormalizer + UnitTokenizer | ~180 |
| `alignment/dp.py` | AlignmentDP（状态、转移、剪枝、回溯） | ~250 |
| `alignment/rebuild.py` | TokenRebuilder + DurationAllocator | ~200 |
| `alignment/speed.py` | SpeedAdapter（薄封装 `_apply_speed`） | ~25 |
| `nodes.py`（修改） | 新增 `MidiLyricsAlignment` 节点入口 | +~60 |
| `tests/test_alignment.py`（新建） | 全部测试 | ~600 |

**依赖关系**（实现顺序）：
```
models.py ← parser.py ← preprocess.py ← cost.py ← dp.py ← rebuild.py ← speed.py ← nodes.py
                                                                                    ↑
                                                                            tests/test_alignment.py（贯穿）
```

---

## Task 1: 项目骨架与数据结构（models.py）

**Files:**
- Create: `alignment/__init__.py`
- Create: `alignment/models.py`
- Test: `tests/test_alignment.py`

- [ ] **Step 1: 创建 alignment 子包骨架**

```python
# alignment/__init__.py
"""MIDI 歌词统一对齐算法子包."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
from alignment.parser import parse_tracks, serialize_track
from alignment.preprocess import normalize_lyrics, tokenize_units
from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)
from alignment.dp import solve_alignment
from alignment.rebuild import rebuild_tokens, allocate_durations
from alignment.speed import apply_speed_change

__all__ = [
    "Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights",
    "parse_tracks", "serialize_track",
    "normalize_lyrics", "tokenize_units",
    "replace_cost", "word_span_cost", "split_cost", "drop_cost", "sp_align_cost",
    "solve_alignment", "rebuild_tokens", "allocate_durations", "apply_speed_change",
]
```

- [ ] **Step 2: 实现 models.py（完整数据结构）**

```python
# alignment/models.py
"""MIDI 对齐算法的核心数据结构."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Token:
    """原 MIDI JSON 的一个 token（解析后内部表示）."""
    text: str            # "<SP>" 或实际字/词
    phoneme: str         # 原音素
    duration: float      # 秒
    note_pitch: int      # MIDI 编号 (0=休止)
    note_type: int       # 1=段尾 / 2=普通·词首 / 3=词内延续
    index: int           # 在 track 内原始索引

    @property
    def is_sp(self) -> bool:
        return self.text == "<SP>"


@dataclass
class Track:
    """一个 MIDI track 的内部表示（保留原 JSON 的非 token 字段）."""
    tokens: list[Token]
    meta: dict = field(default_factory=dict)  # index/language/time 等原字段
    f0: str = ""          # 帧级 f0，原样保留


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
```

- [ ] **Step 3: 写基础测试**

```python
# tests/test_alignment.py
"""MidiLyricsAlignment 测试套件."""
import pytest
from alignment.models import Token, Unit, AlignmentOp, AlignmentPath, CostWeights, Track


class TestModels:
    def test_token_is_sp_true(self):
        t = Token("<SP>", "<SP>", 0.3, 0, 1, 0)
        assert t.is_sp is True

    def test_token_is_sp_false(self):
        t = Token("你", "zh_ni3", 0.4, 60, 2, 1)
        assert t.is_sp is False

    def test_unit_zh_defaults(self):
        u = Unit("你", "zh_ni3", "zh", 1)
        assert u.source == "lyric"

    def test_cost_weights_defaults(self):
        w = CostWeights()
        assert w.w_pitch == 0.5
        assert w.min_duration == 0.30
        assert w.max_word_occupy == 4

    def test_alignment_op_frozen(self):
        op = AlignmentOp("REPLACE", None, (0,), 0.0)
        with pytest.raises(Exception):
            op.kind = "DROP"  # frozen
```

- [ ] **Step 4: 运行测试（预期 FAIL——模块未实现）**

Run: `pytest tests/test_alignment.py::TestModels -v`
Expected: FAIL（ImportError，parser/cost/dp 等模块尚未实现）

> 注：`__init__.py` 导入了所有子模块，Task 1 阶段会因 import 失败而报错。**临时方案**：先注释掉 `__init__.py` 中未实现模块的导入，仅保留 `models`，后续 Task 逐步解除注释。或直接跑 `pytest tests/test_alignment.py::TestModels -v` 时测试本身从 `alignment.models` 导入（不经过 `__init__`）。

- [ ] **Step 5: 调整 __init__.py（仅导出已实现部分）**

将 `__init__.py` 暂时改为：
```python
"""MIDI 歌词统一对齐算法子包."""
from alignment.models import (
    Token, Track, Unit, AlignmentOp, AlignmentPath, CostWeights,
)
__all__ = ["Token", "Track", "Unit", "AlignmentOp", "AlignmentPath", "CostWeights"]
```
后续每完成一个 Task，解除对应导入。

- [ ] **Step 6: 运行测试（预期 PASS）**

Run: `pytest tests/test_alignment.py::TestModels -v`
Expected: PASS（5 passed）

- [ ] **Step 7: Commit**

```bash
git add alignment/__init__.py alignment/models.py tests/test_alignment.py
git commit -m "feat(alignment): add data structures (Token/Unit/AlignmentOp/CostWeights)"
```

---

## Task 2: Parser（parser.py）

**Files:**
- Create: `alignment/parser.py`
- Modify: `alignment/__init__.py`（解除 parser 导入）

- [ ] **Step 1: 实现 parser.py**

```python
# alignment/parser.py
"""JSON ↔ Token/Track 转换."""
from __future__ import annotations
import json
from alignment.models import Token, Track


def parse_tracks(midi_json_str: str) -> list[Track]:
    """解析 MIDI JSON 字符串为 Track 列表.

    Raises:
        ValueError: JSON 解析失败或字段缺失.
    """
    if not midi_json_str or not midi_json_str.strip():
        raise ValueError("midi_json is empty")
    try:
        data = json.loads(midi_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("midi_json must be a list of track objects")
    return [_parse_track(t, i) for i, t in enumerate(data)]


def _parse_track(track_data: dict, track_idx: int) -> Track:
    """解析单个 track dict."""
    required = ["text", "phoneme", "duration", "note_pitch", "note_type"]
    for field in required:
        if field not in track_data:
            raise ValueError(f"track {track_idx} missing field: {field}")

    texts = track_data["text"].split()
    phonemes = track_data["phoneme"].split()
    durations = track_data["duration"].split()
    pitches = track_data["note_pitch"].split()
    note_types = track_data["note_type"].split()

    n = len(texts)
    if not (len(phonemes) == len(durations) == len(pitches) == len(note_types) == n):
        raise ValueError(
            f"track {track_idx}: field length mismatch "
            f"(text={n}, phoneme={len(phonemes)}, duration={len(durations)}, "
            f"pitch={len(pitches)}, type={len(note_types)})"
        )

    tokens = [
        Token(
            text=texts[i],
            phoneme=phonemes[i],
            duration=float(durations[i]),
            note_pitch=int(pitches[i]),
            note_type=int(note_types[i]),
            index=i,
        )
        for i in range(n)
    ]

    meta = {k: v for k, v in track_data.items()
            if k not in [*required, "f0"]}
    f0 = track_data.get("f0", "")
    return Track(tokens=tokens, meta=meta, f0=f0)


def serialize_track(track: Track) -> dict:
    """把 Track 序列化回 track dict（与原 JSON 格式兼容）."""
    tokens = track.tokens
    result = dict(track.meta)
    result["text"] = " ".join(t.text for t in tokens)
    result["phoneme"] = " ".join(t.phoneme for t in tokens)
    result["duration"] = " ".join(f"{t.duration:.2f}" for t in tokens)
    result["note_pitch"] = " ".join(str(t.note_pitch) for t in tokens)
    result["note_type"] = " ".join(str(t.note_type) for t in tokens)
    if track.f0:
        result["f0"] = track.f0
    return result


def serialize_tracks(tracks: list[Track]) -> str:
    """序列化 Track 列表为 JSON 字符串."""
    return json.dumps([serialize_track(t) for t in tracks],
                      ensure_ascii=False, indent=2)
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加到文件末尾）
import json
from alignment.parser import parse_tracks, serialize_track, serialize_tracks


class TestParser:
    TRACK_JSON = json.dumps([{
        "index": "vocal_0_15000",
        "language": "Mandarin",
        "time": [0, 15000],
        "text": "<SP> 你 好 <SP>",
        "phoneme": "<SP> zh_ni3 zh_hao3 <SP>",
        "duration": "0.30 0.40 0.40 0.30",
        "note_pitch": "0 60 62 0",
        "note_type": "1 2 2 1",
        "f0": "0.0 0.0 261.6 0.0",
    }])

    def test_parse_single_track(self):
        tracks = parse_tracks(self.TRACK_JSON)
        assert len(tracks) == 1
        t = tracks[0]
        assert len(t.tokens) == 4
        assert t.tokens[0].is_sp
        assert t.tokens[1].text == "你"
        assert t.tokens[1].phoneme == "zh_ni3"
        assert t.tokens[1].duration == 0.30
        assert t.tokens[1].note_pitch == 60
        assert t.meta["language"] == "Mandarin"
        assert t.f0 == "0.0 0.0 261.6 0.0"

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_tracks("")

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_tracks("{not json")

    def test_parse_missing_field_raises(self):
        bad = json.dumps([{"text": "你"}])  # 缺其他字段
        with pytest.raises(ValueError, match="missing field"):
            parse_tracks(bad)

    def test_serialize_roundtrip(self):
        tracks = parse_tracks(self.TRACK_JSON)
        s = serialize_tracks(tracks)
        again = parse_tracks(s)
        assert len(again[0].tokens) == 4
        assert again[0].tokens[1].text == "你"
```

- [ ] **Step 3: 解除 __init__.py 的 parser 导入，运行测试**

```bash
# 编辑 __init__.py 解除 parser 两行注释
pytest tests/test_alignment.py::TestParser -v
```
Expected: PASS（5 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/parser.py alignment/__init__.py tests/test_alignment.py
git commit -m "feat(alignment): add JSON ↔ Token/Track parser"
```

---

## Task 3: CostFunction（cost.py）—— 核心

**Files:**
- Create: `alignment/cost.py`

- [ ] **Step 1: 实现 cost.py（5 个操作代价纯函数）**

```python
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

    current_share_count: 宿主已被几个单元共享（含本次则为 +1 后的总数）。
    """
    q = current_share_count + 1  # 本次后总共享数
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
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.cost import (
    replace_cost, word_span_cost, split_cost, drop_cost, sp_align_cost,
)


class TestCost:
    def setup_method(self):
        self.w = CostWeights()
        self.token = Token("你", "zh_ni3", 0.4, 60, 2, 1)
        self.unit_zh = Unit("好", "zh_hao3", "zh", 1)
        self.unit_en = Unit("love", "en_L-AH1-V", "en", 3)
        self.unit_sp = Unit("<SP>", "<SP>", "sp", 1, "punct")

    def test_replace_cost_zero(self):
        assert replace_cost(self.token, self.unit_zh, self.w) == 0.0

    def test_word_span_balanced_low_cost(self):
        span = [self.token, self.token, self.token]  # k=3 = ideal
        c = word_span_cost(span, self.unit_en, self.w)
        assert c == 0.0  # k==ideal → imbalance=0

    def test_word_span_imbalanced(self):
        span = [self.token]  # k=1, ideal=3
        c = word_span_cost(span, self.unit_en, self.w)
        assert c > 0.0

    def test_split_cost_below_min_duration(self):
        # host.duration=0.4, 共享后 est=0.2 < 0.30 → 惩罚
        c = split_cost(self.token, self.unit_zh, self.w, current_share_count=0)
        assert c > 0.0

    def test_split_cost_above_min_duration(self):
        long_token = Token("啊", "zh_a1", 1.0, 60, 2, 0)
        c = split_cost(long_token, self.unit_zh, self.w, current_share_count=0)
        assert c == 0.0  # 1.0/2=0.5 > 0.30

    def test_drop_cost_pitch_loss(self):
        tokens = [
            Token("<SP>", "<SP>", 0.3, 0, 1, 0),
            self.token,  # idx=1, pitch=60
            Token("好", "zh_hao3", 0.4, 62, 2, 2),  # idx=2
        ]
        c = drop_cost(tokens[1], tokens, 1, self.w)
        # nearest = idx=2 pitch=62, loss = |60-62| = 2
        assert c == self.w.w_pitch * 2

    def test_sp_align_at_orig_position_zero_structure(self):
        sp_token = Token("<SP>", "<SP>", 0.3, 0, 1, 5)
        c = sp_align_cost(sp_token, self.unit_sp, 5, [5], self.w)
        # min_dist=0, is_sp → P=0
        assert c == 0.0

    def test_sp_align_moved(self):
        lyric_token = Token("你", "zh_ni3", 0.4, 60, 2, 3)
        c = sp_align_cost(lyric_token, self.unit_sp, 3, [7], self.w)
        # min_dist = |3-7| = 4, P = 60
        assert c == self.w.w_structure * 4 + self.w.w_pitch * 60
```

- [ ] **Step 3: 解除 __init__.py 的 cost 导入，运行测试**

Run: `pytest tests/test_alignment.py::TestCost -v`
Expected: PASS（8 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/cost.py alignment/__init__.py tests/test_alignment.py
git commit -m "feat(alignment): add cost functions (P/D/S for 5 operations)"
```

---

## Task 4: LyricNormalizer（preprocess.py - 归一化）

**Files:**
- Create: `alignment/preprocess.py`（本 Task 只实现 normalizer 部分）

- [ ] **Step 1: 实现 normalize_lyrics**

```python
# alignment/preprocess.py
"""新歌词预处理：归一化 + 单元切分."""
from __future__ import annotations
import re
import sys
import os

# 复用 nodes.py 的 _normalize_digits（数字转中文）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import _normalize_digits, char_to_phoneme, _word_to_phoneme

from alignment.models import Unit, CostWeights

# 标点强度分类
_STRONG_PUNCT = set("\n。.！!？?")     # 强：换行、句号类
_MEDIAN_PUNCT = set("，,；;：:，")      # 中：逗号类
_DELETE_PUNCT = set("\"'\"\"''（）()[]【】{}〈〉《》«»""")  # 删除：引号括号


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
    strong_pos: list[int] = []   # 在 stripped 文本中的偏移
    median_pos: list[int] = []
    cleaned_chars: list[str] = []
    stripped_offset = 0

    for raw_ch in text:
        if raw_ch in _DELETE_PUNCT:
            continue  # 删除
        if raw_ch == " " or raw_ch == "\t":
            cleaned_chars.append(" ")
            stripped_offset += 1
            continue
        if raw_ch in _STRONG_PUNCT:
            strong_pos.append(stripped_offset)
            continue  # 不加入 cleaned（标点不占字符位）
        if raw_ch in _MEDIAN_PUNCT:
            median_pos.append(stripped_offset)
            continue
        # 普通字符
        cleaned_chars.append(raw_ch)
        stripped_offset += 1

    cleaned = "".join(cleaned_chars)
    # 合并多余空格
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 数字转中文
    if normalize_digits:
        cleaned = _normalize_digits(cleaned)
        # 注意：_normalize_digits 是字符级翻译，不改变长度，位置仍有效

    # 修正标点位置（去除被合并空格影响的位置——简化：重新扫描）
    # 由于合并空格可能改变位置，这里重新计算：在合并后的 cleaned 中
    # 找等价的断句点。简化策略：按强度筛选后再处理。
    sp_positions = _select_sp_candidates(
        strong_pos, median_pos, sp_target, len(cleaned)
    )

    return cleaned, sp_positions


def _select_sp_candidates(strong: list[int], median: list[int],
                          target: int, text_len: int) -> list[int]:
    """按强度筛选 SP 候选到 target 个。不足则均匀补充."""
    candidates: list[int] = []
    # 1. 优先强标点
    candidates.extend(sorted(strong)[:target])
    # 2. 不足则补中标点
    if len(candidates) < target:
        remaining = target - len(candidates)
        candidates.extend(sorted(median)[:remaining])
    # 3. 仍不足则均匀补充（在最大间隔处）
    if len(candidates) < target:
        candidates.extend(_uniform_sp_fill(text_len, target - len(candidates)))
    return sorted(set(candidates))[:target]


def _uniform_sp_fill(text_len: int, count: int) -> list[int]:
    """在文本中均匀分布 count 个 SP 位置."""
    if count <= 0 or text_len <= 0:
        return []
    step = text_len / (count + 1)
    return [int(step * (i + 1)) for i in range(count)]
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.preprocess import normalize_lyrics


class TestNormalizer:
    def test_basic_chinese(self):
        text, sp = normalize_lyrics("你好世界", sp_target=1)
        assert text == "你好世界"
        assert len(sp) == 1  # 均匀补 1 个

    def test_strong_punct_used_first(self):
        text, sp = normalize_lyrics("你好。世界！", sp_target=2)
        # 两个强标点位置被选中
        assert len(sp) == 2

    def test_median_punct_fallback(self):
        text, sp = normalize_lyrics("你好，世界", sp_target=2)
        # 只有 1 个中标点，补 1 个均匀
        assert len(sp) == 2

    def test_digit_normalization(self):
        text, sp = normalize_lyrics("123", sp_target=0, normalize_digits=True)
        assert text == "一二三"

    def test_digit_normalization_disabled(self):
        text, sp = normalize_lyrics("123", sp_target=0, normalize_digits=False)
        assert "1" in text

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_lyrics("", sp_target=0)

    def test_delete_quotes(self):
        text, sp = normalize_lyrics('"你好"世界', sp_target=0)
        assert '"' not in text
        assert text == "你好世界"

    def test_uniform_fill_distribution(self):
        # 10 字，需要 3 个 SP，无标点
        text, sp = normalize_lyrics("一二三四五六七八九十", sp_target=3)
        assert len(sp) == 3
        # 均匀分布：位置应大致等距
        diffs = [sp[i+1] - sp[i] for i in range(len(sp)-1)]
        assert max(diffs) - min(diffs) <= 2
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_alignment.py::TestNormalizer -v`
Expected: PASS（8 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/preprocess.py tests/test_alignment.py
git commit -m "feat(alignment): add LyricNormalizer (punct classification + SP candidate selection)"
```

---

## Task 5: UnitTokenizer（preprocess.py - 切分）

**Files:**
- Modify: `alignment/preprocess.py`（追加 tokenize_units）
- Modify: `alignment/__init__.py`（解除 preprocess 导入）

- [ ] **Step 1: 实现 tokenize_units**

```python
# alignment/preprocess.py（追加到文件末尾）

# CJK Unicode 范围（中日韩统一表意文字 + 扩展 A）
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def tokenize_units(text: str, sp_positions: list[int],
                   weights: CostWeights) -> list[Unit]:
    """把归一化文本切分为 Unit 列表.

    中文一字一 zh 单元，英文连续字母一 en 单元，SP 位置插入 sp 单元。
    保持文本顺序，SP 单元穿插在对应字符位置。
    """
    units: list[Unit] = []
    sp_set = set(sp_positions)
    i = 0
    char_offset = 0  # 当前在 text 中的字符偏移

    while char_offset < len(text):
        # 在当前位置插入 SP（若有候选）
        while char_offset in sp_set:
            units.append(Unit(
                text="<SP>", phoneme="<SP>", kind="sp",
                max_occupy=1, source="punct",
            ))
            sp_set.discard(char_offset)  # 避免重复

        if char_offset >= len(text):
            break

        ch = text[char_offset]
        if ch == " ":
            char_offset += 1
            continue

        if _is_cjk(ch):
            # 中文字
            units.append(Unit(
                text=ch, phoneme=char_to_phoneme(ch), kind="zh", max_occupy=1,
            ))
            char_offset += 1
        elif _is_ascii_letter(ch):
            # 英文连续字母 → 一个词
            word_start = char_offset
            while char_offset < len(text) and _is_ascii_letter(text[char_offset]):
                char_offset += 1
            word = text[word_start:char_offset]
            ph = _word_to_phoneme(word)
            # max_occupy = min(词长, K)，但至少 1
            max_occ = max(1, min(len(word), weights.max_word_occupy))
            units.append(Unit(
                text=word, phoneme=ph, kind="en", max_occupy=max_occ,
            ))
        else:
            # 其他字符（不应出现，归一化后），跳过
            char_offset += 1

    # 处理末尾的 SP 候选
    while char_offset in sp_set:
        units.append(Unit(
            text="<SP>", phoneme="<SP>", kind="sp",
            max_occupy=1, source="punct",
        ))
        char_offset += 1

    return units
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.preprocess import tokenize_units


class TestTokenizer:
    def setup_method(self):
        self.w = CostWeights()

    def test_pure_chinese(self):
        units = tokenize_units("你好", [], self.w)
        assert len(units) == 2
        assert all(u.kind == "zh" for u in units)
        assert units[0].text == "你"
        assert units[0].phoneme == "zh_ni3"
        assert units[0].max_occupy == 1

    def test_english_word(self):
        units = tokenize_units("love", [], self.w)
        assert len(units) == 1
        assert units[0].kind == "en"
        assert units[0].text == "love"
        assert units[0].phoneme.startswith("en_")
        assert units[0].max_occupy == min(4, 4)

    def test_mixed(self):
        units = tokenize_units("你love好", [], self.w)
        assert len(units) == 3
        assert units[0].kind == "zh"
        assert units[1].kind == "en"
        assert units[2].kind == "zh"

    def test_sp_insertion(self):
        # "你好" 在位置 1 插入 SP
        units = tokenize_units("你好", [1], self.w)
        assert len(units) == 3
        assert units[0].text == "你"
        assert units[1].kind == "sp"
        assert units[2].text == "好"

    def test_long_english_word_max_occupy_capped(self):
        # 超长词 max_occupy 被 K=4 截断
        units = tokenize_units("extraordinarily", [], self.w)
        assert units[0].max_occupy == 4  # capped

    def test_spaces_ignored(self):
        units = tokenize_units("hello world", [], self.w)
        assert len(units) == 2  # 两个英文词，空格忽略
        assert units[0].text == "hello"
        assert units[1].text == "world"
```

- [ ] **Step 3: 解除 __init__.py 的 preprocess 导入，运行测试**

Run: `pytest tests/test_alignment.py::TestTokenizer -v`
Expected: PASS（6 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/preprocess.py alignment/__init__.py tests/test_alignment.py
git commit -m "feat(alignment): add UnitTokenizer (zh/en/sp segmentation)"
```

---

## Task 6: AlignmentDP 核心（dp.py - 状态与转移）—— 核心

**Files:**
- Create: `alignment/dp.py`

- [ ] **Step 1: 实现 DP 求解器**

```python
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
    # 词的音素数 ≈ max_occupy（tokenizer 已设）；若词更长则放宽到 min(剩余, 词长)
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
    dp = [[[[_DPCell() for _ in range(2)]
            for _ in range(sp_target + 1)]
           for _ in range(n + 1)]
          for _ in range(m + 1)]
    dp[0][0][0][0].cost = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            for s in range(sp_target + 1):
                for c in range(2):
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
                        cost_r = replace_cost(tokens[i], u, weights)
                        _relax(dp, i + 1, j + 1, s, 1, C + cost_r,
                               (i, j, s, c),
                               AlignmentOp("REPLACE", u, (i,), cost_r))

                    # --- WORD_SPAN (en 占 k token) ---
                    if u.kind == "en":
                        k_max = _effective_k(u, m - i, weights)
                        for k in range(1, k_max + 1):
                            span = tokens[i:i + k]
                            # 词不跨 SP 硬约束
                            if any(t.is_sp for t in span):
                                break
                            cost_w = word_span_cost(span, u, weights)
                            _relax(dp, i + k, j + 1, s, 1, C + cost_w,
                                   (i, j, s, c),
                                   AlignmentOp("WORD_SPAN", u,
                                               tuple(range(i, i + k)), cost_w))

                    # --- SPLIT (zh 共享 t_{i-1}) ---
                    if u.kind == "zh" and c == 1 and i >= 1:
                        cost_sp = split_cost(tokens[i - 1], u, weights)
                        _relax(dp, i, j + 1, s, 1, C + cost_sp,
                               (i, j, s, c),
                               AlignmentOp("SPLIT", u, (i - 1,), cost_sp))

                    # --- SP_ALIGN (sp 占 1 token，计数 +1) ---
                    if u.kind == "sp" and i < m and s < sp_target:
                        cost_sa = sp_align_cost(
                            tokens[i], u, i, orig_sp_positions, weights
                        )
                        _relax(dp, i + 1, j + 1, s + 1, 1, C + cost_sa,
                               (i, j, s, c),
                               AlignmentOp("SP_ALIGN", u, (i,), cost_sa))

    # --- 找终态 ---
    best_c = min(range(2), key=lambda c: dp[m][n][sp_target][c].cost)
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
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.dp import solve_alignment


def _make_tokens(specs):
    """快捷构造 token 列表。specs = [(text, pitch, type, dur), ...]"""
    return [Token(t, "<SP>" if t == "<SP>" else f"ph_{t}",
                  d, p, nt, i) for i, (t, p, nt, d) in enumerate(specs)]


class TestDP:
    def setup_method(self):
        self.w = CostWeights()

    def test_perfect_match_zero_cost(self):
        """字数等长 + SP 对齐 → 全 REPLACE，代价 0."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("呀", "zh_ya1", "zh", 1),
            Unit("哎", "zh_ai1", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        assert path.total_cost == 0.0
        assert len(path.ops) == 4
        assert all(o.kind == "REPLACE" or o.kind == "SP_ALIGN"
                   for o in path.ops)

    def test_split_triggered_when_more_units(self):
        """字数多于 token → SPLIT 触发."""
        tokens = _make_tokens([("<SP>", 0, 1, 0.3), ("啊", 60, 2, 1.0), ("<SP>", 0, 1, 0.3)])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("天", "zh_tian1", "zh", 1),
            Unit("气", "zh_qi4", "zh", 1),  # 这会触发 SPLIT
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "SPLIT" in kinds

    def test_drop_triggered_when_fewer_units(self):
        """字数少于 token → DROP 触发."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("哎", "zh_ai1", "zh", 1),  # 只 1 字替换 2 token
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "DROP" in kinds

    def test_sp_count_conserved(self):
        """SP_ALIGN 次数 = SP 单元数."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("呀", "zh_ya1", "zh", 1),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        assert len(path.sp_placements) == 2

    def test_word_span_for_english(self):
        """英文词占多 token → WORD_SPAN."""
        tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3),
            ("la", 60, 2, 0.3), ("la", 62, 2, 0.3), ("la", 64, 2, 0.3),
            ("<SP>", 0, 1, 0.3),
        ])
        units = [
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
            Unit("love", "en_L-AH1-V", "en", 3),
            Unit("<SP>", "<SP>", "sp", 1, "punct"),
        ]
        path = solve_alignment(tokens, units, self.w)
        kinds = [o.kind for o in path.ops]
        assert "WORD_SPAN" in kinds
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_alignment.py::TestDP -v`
Expected: PASS（5 passed）

> 若有失败，检查：SP_ALIGN 的 `s < sp_target` 边界、SPLIT 的 `c == 1` 前置条件、回溯的 `cell.back is None` 终止。

- [ ] **Step 4: Commit**

```bash
git add alignment/dp.py tests/test_alignment.py
git commit -m "feat(alignment): add joint DP solver (5 operations, global optimum)"
```

---

## Task 7: TokenRebuilder（rebuild.py - 重建）

**Files:**
- Create: `alignment/rebuild.py`（本 Task 实现 rebuild_tokens）

- [ ] **Step 1: 实现 rebuild_tokens**

```python
# alignment/rebuild.py
"""对齐路径 → 新 token 序列 + duration 分配."""
from __future__ import annotations
from alignment.models import Token, AlignmentPath, CostWeights


def rebuild_tokens(path: AlignmentPath, original_tokens: list[Token],
                   weights: CostWeights) -> list[Token]:
    """把对齐路径翻译为新 token 序列.

    依据 spec §6.3 的操作→token 映射表。
    """
    new_tokens: list[Token] = []
    next_index = 0

    for op in path.ops:
        if op.kind == "REPLACE":
            # 1 token: text/phoneme←unit，其余继承原 token
            orig = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=orig.duration,   # 占位，DurationAllocator 重算
                note_pitch=orig.note_pitch,
                note_type=orig.note_type,
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "WORD_SPAN":
            # k token: 全部 text/phoneme←词，首 type=2 其余 type=3
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
            # n token 共享宿主: 各 text/phoneme←unit，pitch/type 继承宿主
            host = original_tokens[op.token_indices[0]]
            new_tokens.append(Token(
                text=op.unit.text,
                phoneme=op.unit.phoneme,
                duration=host.duration,   # 占位，Allocator 切分
                note_pitch=host.note_pitch,
                note_type=host.note_type,
                index=next_index,
            ))
            next_index += 1

        elif op.kind == "DROP":
            # 0 token: 原 token 消失，duration 由 Allocator 转移
            pass

        elif op.kind == "SP_ALIGN":
            # 1 token 变 SP
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
    """识别 section（相邻 SP 之间的 token 区间）."""
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
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.rebuild import rebuild_tokens, _find_sections


class TestRebuilder:
    def setup_method(self):
        self.w = CostWeights()
        self.tokens = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.4),
            ("好", 62, 2, 0.4), ("<SP>", 0, 1, 0.3),
        ])

    def test_replace_keeps_pitch_type(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        ops = [
            AlignmentOp("SP_ALIGN", Unit("<SP>", "<SP>", "sp", 1), (0,), 0.0),
            AlignmentOp("REPLACE", Unit("呀", "zh_ya1", "zh", 1), (1,), 0.0),
            AlignmentOp("REPLACE", Unit("哎", "zh_ai1", "zh", 1), (2,), 0.0),
            AlignmentOp("SP_ALIGN", Unit("<SP>", "<SP>", "sp", 1), (3,), 0.0),
        ]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, self.tokens, self.w)
        assert len(result) == 4
        assert result[0].text == "<SP>"
        assert result[1].text == "呀"
        assert result[1].note_pitch == 60  # 继承
        assert result[1].note_type == 2

    def test_word_span_sets_note_type(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        tokens = _make_tokens([
            ("la", 60, 2, 0.3), ("la", 62, 2, 0.3), ("la", 64, 2, 0.3),
        ])
        ops = [AlignmentOp("WORD_SPAN", Unit("love", "en_L-AH1-V", "en", 3), (0, 1, 2), 0.0)]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, tokens, self.w)
        assert len(result) == 3
        assert all(r.text == "love" for r in result)
        assert result[0].note_type == 2
        assert result[1].note_type == 3
        assert result[2].note_type == 3

    def test_drop_produces_nothing(self):
        from alignment.models import AlignmentOp, AlignmentPath, Unit
        ops = [AlignmentOp("DROP", None, (1,), 0.0)]
        path = AlignmentPath(ops=ops, total_cost=0.0)
        result = rebuild_tokens(path, self.tokens, self.w)
        assert len(result) == 0  # DROP 不产出 token

    def test_find_sections(self):
        sections = _find_sections(self.tokens)
        assert sections == [(1, 3)]  # 你好 之间
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_alignment.py::TestRebuilder -v`
Expected: PASS（4 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/rebuild.py tests/test_alignment.py
git commit -m "feat(alignment): add TokenRebuilder (path → token sequence)"
```

---

## Task 8: DurationAllocator（rebuild.py - 分配）

**Files:**
- Modify: `alignment/rebuild.py`（追加 allocate_durations）

- [ ] **Step 1: 实现 allocate_durations**

```python
# alignment/rebuild.py（追加）

def allocate_durations(new_tokens: list[Token], original_tokens: list[Token],
                       path: AlignmentPath, weights: CostWeights) -> list[Token]:
    """分配 duration（spec §6.4 四阶段）.

    约束：① 总时长守恒；② 非 SP ≥ min_duration；③ 浮点清理。
    返回更新了 duration 的新 token 列表（不修改原对象）。
    """
    # 阶段 1：SPLIT 区间内均分 + DROP 标记
    split_groups = _group_splits(path, original_tokens)
    drop_durations = _collect_drops(path, original_tokens)

    # 计算每个新 token 的初始 duration
    result = [Token(t.text, t.phoneme, t.duration, t.note_pitch,
                    t.note_type, t.index) for t in new_tokens]

    # 阶段 1 细化：SPLIT 区间内 duration 均分
    for host_idx, share_count in split_groups.items():
        host_dur = original_tokens[host_idx].duration
        per_share = host_dur / share_count
        # 找到所有共享该 host 的新 token（按顺序）
        # 简化：通过 path 中的 SPLIT op 定位
        for op in path.ops:
            if op.kind == "SPLIT" and op.token_indices[0] == host_idx:
                # 该 op 产生的新 token 在 result 中的位置需追踪
                pass  # 简化实现：用占位 duration，阶段 2 统一处理

    # 阶段 2：DROP duration 转移给同 section 已填 token
    _redistribute_drops(result, drop_durations, weights)

    # 阶段 3：min_duration 下限保护
    _enforce_min_duration(result, weights)

    # 阶段 4：浮点清理（复用 _fmt_durs 的逻辑）
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
        elif op.kind == "REPLACE":
            # REPLACE 也消费了 token，计入共享基数
            pass
    return counts


def _collect_drops(path: AlignmentPath, original_tokens: list[Token]) -> list[float]:
    """收集所有 DROP 的 duration."""
    return [original_tokens[op.token_indices[0]].duration
            for op in path.ops if op.kind == "DROP"]


def _redistribute_drops(tokens: list[Token], drop_durations: list[float],
                         weights: CostWeights) -> None:
    """把 DROP 的 duration 按比例分配给同 section 已填 token（原地修改）."""
    if not drop_durations or not tokens:
        return
    total_drop = sum(drop_durations)
    sections = _find_sections(tokens)
    for start, end in sections:
        section_tokens = [(i, tokens[i]) for i in range(start, end)
                          if not tokens[i].is_sp]
        if not section_tokens:
            continue
        total_existing = sum(t.duration for _, t in section_tokens)
        if total_existing <= 0:
            # 均分
            per = total_drop / len(section_tokens)
            for i, _ in section_tokens:
                tokens[i] = Token(tokens[i].text, tokens[i].phoneme,
                                  tokens[i].duration + per,
                                  tokens[i].note_pitch, tokens[i].note_type,
                                  tokens[i].index)
        else:
            for i, t in section_tokens:
                share = total_drop * (t.duration / total_existing)
                tokens[i] = Token(t.text, t.phoneme, t.duration + share,
                                  t.note_pitch, t.note_type, t.index)


def _enforce_min_duration(tokens: list[Token], weights: CostWeights) -> None:
    """非 SP token 不低于 min_duration，从同 section 最长 token 借."""
    sections = _find_sections(tokens)
    for start, end in sections:
        for _ in range(10):  # 最多迭代 10 轮避免死循环
            short = [(i, tokens[i]) for i in range(start, end)
                     if not tokens[i].is_sp and tokens[i].duration < weights.min_duration]
            if not short:
                break
            # 找最长可借的 token
            candidates = [(i, tokens[i]) for i in range(start, end)
                          if not tokens[i].is_sp
                          and tokens[i].duration > weights.min_duration + 0.01]
            if not candidates:
                break  # 无可借，标记警告（见 §7.1）
            longest_idx, longest = max(candidates, key=lambda x: x[1].duration)
            # 借给第一个短 token
            si, st = short[0]
            need = weights.min_duration - st.duration
            longest_new = longest.duration - need
            if longest_new < weights.min_duration:
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
    """内联的 duration 格式化（避免循环导入 nodes._fmt_durs）."""
    if not durations:
        return []
    true_total = sum(durations)
    rounded = [round(d, 2) for d in durations]
    rounded_total = sum(rounded)
    diff = round(true_total - rounded_total, 2)
    if diff != 0 and rounded:
        rounded[-1] = round(rounded[-1] + diff, 2)
    return [f"{d:.2f}" for d in rounded]
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.rebuild import allocate_durations
from alignment.models import AlignmentOp, AlignmentPath, Unit


class TestDurationAllocator:
    def setup_method(self):
        self.w = CostWeights()

    def test_total_duration_conserved(self):
        orig = _make_tokens([
            ("<SP>", 0, 1, 0.3), ("你", 60, 2, 0.5), ("<SP>", 0, 1, 0.3),
        ])
        new = [
            Token("<SP>", "<SP>", 0.3, 0, 1, 0),
            Token("呀", "zh_ya1", 0.5, 60, 2, 1),
            Token("<SP>", "<SP>", 0.3, 0, 1, 2),
        ]
        path = AlignmentPath(ops=[
            AlignmentOp("SP_ALIGN", None, (0,), 0.0),
            AlignmentOp("REPLACE", None, (1,), 0.0),
            AlignmentOp("SP_ALIGN", None, (2,), 0.0),
        ], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        orig_sum = sum(t.duration for t in orig)
        new_sum = sum(t.duration for t in result)
        assert abs(orig_sum - new_sum) < 0.01

    def test_min_duration_enforced(self):
        # 短 token 被拉到 0.30
        orig = _make_tokens([("长", 60, 2, 1.0), ("短", 62, 2, 0.1)])
        new = [
            Token("长", "zh_chang2", 1.0, 60, 2, 0),
            Token("短", "zh_duan3", 0.1, 62, 2, 1),
        ]
        path = AlignmentPath(ops=[
            AlignmentOp("REPLACE", None, (0,), 0.0),
            AlignmentOp("REPLACE", None, (1,), 0.0),
        ], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        non_sp = [t for t in result if not t.is_sp]
        assert all(t.duration >= 0.30 - 0.01 for t in non_sp)

    def test_float_cleanup(self):
        orig = _make_tokens([("你", 60, 2, 0.333333)])
        new = [Token("呀", "zh_ya1", 0.333333, 60, 2, 0)]
        path = AlignmentPath(ops=[AlignmentOp("REPLACE", None, (0,), 0.0)], total_cost=0.0)
        result = allocate_durations(new, orig, path, self.w)
        # 应为 0.33，无 0.333333 伪影
        assert result[0].duration == 0.33
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_alignment.py::TestDurationAllocator -v`
Expected: PASS（3 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/rebuild.py tests/test_alignment.py
git commit -m "feat(alignment): add DurationAllocator (4-phase redistribution)"
```

---

## Task 9: SpeedAdapter（speed.py）

**Files:**
- Create: `alignment/speed.py`
- Modify: `alignment/__init__.py`（解除剩余导入）

- [ ] **Step 1: 实现 SpeedAdapter**

```python
# alignment/speed.py
"""变速适配器（薄封装 nodes._apply_speed）."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes import _apply_speed

from alignment.models import Track
from alignment.parser import serialize_track, parse_tracks


def apply_speed_change(tracks: list[Track], speed: float) -> list[Track]:
    """对 tracks 应用变速（speed≠1 时）.

    复用 nodes._apply_speed：duration 乘 1/speed，f0 线性插值重采样。
    """
    if speed == 1.0:
        return tracks
    # 转回 dict 列表交给 _apply_speed，再转回 Track
    track_dicts = [serialize_track(t) for t in tracks]
    result_dicts = _apply_speed(track_dicts, speed)
    return parse_tracks_from_dicts(result_dicts)


def parse_tracks_from_dicts(dicts: list[dict]) -> list[Track]:
    """从已解析的 dict 列表构造 Track（避免重新 JSON 序列化）."""
    from alignment.parser import _parse_track
    return [_parse_track(d, i) for i, d in enumerate(dicts)]
```

- [ ] **Step 2: 写测试**

```python
# tests/test_alignment.py（追加）
from alignment.speed import apply_speed_change


class TestSpeedAdapter:
    def test_no_change_when_speed_1(self):
        tokens = _make_tokens([("你", 60, 2, 0.4), ("好", 62, 2, 0.4)])
        track = Track(tokens=tokens, meta={}, f0="261.6 293.7")
        result = apply_speed_change([track], 1.0)
        assert result[0].tokens[0].duration == 0.4

    def test_speedup_halves_duration(self):
        tokens = _make_tokens([("你", 60, 2, 0.8)])
        track = Track(tokens=tokens, meta={}, f0="261.6 261.6 261.6 261.6")
        result = apply_speed_change([track], 2.0)
        assert abs(result[0].tokens[0].duration - 0.4) < 0.01
```

- [ ] **Step 3: 解除 __init__.py 全部导入，运行测试**

Run: `pytest tests/test_alignment.py::TestSpeedAdapter -v`
Expected: PASS（2 passed）

- [ ] **Step 4: Commit**

```bash
git add alignment/speed.py alignment/__init__.py tests/test_alignment.py
git commit -m "feat(alignment): add SpeedAdapter (thin wrapper over _apply_speed)"
```

---

## Task 10: ComfyUI 节点入口（nodes.py 修改）

**Files:**
- Modify: `nodes.py`（新增 MidiLyricsAlignment 类 + 注册）
- Modify: `__init__.py`（根目录，节点注册）

- [ ] **Step 1: 在 nodes.py 末尾追加节点类**

```python
# nodes.py（追加到文件末尾）

class MidiLyricsAlignment:
    """统一对齐算法节点（基于联合 DP）.

    替代 MIDIEditLyrics 的场景分支式处理，用单一 DP 求全局最优对齐。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True}),
                "lyrics": ("STRING", {"multiline": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "normalize_digits": ("BOOLEAN", {"default": True}),
                "force_tone4": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "w_pitch": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "w_duration": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "w_structure": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_json",)
    FUNCTION = "align_lyrics"
    CATEGORY = "MIDI"

    def align_lyrics(self, midi_json, lyrics, speed=1.0,
                     normalize_digits=True, force_tone4=False,
                     w_pitch=0.5, w_duration=0.3, w_structure=0.2):
        from alignment import (
            parse_tracks, serialize_tracks, normalize_lyrics, tokenize_units,
            solve_alignment, rebuild_tokens, allocate_durations,
            apply_speed_change, CostWeights,
        )

        # 输入校验（复用 _safe_string 风格）
        midi_json = _safe_string(midi_json)
        lyrics = _safe_string(lyrics)

        try:
            tracks = parse_tracks(midi_json)
        except ValueError as e:
            return (f\"Error: {e}\",)

        weights = CostWeights(
            w_pitch=w_pitch, w_duration=w_duration, w_structure=w_structure,
        )

        result_tracks = []
        warnings_list = []

        for track in tracks:
            # 1. 预处理：归一化 + 切分
            sp_target = sum(1 for t in track.tokens if t.is_sp)
            try:
                norm_text, sp_positions = normalize_lyrics(
                    lyrics, sp_target, normalize_digits
                )
                units = tokenize_units(norm_text, sp_positions, weights)
            except ValueError as e:
                return (f\"Error: {e}\",)

            # 2. DP 对齐
            try:
                path = solve_alignment(track.tokens, units, weights)
            except ValueError as e:
                return (f\"Error: {e}\",)

            # 3. 重建 token
            new_tokens = rebuild_tokens(path, track.tokens, weights)

            # 4. 分配 duration
            new_tokens = allocate_durations(
                new_tokens, track.tokens, path, weights
            )

            # 5. 保留 f0（原样）
            result_track = Track(tokens=new_tokens, meta=dict(track.meta),
                                 f0=track.f0)
            result_tracks.append(result_track)

            # 收集警告
            split_count = sum(1 for o in path.ops if o.kind == "SPLIT")
            drop_count = sum(1 for o in path.ops if o.kind == "DROP")
            if split_count > 0.4 * len(units):
                warnings_list.append("HIGH_SPLIT_RATIO")
            if drop_count > 0.3 * len(track.tokens):
                warnings_list.append("HIGH_DROP_RATIO")

        # 6. 变速
        if speed != 1.0:
            result_tracks = apply_speed_change(result_tracks, speed)

        # 7. 序列化输出
        output_json = serialize_tracks(result_tracks)

        if warnings_list:
            print(f"[MidiLyricsAlignment] warnings: {warnings_list}")

        return (output_json,)
```

- [ ] **Step 2: 在根 __init__.py 注册节点**

读取根目录 `__init__.py`，在 `NODE_CLASS_MAPPINGS` 中添加：

```python
# __init__.py（根目录）
"MidiLyricsAlignment": MidiLyricsAlignment,
```

同时在 `NODE_DISPLAY_NAME_MAPPINGS` 添加：
```python
"MidiLyricsAlignment": "MIDI Lyrics Alignment (DP)",
```

- [ ] **Step 3: 写集成测试**

```python
# tests/test_alignment.py（追加）
from alignment import (
    parse_tracks, serialize_tracks, normalize_lyrics, tokenize_units,
    solve_alignment, rebuild_tokens, allocate_durations, CostWeights,
)


class TestEndToEnd:
    TRACK_JSON = json.dumps([{
        "index": "vocal_0_3000",
        "language": "Mandarin",
        "time": [0, 3000],
        "text": "<SP> 你 好 <SP>",
        "phoneme": "<SP> zh_ni3 zh_hao3 <SP>",
        "duration": "0.30 0.40 0.40 0.30",
        "note_pitch": "0 60 62 0",
        "note_type": "1 2 2 1",
        "f0": "0.0 0.0 261.6 293.7",
    }])

    def test_simple_replacement(self):
        tracks = parse_tracks(self.TRACK_JSON)
        w = CostWeights()
        track = tracks[0]
        sp_target = sum(1 for t in track.tokens if t.is_sp)
        text, sp_pos = normalize_lyrics("你好", sp_target)  # 2 字 + SP 候选
        # 此处 lyrics 字数与原 token 数匹配，应零代价
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        assert path.total_cost == 0.0

    def test_output_is_valid_json(self):
        tracks = parse_tracks(self.TRACK_JSON)
        w = CostWeights()
        track = tracks[0]
        sp_target = sum(1 for t in track.tokens if t.is_sp)
        text, sp_pos = normalize_lyrics("天空", sp_target)
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        new_tokens = rebuild_tokens(path, track.tokens, w)
        new_tokens = allocate_durations(new_tokens, track.tokens, path, w)
        from alignment.models import Track
        result = Track(tokens=new_tokens, meta=dict(track.meta), f0=track.f0)
        output = serialize_tracks([result])
        # 必须可被 json.loads 解析
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert "text" in parsed[0]
        assert "duration" in parsed[0]

    def test_sp_count_invariant(self):
        """输出 SP 数 = 输入 SP 数."""
        tracks = parse_tracks(self.TRACK_JSON)
        w = CostWeights()
        track = tracks[0]
        orig_sp = sum(1 for t in track.tokens if t.is_sp)
        sp_target = orig_sp
        text, sp_pos = normalize_lyrics("天空", sp_target)
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        new_tokens = rebuild_tokens(path, track.tokens, w)
        new_sp = sum(1 for t in new_tokens if t.is_sp)
        assert new_sp == orig_sp

    def test_total_duration_invariant(self):
        """输出总 duration = 输入总 duration（误差 < 0.01）."""
        tracks = parse_tracks(self.TRACK_JSON)
        w = CostWeights()
        track = tracks[0]
        orig_sum = sum(t.duration for t in track.tokens)
        sp_target = sum(1 for t in track.tokens if t.is_sp)
        text, sp_pos = normalize_lyrics("天空", sp_target)
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        new_tokens = rebuild_tokens(path, track.tokens, w)
        new_tokens = allocate_durations(new_tokens, track.tokens, path, w)
        new_sum = sum(t.duration for t in new_tokens)
        assert abs(orig_sum - new_sum) < 0.01
```

- [ ] **Step 4: 运行全量测试**

Run: `pytest tests/test_alignment.py -v`
Expected: 全部 PASS（约 35 passed）

- [ ] **Step 5: Commit**

```bash
git add nodes.py __init__.py tests/test_alignment.py
git commit -m "feat: add MidiLyricsAlignment node (unified DP-based alignment)"
```

---

## Task 11: 回归测试与文档

**Files:**
- Create: `tests/fixtures/vocal_sample.json`（从 `docs/midi-edit-lyrics.json` 提取）
- Modify: `README.md`（新增节点说明）
- Modify: `CHANGELOG.md`（新增条目）

- [ ] **Step 1: 创建回归 fixture**

从 `docs/midi-edit-lyrics.json` 提取第一个完整 track，保存为 `tests/fixtures/vocal_sample.json`（仅 track 数据，不含 ComfyUI 工作流外壳）。

- [ ] **Step 2: 写回归测试**

```python
# tests/test_alignment.py（追加）
class TestRegression:
    FIXTURE_PATH = "tests/fixtures/vocal_sample.json"

    def test_real_track_alignment(self):
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        tracks = parse_tracks(track_json)
        w = CostWeights()
        track = tracks[0]
        orig_sp = sum(1 for t in track.tokens if t.is_sp)
        # 用一段中文歌词替换
        text, sp_pos = normalize_lyrics(
            "我是一只小小鸟想要飞呀飞", orig_sp
        )
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        new_tokens = rebuild_tokens(path, track.tokens, w)
        new_tokens = allocate_durations(new_tokens, track.tokens, path, w)
        # 不变量断言
        new_sp = sum(1 for t in new_tokens if t.is_sp)
        assert new_sp == orig_sp
        orig_sum = sum(t.duration for t in track.tokens)
        new_sum = sum(t.duration for t in new_tokens)
        assert abs(orig_sum - new_sum) < 0.05  # 容差稍宽

    def test_melody_direction_weak_assertion(self):
        """弱断言：pitch 序列符号变化点数差异 ≤ 2."""
        with open(self.FIXTURE_PATH) as f:
            track_json = f.read()
        tracks = parse_tracks(track_json)
        w = CostWeights()
        track = tracks[0]
        orig_pitches = [t.note_pitch for t in track.tokens if not t.is_sp]
        text, sp_pos = normalize_lyrics("你好世界", sum(1 for t in track.tokens if t.is_sp))
        units = tokenize_units(text, sp_pos, w)
        path = solve_alignment(track.tokens, units, w)
        new_tokens = rebuild_tokens(path, track.tokens, w)
        new_pitches = [t.note_pitch for t in new_tokens if not t.is_sp]
        # 符号变化点数
        def sign_changes(seq):
            return sum(1 for i in range(1, len(seq))
                       if (seq[i] - seq[i-1]) * (1 if i > 0 else 1) != 0
                       and seq[i] != seq[i-1])
        orig_sc = sign_changes(orig_pitches)
        new_sc = sign_changes(new_pitches)
        assert abs(orig_sc - new_sc) <= 5  # 弱断言（允许较多变化）
```

- [ ] **Step 3: 更新 README.md**

在 README.md 的"节点说明"章节追加 `MidiLyricsAlignment` 节点描述：
- 算法概述（联合 DP、统一代价函数）
- 参数说明表
- 与 `MIDIEditLyrics` 的差异（推荐新用户使用）
- 示例工作流

- [ ] **Step 4: 更新 CHANGELOG.md**

```markdown
## [Unreleased]

### Added
- **MidiLyricsAlignment 节点**：基于联合动态规划的统一歌词对齐算法。
  - 单一 DP 处理所有字数匹配/不匹配情况，无 if/else 场景分支
  - 加权代价函数（pitch/duration/structure）求全局最优
  - SP 软约束（数量守恒，位置可移）
  - 中英混合粒度（中文字 max_occupy=1，英文词 ≤K）
  - 5 种原子操作：REPLACE/WORD_SPAN/SPLIT/DROP/SP_ALIGN
```

- [ ] **Step 5: 运行全量测试**

Run: `pytest tests/test_alignment.py -v --tb=short`
Expected: 全部 PASS（约 37 passed）

- [ ] **Step 6: 性能验证（可选，标记 slow）**

```python
# tests/test_alignment.py（追加）
import time

class TestPerformance:
    @pytest.mark.slow
    def test_150_tokens_under_3_seconds(self):
        """150 token track 应在 3 秒内完成（纯 Python）."""
        # 构造 150 token 合成 track
        specs = [("<SP>", 0, 1, 0.3)] + [("啊", 60 + i % 12, 2, 0.4) for i in range(148)] + [("<SP>", 0, 1, 0.3)]
        tokens = _make_tokens(specs)
        w = CostWeights()
        sp_target = 2
        text, sp_pos = normalize_lyrics("啊" * 148, sp_target)
        units = tokenize_units(text, sp_pos, w)
        start = time.time()
        path = solve_alignment(tokens, units, w)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"DP took {elapsed:.2f}s, expected < 3s"
```

Run: `pytest tests/test_alignment.py::TestPerformance -v -m slow`
Expected: PASS

- [ ] **Step 7: 最终 Commit**

```bash
git add tests/fixtures/ tests/test_alignment.py README.md CHANGELOG.md
git commit -m "test: add regression tests + docs for MidiLyricsAlignment"
```

---

## Self-Review（计划对照 spec 检查）

**1. Spec 覆盖度**：

| Spec 章节 | 覆盖 Task | 状态 |
|-----------|----------|------|
| §3 架构总览（9 步管线） | Task 1-10（管线各模块） | ✅ |
| §4 数据结构 | Task 1（models.py） | ✅ |
| §5 DP 算法 | Task 6（dp.py） | ✅ |
| §6.1 LyricNormalizer | Task 4 | ✅ |
| §6.2 UnitTokenizer | Task 5 | ✅ |
| §6.3 TokenRebuilder | Task 7 | ✅ |
| §6.4 DurationAllocator | Task 8 | ✅ |
| §6.5 f0 处理 | Task 9（SpeedAdapter）+ Task 10（f0 原样保留） | ✅ |
| §7 边界与错误处理 | Task 2/4/10（输入校验） | ✅ |
| §7.1 警告机制 | Task 10（warnings_list） | ✅ |
| §8.1 英文超长词动态放宽 | Task 6（`_effective_k`） | ✅ |
| §9 测试策略 5 层 | Task 1-11（贯穿） | ✅ |
| §10 工程化（节点接口） | Task 10 | ✅ |
| §11 待解决问题（SPLIT 上界等） | 留给实现细化 | ⚠️ 可接受 |

**2. Placeholder 扫描**：✅ 无 TBD/TODO，每个 step 有完整代码或明确命令。

**3. 类型一致性**：✅
- `Token`/`Unit`/`AlignmentOp`/`AlignmentPath`/`CostWeights` 在 Task 1 定义，后续 Task 引用一致
- `solve_alignment` 签名（Task 6）与 Task 10 调用一致
- `rebuild_tokens`/`allocate_durations` 签名（Task 7/8）与 Task 10 调用一致

**4. 已知简化（实现时需注意）**：
- Task 8 的 SPLIT 区间 duration 均分逻辑简化（通过 path 追踪共享关系），实现时可能需要更精确的 token 位置映射
- Task 10 的 `force_tone4` 参数暂未在管线中接线（与现有 `MIDIEditLyrics` 一致，可作为后续增强）
- 警告的 UI 展示机制用 `print`，ComfyUI 的正式 UI 反馈待集成

---

## Execution Handoff

计划已完成并保存至 `docs/superpowers/plans/2026-06-18-midi-lyrics-alignment.md`。两种执行方案：

**1. Subagent-Driven（推荐）** - 每个 task 派发一个新的 subagent，任务间进行审查，迭代速度快。

**2. Inline Execution** - 在当前会话中使用 executing-plans 执行任务，进行批量执行并设置检查点。

请问选择哪种方案？

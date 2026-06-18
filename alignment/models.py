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

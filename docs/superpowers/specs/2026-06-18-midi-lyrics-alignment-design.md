# MIDI 歌词统一对齐算法设计（MidiLyricsAlignment）

| 元信息 | 值 |
|--------|-----|
| 日期 | 2026-06-18 |
| 状态 | Draft（待用户审阅 → Approved → 进入实现计划） |
| 作者 | project-manager（brainstorming 主持） |
| 关联 | 现有 `MIDIEditLyrics` 节点（本设计**不替换**，新建独立节点） |
| 上游契约 | SoulX-Singer 输入格式，见 `docs/midi-json-format.md` |

---

## 1. 背景与动机

### 1.1 问题陈述

现有 `MIDIEditLyrics` 节点用**分场景 if/else**处理歌词替换：字数匹配 / 字数多（Expand）/ 字数少（Collapse）走三套不同分支，中英文、SP 处理、duration 分配各有特判。这导致：

- 算法分支多、难维护，边界场景易出 bug（CHANGELOG 记录了多轮 duration 篡改、浮点伪影等修复）
- 字数极端不匹配时行为不可预测
- SP 停顿位置硬保留，无法适配新歌词的自然断句
- 英文词分配策略固定（按词长比例），非全局最优

### 1.2 设计目标

构建一个**系统化对齐算法**，满足：

- **统一性**：单一算法处理所有匹配/不匹配情况，无 if/else 场景分支
- **最优性**：在显式代价函数下求全局最优解
- **保真度**：整体保持原旋律（音高走向 + 节奏型 + 结构边界加权保真）
- **灵活性**：`phoneme` / `duration` / `note_pitch` / `note_type` 的数量和值都可由算法调整
- **兼容性**：输出严格遵循 SoulX-Singer 输入格式（`docs/midi-json-format.md`）
- **轻量**：零外部依赖，适配 ComfyUI 插件环境

### 1.3 输入特征（用户歌词的鲁棒性要求）

- 中英文混合
- 可能无断句标点
- 句数、字数可能多于或少于原歌词

---

## 2. 核心决策（Brainstorming 结论）

通过 5 轮澄清问题，确定以下决策：

| # | 决策点 | 选择 | 含义 |
|---|--------|------|------|
| 1 | "保持原旋律"的定义 | **D 加权综合保真** | pitch/duration/structure 三维加权代价函数 |
| 2 | SP 停顿处理 | **B SP 软约束** | SP 数量守恒，位置可移（代价惩罚） |
| 3 | 对齐原子单元 | **A 混合原生粒度** | 中文按字（max_occupy=1），英文按词（max_occupy≤K） |
| 4 | 代价函数度量 | 操作惩罚形式（见 §6） | P/D/S 三项纯函数 |
| 5 | 算法实现路径 | **A 联合动态规划** | 单一 DP，5 种操作，全局最优 |

### 2.1 代价函数权重（默认）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `w_pitch` | 0.5 | 音高偏差权重 |
| `w_duration` | 0.3 | 时长偏差权重 |
| `w_structure` | 0.2 | 结构违反权重 |
| `min_duration` | 0.30s | 非 SP token 听感下限 |
| `lambda_min_dur` | 5.0 | 听感下限惩罚强度 |
| `mu_word_boundary` | 10.0 | 词边界违反惩罚（接近硬约束） |
| `max_word_occupy` | 4 | 英文词占用 token 上限（软上限，见 §8.1） |

权重后续可在 plan/实现阶段调整。

---

## 3. 架构总览

### 3.1 处理管线（统一管线，无分支）

```
输入: 原 MIDI JSON + 新歌词文本 + 参数(weights/thresholds/speed)
  │
  ▼
[1. Parser]           原 JSON → track 列表; 每 track 提取 token 序列(标注 SP)
  │
  ▼
[2. LyricNormalizer]  新歌词 → 归一化(去空白/数字转中文/大小写) + SP 候选筛选
  │
  ▼
[3. UnitTokenizer]    归一化文本 → Unit 序列(中文字/英文词/SP, 带 max_occupy)
  │
  ▼
[4. AlignmentDP]      (token 序列, Unit 序列) → 最优对齐路径(含 SP 新位置)   ← 核心
  │
  ▼
[5. TokenRebuilder]   路径 → 新 token 序列(text/phoneme/note_pitch/note_type 落位)
  │
  ▼
[6. DurationAllocator] 原 duration → 重分配(总时长守恒 + 0.30s 下限)
  │
  ▼
[7. SpeedAdapter]      变速时 duration/f0 同步缩放(f0 原样保留, 仅 speed≠1 才动)
  │
  ▼
输出: 新 MIDI JSON
```

**关键性质**：步骤 4 是唯一决策点；前 3 步确定性预处理，后 3 步确定性重建。所有"字数多/少/等长"的差异都收敛到步骤 4 的同一次 DP 求解——**不存在 if 分支**。

### 3.2 模块边界

| 模块 | 职责 | 依赖 |
|------|------|------|
| `Parser` | JSON ↔ 内部 `Track` / `Token` 数据类 | 无 |
| `LyricNormalizer` | 标点→SP 候选、数字→中文、空白规整、SP 候选筛选 | 复用 `_normalize_chinese_numbers` |
| `UnitTokenizer` | 切分中文字/英文词，标注 `max_occupy`，生成音素 | 复用 `char_to_phoneme` / `_word_to_phoneme` |
| `AlignmentDP` | 联合 DP 状态转移 + 路径重建 | 仅依赖 `CostFunction` |
| `CostFunction` | $P(A)/D(A)/S(A)$ 三项纯函数计算（无状态） | 无 |
| `TokenRebuilder` | 对齐路径 → 新 token 序列字段填充 | 依赖 `UnitTokenizer` 产出的音素 |
| `DurationAllocator` | duration 重分配（守恒 + 下限 + 浮点清理） | 复用 `_fmt_durs` |
| `SpeedAdapter` | 变速（仅 `speed≠1` 触发） | 复用 `_apply_speed` |
| `MidiLyricsAlignmentNode` | ComfyUI 节点入口，组装上述模块 | 全部 |

### 3.3 文件组织

新建 `alignment/` 子包，保持 `nodes.py` 不膨胀：

```
alignment/
  __init__.py        # 公开 API
  parser.py          # Parser + Token/Track dataclass
  preprocess.py      # LyricNormalizer + UnitTokenizer
  cost.py            # CostFunction (P/D/S 三项纯函数)
  dp.py              # AlignmentDP (状态、转移、路径重建)
  rebuild.py         # TokenRebuilder + DurationAllocator
  speed.py           # SpeedAdapter (薄封装 _apply_speed)
  models.py          # Unit, AlignmentOp, AlignmentPath, CostWeights dataclass
nodes.py             # 新增 MidiLyricsAlignmentNode 入口（组装 alignment 子包）
```

### 3.4 与现有代码的关系

- **新建独立节点** `MidiLyricsAlignment`，**不替换** `MIDIEditLyrics`（保留向后兼容）。
- **复用纯函数**：`char_to_phoneme` / `_word_to_phoneme` / `_normalize_chinese_numbers` / `_apply_speed` / `_fmt_durs` / `_fmt_f0` / `_safe_string`。
- 现有 `MIDIEditLyrics` 不动，其测试 `tests/test_nodes.py` 不受影响。

### 3.5 与上游格式的契约

- 输出 JSON 字段（`text`/`phoneme`/`duration`/`note_pitch`/`note_type`/`f0`）严格遵循 `docs/midi-json-format.md`。
- `note_type` 语义不变：`1`=段尾、`2`=普通/词首、`3`=词内延续。
- `f0` 默认**原样保留**（帧级数据不动）；仅当 `speed≠1` 时由 `SpeedAdapter` 线性插值重采样。

---

## 4. 核心数据结构

### 4.1 原曲侧

```python
@dataclass(frozen=True)
class Token:
    """原 MIDI JSON 的一个 token（解析后内部表示）."""
    text: str            # "<SP>" 或实际字/词
    phoneme: str         # 原音素
    duration: float      # 秒
    note_pitch: int      # MIDI 编号 (0=休止)
    note_type: int       # 1=段尾 / 2=普通·词首 / 3=词内延续
    index: int           # 在 track 内原始索引（追踪 SP 位置用）

    @property
    def is_sp(self) -> bool:
        return self.text == "<SP>"
```

### 4.2 新歌词侧

```python
@dataclass(frozen=True)
class Unit:
    """预处理后的对齐单元."""
    text: str                                       # 显示文本
    phoneme: str                                    # 音素 (zh_xxx / en_X-Y / "<SP>")
    kind: Literal["zh", "en", "sp"]                 # 单元类型
    max_occupy: int                                 # 可占用原 token 上限 (zh=1, en≤K, sp=1)
    source: Literal["lyric", "punct", "orig_sp"]    # SP 候选来源（非 SP 单元为 "lyric"）
```

### 4.3 对齐路径

```python
@dataclass(frozen=True)
class AlignmentOp:
    """DP 转移的原子操作（5 种，覆盖全部场景）."""
    kind: Literal["REPLACE", "WORD_SPAN", "SPLIT", "DROP", "SP_ALIGN"]
    unit: Unit | None              # 涉及的新单元（DROP 时为 None）
    token_indices: tuple[int, ...] # 涉及的原 token 索引
    op_cost: float                 # 该操作代价（已乘权重）

@dataclass
class AlignmentPath:
    ops: list[AlignmentOp]         # 有序操作序列
    total_cost: float              # Σ op_cost
    sp_placements: list[int]       # SP 最终落到哪些原 token 索引
```

#### 5 种操作的精确语义

| 操作 | 单元 → token | 触发场景 | duration / pitch 处理 |
|------|-------------|----------|----------------------|
| `REPLACE` | 1 zh ↔ 1 token | 中文标准替换 | 继承原 token |
| `WORD_SPAN` | 1 en ↔ k token | 英文词跨音符 | k 个 token 各继承原值，首 `type=2` 其余 `type=3` |
| `SPLIT` | 1 token → n zh | 字数多于 token（扩容） | duration 切分给 n 字，pitch 继承宿主 |
| `DROP` | 0 unit ↔ 1 token | 冗余原 token（压缩） | 变空，duration 转移给同 section 已填 token |
| `SP_ALIGN` | 1 sp ↔ 1 token（被标记为 SP） | SP 放置（位置可移） | 位置移动量计入结构代价 |

**核心不变量**：
- 中文单元 `max_occupy=1`（强制 `REPLACE` 或被 `SPLIT` 覆盖）
- 英文单元 `max_occupy ∈ [1, K]`，DP 选最优占用数
- SP 数量硬约束：`SP_ALIGN` 次数 = 原 SP 数（否则路径不可行）

### 4.4 配置

```python
@dataclass(frozen=True)
class CostWeights:
    w_pitch: float = 0.5
    w_duration: float = 0.3
    w_structure: float = 0.2
    min_duration: float = 0.30          # 听感下限
    lambda_min_dur: float = 5.0         # 下限惩罚强度
    mu_word_boundary: float = 10.0      # 词边界违反惩罚
    max_word_occupy: int = 4            # 英文词最大占用（软上限）
```

---

## 5. DP 算法详解（核心）

### 5.1 符号

- 原 token 序列 $T = [t_0, ..., t_{m-1}]$，长度 $m$
- 新单元序列 $U = [u_0, ..., u_{n-1}]$，长度 $n$（含 SP 候选单元）
- $S^{\star}$ = 目标 SP_ALIGN 次数 $=$ 原 SP 数（守恒约束）
- $K$ = `max_word_occupy`（默认 4，遇超长词动态放宽，见 §8.1）

### 5.2 DP 状态

$$f(i, j, s, c) = \text{已处理 } T[:i] \text{ 与 } U[:j] \text{、已 SP\_ALIGN } s \text{ 次、} c \in \{0,1\} \text{ 表示 } t_{i-1} \text{ 是否可被 SPLIT 共享}$$

- $c=1$：上一步消费了 $t_{i-1}$（REPLACE/WORD_SPAN/SP_ALIGN），允许后续 SPLIT 共享它
- $c=0$：上一步是 DROP 或初始态，禁止 SPLIT（避免共享空 token）

**为何加 $c$ 维**：SPLIT（多字共享一 token）必须依附于已消费的 token。这 1 bit 是保证 SPLIT 合理性的最小扩展。

### 5.3 转移方程（前向松弛）

```
初始:  f(0, 0, 0, 0) = 0
目标:  answer = min_c f(m, n, S*, c)

对每个状态 f(i, j, s, c) with cost C:
┌─────────────────────────────────────────────────────────────┐
│ DROP (推进 i, 不推进 j)                                       │
│   if i < m:  relax f(i+1, j, s, 0) ← C + drop_cost(t_i)     │
├─────────────────────────────────────────────────────────────┤
│ REPLACE  zh 单元占 1 token                                   │
│   if u_j.kind=="zh" and i<m:                                │
│     relax f(i+1, j+1, s, 1) ← C + replace_cost(t_i, u_j)    │
├─────────────────────────────────────────────────────────────┤
│ WORD_SPAN  en 单元占 k 个连续 token                           │
│   if u_j.kind=="en":                                        │
│     for k in 1..min(K_eff, m-i):    # K_eff 见 §8.1 动态放宽  │
│       if any(t_{i..i+k-1}.is_sp): break  # 词不跨 SP 硬约束   │
│       relax f(i+k, j+1, s, 1) ← C + word_span_cost(...)     │
├─────────────────────────────────────────────────────────────┤
│ SPLIT  zh 单元共享上一 token t_{i-1}                          │
│   if u_j.kind=="zh" and c==1 and i>=1:                      │
│     relax f(i, j+1, s, 1) ← C + split_cost(t_{i-1}, u_j)    │
├─────────────────────────────────────────────────────────────┤
│ SP_ALIGN  sp 单元占 1 token, 把它变 SP, 计数 s+1              │
│   if u_j.kind=="sp" and i<m:                                │
│     relax f(i+1, j+1, s+1, 1) ← C + sp_align_cost(t_i, u_j) │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 操作代价（映射到 P/D/S 三项）

| 操作 | $P$（音高） | $D$（时长） | $S$（结构） |
|------|------------|------------|------------|
| `REPLACE` | 0（继承） | 0（继承原值） | 0 |
| `WORD_SPAN` | 0（各 token 继承） | 0（各继承） | 词长/token 数失衡的轻微惩罚 |
| `SPLIT` | 0（继承宿主） | $\lambda \cdot \max(0, d_{min} - d_{host}/(q+1))$，$q$=当前共享数 | 0 |
| `DROP` | $\|\text{pitch}_t - \text{pitch}_{\text{nearest kept}}\|$ | 转 duration 给邻居 | 0 |
| `SP_ALIGN` | 原 token 变 SP 后 pitch 归 0；若该 token 原本非 SP，计 pitch 损失（= 原 pitch） | 继承原 token duration | $\|\text{new pos} - \text{orig pos}\|$（移动惩罚；原位为 0） |

每操作 `op_cost` = $w_p P + w_d D + w_s S$。

**关键洞察**：`REPLACE` 和"原位 `SP_ALIGN`"代价为 0，是算法的"吸引子"——字数匹配且 SP 不动时，DP 自然走到零代价路径，等价于直接替换。所有不匹配情况都通过正代价操作吸收，**无需 if 分支**。

### 5.5 剪枝（保证典型 track 秒级求解）

1. **SP 计数可行性**：剩余 SP 单元数 $\leq$ 剩余 token 数；已 ALIGN 数 $\leq S^{\star}$
2. **词边界硬约束**：`WORD_SPAN` 区间遇原 SP 立即 `break`
3. **max_occupy 上限**：`WORD_SPAN` 的 $k \leq \min(K_{eff}, m-i)$
4. **SPLIT 共享上限**：单个 token 被共享次数 $\leq \lceil d_{host}/d_{min} \rceil$（避免 duration 被切到负数）
5. **beam search（可选退化）**：极长 track（>500 token）时每步保留 top-B 状态，牺牲最优换性能；默认关闭，由参数 `max_tokens_for_optimal` 控制（默认 500）

### 5.6 路径重建（回溯）

每个状态记录最优前驱 `(prev_state, op)`。从 `f(m, n, S*, c*)` 回溯到 `f(0, 0, 0, 0)`，按时间顺序输出 `AlignmentOp` 列表 → `AlignmentPath`。

```python
def reconstruct(f_table, m, n, S_star) -> AlignmentPath:
    c_end = argmin_c f_table[m][n][S_star][c]
    ops, i, j, s, c = [], m, n, S_star, c_end
    while (i, j, s, c) != (0, 0, 0, 0):
        prev, op = f_table[i][j][s][c].back_ptr
        ops.append(op)
        i, j, s, c = prev
    ops.reverse()
    sp_placements = [op.token_indices[0] for op in ops if op.kind == "SP_ALIGN"]
    return AlignmentPath(ops, f_table[m][n][S_star][c_end], sp_placements)
```

### 5.7 复杂度

| 量 | 表达 | 典型值 |
|----|------|--------|
| 状态数 | $m \cdot n \cdot S^{\star} \cdot 2$ | $150 \times 150 \times 10 \times 2 = 450K$ |
| 每状态转移 | REPLACE/DROP/SP_ALIGN $O(1)$；WORD_SPAN $O(K)$；SPLIT $O(1)$ | — |
| 总操作 | $O(m \cdot n \cdot S^{\star} \cdot K)$ | $\sim$1.8M |
| 纯 Python 预估 | — | 1-3 秒/track |
| numpy 向量化 | — | <0.5 秒/track |

---

## 6. 预处理与后处理

### 6.1 LyricNormalizer（新歌词归一化）

输入：用户原始歌词串（中英混合、可能含数字/标点/换行）
输出：`(归一化文本, SP候选位置列表)`

```
步骤:
1. 合并多余空白（连续空格/制表符/换行 → 单空格；但记录换行位置为强 SP 候选）
2. 阿拉伯数字 → 中文（复用 _normalize_chinese_numbers，如 "123" → "一二三"）
   - 由参数 normalize_digits 控制（默认 True）
3. 英文统一转小写（g2p_en 对大小写不敏感；统一显示）
4. 标点分类标记（不删除，记录位置与强度）:
   - 强: \n(换行) 。 . ！ ! ？
   - 中: ， , ； ; ：
   - 删除: 引号/括号/其他符号（不参与 SP 候选）
5. SP 候选筛选（目标数量 = 原 SP 数 S*）:
   if 强标点数 ≥ S*: 取前 S* 个强标点位置
   elif 强+中 ≥ S*: 取全部强 + 补中到 S*
   else: 不足部分, 在剩余字间隔最大处均匀补 SP 候选
6. 输出: 去除标点的纯文本 + SP 候选位置列表(按文本顺序)
```

**关键**：SP 候选筛选是**确定性预处理**（非 DP 决策），DP 只看到数量已对齐的 SP 单元序列，守恒约束自然成立。

### 6.2 UnitTokenizer（单元切分）

```
输入: 归一化文本 + SP 候选位置
输出: Unit 列表

逐字扫描:
  中文字符(CJK Unicode range) → Unit(kind="zh", max_occupy=1,
                                       phoneme=char_to_phoneme(ch))
  连续英文字母               → Unit(kind="en", max_occupy=min(len,K),
                                       phoneme=_word_to_phoneme(word))
  SP 候选位置                → Unit(kind="sp", max_occupy=1,
                                       phoneme="<SP>", source="punct")
合并相邻同类型(可选优化), 保持文本顺序
```

复用现有 `char_to_phoneme` / `_word_to_phoneme`，零新增音素逻辑。

### 6.3 TokenRebuilder（路径 → 新 token 序列）

按 `AlignmentPath.ops` 顺序，把每个 op 翻译为输出 token：

| Op | 输出 token 生成 | note_type | note_pitch |
|----|----------------|-----------|------------|
| `REPLACE` | 1 token: text/phoneme←unit, 其余继承原 token | 继承原值 | 继承原值 |
| `WORD_SPAN`(k token) | k token: 全部 text/phoneme←unit(英文词) | 首=2, 其余=3 | 各继承原值 |
| `SPLIT`(n unit 共享 1 原 token) | n token: 各 text/phoneme←各 unit | 继承宿主 | 继承宿主 |
| `DROP` | 0 token（原 token 消失，duration 转移） | — | — |
| `SP_ALIGN` | 1 token: text/phoneme="<SP>" | 1（段尾标记） | 0（休止） |

**输出 token 数** $m' = m - \text{DROP数} + \sum_{\text{SPLIT}}(n-1)$

duration 此阶段全部置占位 `None`，由 DurationAllocator 填充。

### 6.4 DurationAllocator（duration 重分配）

**三约束**：① 总时长守恒 $\sum d^{new} = \sum d^{orig}$；② 非 SP $\geq 0.30s$；③ 浮点清理。

```
阶段 1 — 区间内分配:
  对每个 SPLIT 区间(宿主 token 被 n unit 共享):
    宿主原 duration 按 unit 数均分(中文等宽)
  对每个 WORD_SPAN 区间(k token 共享一词):
    各 token duration 保持原值不变(已是 token 级)
  对每个 REPLACE/SP_ALIGN:
    duration 继承原 token

阶段 2 — DROP 补偿:
  对每个被 DROP 的原 token:
    其 duration 按同 section 内已填 token 的现有 duration 比例分配给它们
    (section = 相邻 SP_ALIGN 之间的 token 组)

阶段 3 — min_duration 下限保护:
  while 存在非 SP token with d < 0.30:
    找同 section 内最长的非 SP token(>0.30)
    借出 (0.30 - d_short) 给短 token
    若无足够余量可借, 标记警告 MIN_DURATION_UNRESOLVED, 不强行

阶段 4 — 浮点清理:
  复用 _fmt_durs: 保留 2 位小数, 末元素校正以确保 Σ 不变
```

### 6.5 f0 处理

- **默认原样保留**：f0 是帧级数据，与 token 数无关，token 重排不影响 f0 序列。总时长守恒保证 f0 时间轴不变。
- **变速（speed≠1）**：`SpeedAdapter` 复用现有 `_apply_speed`，对 duration 乘 `1/speed`，对 f0 做 `numpy.interp` 线性插值重采样（帧数等比变化）。
- **不修改 f0 的硬约束**：除变速外，任何模块不得修改 f0 字段。

---

## 7. 边界情况与错误处理

| # | 边界情况 | 处理策略 | 严重度 |
|---|---------|----------|--------|
| 1 | 输入 `None` / 空字符串 / 非 JSON | 节点返回错误信息（复用 `_safe_string`） | **报错** |
| 2 | 原 JSON 解析失败 / 字段缺失 | 报错 + 指出缺失字段 | **报错** |
| 3 | 新歌词为空 | 报错（无歌词可对齐） | **报错** |
| 4 | 新歌词全标点/全空白（归一化后无内容） | 报错 | **报错** |
| 5 | 原 track 无 SP（$S^{\star}=0$） | 正常处理，新歌词标点全部 DROP，整段连续对齐 | 降级 |
| 6 | 新歌词标点 < $S^{\star}$ | 预处理在最大字间隔处均匀补 SP 候选 | 降级 |
| 7 | 新歌词标点 > $S^{\star}$ | 按强度筛选到 $S^{\star}$ 个（强优先） | 降级 |
| 8 | 字数远多于原 token（>2×） | SPLIT 大量触发，触发 min_duration 警告 | 降级+警告 |
| 9 | 字数远少于原 token（<0.5×） | DROP 大量触发，duration 集中 | 降级+警告 |
| 10 | 单 token track（$m=1$） | 整字 SPLIT 或单 REPLACE | 降级 |
| 11 | 英文词长 > $K$ | 动态放宽 $K_{eff}$（见 §8.1） | 降级 |
| 12 | min_duration 不可解 | 标记警告，不强行借时间 | 降级+警告 |
| 13 | duration 总和为 0 / 负 / NaN | 报错（原数据损坏） | **报错** |
| 14 | 浮点精度（如 `0.3200000006`） | `_fmt_durs` 清理 | 自动 |

### 7.1 警告机制

节点返回 `(输出 JSON, warnings_list)`，通过 ComfyUI UI 反馈展示。警告类型：
- `HIGH_SPLIT_RATIO`：SPLIT 产生的 token 占比 > 40%
- `HIGH_DROP_RATIO`：DROP 产生的 duration 占比 > 30%
- `MIN_DURATION_UNRESOLVED`：存在无法满足 0.30s 下限的 token
- `SP_REDISTRIBUTED`：SP 位置偏离原位 > 平均 3 token

---

## 8. 关键设计细节

### 8.1 英文超长词处理（决策 #11）

当英文词音素数 > `max_word_occupy`（默认 4）时，采用**动态放宽**：

```python
K_eff = min(max_word_occupy, 词内音素数, m - i)
# 若词内音素数 > max_word_occupy, 放宽到 min(词内音素数, m-i)
```

理由：硬截断会丢歌词，硬拆词违背"整词"假设；动态放宽让代价函数自然惩罚长占用（`word_balance_penalty` 随 $k$ 增大而增大），DP 仍选全局最优。

### 8.2 Section 定义（用于 DROP 补偿与 min_duration 借调）

Section = 相邻两个 `SP_ALIGN` 操作之间的 token 组（不含 SP token 本身）。DROP 的 duration 转移与 min_duration 借调都在 section 内部进行，不跨 SP 边界。

### 8.3 与现有实现的根本差异

| 维度 | 现有 `MIDIEditLyrics` | 本设计 |
|------|---------------------|--------|
| 决策方式 | 按场景 if/else（Expand/Collapse/匹配） | 单一 DP，5 操作统一 |
| 字数匹配 | 三套分支处理 | 同一 DP 自动选操作组合 |
| SP 处理 | 硬保留 | 位置可移，代价权衡 |
| 英文词 | 固定按词长比例分配 | DP 在 $[1,K_{eff}]$ 选最优占用 |
| 最优性 | 局部贪心 | 全局最优（给定代价函数） |

---

## 9. 测试策略

新建独立测试文件 `tests/test_alignment.py`（不混入现有 `test_nodes.py`）。

### 9.1 测试层次

**第 1 层：模块单元测试**（每个模块独立，纯函数易测）
- `test_cost.py`：P/D/S 三项计算的数值正确性（边界值、零值、极端值）
- `test_normalizer.py`：归一化规则（数字转中文、标点分类、空白合并、SP 候选筛选）
- `test_tokenizer.py`：中英切分、max_occupy 标注、音素生成
- `test_dp.py`：DP 状态转移正确性（构造小规模用例手算 vs 算法输出）
- `test_rebuilder.py`：5 种操作的 token 生成
- `test_duration.py`：守恒、下限、浮点清理

**第 2 层：不变量测试**（手写断言，不引入 hypothesis 依赖）
对所有输入成立的不变量：
- ✅ SP 数量守恒：`output.SP_count == input.SP_count`
- ✅ 总 duration 守恒：`|Σ output.duration − Σ input.duration| < 0.01s`
- ✅ 字段完整性：每个输出 token 有 text/phoneme/duration/note_pitch/note_type
- ✅ 类型正确：duration 是 float、note_pitch 是 int、note_type ∈ {0,1,2,3}
- ✅ f0 未修改（speed=1 时）：`output.f0 == input.f0`
- ✅ 输出 JSON 可被 `json.loads` 解析

**第 3 层：场景集成测试**（端到端，关键路径覆盖）
| 场景 | 输入特征 | 预期主导操作 |
|------|----------|-------------|
| 字数等长 | 中文字数 = 原 token 数 | 全 REPLACE |
| 字数多于原 | 2× 字数 | SPLIT 主导 |
| 字数少于原 | 0.5× 字数 | DROP 主导 |
| 纯英文 | 全英文歌词 | WORD_SPAN 主导 |
| 中英混合 | 交错 | REPLACE + WORD_SPAN |
| SP 移动 | 标点位置 ≠ 原 SP | SP_ALIGN 偏移 |
| 无标点 | 纯文字 | SP 候选均匀补 |
| 极端不匹配 | 3× 字数 | SPLIT + min_duration 警告 |

**第 4 层：回归测试**
- 用现有 `docs/midi-edit-lyrics.json` 的真实 track 作输入，新歌词替换后断言：
  - 旋律走向（pitch 序列的符号变化点数）与原曲差异 ≤ 2（弱断言）
  - duration 总和守恒
  - 输出可被 SoulX-Singer 消费（字段格式校验）

**第 5 层：性能测试**
- 构造 50 / 150 / 300 / 500 token 的合成 track，断言：
  - 150 token track < 1 秒（numpy 向量化后）
  - 500 token track < 5 秒（或触发 beam search 退化）
- 标记为 `@pytest.mark.slow`，CI 可选跳过

### 9.2 测试数据管理
- 合成小用例：直接在测试代码里构造（易读、自解释）
- 真实回归用例：放 `tests/fixtures/`（JSON 文件）
- 复用现有 `docs/midi-edit-lyrics.json` 作真实 fixture

---

## 10. 工程化

### 10.1 ComfyUI 节点接口

```python
class MidiLyricsAlignment:
    """统一对齐算法节点（基于联合 DP）."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_json": ("STRING", {"multiline": True}),
                "lyrics": ("STRING", {"multiline": True}),
                # 注: 移除 split_mode — 新 DP 算法自动决定字数分配，无需该参数
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "normalize_digits": ("BOOLEAN", {"default": True}),
                "force_tone4": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # 高级参数（权重微调），默认隐藏
                "w_pitch": ("FLOAT", {"default": 0.5}),
                "w_duration": ("FLOAT", {"default": 0.3}),
                "w_structure": ("FLOAT", {"default": 0.2}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_json",)
    # warnings 通过 logging 或 PreviewAny 展示
```

### 10.2 性能优化路径

1. **第一版**：纯 Python + dataclass，先保证正确性
2. **若性能不足**：DP 内循环用 numpy 向量化（状态转移批量计算）
3. **若仍不足**：极长 track 触发 beam search 退化（牺牲最优换性能）

### 10.3 渐进式集成

- Phase 1：实现核心 DP + 单元测试，先用合成小用例验证正确性
- Phase 2：集成预处理/后处理，跑场景测试
- Phase 3：真实回归测试（`docs/midi-edit-lyrics.json`）
- Phase 4：性能优化（若需要）
- Phase 5：文档更新（README + CHANGELOG）

---

## 11. 待解决问题（留给实现计划）

以下细节留给 writing-plans 阶段进一步拆解：

- SPLIT 共享数的精确上界计算（剪枝 #4 的公式细化）
- beam search 的 B 值选择与质量评估
- DurationAllocator 阶段 2 的 section 内比例分配的浮点稳定性
- 警告的 UI 展示机制（ComfyUI 的具体 API）
- 与现有 `MIDIEditLyrics` 的并行存在策略（是否共享某些 helper）

---

## 12. 参考与依据

- 数据格式：`docs/midi-json-format.md`
- 现有实现：`nodes.py`（`MIDIEditLyrics`、`_apply_speed`、`_fmt_durs`、`char_to_phoneme` 等）
- 历史决策：`CHANGELOG.md`（duration 篡改修复、英文词处理等）
- 需求来源：`docs/REQUIREMENT.md`
- 上游音素表：[SoulX-Singer phone_set.json](https://github.com/Soul-AILab/SoulX-Singer/blob/main/soulxsinger/utils/phoneme/phone_set.json)
- 算法范式：序列对齐 DP（类 Needleman-Wunsch / DTW 的扩展，支持一对多/多对一）

---

## 13. 决策溯源（Brainstorming 记录）

本设计通过 brainstorming skill 的 5 轮澄清问题收敛而成：

1. **"保持原旋律"的定义** → D 加权综合保真（pitch/duration/structure 三维代价）
2. **SP 停顿处理** → B SP 软约束（数量守恒，位置可移）
3. **对齐原子单元** → A 混合原生粒度（中文字/英文词）
4. **代价函数度量** → 操作惩罚形式（P/D/S 三项纯函数）
5. **算法实现路径** → A 联合动态规划（单一 DP，5 操作，全局最优）

每轮决策的完整备选与权衡见 brainstorming 会话记录。

---

**文档结束**

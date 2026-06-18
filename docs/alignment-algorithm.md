# MIDI 歌词统一对齐算法（MidiLyricsAlignment）

> 本文是 `MidiLyricsAlignment` 节点（统一歌词对齐算法）的**算法设计说明**，
> 面向项目贡献者、开发维护者，以及希望理解算法工作原理的用户。
> 文档自包含：不阅读 spec / plan 也能完整理解算法。

---

## 1. 概述

`MidiLyricsAlignment` 是一个**基于联合动态规划（Joint Dynamic Programming）的歌词对齐算法**。它用**单一代价函数的最小化**，取代传统歌词替换中按场景分流的 `if/else` 处理，把"把新歌词贴回原曲旋律"这件事统一建模成一次序列对齐求最优解。

它要解决的核心问题是：当用户给出一段**中英文混合、可能没有断句标点、字数可能多于或少于原歌词**的新歌词时，如何让它**贴合原曲的旋律（音高走向 + 节奏型 + 结构边界）**，而不是机械地逐字覆盖。

与现有 `MIDIEditLyrics` 节点的根本差异在于：现有节点按"字数匹配 / 字数多（Expand）/ 字数少（Collapse）"三套分支分别处理，SP 停顿位置硬保留、英文词按词长比例固定分配；本算法把所有这些情况收敛到**同一次 DP 求解**——由代价函数在 5 种原子操作之间自动选择全局最优组合，不存在场景分支（详见第 9 节对比）。

本节点为**独立新增节点**，不替换 `MIDIEditLyrics`，二者可并存。

---

## 2. 设计目标

| 目标 | 含义 |
|------|------|
| 统一性 | 单一算法处理所有匹配 / 不匹配情况，无 `if/else` 场景分支 |
| 最优性 | 在显式代价函数下求**全局最优**对齐，而非局部贪心 |
| 保真度 | 从音高（pitch）、时长（duration）、结构（structure）三个维度加权保持原旋律 |
| 灵活性 | `phoneme` / `duration` / `note_pitch` / `note_type` 的数量与取值均可由算法调整 |
| 兼容性 | 输出严格遵循 SoulX-Singer 输入格式（见 `docs/midi-json-format.md`） |
| 轻量 | 零外部算法依赖（除复用现有 `numpy` 变速逻辑外），适配 ComfyUI 插件环境 |

---

## 3. 数据格式前提

算法的输入输出均为 **MIDI JSON**，即一个 **track 对象数组**，每个 track 表示一段人声片段。字段分两类粒度：

- **token 级字段（5 个，1:1 对齐）**：`text` / `phoneme` / `duration` / `note_pitch` / `note_type`，按相同索引一一对应，数组长度相同。
- **帧级字段（1 个，独立）**：`f0`，长度由 track 总时长决定，**不与 token 1:1 对齐**。

字段的具体类型、单位、取值与样例，见 [`docs/midi-json-format.md`](./midi-json-format.md)，本文不重复定义。以下仅说明算法实际依赖的字段语义要点：

- 算法主要在 **token 级 5 字段** 上工作：`text` 与 `phoneme` 是歌词载体（会被改写），`duration` / `note_pitch` / `note_type` 是旋律载体（保真目标，必要时在守恒约束下重分配）。
- `<SP>` 表示停顿（silence / pause），是 `text` 的特殊取值；`note_pitch=0`、`note_type=1` 通常出现在 `<SP>` 位置。
- `note_type` 语义：`1` = 段尾 / 乐句收尾，`2` = 普通音符 / 词首音符，`3` = 词内延续音符（英文词跨多音符时的后续音符）。
- `f0` 默认**原样保留**：token 重排不影响帧级 f0 序列；仅当用户指定变速（`speed != 1`）时，才对 f0 做线性插值重采样。

---

## 4. 核心思想

算法建立在三个核心设计决策之上。本节说明它们的**是什么**；为什么这样选的完整论证见 spec 的决策溯源。

### 4.1 加权综合保真

"保持原旋律"被定义为一个**三维加权代价函数**：从**音高偏差 $P$**、**时长偏差 $D$**、**结构违反 $S$** 三个维度度量一条对齐方案 $A$ 偏离原曲的程度，再加权求和：

$$
\text{Cost}(A) \;=\; w_p \, P(A) \;+\; w_d \, D(A) \;+\; w_s \, S(A)
$$

算法的目标是找到使 $\text{Cost}(A)$ 最小的对齐方案。默认权重如下表：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `w_pitch` (`$w_p$`) | 0.5 | 音高偏差权重 |
| `w_duration` (`$w_d$`) | 0.3 | 时长偏差权重 |
| `w_structure` (`$w_s$`) | 0.2 | 结构违反权重 |
| `min_duration` | 0.30 s | 非 SP token 的听感下限 |
| `lambda_min_dur` | 5.0 | 听感下限违反的惩罚强度 |
| `mu_word_boundary` | 10.0 | 词边界违反惩罚（接近硬约束） |
| `max_word_occupy` | 4 | 英文词占用原 token 数的软上限 |

三项 $P$ / $D$ / $S$ 都是**无状态的纯函数**，针对每一种"原子操作"给出确定代价（见 6.5 节的代价映射表）。

### 4.2 SP 软约束

原曲中的 `<SP>` 停顿被当作**软约束**处理：

- **数量守恒**：输出中的 `<SP>` 数量必须等于原曲的 `<SP>` 数量 $S^{\star}$（硬约束，由 DP 状态保证）。
- **位置可移**：SP 落在哪个 token 上由 DP 决定，不必保留原位；但**位置移动会产生结构代价 $S$**（移动越远惩罚越大，原位则惩罚为 0）。

这使得新歌词的自然断句能够"借用"原曲的停顿额度，而不被原 SP 位置硬性卡死。

### 4.3 混合原生粒度

对齐的**原子单元**采用混合原生粒度，尊重两种语言的天然边界：

- **中文按字（character）**：一个汉字是一个对齐单元，`max_occupy = 1`（最多占用 1 个原 token）。
- **英文按词（word）**：连续字母组成一个完整单词作为一个对齐单元，`max_occupy <= K`（可跨多个原 token，$K$ = `max_word_occupy`，默认 4）。
- **SP 单元**：`max_occupy = 1`，对应一个被标记为 `<SP>` 的原 token。

这样避免了"英文按字母切"带来的合成噪声，也避免了"把英文词硬拆"破坏整词假设。

---

## 5. 处理管线

算法是一条 **9 步确定性管线**。前 5 步是确定性预处理，步骤 6 是**唯一决策点**，后 3 步是确定性重建（最后序列化为输出 JSON）。

```
输入: 原 MIDI JSON + 新歌词文本 + 参数(weights / thresholds / speed)
  │
  ▼
[1. Parser]            原 JSON 字符串 → track 对象列表
  │
  ▼
[2. Token 提取]        每 track 提取 token 序列（标注 SP，保留 f0 / meta）
  │
  ▼
[3. LyricNormalizer]   新歌词 → 归一化（去空白 / 数字转中文 / 英文小写 / 标点分类）
  │
  ▼
[4. SP 候选筛选]       按强度选出 S* 个 SP 候选位置（确定性，非 DP 决策）
  │
  ▼
[5. UnitTokenizer]     归一化文本 + SP 候选 → Unit 序列（中文字 / 英文词 / SP，带 max_occupy）
  │
  ▼
[6. AlignmentDP]       (token 序列, Unit 序列) → 最优对齐路径（含 SP 新位置）   ← 核心决策点
  │
  ▼
[7. TokenRebuilder]    路径 → 新 token 序列（text / phoneme / note_pitch / note_type 落位）
  │
  ▼
[8. DurationAllocator] 原 duration → 重分配（总时长守恒 + 0.30s 下限 + 浮点清理）
  │
  ▼
[9. SpeedAdapter]      变速时 duration / f0 同步缩放（speed = 1 时完全不动）
  │
  ▼
序列化: 新 track → 输出 MIDI JSON
```

> 说明：spec 把"解析 + token 提取"合写在 Parser 框、把"归一化 + SP 候选筛选"合写在 LyricNormalizer 框；此处按原子动作展开为独立的 9 步，便于理解数据流，不改变模块边界（步骤 1–2 同属 `Parser`，步骤 3–4 同属 `LyricNormalizer`，见第 11 节）。

**关键性质**：步骤 6（AlignmentDP）是管线中唯一的决策点。所有"字数多 / 字数少 / 等长"的差异，都收敛到这**同一次 DP 求解**——前 5 步预处理与后 3 步重建都不含基于字数匹配情况的 `if` 分支。这是本算法"统一性"的工程体现。

---

## 6. 联合动态规划

本节是算法的技术核心。

### 6.1 符号定义

| 符号 | 含义 |
|------|------|
| $T = [t_0, \dots, t_{m-1}]$ | 原曲 token 序列，长度 $m$ |
| $U = [u_0, \dots, u_{n-1}]$ | 新歌词单元序列，长度 $n$（含 SP 候选单元） |
| $S^{\star}$ | 目标 SP_ALIGN 次数 = 原 SP 数（守恒约束） |
| $K$ | `max_word_occupy`（默认 4；遇超长英文词动态放宽为 $K_{\text{eff}}$，见 8.1） |

### 6.2 DP 状态

$$
f(i, j, s, c) \;=\; \text{已处理 } T[:i] \text{ 与 } U[:j] \text{、已执行 } s \text{ 次 SP\_ALIGN、共享标志 } c \text{ 时的最小代价}
$$

其中：

- $i \in [0, m]$：已消费的原 token 前缀长度。
- $j \in [0, n]$：已消费的新单元前缀长度。
- $s \in [0, S^{\star}]$：已执行的 SP_ALIGN 次数（用于强制数量守恒）。
- $c \in \{0, 1\}$：**共享标志**，表示 $t_{i-1}$ 是否可被 SPLIT 共享。
  - $c = 1$：上一步消费了 $t_{i-1}$（REPLACE / WORD_SPAN / SP_ALIGN），允许后续 SPLIT 继续共享它。
  - $c = 0$：上一步是 DROP 或初始态，禁止 SPLIT（避免共享一个"空"的 token）。

引入 $c$ 这 1 个 bit 是为了让 SPLIT（多字共享一个 token）只能依附于一个**已被实际消费**的 token，这是保证 SPLIT 合理性的最小状态扩展。

**初始与目标**：

$$
f(0, 0, 0, 0) = 0, \qquad \text{answer} = \min_{c} f(m, n, S^{\star}, c)
$$

### 6.3 五种原子操作

DP 的每一次转移都从下列 5 种操作中选一种。它们覆盖了所有对齐场景：

| 操作 | 单元 → 原 token | 触发场景 | pitch / duration 处理 |
|------|----------------|----------|----------------------|
| `REPLACE` | 1 中文单元 ↔ 1 token | 中文标准替换 | 继承原 token 的 pitch / duration |
| `WORD_SPAN` | 1 英文单元 ↔ k 个连续 token | 英文词跨音符 | k 个 token 各继承原 pitch / duration；首 token `note_type=2`，其余 `note_type=3` |
| `SPLIT` | n 个中文单元 共享 1 个 token | 字数多于 token（扩容） | pitch 继承宿主 token；duration 在 n 字间切分 |
| `DROP` | 0 单元 ↔ 1 token | 冗余原 token（压缩） | 原 token 消失，其 duration 转移给同 section 内已填 token |
| `SP_ALIGN` | 1 SP 单元 ↔ 1 token（标记为 SP） | SP 放置（位置可移） | 该 token 变 SP（pitch=0）；位置移动量计入结构代价 |

**三条核心不变量**：

- 中文单元 `max_occupy = 1`：只能 REPLACE，或被 SPLIT 覆盖（多字共享）。
- 英文单元 `max_occupy ∈ [1, K_{\text{eff}}]`：由 DP 在区间内选最优占用数 $k$。
- SP 数量硬约束：整条路径中 `SP_ALIGN` 的总次数必须 $= S^{\star}$，否则路径不可行。

### 6.4 状态转移（前向松弛）

对每个状态 $f(i, j, s, c)$（设其当前累计代价为 $C$），向前松弛 5 种操作：

```
初始:  f(0, 0, 0, 0) = 0
目标:  answer = min_c f(m, n, S*, c)

对每个状态 f(i, j, s, c) with cost C:
┌────────────────────────────────────────────────────────────────┐
│ DROP（推进 i，不推进 j）                                          │
│   if i < m:                                                     │
│     relax f(i+1, j, s, 0) ← C + drop_cost(t_i)                 │
├────────────────────────────────────────────────────────────────┤
│ REPLACE（中文单元占 1 token）                                     │
│   if u_j.kind == "zh" and i < m:                                │
│     relax f(i+1, j+1, s, 1) ← C + replace_cost(t_i, u_j)       │
├────────────────────────────────────────────────────────────────┤
│ WORD_SPAN（英文单元占 k 个连续 token）                            │
│   if u_j.kind == "en":                                          │
│     for k in 1 .. min(K_eff, m - i):        # K_eff 见 8.1      │
│       if any(t_{i..i+k-1}.is_sp): break     # 词不跨 SP（硬约束）│
│       relax f(i+k, j+1, s, 1) ← C + word_span_cost(t_{i..i+k-1}, u_j, k) │
├────────────────────────────────────────────────────────────────┤
│ SPLIT（中文单元共享上一个 token t_{i-1}）                         │
│   if u_j.kind == "zh" and c == 1 and i >= 1:                    │
│     relax f(i, j+1, s, 1) ← C + split_cost(t_{i-1}, u_j)       │
├────────────────────────────────────────────────────────────────┤
│ SP_ALIGN（SP 单元占 1 token，把它变 SP，计数 s+1）                │
│   if u_j.kind == "sp" and i < m:                                │
│     relax f(i+1, j+1, s+1, 1) ← C + sp_align_cost(t_i, u_j)    │
└────────────────────────────────────────────────────────────────┘
```

其中 `relax(target) ← value` 表示"若 `value` 小于 `target` 当前值，则更新 `target` 并记录前驱指针"。每种 `*_cost` 都是 4.1 节代价函数在该操作上的取值（$w_p P + w_d D + w_s S$）。

### 6.5 操作代价映射

下表给出每种操作的三项 $P$ / $D$ / $S$ 如何计算。每条操作最终代价为 `op_cost =` $w_p P + w_d D + w_s S$。

| 操作 | $P$（音高） | $D$（时长） | $S$（结构） |
|------|------------|------------|------------|
| `REPLACE` | 0（继承原 pitch） | 0（继承原 duration） | 0 |
| `WORD_SPAN` | 0（各 token 继承原 pitch） | 0（各 token 继承原 duration） | 词长 / token 数失衡的轻微惩罚（`word_balance_penalty`，随 $k$ 增大而增大） |
| `SPLIT` | 0（继承宿主 pitch） | $\lambda \cdot \max\bigl(0,\; d_{\min} - d_{\text{host}}/(q+1)\bigr)$，其中 $q$ 为该 token 当前被共享的次数，$\lambda$ = `lambda_min_dur` | 0 |
| `DROP` | $\lvert \text{pitch}_t - \text{pitch}_{\text{nearest kept}} \rvert$（相对最近的保留 token） | 其 duration 转移给邻居（在 DurationAllocator 阶段完成） | 0 |
| `SP_ALIGN` | 若该 token 原本非 SP：损失 = 原 pitch（因变 SP 后 pitch 归 0）；若原本即 SP：0 | 继承原 token duration | $\lvert \text{new pos} - \text{orig pos} \rvert$（位置移动惩罚；原位为 0） |

**关键洞察**：`REPLACE` 与"原位 `SP_ALIGN`"的代价都为 0，是算法的**吸引子**。当字数匹配且 SP 不动时，DP 自然收敛到一条零代价路径，等价于"直接逐字替换"；而所有不匹配情况都通过正代价操作吸收。正因如此，算法**不需要任何基于字数是否匹配的 `if` 分支**。

### 6.6 剪枝策略

为保证典型 track 在秒级内求解，DP 应用以下剪枝规则：

1. **SP 计数可行性**：剩余 SP 单元数 $\leq$ 剩余 token 数；已 ALIGN 次数 $\leq S^{\star}$。
2. **词边界硬约束**：`WORD_SPAN` 的区间内只要遇到原 SP，立即 `break`（词不跨 SP）。
3. **max_occupy 上限**：`WORD_SPAN` 的 $k \leq \min(K_{\text{eff}},\; m - i)$。
4. **SPLIT 共享上限**：单个 token 被共享次数 $\leq \lceil d_{\text{host}} / d_{\min} \rceil$，避免 duration 被切到负数 / 过小。
5. **beam search（可选退化）**：极长 track（> 500 token）时，每步只保留代价最低的 top-$B$ 个状态，以牺牲最优性换取性能。默认关闭，由参数 `max_tokens_for_optimal`（默认 500）控制是否触发。

### 6.7 路径重建（回溯）

DP 求解时，每个状态记录其最优前驱 `(prev_state, op)`。求解完成后，从终点 $f(m, n, S^{\star}, c^{\ast})$（其中 $c^{\ast} = \arg\min_c f(m, n, S^{\star}, c)$）沿前驱指针回溯到起点 $f(0,0,0,0)$，把沿途的操作逆序，即得到按时间顺序排列的 `AlignmentOp` 序列，封装为 `AlignmentPath`：

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

`sp_placements` 记录 SP 最终落到哪些原 token 索引上，供后续 DurationAllocator 划分 section 使用。

### 6.8 复杂度

| 量 | 表达式 | 典型 track（$m \approx n \approx 150$，$S^{\star} \approx 10$） |
|----|--------|------|
| 状态数 | $m \cdot n \cdot S^{\star} \cdot 2$ | $150 \times 150 \times 10 \times 2 = 4.5 \times 10^{5}$ |
| 单状态转移 | REPLACE / DROP / SP_ALIGN 为 $O(1)$；WORD_SPAN 为 $O(K)$；SPLIT 为 $O(1)$ | — |
| 总操作数 | $O(m \cdot n \cdot S^{\star} \cdot K)$ | 约 $1.8 \times 10^{6}$ |
| 纯 Python 预估 | — | 1 ~ 3 秒 / track |
| numpy 向量化后 | — | < 0.5 秒 / track |

---

## 7. 预处理与后处理

### 7.1 歌词归一化（LyricNormalizer）

输入：用户原始歌词串（中英混合、可能含数字 / 标点 / 换行）。
输出：`(归一化文本, SP 候选位置列表)`。

步骤：

1. **合并空白**：连续空格 / 制表符 / 换行合并为单个空格；但记录换行位置作为强 SP 候选。
2. **数字转中文**：阿拉伯数字转中文（如 `123` → `一二三`），复用现有 `_normalize_digits`；由参数 `normalize_digits` 控制（默认开启）。
3. **英文转小写**：统一显示，g2p_en 对大小写不敏感。
4. **标点分类**（不删除，记录位置与强度）：
   - 强：`\n`、`。`、`.`、`！`、`!`、`？`
   - 中：`，`、`,`、`；`、`;`、`：`
   - 删除：引号 / 括号 / 其他符号（不参与 SP 候选）
5. **SP 候选筛选**（目标数量 $= S^{\star}$）：
   - 若强标点数 $\geq S^{\star}$：取前 $S^{\star}$ 个强标点位置。
   - 否则若强 + 中 $\geq S^{\star}$：取全部强标点，再用中标点补到 $S^{\star}$。
   - 否则（不足）：在剩余字间隔最大处均匀补 SP 候选。
6. **输出**：去除标点的纯文本 + SP 候选位置列表（按文本顺序）。

**关键**：SP 候选筛选是**确定性预处理**（不是 DP 决策）。DP 看到的 SP 单元数量已经被对齐到 $S^{\star}$，因此守恒约束天然成立。

### 7.2 单元切分（UnitTokenizer）

输入：归一化文本 + SP 候选位置。
输出：`Unit` 列表。

逐字扫描：

- 中文字符（CJK Unicode 范围）→ `Unit(kind="zh", max_occupy=1, phoneme=char_to_phoneme(ch))`
- 连续英文字母 → `Unit(kind="en", max_occupy=min(len, K), phoneme=_word_to_phoneme(word))`
- SP 候选位置 → `Unit(kind="sp", max_occupy=1, phoneme="<SP>", source="punct")`

复用现有 `char_to_phoneme` / `_word_to_phoneme`，**不引入任何新的音素逻辑**。

### 7.3 Token 重建（TokenRebuilder）

按 `AlignmentPath.ops` 顺序，把每个操作翻译为输出 token：

| 操作 | 输出 token 生成 | note_type | note_pitch |
|------|----------------|-----------|------------|
| `REPLACE` | 1 token：text / phoneme ← unit，其余继承原 token | 继承原值 | 继承原值 |
| `WORD_SPAN`（k token） | k token：全部 text / phoneme ← 英文 unit | 首 = 2，其余 = 3 | 各继承原值 |
| `SPLIT`（n unit 共享 1 原 token） | n token：各 text / phoneme ← 各 unit | 继承宿主 | 继承宿主 |
| `DROP` | 0 token（原 token 消失，duration 转移） | — | — |
| `SP_ALIGN` | 1 token：text / phoneme = `<SP>` | 1（段尾标记） | 0（休止） |

**输出 token 数**：

$$
m' \;=\; m \;-\; \text{DROP 数} \;+\; \sum_{\text{SPLIT}} (n - 1)
$$

此阶段所有 token 的 `duration` 先置占位 `None`，统一由 DurationAllocator 填充。

### 7.4 Duration 分配（DurationAllocator）

三条约束：① **总时长守恒** $\sum d^{\text{new}} = \sum d^{\text{orig}}$；② 非 SP token $\geq$ `min_duration`（0.30 s）；③ 浮点清理。分四个阶段：

**阶段 1 — 区间内分配**

- 对每个 SPLIT 区间（宿主 token 被 $n$ 个单元共享）：宿主原 duration 按 unit 数均分（中文等宽）。
- 对每个 WORD_SPAN 区间（$k$ 个 token 共享一词）：各 token duration 保持原值不变（已是 token 级）。
- 对每个 REPLACE / SP_ALIGN：duration 继承原 token。

**阶段 2 — DROP 补偿**

- 对每个被 DROP 的原 token：其 duration 按同 section 内已填 token 的现有 duration **比例**分配给它们。
- section 定义见 8.2。

**阶段 3 — min_duration 保护**

- 当存在非 SP token 且 $d < 0.30$ 时：找同 section 内最长的非 SP token（$> 0.30$），借出 $(0.30 - d_{\text{short}})$ 给短 token。
- 若同 section 内无足够余量可借：标记警告 `MIN_DURATION_UNRESOLVED`，**不强行**借时间（避免破坏总时长守恒）。

**阶段 4 — 浮点清理**

- 复用现有 `_fmt_durs`：duration 保留 2 位小数，并校正末元素以确保四舍五入后 $\sum d^{\text{new}}$ 与目标值一致（消除 `0.3200000006` 之类的浮点伪影）。

### 7.5 f0 处理

- **默认原样保留**：f0 是帧级数据，与 token 数无关；token 重排不影响 f0 序列。总时长守恒保证了 f0 的时间轴不变。
- **变速（`speed != 1`）**：`SpeedAdapter` 复用现有 `_apply_speed`——对 duration 按速度比例缩放（加速 → 时长缩短），对 f0 用 `numpy.interp` 线性插值重采样（帧数随速度等比变化，音高轮廓的 Hz 数值不变，仅拉伸 / 压缩时间轴）。
- **硬约束**：除变速外，任何模块都不得修改 `f0` 字段。

---

## 8. 边界情况处理

边界处理按严重度分为三类：

| 类别 | 处理方式 | 涉及情况 |
|------|----------|----------|
| 报错 | 节点返回错误信息，不产出结果 | 输入为 `None` / 空串 / 非 JSON；原 JSON 解析失败 / 必填字段缺失；新歌词为空或归一化后无内容；原 duration 总和为 0 / 负 / NaN |
| 降级 | 算法自动处理，无需用户介入 | 原 track 无 SP（$S^{\star}=0$）；新歌词标点少于 $S^{\star}$（均匀补 SP 候选）或多于 $S^{\star}$（按强度筛选）；单 token track（$m=1$）；英文词长 $> K$（动态放宽 $K_{\text{eff}}$，见 8.1）；浮点精度清理 |
| 降级 + 警告 | 算法尽力处理并通过警告反馈 | 字数远多于原 token（$> 2\times$，SPLIT 大量触发）；字数远少于原 token（$< 0.5\times$，DROP 大量触发）；`min_duration` 不可解 |

**警告机制**：节点除输出 JSON 外，还返回一个警告列表，通过 ComfyUI UI 反馈展示。警告类型有四种：

| 警告类型 | 触发条件 |
|----------|----------|
| `HIGH_SPLIT_RATIO` | SPLIT 产生的 token 占比 > 40% |
| `HIGH_DROP_RATIO` | DROP 产生的 duration 占比 > 30% |
| `MIN_DURATION_UNRESOLVED` | 存在无法满足 0.30 s 下限的 token |
| `SP_REDISTRIBUTED` | SP 位置偏离原位平均 > 3 token |

### 8.1 英文超长词处理

当英文词的音素数 $>$ `max_word_occupy`（默认 4）时，采用**动态放宽**而非硬截断：

$$
K_{\text{eff}} \;=\; \min\bigl(\max(\text{max\_word\_occupy},\; \text{词内音素数}),\; m - i\bigr)
$$

即把 $K$ 放宽到"词内音素数"与"剩余 token 数"的较小者。这样既不丢歌词（不截断）、也不硬拆词（保留整词假设），而让代价函数通过 `word_balance_penalty`（随 $k$ 增大而增大）自然惩罚长占用，DP 仍能选出全局最优。

### 8.2 Section 定义

**Section** = 相邻两个 `SP_ALIGN` 操作之间的 token 组（不含 SP token 本身）。DROP 的 duration 转移（7.4 阶段 2）与 `min_duration` 借调（7.4 阶段 3）都只在 section 内部进行，**不跨 SP 边界**。这保证了一段歌词内部的时长调整不会"泄漏"到相邻乐句。

---

## 9. 与现有 MIDIEditLyrics 的对比

| 维度 | 现有 `MIDIEditLyrics` | 本算法（`MidiLyricsAlignment`） |
|------|---------------------|--------------------------------|
| 决策方式 | 按场景 `if/else`（Expand / Collapse / Collapse+Distribute） | 单一 DP，5 种操作统一处理 |
| 字数匹配 | 三套分支分别处理字数多 / 少 / 等长 | 同一次 DP 自动选择操作组合，无分支 |
| SP 处理 | SP 位置**硬保留** | SP **位置可移**，由代价权衡（数量仍守恒） |
| 英文词分配 | 固定按词长**比例**分配到多个音符 | DP 在 $[1, K_{\text{eff}}]$ 区间内选**最优**占用数 |
| 最优性 | 局部贪心 | 给定代价函数下的**全局最优** |
| 句数适配 | 依赖 CT-Transformer 智能拆句 + 比例硬切 | 无需外部模型，SP 软约束 + DP 统一吸收 |
| 与原节点关系 | 保留不动 | 独立新增，不替换；复用其纯函数 helper |

两者并存：本算法不修改 `MIDIEditLyrics` 及其测试，只复用其**纯函数**辅助工具（`char_to_phoneme` / `_word_to_phoneme` / `_normalize_digits` / `_apply_speed` / `_fmt_durs` / `_fmt_f0` 等）。

---

## 10. 性能特征

- **典型 track**（约 150 token，$S^{\star} \approx 10$）：纯 Python 实现 1 ~ 3 秒 / track；numpy 向量化后 < 0.5 秒 / track。
- **极长 track**（> 500 token）：状态数随 $m \cdot n \cdot S^{\star}$ 增长，纯精确求解会变慢。此时触发 **beam search 退化**（6.6 剪枝 #5），每步只保留 top-$B$ 个状态，以可控地牺牲最优性换取性能。退化由参数 `max_tokens_for_optimal`（默认 500）控制：token 数超过该阈值即启用 beam search。
- **优化路径**：实现分三步走——① 纯 Python + dataclass 先保证正确性；② 若性能不足，把 DP 内循环用 numpy 向量化（状态转移批量计算）；③ 若仍不足，对极长 track 启用 beam search。

---

## 11. 工程实现要点

### 11.1 模块划分

算法放在新建的 `alignment/` 子包中，保持 `nodes.py` 不膨胀。每个文件单一职责：

| 文件 | 职责 |
|------|------|
| `alignment/__init__.py` | 子包公开 API |
| `alignment/models.py` | 核心数据结构（`Token` / `Track` / `Unit` / `AlignmentOp` / `AlignmentPath` / `CostWeights`） |
| `alignment/parser.py` | JSON ↔ `Token` / `Track` 转换 |
| `alignment/preprocess.py` | `LyricNormalizer` + `UnitTokenizer` |
| `alignment/cost.py` | 5 种操作代价的纯函数（$P$ / $D$ / $S$ 三项） |
| `alignment/dp.py` | `AlignmentDP`（状态、转移、剪枝、回溯） |
| `alignment/rebuild.py` | `TokenRebuilder` + `DurationAllocator` |
| `alignment/speed.py` | `SpeedAdapter`（薄封装 `_apply_speed`） |
| `nodes.py`（新增节点） | `MidiLyricsAlignment` 节点入口，组装上述模块 |

### 11.2 与现有代码的复用关系

- **复用纯函数**：`char_to_phoneme` / `_word_to_phoneme` / `_normalize_digits` / `_apply_speed` / `_fmt_durs` / `_fmt_f0`。
- **不修改**现有 `MIDIEditLyrics` 及其测试 `tests/test_nodes.py`。
- **新增独立测试** `tests/test_alignment.py`。

### 11.3 节点接口

新建节点 `MidiLyricsAlignment`，主要输入：`midi_json`、`lyrics`、`speed`、`normalize_digits`、`force_tone4`，以及可选的高级权重参数（`w_pitch` / `w_duration` / `w_structure`，默认隐藏）。注意：相比 `MIDIEditLyrics`，本节点**移除了 `split_mode` 参数**——字数如何分配由 DP 自动决定，无需用户选择模式。输出为修改后的 `midi_json` 字符串，附带警告列表。

### 11.4 输出契约

输出 JSON 的字段（`text` / `phoneme` / `duration` / `note_pitch` / `note_type` / `f0`）严格遵循 [`docs/midi-json-format.md`](./midi-json-format.md)；`note_type` 语义不变（`1` = 段尾、`2` = 普通 / 词首、`3` = 词内延续）。

---

## 12. 参考

- **数据格式**：[`docs/midi-json-format.md`](./midi-json-format.md) —— MIDI JSON 各字段的类型、单位、取值与样例。
- **完整设计决策**：`docs/superpowers/specs/2026-06-18-midi-lyrics-alignment-design.md` —— 含每项决策的备选方案对比、权衡过程与 brainstorming 记录。
- **实现细节与代码**：`docs/superpowers/plans/2026-06-18-midi-lyrics-alignment.md` —— TDD 步骤、完整代码与提交命令。
- **现有实现**：`nodes.py`（`MIDIEditLyrics` 及 `_apply_speed` / `_fmt_durs` / `char_to_phoneme` 等纯函数）。
- **上游音素表**：SoulX-Singer [phone_set.json](https://github.com/Soul-AILab/SoulX-Singer/blob/main/soulxsinger/utils/phoneme/phone_set.json) —— `phoneme` 字段的音素集合与命名规范。
- **算法范式**：序列对齐动态规划（Needleman-Wunsch / DTW 的扩展形式，支持一对多 / 多对一对齐）。

---

**文档结束**

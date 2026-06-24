# 歌词对齐算法重写设计（v3：顺序映射 + 贪心压缩）

| 元信息 | 值 |
|--------|-----|
| 日期 | 2026-06-24 |
| 状态 | Draft（待用户审阅） |
| 作者 | project-manager（brainstorming 主持） |
| 关联 | 替换 v1 DP 设计（`docs/superpowers/specs/2026-06-18-midi-lyrics-alignment-design.md`） |
| 上游契约 | SoulX-Singer 输入格式，`data_processor.py` 确认 f0 = 50fps |

---

## 1. 背景与动机

v1 DP 算法在真实测试中暴露多个根本性问题：
- DP 的"全局最优"在音乐场景反复被证明效果差
- SP 软约束/硬保留都导致 f0 与 token 错位 → 唱不出来
- SPLIT 无限堆叠 → 0.03s/字
- 歌词分配机械 → section 过载

**v3 彻底放弃 DP**，改为确定性顺序映射 + 贪心压缩。核心原则：**效果优先，简单可预测**。

---

## 2. f0 与 token 的对应关系（源码确认）

从 SoulX-Singer `data_processor.py` 源码确认：

```python
hop_size = 480
sample_rate = 24000
# 帧率 = sample_rate / hop_size = 50fps（每帧 20ms）
# 每个 token 拥有 round(duration × 50) 个 f0 帧
# f0 总长度 = round(Σ(duration) × 50)
```

`preprocess` 方法中：
```python
duration = sum(note_duration) * sample_rate / hop_size  # = Σduration × 50
mel2note = torch.zeros(int(duration))
# 每个 token 的帧边界: round(累积duration × 50)
```

`process` 方法中 f0 被 truncate 到 mel2note 长度。f0 值原样使用，不做转换。

**结论**：可以安全地按 token duration 切分 f0，按新结构重新拼接。

---

## 3. 核心算法

### 3.1 准备

1. 解析原 track，分离原 SP token 和非 SP token
2. 原 SP token：记录 duration（算 AVG/MAX 用），丢弃其 f0 段
3. 非 SP token：作为映射目标池，各带 duration/pitch/f0 段
4. 原总字数 M = 原 track 总 token 数（含 SP）

### 3.2 新歌词断句

```
Step 1: 按标点断句（。！？，、；：\n）
Step 2: 句数 ≥ 原 SP 数？
  是 → 完成
  否 → Step 3
Step 3: 找最长句，用 jieba 在中间附近找词边界切分
Step 4: 重复 Step 2-3，直到句数 ≥ 原 SP 数或最长句 ≤ 3 字无法再切
```

### 3.3 SP 结构构建

```
K 句歌词 → [SP] 句1 [SP] 句2 [SP] ... [SP] 句K [SP]
SP 数 = K + 1
新字数 N = (K+1) + 总字数（SP 也算一字）
```

### 3.4 SPD 计算

SP 的 duration 是固定计算值（不从 DROP pool 匀出，由公式决定）：

```
M = 原 track 总 token 数（含 SP）
N = 新歌词总 unit 数（含 SP）

SPD = AVG(原 SP 的 duration) × (M / N)
限制: 0.1 ≤ SPD ≤ MAX(原 SP 的 duration)
```

原 SP token 的 duration 在移除后丢弃（不重分配给字）。

### 3.5 分配

#### Case 1: 新字数（含 SP）≤ 原 token 数（含 SP）

新单元（字 + SP）按顺序 1:1 映射到原 token（含原 SP）：
- 字继承原 token 的 duration、pitch、f0 段
- SP 替换原 token（duration=SPD，pitch=0，f0=0）
- 多余的原 token 丢弃

#### Case 2: 新字数（含 SP）> 原 token 数（含 SP）

初始：前 M 个字 1:1 映射。剩余字需要共享 token。

贪心压缩：
```
Step 1: 原 token 按 duration 降序排列
Step 2: 取最长 token T_k
        查: 新歌词第 k 个字是否属于多字词（jieba）
        是 → 这个词的所有字共享 T_k，duration 平分
             释放的 token 位置给多余字填充
        否 → 取第二长 token，重复
Step 3: 直到所有字都分配完

特殊情况: 没有多字词可压缩
        → 从最长 token 开始，允许"单字+相邻字"组合共享
```

SPLIT 后某字 duration < 0.1 → 给 0.1 最低保障，从同 token 其他字匀。

### 3.6 f0 重建

1. 按 duration × 50 切分原 f0 → 每个 token 一段
2. 原 SP token 的 f0 段：丢弃
3. 非 SP token 的 f0 段：保留（旋律）
4. 新字：用映射的原 token 的 f0 段（原样保留）
5. SPLIT 的字：原 f0 段按字数切片
6. 新 SP：插入全 0 帧（round(SPD × 50) 帧）
7. 拼接 → 新 f0 序列

**f0 值不修改、不生成、不插值——只是原段的拼接和切片。**

### 3.7 note_pitch / note_type / phoneme

| 新单元 | note_pitch | note_type | phoneme |
|--------|-----------|-----------|---------|
| 字（继承原 token） | 原 token pitch | _note_type()（2 普通 / 3 重复字 / 排除叠词） | char_to_phoneme() |
| SP | 0 | 1 | `<SP>` |

### 3.8 输出

```json
{
  "text": "<SP> 哥 哥 <SP> ...",
  "phoneme": "<SP> zh_ge1 zh_ge1 <SP> ...",
  "duration": "0.49 0.26 0.24 0.49 ...",
  "note_pitch": "0 59 64 0 ...",
  "note_type": "1 2 2 1 ...",
  "f0": "0.0 0.0 ... 485.3 511.9 ... 0.0 ..."
}
```

`time` 字段更新为 `[0, round(new_total_duration × 1000)]`。

---

## 4. 文件变更

### 新建

| 文件 | 职责 |
|------|------|
| `alignment/align.py` | 新算法核心：断句、SPD、映射、压缩、f0 重建 |

### 删除

| 文件 | 原因 |
|------|------|
| `alignment/cost.py` | DP 代价函数，不再使用 |
| `alignment/dp.py` | DP 求解器，不再使用 |
| `alignment/rebuild.py` | rebuild_tokens / allocate_durations，被新算法替代 |
| `alignment/preprocess.py` | normalize_lyrics / tokenize_units，被新断句+映射替代 |

### 保留（不修改）

| 文件 | 原因 |
|------|------|
| `alignment/models.py` | Token / Unit 数据结构 |
| `alignment/parser.py` | JSON ↔ Token 转换 |
| `alignment/phoneme.py` | char_to_phoneme / is_reduplication |
| `alignment/speed.py` | 变速封装 |
| `alignment/speed_impl.py` | _fmt_durs / _fmt_f0 / apply_speed |

### 修改

| 文件 | 变更 |
|------|------|
| `alignment/__init__.py` | 移除旧导出，加入 align |
| `nodes.py` | 重写 `MidiLyricsAlignment.align_lyrics`，用新算法 |
| `tests/test_alignment.py` | 移除 DP/cost/rebuild 测试，新增 align 测试 |

**MIDIEditLyrics 不受影响**——它只用 `nodes.py` 内部函数，不依赖 `alignment/` 子包。

---

## 5. 边界处理

| 情况 | 处理 |
|------|------|
| 新歌词为空 | 报错 |
| 原 track 无 SP | AVG_SP = MAX_SP = 0.3（默认） |
| 多 track 输入 | 按 duration 比例分配歌词到各 track |
| 断句后句数 > 原 SP 数 | 允许，SPD 相应变小 |
| 最长句无法再切（≤3字） | 不强行切 |
| SPLIT 后 dur < 0.1 | 给 0.1 最低保障 |
| 英文词 | char_to_phoneme → en_ 前缀，note_type 首=2/续=3 |
| force_tone4 | 高音(≥79) 中文音素改四声 |
| speed ≠ 1 | duration × 1/speed，f0 线性插值重采样 |

---

## 6. 决策溯源

| 决策 | 选择 | 理由 |
|------|------|------|
| 对齐方式 | 顺序映射 + 贪心（放弃 DP） | DP 反复证明"数学最优"≠"听感好" |
| SP 处理 | 原始移除，按新歌词重建 | 原 SP 位置与新歌词无关，强行保留导致 f0 错位 |
| f0 | 按 token 切段保留，SP 处插 0 | 源码确认 50fps 对应关系，安全操作 |
| SPLIT | 贪心压缩多字词到最长 token | 保住词的完整性，duration 均分可唱 |
| SPD | 公式计算，有上下限 | 适应不同歌词长度，保证 SP 不太长不太短 |
| 断句 | 标点优先，不够参照原 SP 数切 | 新歌词自然断句，必要时贴近原曲结构 |

---

**文档结束**

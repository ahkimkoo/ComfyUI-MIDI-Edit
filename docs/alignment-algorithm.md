# MIDI 歌词对齐算法（MidiLyricsAlignment v3）

> 本文是 `MidiLyricsAlignment` 节点（v3 歌词对齐算法）的**算法实现说明**，
> 面向项目贡献者、开发维护者，以及希望理解算法工作原理的用户。
> 文档自包含：不阅读 spec / plan 也能完整理解算法。
>
> 对应代码：[`core/align_algorithm.py`](../core/align_algorithm.py)。

---

## 1. 概述

v3 彻底放弃 v1/v2 的联合动态规划（DP），改为**确定性顺序映射 + 贪心压缩 + CT-Transformer 智能断句**。设计取向从"求全局最优"转向**效果优先、简单可预测**：

- 不再有代价函数、5 种原子操作、状态网格搜索。
- 原 `<SP>` 停顿全部丢弃，按**新歌词自身的断句**重建 SP 结构。
- 新字与原曲旋律 token（非 SP）建立顺序映射，按字数关系走两种确定性分支。
- f0 按 token 的 `duration × 50` 帧切段重建，不再整段保留。

相比 `MIDIEditLyrics` 的 Collapse / Expand / Collapse+Distribute 三套规则分支，v3 用单一管线覆盖所有字数匹配情况，且 SP 边界完全由新歌词决定，与原曲句数解耦。

---

## 2. f0 与 token 的对应关系

数据格式细节见 [`docs/midi-json-format.md`](./midi-json-format.md)，这里只复述 v3 算法强依赖的契约：

- **帧率 = 50 fps**（SoulX-Singer：`sample_rate=24000`，`hop_size=480`）。
- **每个 token 拥有 `round(duration × 50)` 个 f0 帧**。
- `f0` 是帧级字段，与 token 序列**不**做 1:1 对齐；但通过上面的帧数公式，每个 token 在 f0 序列上对应一个连续帧段。
- `<SP>` token 的帧通常为 `0.0`（清音/无声）。

v3 把 f0 当作"按 token 切片再重组"的素材，而非整段不可动的整体（这是与 v1/v2 的根本差异之一）。

---

## 3. 断句（`segment_sentences`）

断句**只看新歌词本身**，不参照原 track 的句数或结构。

1. **按标点切分**：用正则 `[。！？，、；：\n]` 把歌词切成若干段，去空白。
2. **长句二次切分**：对每段长度超过 10 字的句子，调用 **CT-Transformer 标点恢复模型**（`punctuate_fn`，节点层注入 `_restore_punctuation`）给它加标点，然后按所有标点（`[，。！？；：、]`）**全切**——一次产出多段，无需递归。
3. **降级**：若 CT-Transformer 不可用或加标点后只有一段，原句保留不切。
4. **多 track**：整段歌词先按各 track 的非 SP duration 比例分配（`_distribute_lyrics`），每个 track 对自己分到的歌词独立走断句。

> 断句结果决定新结构的 SP 数量（见 §4），因此 CT-Transformer 是否启用直接影响最终句数。10 字阈值是经验值，避免短句被过度切碎。

---

## 4. SP 结构与 SPD

### 4.1 新 SP 结构

v3 丢弃原曲所有 `<SP>` token，按断句结果重建：

```
[SP] 句1 [SP] 句2 ... [SP] 句k [SP]
```

- SP 总是**首尾各一个**，句与句之间各一个。
- 新 SP 数量 = `k + 1`（k 为断句后的句子数）。
- 新 token 总数 `N` = 新 SP 数 + 新字数 `C`。

### 4.2 SPD（新 SP 的 duration）

SPD 由原 SP 的时长统计反算，而非守恒原 SP 总时长：

```
SPD = AVG(原 SP durations) × (原 token 总数 M / 新 token 总数 N)
```

限制范围 `[0.1, MAX(原 SP durations)]`：

- 比例因子 `M/N` 让 SP 时长随"字数变多 / 变少"自适应缩放。
- 下限 0.1s 防 SP 过短；上限取原 SP 最大值防 SP 过长。
- 原曲无 SP 时，AVG/MAX 退化为默认 0.3s。

**注意**：v3 **不保证**总 duration 守恒。SP 数变了、SPD 是估计值，新 track 总时长会与原曲不同。

---

## 5. 映射（`_compute_pack`）

映射发生在**新字**与**原曲非 SP token**（旋律来源）之间。设：

- `C` = 新字数（不含新 SP）
- `nonsl` = 原曲非 SP token 数
- `pack[t]` = 第 t 个非 SP token 承载的字数，`sum(pack) = C`

### Case 1: `C ≤ nonsl`（顺序 1:1 + 丢弃）

```
pack = [1] * C + [0] * (nonsl - C)
```

- 前 `C` 个非 SP token 各承载 1 个字，**顺序对应**。
- 多余的原 token（尾部）`pack=0`，直接丢弃。
- 每个字继承其映射到的原 token 的 `duration` / `note_pitch` / f0 帧。

### Case 2: `C > nonsl`（按 duration 比例贪心压缩）

按每个原 token 的时长比例分配字数，让长 token 多扛字、短 token 少扛字：

1. `target = Σ(非 SP duration) / C`（每字的目标 duration）
2. 初始：`pack[t] = max(1, round(D_t / target))`
3. 修正 `sum(pack)` 与 `C` 的差 `diff`：
   - `diff > 0`（字没分完）：给 `D_t / pack[t]` 最大的 token 加字（它们分得起）。
   - `diff < 0`（分超了）：从 `D_t / pack[t]` 最小的 token 减字，保持每 token 至少 1 字。

#### SPLIT 组的 duration

被多字共享的原 token（`pack[t] ≥ 2`）称为一个 SPLIT 组：

- 每字 duration = `D_t / pack[t]`，组内等分。
- 组内施加 **0.1s 下限保障**（`_enforce_min_split`）：把过短的字抬到 0.1s，从同组更宽裕的字匀出；若组内无余量，标记 `MIN_DURATION_UNRESOLVED` 警告，不强行破坏等分。

> Case 2 下每字至少 `~target` 秒，避免 v1 DP 时代 0.03s/字的灾难。分配只依赖时长比例，是确定性贪心，不再搜索最优组合。

---

## 6. f0 重建

v3 把原 f0 按 token 切成段再重新拼装，是新结构而非原样保留：

1. **切原 f0**（`_segment_f0`）：按 `round(orig_token.duration × FPS)` 把原 f0 序列切成段，每段对应一个原 token（含原 SP 段）。舍入误差产生的尾部帧并入最后一段。
2. **丢弃原 SP 段**：只保留非 SP token 对应的 f0 段，作为旋律素材。
3. **字的 f0**：
   - `pack[t] == 1`（未 SPLIT）：字直接继承其映射 token 的整段 f0。
   - `pack[t] ≥ 2`（SPLIT）：把该 token 的 f0 段尽可能均分成 `pack[t]` 份（前 `rem` 份各多 1 帧），取字在组内位置 `pos` 对应的那份。**直接切，不插值**。
4. **新 SP 的 f0**：插入 `round(SPD × 50)` 个全 `0.0` 帧。
5. **time 字段反算**：track 的 `meta["time"] = [0, round(len(new_f0) / FPS × 1000)]`，用实际 f0 帧数反算毫秒。这是为了消除 `Σ round(d×50) ≠ round(Σd×50)` 的累积误差，避免 SoulX-Singer 因 time 与帧数不匹配报 `"could not broadcast"` 错误。

变速（`speed ≠ 1`）在对齐完成后由 `apply_speed_change` 统一处理：duration 按比例缩放，f0 用 `numpy.interp` 线性插值重采样。

---

## 7. note_type

v3 的 `note_type` 语义：

| 取值 | 含义 | 触发 |
|------|------|------|
| `1` | SP（停顿） | 所有新 `<SP>` token |
| `3` | 重复字（非叠词） | 单字与前一个非 SP 字相同，且不是叠词（`哥哥`/`妹妹` 等独立词汇） |
| `2` | 普通字 / 词首 | 其他所有字 |

判断在 `_note_type` 中完成：仅单字参与重复检测，英文词整词独立（不标 `3`）。叠词由 `core.text_utils.is_reduplication` 识别并排除。

---

## 8. force_tone4（可选后处理）

开启后（`force_tone4=True`，阈值默认 79 = G5）：对所有非 SP、`note_pitch ≥ 79`、且音素以 `zh_` 开头并带声调数字的 token，把末位声调改写为 `4`。英文、低音、SP 不受影响。

---

## 9. 边界情况与警告

| 类别 | 处理 | 涉及情况 |
|------|------|----------|
| 报错 | 返回错误 | 输入为空 / 非 JSON / 必填字段缺失 / 归一化后歌词为空 |
| 降级 | 自动处理 | 原 track 无 SP（SPD 用默认 0.3）；CT-Transformer 不可用（长句不切）；多 track 分配（按 duration 比例） |
| 降级 + 警告 | 通过 `warnings` 输出 | 高压缩（字数远多于非 SP token）；SPLIT 组内 duration 无法抬到 0.1s |

**警告类型**（通过节点第二个输出 `warnings` 返回，分号分隔，带 `t{idx}` track 索引前缀）：

| 警告 | 触发条件 |
|------|----------|
| `HIGH_SPLIT_RATIO(chars=C,slots=nonsl)` | `C > nonsl` 且 `C - nonsl > 0.4 × nonsl`（字数远多于原 token） |
| `MIN_DURATION_UNRESOLVED` | 存在 SPLIT 组内字 duration 低于 0.1s 且组内无余量可借 |

这些警告不影响输出 JSON 的生成，但提示演唱效果可能受限。

---

## 10. 处理管线（单 track）

```
输入: 原 Track + 该 track 分到的新歌词 + 参数(speed / normalize_digits / force_tone4)
  │
  ▼
[1. 分离 token]      原非 SP token（旋律来源） / 原 SP token（算 SPD）
  │
  ▼
[2. 归一化 + 断句]   数字转中文 → 标点切 → >10字走 CT-Transformer 全切
  │
  ▼
[3. 建新单元结构]    [SP] 句1 [SP] ... [SP]，统计 C / N
  │
  ▼
[4. SPD]             AVG(原SP) × (M/N)，限制 [0.1, MAX(原SP)]
  │
  ▼
[5. 切原 f0]         按 round(d×50) 切段，丢原 SP 段，留非 SP 段
  │
  ▼
[6. 贪心压缩]        pack[t]：C≤nonsl 顺序1:1丢弃；C>nonsl 按 D 比例
  │
  ▼
[7. SPLIT duration]  组内等分 + 0.1s 下限保障
  │
  ▼
[8. 组装新 token+f0] 字继承 token 的 pitch/dur/f0，SPLIT 切 f0，SP 插 0
  │
  ▼
[9. note_type]       SP=1 / 重复非叠词=3 / 其他=2
  │
  ▼
[10. force_tone4]    可选：高音 zh_ 音素末位声调改 4
  │
  ▼
[11. time 反算]      meta["time"] = [0, round(帧数/50×1000)]
  │
  ▼
（节点层）speed≠1 → apply_speed_change(duration 缩放, f0 插值重采样)
```

全程无 DP、无代价函数、无状态搜索，每一步都是确定性的。

---

## 11. 模块结构

v3 把算法收敛进 `core/` 包，`nodes.py` 仅作 ComfyUI 入口与历史符号再导出。

| 文件 | 职责 |
|------|------|
| [`core/g2p.py`](../core/g2p.py) | G2P 纯函数：`char_to_phoneme` / `word_to_phoneme` / `normalize_digits` / `is_chinese_char` |
| [`core/ct_transformer.py`](../core/ct_transformer.py) | CT-Transformer 标点恢复模型加载与 `restore_punctuation`（ModelScope 下载，ONNX 推理） |
| [`core/midi_format.py`](../core/midi_format.py) | `Token` / `Track` 数据结构，JSON parse/serialize，`FPS=50` |
| [`core/text_utils.py`](../core/text_utils.py) | `clean_lyrics` / `split_lyrics_to_sentences` / `is_reduplication`（叠词识别） |
| [`core/speed.py`](../core/speed.py) | `apply_speed` / `apply_speed_change`（duration 缩放 + f0 插值）/ `format_durations` / `format_f0` |
| [`core/edit_algorithm.py`](../core/edit_algorithm.py) | `MIDIEditLyrics` 的实现：`replace_lyrics` / `extract_lyrics` / `merge_repeated_chars` 及智能拆句 |
| [`core/align_algorithm.py`](../core/align_algorithm.py) | **本文主题**：`align_track` / `segment_sentences` / `calculate_spd` / `_compute_pack` 等 |

`nodes.py` 中 `MidiLyricsAlignment.align_lyrics` 负责多 track 歌词分配（`_distribute_lyrics`）、逐 track 调用 `align_track`、汇总 warnings、并在末尾统一施加变速。

---

## 12. 参考

- **数据格式**：[`docs/midi-json-format.md`](./midi-json-format.md) —— MIDI JSON 各字段、50fps 帧率契约、f0 帧段对应关系。
- **v3 设计决策**：`docs/superpowers/specs/2026-06-24-alignment-v3-design.md` —— 顺序映射 + 贪心压缩 + CT-Transformer 断句的设计溯源。
- **实现**：`core/align_algorithm.py`（算法）/ `nodes.py`（节点入口）。
- **上游音素表**：SoulX-Singer [phone_set.json](https://github.com/Soul-AILab/SoulX-Singer/blob/main/soulxsinger/utils/phoneme/phone_set.json)。
- **历史算法**：v1/v2 的联合 DP 描述已废弃，归档于 git 历史。

---

**文档结束**

# MIDI JSON 格式说明

## 1. 概述

MIDI JSON 是 [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) 歌声合成系统的输入格式。其顶层为一个 **track 对象数组**，数组中的每个元素表示一段人声片段（一个 vocal track）。

每个 track 的字段可分为两类粒度：

- **token 级字段**（共 5 个）：`text` / `phoneme` / `duration` / `note_pitch` / `note_type`，按相同索引 **1:1 对齐**，空格分隔，数组长度相同。
- **帧级字段**（1 个）：`f0`，独立于 token 序列，长度由 track 总时长决定，**不与 token 1:1 对齐**。

下面是一段真实样例（节选自 `docs/REQUIREMENT.md`）：

```json
[
  {
    "index": "vocal_0_15000",
    "language": "Mandarin",
    "time": [0, 15000],
    "text": "<SP> 你 个 小 毛 驴 <SP> 发 语 音 还 在 唱 歌 <SP>",
    "phoneme": "<SP> zh_ni3 zh_ge4 zh_xiao3 zh_mao2 zh_lu:4 <SP> zh_fa1 zh_yu3 zh_yin1 zh_hai2 zh_zai4 zh_chang4 zh_ge1 <SP>",
    "duration": "0.27 0.36 0.48 0.36 0.24 0.98 0.58 0.24 0.24 0.36 0.24 0.20 0.44 0.62 0.40 0.24",
    "note_pitch": "0 60 63 65 67 67 0 67 63 62 61 60 58 58 0 57",
    "note_type": "1 2 2 2 2 2 1 2 2 3 2 2 2 2 1 2",
    "f0": "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 267.2 264.1 263.6 262.0 ..."
  }
]
```

> 说明：除上述 6 个核心字段外，track 还可能携带 `index` / `language` / `time` 等元信息。`time` 是 `[起, 止]` 毫秒时间区间，供下游预分配音频缓冲使用。本插件替换歌词时**只修改 `text` 与 `phoneme`**，其余字段（含 `f0`）一律保留。

## 2. 字段分类

| 字段 | 类型 | 粒度 | 单位 / 取值 | 歌词相关 |
|------|------|------|-------------|----------|
| `text` | string（空格分隔） | token 级 | 字 / 词 / `<SP>` | 是 |
| `phoneme` | string（空格分隔） | token 级 | `zh_<拼音>` / `en_<ARPAbet>` / `<SP>` | 是 |
| `duration` | string（空格分隔） | token 级 | 秒（float） | 否 |
| `note_pitch` | string（空格分隔） | token 级 | MIDI 音高编号（int，0 = 休止） | 否 |
| `note_type` | string（空格分隔） | token 级 | 1 / 2 / 3 | 否 |
| `f0` | string（空格分隔） | 帧级（~50fps） | 基频 Hz（float，0.0 = 清音/无声） | 否 |

## 3. 字段详解

### 3.1 `text`

歌词文本，是其余 4 个 token 级字段的**对齐基准轴**。

- 空格分隔，每个 token 表示一个字（中文）/ 一个词（英文）/ 一个停顿标记。
- `<SP>` 表示停顿（silence / pause），保留原样不参与歌词替换。
- 示例：`"<SP> 你 个 小 毛 驴 <SP>"`

### 3.2 `phoneme`

音素序列，与 `text` **严格 1:1 对应**（数组长度相同、索引一致）。

- 中文 → `zh_<带声调拼音>`，如 `你` → `zh_ni3`，`驴` → `zh_lu:4`（`ü` 写作 `u:` 或 `v`，依 phone_set）。
- 英文单词 → `en_<ARPAbet>`，如 `wish` → `en_W-IH1-SH`（多音素单词以 `-` 连接，含重音标记数字）。
- `<SP>` → `<SP>`（保留）。
- 前缀 `zh_` / `en_` 标识语种，音素集合遵循 SoulX-Singer 的 [phone_set.json](https://github.com/Soul-AILab/SoulX-Singer/blob/main/soulxsinger/utils/phoneme/phone_set.json)。

### 3.3 `duration`

每个 token 的持续时长（单位：秒）。

- `<SP>` 同样拥有 duration。
- track 总时长 ≈ `Σ(duration)`。
- 变速（speed 参数）时，每个 duration 乘以 `1/speed`（加速 → 时长变短）。参见 `nodes.py:1279-1327` 的 `_apply_speed`。

### 3.4 `note_pitch`

每个 token 对应的 MIDI 音高编号（离散目标音高）。

- `60` = C4（中央 C），每 `+1` 升一个半音，每 `+12` 升一个八度。
- `0` = 休止 / 无音高，通常对应 `<SP>` 位置。
- 与 `f0`（连续基频）互补：`note_pitch` 是离散目标，`f0` 是连续轨迹。

### 3.5 `note_type`

音符类型，取值 1 / 2 / 3，用于区分音符在乐句与词中的角色。

| 值 | 含义 | 出现场景 |
|----|------|----------|
| `1` | 段落 / 乐句收尾音 | 通常出现在 `<SP>` 位置或长音结尾，标记一个乐句的结束 |
| `2` | 普通音符 / 词首音符 | 默认值；英文单词的**第一个**音符 |
| `3` | 延续音符 | 英文单词跨多个音符时的**后续**音符，共享同一音素 |

**英文单词比例分配场景**（参见 `CHANGELOG.md`）：当一个英文单词需要分配到多个音符时，单词按词长比例切分——首个音符记 `note_type=2`，后续延续音符记 `note_type=3`，且这些音符共享同一音素。

**代码引用**（`nodes.py:811-814`）：

```python
if is_continuation:
    tokens[idx]["note_type"] = 3
elif tokens[idx].get("note_type") == 3:
    tokens[idx]["note_type"] = 2
```

即：当前 token 被判定为延续音时设为 `3`；若原本是 `3` 但本次不再是延续音，则改回默认的 `2`。

### 3.6 `f0`

基频曲线（单位：Hz），是**帧级数据**。

#### f0 与 duration 的精确对应关系

从 SoulX-Singer 源码（`soulxsinger/utils/data_processor.py`）确认：

```python
hop_size = 480       # 音频帧移（采样点）
sample_rate = 24000  # 采样率

# 帧率 = sample_rate / hop_size = 24000 / 480 = 50fps（每帧 20ms）
```

**每个 token 拥有 `round(duration × 50)` 个 f0 帧**：

```
Token 序列:  [SP d=0.33]  [老 d=0.26]  [师 d=0.24]  [SP d=0.28]  ...
帧范围:       帧 0~16       帧 17~29      帧 30~41      帧 42~55      ...
帧数:         17 (0.33×50)  13 (0.26×50)  12 (0.24×50)  14 (0.28×50)  ...
f0 值:        [0,0,...]     [485,511,...] [297,327,...] [0,0,...]     ...
```

`data_processor.preprocess()` 中的关键代码：

```python
# 总帧数 = Σ(duration) × sample_rate / hop_size = Σ(duration) × 50
duration = sum(note_duration) * sample_rate / hop_size
mel2note = torch.zeros(int(duration), dtype=torch.long)

# 每个 token 的帧边界
dur = int(np.round(dur_sum * sample_rate / hop_size))  # = round(累积duration × 50)
```

`data_processor.process()` 中 f0 被 truncate 到 `mel2note` 长度后原样使用。

#### 其他 f0 要点

- `0.0` 表示清音 / 无声段（通常对应 `<SP>` 或气声段）。
- 变速时通过 `numpy.interp` 线性插值重采样：新帧数 = `round(原帧数 × 1/speed)`。
- SP token 的 f0 段**多数为 0**，但边界处可能有少量非零值（前一个音符的尾音泄漏）。
- 歌词替换时，可按 token duration 切分 f0 段，按新结构重新拼接（保留旋律段，SP 处插 0）。

## 4. 字段对应关系

下面以 `<SP> 你 个 小 毛 驴 <SP>`（节选自 `docs/REQUIREMENT.md` 真实数据）为例，展示 token 级 5 字段的 1:1 对齐，以及 `f0` 的帧级独立结构：

```
text:        <SP>     你       个       小       毛       驴       <SP>
phoneme:     <SP>     zh_ni3   zh_ge4   zh_xiao3 zh_mao2  zh_lu:4  <SP>      ← 1:1
duration:    0.27     0.36     0.48     0.36     0.24     0.98     0.58      ← 1:1（秒）
note_pitch:  0        60       63       65       67       67       0         ← 1:1（MIDI#）
note_type:   1        2        2        2        2        2        1         ← 1:1（1/2/3）

                                                                 时间轴→
f0:          [0.0 0.0 ... 0.0 267.2 264.1 263.6 262.0 ... 0.0]            ← 帧级（~50fps）
             └──────── Σ(duration) ≈ 3.27s × 50 ≈ 164 帧 ────────┘
```

> 上图中 `f0` 行的数值为示意（真实 f0 不按 token 切分，而是一条连续帧序列）；开头若干 `0.0` 帧对应第一个 `<SP>` 与清音段，随后出现的 `267.2 264.1 ...` 等为有声段的连续基频采样。

**4 条关键关系要点**：

1. **5 个 token 级字段按相同索引一一对应**：`text` / `phoneme` / `duration` / `note_pitch` / `note_type` 均以空格分隔，**数组长度相同**，第 `i` 个 token 的五个属性共享下标 `i`。
2. **`f0` 独立**：长度由 track 总时长（≈ `Σ(duration)`）决定，与 token 数无关；替换歌词时不修改。
3. **`note_pitch`（离散目标）与 `f0`（连续轨迹）互补**：`note_pitch` 告诉模型"该唱哪个音"，`f0` 给出实际基频曲线；`note_pitch=0` 处 `f0` 通常也是 `0.0`。
4. **`text` 与 `phoneme` 是歌词载体**：本插件的核心工作就是只改这两个字段，其余字段保留即可让新歌词贴合原曲的节奏与旋律。

## 5. 工程含义

本格式是 SoulX-Singer 歌声合成系统的输入。各字段在系统中的作用大致如下：

- **token 级字段** → 音素编码器 + MIDI 条件编码器，控制"唱什么字、唱什么音、每个音多长"。
  - `phoneme`：驱动音素级语言学表征。
  - `note_pitch` / `note_type`：作为 MIDI 条件，约束目标音高与音符角色。
  - `duration`：确定每个 token 占据的时间跨度。
- **帧级 `f0`** → 声码器 / 扩散模型的基频目标或辅助特征，提供连续的音高轨迹。

因此，**替换歌词时只需修改 `text` 与 `phoneme`**，保留 `duration` / `note_pitch` / `note_type` / `f0` 即可让新歌词贴合原曲的节奏与旋律——这正是本插件的工作原理。

## 6. 参考来源

本文档的所有断言均可在以下来源中验证：

- `README.md`：第 153-176 行（简版字段说明）、第 280 行（声明 f0 完全不修改）。
- `CHANGELOG.md`：英文单词比例分配与 `note_type=2/3` 语义、`f0` 线性插值重采样逻辑。
- `nodes.py`：
  - 第 707-731 行：`_split_into_segments`，注释明确 f0 是 frame-level、不与 token 1:1。
  - 第 811-814 行：`note_type` 设为 `3` / 改回 `2` 的延续音判定逻辑。
  - 第 1279-1327 行：`_apply_speed`，duration 按 `1/speed` 缩放、f0 以 `numpy.interp` 在 ~50fps 下重采样。
- SoulX-Singer [phone_set.json](https://github.com/Soul-AILab/SoulX-Singer/blob/main/soulxsinger/utils/phoneme/phone_set.json)：`phoneme` 字段的音素集合与命名规范来源。
- `docs/REQUIREMENT.md`：完整真实数据样例。

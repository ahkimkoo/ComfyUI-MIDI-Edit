# ComfyUI-MIDI-Edit

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 自定义节点插件，提供 **MIDI 歌词编辑** 与 **SoulX-Singer 歌声合成** 端到端能力：把音频转写成 MIDI JSON、替换/对齐/提取歌词、并以参考音色合成歌声。适用于 MIDI 歌曲生成与"魔改歌词"工作流，支持中文、英文、粤语及中英混合歌词。

提供 **六个节点**，按 ComfyUI 分类：

| 分类 | 节点 | 作用 |
|------|------|------|
| `MIDI-Edit` | MIDI Edit Lyrics | 替换歌词并自动生成音素 |
| `MIDI-Edit` | MIDI Extract Lyrics | 提取歌词文本（去空格，`<SP>` 转换行） |
| `MIDI-Edit` | MIDI Merge Repeated Chars | 合并连续重复字符 |
| `MIDI` | MIDI Lyrics Alignment | 顺序映射 + 贪心压缩 + CT-Transformer 智能断句的对齐算法 |
| `MIDI-SoulX` | MIDI Transcribe Audio | 用 SoulX-Singer 把音频转写成 MIDI JSON |
| `MIDI-SoulX` | MIDI Synthesize Audio | 用 SoulX-Singer 把 MIDI JSON + 参考音色合成歌声 |

> 最新版本：**v3.2.0**（2026-06-26）— 修复 SVS 口齿不清（prompt 元数据来源）、`control` 默认改 `melody`、推理默认 FP32、新增 `prompt_metadata` / `use_fp16` / `cfg` / `n_steps` 输入；`MIDI Extract Lyrics` 新增 `resegment` 重新断句开关。详见 [CHANGELOG.md](CHANGELOG.md)。

---

## 功能特性

本插件提供 **六个节点**，覆盖从音频转写、歌词编辑到歌声合成的完整工作流。支持中文（`zh_` 前缀拼音）、英文（`en_` 前缀 ARPAbet 音素）、粤语及中英混合歌词。

### MIDI Transcribe Audio

用 SoulX-Singer 预处理管线把音频转写成 MIDI JSON。管线依次执行：人声分离（mel-band-roformer）→ 去混响（dereverb）→ F0 提取（RMVPE）→ VAD 语音检测 → 歌词转录（Paraformer/Parakeet ASR）→ 音符转录（ROSVOT）。产出的 MIDI JSON 可用后续节点编辑歌词后送入合成。

**输入：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `audio` | AUDIO | ✅ | — | 输入音频 |
| `max_merge_duration` | INT | — | `30000` | 最大合并段时长，单位 ms（范围 1000–120000） |
| `language` | [`Mandarin`, `English`, `Cantonese`] | — | `Mandarin` | 歌词语种 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING | 转写得到的 MIDI JSON 字符串 |

**分类：** `MIDI-SoulX`

---

### MIDI Synthesize Audio

用 SoulX-Singer 把 **目标 MIDI JSON**（要唱的内容）+ **参考音色音频** 合成歌声。以 MIDI JSON 为目标（歌词/音素/duration/pitch/f0），以参考音频提供音色，通过 flow-matching 扩散模型生成波形。

**输入：**

| 字段 | 类型 | 必填 | 默认值 | 范围/选项 | 说明 |
|------|------|------|--------|-----------|------|
| `midi_json` | STRING, multiline | ✅ | — | — | **目标**歌词/音素/duration/pitch/f0 |
| `prompt_audio` | AUDIO | ✅ | — | — | 参考音色音频（提供目标音色） |
| `prompt_metadata` | STRING, multiline | — | 空 | — | 描述参考音频真实声学内容的元数据。**推荐**：把 `MIDI Transcribe Audio`（对参考音频跑一遍）的 `midi_json` 输出接到这里，避免每次重复预处理；留空时节点会自动对参考音频预处理 |
| `control` | [`melody`, `score`] | — | `melody` | — | 控制模式（见下方 **control 选择策略**） |
| `seed` | INT | — | `12306` | 0–2147483647 | 随机种子 |
| `auto_shift` | BOOLEAN | — | ON | ON/OFF | 自动把目标音高对齐到参考音色音域 |
| `pitch_shift` | INT | — | `0` | -36 ~ 36 半音 | 整体音高平移 |
| `use_fp16` | BOOLEAN | — | **OFF/FP32** | FP16/FP32 | OFF=FP32（与参考实现一致，音质最稳）；ON=FP16 autocast（GPU 加速） |
| `cfg` | FLOAT | — | `3.0` | 1.0–10.0 | classifier-free guidance 强度（见下方 **cfg 调优建议**） |
| `n_steps` | INT | — | `32` | 8–128 | flow-matching 反向扩散步数；提高略改善质量但变慢 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio` | AUDIO | 合成的歌声 |

**分类：** `MIDI-SoulX`

#### control 选择策略

| 场景 | 选择 | 理由 |
|------|------|------|
| 目标含真实人声 F0（如从人声改歌词得到的 target） | **`melody`**（默认） | 音色更贴近参考音频 |
| 目标是乐谱/纯音乐/无可靠 F0 | **`score`** | 吐字更清晰，但音色更偏离参考 |

> `melody` 模式下若出现吞字/口齿不清，按下方 `cfg` 调优。

#### cfg 调优建议

- 默认 `cfg=3.0`。
- `melody` 模式下若出现吞字/发音含糊，把 `cfg` 从 3 提到 **4–5**（增强对歌词音素的服从）。
- `cfg` 过高可能导致过饱和或伪影，不建议超过 5。
- `n_steps` 提高可略改善质量，但合成时间线性增加；一般保持默认 32 即可。

> `rescale_cfg` 暂不支持：上游 `SoulXSinger.infer` 接口未透出该参数（内部硬编码为 0.75）。

---

### MIDI Edit Lyrics

替换 MIDI JSON 中的歌词并自动生成对应音素。支持任意长度差异的新旧歌词，3 种模式自动适配，配合 CT-Transformer 智能拆句处理句子数不匹配的场景。

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING, multiline | MIDI JSON 字符串 |
| `new_lyrics` | STRING, multiline | 新歌词文本 |
| `force_tone4` | BOOLEAN | 高音强制第四声（默认 OFF） |
| `high_pitch_threshold` | INT | 高音阈值 0-127（默认 79 = G5） |
| `fixed_pause` | BOOLEAN | 固定停顿模式（默认 Fixed=ON，Flexible=OFF 时 SP 时间可匀给 token） |
| `split_mode` | COMBO [`token`, `duration`] | 字数分配模式：token 按原曲 token 数比例，duration 按原曲时长比例（默认 token） |
| `speed` | FLOAT | 变速倍率（0.1~3.0，默认 1.0），duration 和 f0 同步缩放 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING | 修改后的 MIDI JSON 字符串 |

**处理逻辑：**

1. 新歌词按换行和标点（，。！？；：等）切分为句子
2. 每个句子映射到原曲的一个 MIDI section（`<SP>` 分隔的段落）
3. 如果句子数 ≠ 原 section 数，使用 **比例分配 + AI 切分**算法：
   - 按原曲每个 section 的 token 数（或时长，取决于 `split_mode`）比例计算预期字数
   - 逐 section 处理：CT-Transformer 加标点，取第一个标点切分
   - 切点字数在预期 ±15% 内 → 用 AI 切点；否则按预期硬切
   - 句子数 = 原 section 数时不触发，走下方的 collapse/expand 匹配
4. 每个 section 内使用 3 种模式匹配：
   - **Collapse**（新字数 ≤ 去重 slot 数）：右对齐映射到 slot，保护结尾长音
   - **Collapse+Distribute**（slot 数 < 新字数 ≤ token 数）：slot 内多个 token 分配不同的字（如原曲 `天天` → 新歌词 `把它`）
   - **Expand**（新字数 > token 数）：拆分最长 duration 的 token（减半，pitch 不变）
5. 原曲连续重复字（如 `天 天`）collapse 为 1 个 slot 后展开，新字自动按重复数复制
6. 为每个替换的字自动生成音素（中文 → `zh_` 拼音，英文 → `en_` ARPAbet 单词级音素）
7. 英文单词按词长比例分配到多个音符，首个音符 `note_type=2`，延续音符 `note_type=3`
8. 空 token 的 duration 重分配给同 section 的已填 token
9. Expand（拆分 token）时，非 SP token 的 duration 不低于 0.30s；Collapse 模式保持原 duration 不变
10. 灵活停顿模式下（`fixed_pause=Flexible`）：当 SP ≥ 2 倍 token 平均时长或 token 平均时长 < 0.30s 时，SP 降至 token 平均时长，释放时间按比例分给 token（总时长守恒）
11. 变速模式下（`speed ≠ 1.0`）：所有 duration 乘以速度倍率，f0 同步线性插值重采样（帧数等比变化）

**分类：** `MIDI-Edit`

---

### MIDI Lyrics Alignment

基于**顺序映射 + 贪心压缩 + CT-Transformer 智能断句**的歌词对齐节点。彻底放弃 DP，用确定性管线覆盖所有字数匹配情况：原 `<SP>` 全部丢弃并按新歌词断句重建，新字与原曲非 SP token 建立顺序映射，f0 按 token 切段重组。

**算法概述：**

- **CT-Transformer 智能断句**：先按标点切，超过 10 字用 CT-Transformer 标点模型加标点后全切；断句只看新歌词，不参照原曲句数
- **SP 结构重建**：丢弃原曲所有 `<SP>`，按断句结果重建 `[SP] 句1 [SP] 句2 ... [SP]`
- **SPD 时长**：`AVG(原 SP duration) × (原 token 数 / 新 token 数)`，限制 `[0.1, MAX(原 SP)]`
- **顺序映射（字数 ≤ 原 token）**：前 C 个非 SP token 各承载 1 字，多余 token 丢弃，字继承原 token 的 duration/pitch/f0
- **贪心压缩（字数 > 原 token）**：按原 token duration 比例分配字数，长 token 多扛字、短 token 少扛字；SPLIT 组内 duration 等分并施加 0.1s 下限保障
- **f0 按 token 切段重建**：按 `round(duration×50)` 切原 f0，丢原 SP 段，字用映射 token 的 f0 段，SPLIT 的段按字数切片，新 SP 插全 0 帧
- **time 反算**：`time = [0, round(帧数/50×1000)]`，消除累积误差，避免 SoulX-Singer "could not broadcast" 错误

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING, multiline | MIDI JSON 字符串 |
| `lyrics` | STRING, multiline | 新歌词文本（中英混合，可含标点/换行） |
| `speed` | FLOAT | 变速倍率（0.1~3.0，默认 1.0），duration 和 f0 同步缩放 |
| `normalize_digits` | BOOLEAN | 阿拉伯数字自动转中文（默认 ON） |
| `force_tone4` | BOOLEAN | 高音强制第四声（默认 OFF） |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING | 对齐后的 MIDI JSON 字符串 |
| `warnings` | STRING | 警告信息（见下方"警告类型"；无警告时为空串） |

**与 MIDI Edit Lyrics 的差异：**

| 维度 | MIDI Edit Lyrics | MIDI Lyrics Alignment |
|------|------------------|-----------------------|
| 算法 | 3 模式 + 多分支（Collapse/Expand/Distribute） | 顺序映射 + 贪心压缩，确定性管线 |
| 字数匹配 | 按场景选不同策略 | 字数 ≤ token 顺序 1:1 丢弃；字数 > token 按 duration 比例压缩 |
| SP 处理 | 严格位置保留 | 原曲 SP 全丢弃，按新歌词断句重建 SP |
| 断句依据 | 参照原曲 section 数，AI 辅助切到匹配 | 只看新歌词（标点 + CT-Transformer），不参照原曲句数 |
| f0 处理 | 帧级整体不动 | 按 token 切段重建（丢原 SP 段，SPLIT 切片，SP 插 0） |
| 总时长 | 守恒 | 不守恒（SP 数变、SPD 为估计值） |
| 适用场景 | 已稳定，老工作流兼容 | SP 边界随新歌词自由调整，断句更贴合语义 |

> 注意：不保证总 duration 守恒。SP 数量与 SPD 都随新歌词变化，输出 track 总时长会与原曲不同。

**多 track 行为：**

输入含多个 track 时，整段歌词按各 track 的非 SP duration 比例自动分配——长 track 分到更多歌词，短 track 分到更少。分到空歌词的 track 原样保留（不替换）。例如：track0=30s + track1=1.5s 时，track1 只分到约 5% 的歌词。每个 track 独立走断句 + 对齐管线。

**警告类型（通过 `warnings` 输出）：**

算法在物理限制或极端不匹配场景下会发出警告（分号分隔，带 `t{idx}` track 索引前缀），用户可在 ComfyUI 中通过 `warnings` 输出查看：

| 警告 | 含义 |
|------|------|
| `HIGH_SPLIT_RATIO(chars=C,slots=nonsl)(t{idx})` | 字数 `C` 远多于原非 SP token 数 `nonsl`（`C - nonsl > 0.4 × nonsl`） |
| `MIN_DURATION_UNRESOLVED(t{idx})` | SPLIT 组内存在字 duration 低于 0.1s 且组内无余量可借 |

> 这些警告表示字数与原曲 token 数严重不匹配，输出虽已尽力处理，但演唱效果可能受限。建议调整歌词字数或选择更匹配的原曲。

**示例：**

输入 MIDI JSON 的 text：`<SP> 你 好 <SP>` （2 token + 2 SP）

输入新歌词：`天空`

输出 text：`<SP> 天 空 <SP>`（原 2 个 `<SP>` 丢弃，"天空"为 1 句 → 重建为首尾 SP；字数 ≤ 原 token，顺序 1:1 映射，字继承原 token 的 duration/pitch/f0）

**分类：** `MIDI`

---

### MIDI Extract Lyrics

从 MIDI JSON 中提取歌词文本。默认沿 track 拼接 `text`、去空格、把 `<SP>` 换行，保留原曲的断句；可选开启 `resegment`，用 CT-Transformer 重新加标点并按自然句读输出（一句一行）。

**功能要点：**

- **默认提取**：按 `<SP>` 还原原曲分句（原曲一个 `<SP>` 段 = 输出一行）
- **合并连续重复字**（`merge_repeated`）：可选，等价于对结果跑一次 `MIDI Merge Repeated Chars`
- **重新断句**（`resegment`）：忽略原 `<SP>` 分句，把歌词清洗后由 CT-Transformer 重新加标点，按自然句读输出一句一行；适合原曲断句零碎、希望重新自然分句的场景

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING, multiline | MIDI JSON 字符串 |
| `merge_repeated` | BOOLEAN | 合并连续重复字符（默认 OFF；ON 时相当于对结果跑一次 `MIDI Merge Repeated Chars`）。当 `resegment=ON` 时本开关无效——重新断句流程已内含合并步骤 |
| `resegment` | BOOLEAN | 重新断句（默认 OFF）。ON 时忽略原 `<SP>` 断句：先转中文数字 → 去换行/空格/标点 → 合并连续重复字 → CT-Transformer 重新加标点断句 → 一句一行输出，适合把歌词重新自然分句 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `lyrics_text` | STRING | 提取的歌词文本 |

**处理逻辑：**

默认流程（`resegment=OFF`）：

1. 遍历所有 track，拼接 `text` 字段
2. 去除所有空格
3. 将 `<SP>` 替换为换行符
4. 当 `merge_repeated=ON` 时，合并连续重复字符（仅保留一个）

重新断句流程（`resegment=ON`）：

1. 遍历所有 track，拼接 `text` 字段
2. `normalize_digits`：把阿拉伯数字转为中文数字（避免 ASR 残留的 "1 2 3"）
3. 去掉 `<SP>`、换行、空格及所有标点，仅保留中文字和英文字母
4. 合并连续重复字（`merge_repeated_chars`，如 `向向往` → `向往`）
5. CT-Transformer 重新加标点（`restore_punctuation`；首次使用自动下载模型到 `models/ct-transformer-punc/`）
6. 按标点断句（`split_lyrics_to_sentences`），一句一行输出

> 注意：`resegment=ON` 时 `merge_repeated` 开关无效——重新断句管线已内含合并步骤。

**分类：** `MIDI-Edit`

---

### MIDI Merge Repeated Chars

合并文本中连续重复的字符，只保留一个。常用于清理歌词提取结果中的叠字（如 `向向往` → `向往`）。

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | STRING, multiline | 输入文本 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | STRING | 合并连续重复字符后的文本 |

**分类：** `MIDI-Edit`

---

## 安装

### 1. 克隆（含 submodule）— 最常见的失败原因

> **⚠️ 重要：** 大部分用户无法使用本项目，就是因为克隆时没有拉取 git submodule。请务必按以下步骤操作。

本插件以 **git submodule** 形式集成 [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) 歌声合成引擎。**必须**确保 `SoulX-Singer/` 子目录包含完整源码（不是一个空目录）。

```bash
# 方式一：新克隆（推荐，一步到位）
cd ~/App/ComfyUI/custom_nodes
git clone --recursive https://github.com/ahkimkoo/ComfyUI-MIDI-Edit.git

# 方式二：已经克隆过，但 SoulX-Singer/ 目录为空
cd ~/App/ComfyUI/custom_nodes/ComfyUI-MIDI-Edit
git submodule update --init --recursive
```

**验证 submodule 是否已初始化：**

```bash
ls SoulX-Singer/soulxsinger/
# 应该看到 __init__.py  config/  models/  utils/ 等内容
# 如果目录为空或只有 .git 文件，说明 submodule 没有拉取成功
```

> 若用符号链接方式安装（`ln -s`），也要确保原仓库里的 `SoulX-Singer/` 子目录已被初始化。

### 2. Python 依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含全部依赖，分为两块：

- **MIDI 编辑核心**：`g2pM`、`g2p_en`、`modelscope`、`onnxruntime>=1.17.0`
- **SoulX-Singer 集成**：`funasr`、`einops`、`huggingface_hub`、`omegaconf`、`librosa`、`soundfile`、`torch`、`torchaudio`、`transformers`、`accelerate`、`rotary_embedding_torch`、`sageattention`、`praat-parselmouth`、`pyworld`、`webrtcvad` 等

> 注意：`torch` 和 `torchaudio` 需要与你的 CUDA 版本匹配。如果 `pip install` 报错，请参考 [PyTorch 安装指南](https://pytorch.org/get-started/locally/) 选择对应的 CUDA 版本安装。

### 3. 模型下载

> **仅使用歌词编辑功能（MIDI Edit Lyrics / MIDI Extract Lyrics 等前 4 个节点）不需要下载任何模型。** 以下模型仅在使用 `MIDI Transcribe Audio` 和 `MIDI Synthesize Audio` 时需要。

所有模型都下载到 ComfyUI 的 `models/` 目录下。以下是本项目所需的全部模型目录结构：

```
ComfyUI/models/
├── Soul-AILab/                                    # ← 歌声合成（HuggingFace 整体下载）
│   ├── SoulX-Singer/
│   │   └── model.pt                               # SVS 主模型（~2.6 GB）
│   └── SoulX-Singer-Preprocess/
│       ├── mel-band-roformer-karaoke/
│       │   ├── mel_band_roformer_karaoke_becruily.ckpt   # 人声分离（~1.6 GB）
│       │   └── config_karaoke_becruily.yaml
│       ├── dereverb_mel_band_roformer/
│       │   ├── dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt  # 去混响（~0.85 GB）
│       │   └── dereverb_mel_band_roformer_anvuew.yaml
│       ├── rmvpe/
│       │   └── rmvpe.pt                           # F0 提取（~173 MB）
│       ├── rosvot/
│       │   ├── rosvot/
│       │   │   ├── model.pt                       # 音符转录（~138 MB）
│       │   │   └── config.yaml
│       │   ├── rwbd/
│       │   │   ├── model.pt                       # 词边界检测（~114 MB）
│       │   │   └── config.yaml
│       │   └── rmvpe/
│       │       └── model.pt                       # rosvot 内部 F0 模型
│       ├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
│       │   ├── model.pt                           # 中文 ASR（~0.92 GB）
│       │   ├── config.yaml / configuration.json / tokens.json / seg_dict / am.mvn
│       └── parakeet-tdt-0.6b-v2/
│           └── parakeet-tdt-0.6b-v2.nemo          # 英文 ASR（~2.3 GB）
├── ct-transformer-punc/                           # ← 标点恢复（ModelScope 下载）
│   ├── model_quant.onnx
│   └── tokens.json
└── nltk/                                          # ← NLP 数据（NLTK 自动下载）
    ├── taggers/
    │   ├── averaged_perceptron_tagger/
    │   └── averaged_perceptron_tagger_eng/
    └── tokenizers/
        ├── punkt/
        └── punkt_tab/
```

#### SoulX-Singer 模型（歌声合成必需）

整体下载，两条命令搞定。**以下命令假设你当前在 `ComfyUI/models/` 目录下执行：**

```bash
cd ~/App/ComfyUI/models

# 下载 SVS 主模型（~2.6 GB）
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir Soul-AILab/SoulX-Singer

# 下载全部预处理模型（人声分离 + F0 + ASR + ROSVOT，~7 GB）
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir Soul-AILab/SoulX-Singer-Preprocess
```

ModelScope 下载：

```bash
cd ~/App/ComfyUI/models

modelscope download Soul-AILab/SoulX-Singer --local_dir Soul-AILab/SoulX-Singer
modelscope download Soul-AILab/SoulX-Singer-Preprocess --local_dir Soul-AILab/SoulX-Singer-Preprocess
```

#### CT-Transformer 标点恢复模型（智能断句用）

首次使用智能拆句功能时自动从 ModelScope 下载到 `ComfyUI/models/ct-transformer-punc/`（~270MB）。

手动下载：

```bash
cd ~/App/ComfyUI/models
modelscope download iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx --local_dir ct-transformer-punc
```

#### NLTK 数据（歌词编辑用）

首次运行时自动下载到 `ComfyUI/models/nltk/`（极小，几 MB）。

手动下载：

```bash
cd ~/App/ComfyUI/models/nltk
python -c "import nltk; nltk.download('averaged_perceptron_tagger', download_dir='.'); nltk.download('averaged_perceptron_tagger_eng', download_dir='.'); nltk.download('punkt', download_dir='.'); nltk.download('punkt_tab', download_dir='.')"
```

#### Whisper Encoder（歌声合成用）

SoulX-Singer 的 SVC 模块使用 OpenAI Whisper Base 作为音频编码器，首次运行时由 `huggingface_hub` 自动下载到缓存目录，无需手动操作。如需预下载：

```bash
huggingface-cli download openai/whisper-base
```

#### g2pM 模型（歌词编辑用）

中文 G2P 模型首次使用时自动下载到 `~/.g2pM/`（极小，几 MB），无需手动操作。

### 4. 重启 ComfyUI

---

## MIDI JSON 格式说明

MIDI JSON 为一个 track 对象数组，每个 track 包含以下字段：

```json
[
  {
    "text": "<SP> 我 有 一 只 小 <SP> 毛 驴 <SP>",
    "phoneme": "<SP> zh_wo3 zh_you3 zh_yi1 zh_zhi1 zh_xiao3 <SP> zh_mao2 zh_lu:2 <SP>",
    "duration": "0.27 0.36 0.48 0.36 0.24 0.98 0.24 0.36",
    "note_pitch": "0 60 63 65 67 67 0 60",
    "note_type": "1 2 2 2 2 1 2 2",
    "f0": "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
  }
]
```

字段说明：

| 字段 | 说明 |
|------|------|
| `text` | 歌词文本，空格分隔各字/词，`<SP>` 表示停顿 |
| `phoneme` | 音素序列，与 `text` 一一对应 |
| `duration` / `note_pitch` / `note_type` / `f0` | 非歌词字段，替换时不修改 |

---

## 使用示例

### 示例 1：替换歌词

输入 MIDI JSON 的 text：`<SP> 你 个 小 毛 驴 <SP> 发 语 音 <SP>`

输入新歌词：`红鲤鱼与绿鹦鹉`

输出 text：`<SP> 红 鲤 鱼 与 <SP> 发 语 音 <SP>`

> 前 4 个 slot 替换，多余 2 字忽略，原文后段保留。

### 示例 2：智能拆句（句子数 ≠ 原 section 数）

当新歌词句子数不等于原曲 section 数时，按原曲 token 比例分配字数，AI 辅助切分：

原曲 4 个 section，token 数：8 / 8 / 8 / 7 = 31

新歌词（无标点整段）：`如果你觉得有点累送你个小炸弹把它扔给你的烦恼把烦恼都炸飞拉上你的老闺蜜呀`（36 字）

按比例分配预期字数：9 / 9 / 9 / 9（四舍五入 + 最后兜底）

逐 section AI 切分：
- Section 1：AI 加标点 → `如果你觉得有点累，送你个小炸弹...` → 第一个标点前 8 字，与预期 9 字偏差 1 ≤ 3 → 采用 → `如果你觉得有点累`
- Section 2：AI 加标点 → `送你个小炸弹，把它扔给你的烦恼...` → 6 字，偏差 3 ≤ 3 → 采用 → `送你个小炸弹`
- Section 3：AI 加标点 → `把它扔给你的烦恼，把烦恼都炸飞...` → 8 字，偏差 1 ≤ 3 → 采用 → `把它扔给你的烦恼`
- Section 4：最后 section 收剩余 → `把烦恼都炸飞拉上你的老闺蜜呀`

### 示例 3：Collapse+Distribute 模式

MIDI 中连续重复字表示长音（占多个音节）。用户输入不需要手动写重复字，插件会自动展开匹配。

输入 MIDI JSON 的 text：`敌 敌 人 不 敢 靠 近 近 近 你`

（slots：敌×2, 人×1, 不×1, 敢×1, 靠×1, 近×3, 你×1 = 7 个 slot）

输入新歌词：`一二三四五六七`

输出 text：`一 一 二 三 四 五 近 近 近 你`

> "敌敌"（长音 2 拍）→ 用户写"一"自动展开为"一一"；"近近近"（长音 3 拍）→ 用户写"六"自动展开为"六六六"。

### 示例 4：提取歌词

输入 MIDI JSON 的 text：`<SP> 我 有 一 只 小 <SP> 毛 驴 我 从 来 都 不 <SP> 骑 有 一 天 <SP>`

输出：

```
我有一只小
毛驴我从来都不
骑有一天
```

---

## 工作流示例

### 端到端"魔改歌词"工作流（音频 → 转写 → 编辑 → 合成）

完整链路：**原曲音频** → 转写 MIDI JSON → 编辑歌词 → 用参考音色合成歌声。

```
┌─────────────────┐
│ 原曲音频 (AUDIO) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ MIDI Transcribe Audio   │  ← 人声分离 + F0 + VAD + 转录 + 音符
│ (language=Mandarin)     │
└────────┬────────────────┘
         │ midi_json (target)
         ▼
┌──────────────────────────────────────┐
│ MIDI Edit Lyrics                     │  ← 改歌词 / 或 MIDI Lyrics Alignment
│ 或 MIDI Lyrics Alignment            │
└────────┬─────────────────────────────┘
         │ midi_json (编辑后的 target)
         │
         │                 ┌────────────────────────┐
         │                 │ 参考音色音频 (AUDIO)    │
         │                 └───────────┬────────────┘
         │                             │
         │                             ▼
         │                 ┌─────────────────────────┐
         │                 │ MIDI Transcribe Audio   │  ← 可选：对参考音频跑一遍
         │                 │ (避免每次重复预处理)     │
         │                 └───────────┬─────────────┘
         │                             │ midi_json → prompt_metadata
         │                             │
         ▼                             ▼
┌─────────────────────────────────────────────────┐
│ MIDI Synthesize Audio                           │
│   midi_json     ← target（编辑后）              │
│   prompt_audio  ← 参考音色音频                  │
│   prompt_metadata ← 参考音频的 midi_json（可选）│
│   control=melody / cfg=3.0 / use_fp16=OFF       │
└────────┬────────────────────────────────────────┘
         │ audio (AUDIO)
         ▼
    合成歌声
```

**要点：**

- **target 路径**：原曲音频 → `MIDI Transcribe Audio` → 用 `MIDI Edit Lyrics` / `MIDI Lyrics Alignment` 改歌词 → 作为 `midi_json` 送入合成节点
- **prompt 旁路**：参考音色音频 → `MIDI Transcribe Audio` → 接到合成节点的 `prompt_metadata`（避免每次合成重复跑预处理管线）；参考音频本身接到 `prompt_audio`
- **简化方案**：不接 `prompt_metadata` 也能跑——节点会在内部自动对参考音频做预处理，但速度更慢
- **control 选择**：从人声改歌词得到的 target 选 `melody`（默认）；从纯乐谱/无 F0 数据生成的 target 选 `score`
- **吞字调优**：`melody` 模式下若出现吞字，把 `cfg` 从 3 提到 4–5

> 完整工作流文件：[下载 workflow.json](docs/workflow.json)（拖入 ComfyUI 界面即可使用，含转写 / 编辑 / 对齐 / 合成 / 提取全链路，以及参考音色旁路与试听节点）

### 仅歌词编辑工作流（无需 SoulX-Singer 模型）

如果只做 MIDI JSON 的歌词替换/提取，不需要安装 SoulX-Singer 模型，可只用前 4 个节点：

- [下载工作流 JSON](docs/midi-edit-lyrics.json)（拖入 ComfyUI 界面即可使用）

---

## ComfyUI API 调用示例

通过 HTTP API 调用 `MIDI Edit Lyrics` 节点：

```bash
curl -s http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {
        "class_type": "MIDIEditLyrics",
        "inputs": {
          "midi_json": "[{\"text\":\"<SP> 你 好 <SP>\",\"phoneme\":\"<SP> zh_ni3 zh_hao3 <SP>\"}]",
          "new_lyrics": "红鲤鱼"
        }
      },
      "2": {
        "class_type": "PreviewAny",
        "inputs": {
          "source": ["1", 0]
        }
      }
    }
  }'
```

---

## 核心函数独立使用（HTTP API）

`core/` 下的所有模块不依赖 ComfyUI，可直接用于 HTTP API 服务器：

```python
from core.soulsx_singer import transcribe_audio, synthesize_audio, set_models_base

# 可选：自定义模型路径（默认自动从 ComfyUI models/ 或本地 models/ 查找）
set_models_base("/path/to/models")

# 音频转 MIDI JSON
midi_json = transcribe_audio("/path/to/audio.wav", language="Mandarin")

# MIDI JSON + 参考音色合成歌声
# 推荐：先转写参考音频得到 prompt_metadata，避免每次重复预处理
prompt_metadata = transcribe_audio("/path/to/prompt.wav")
waveform, sample_rate = synthesize_audio(
    midi_json,
    "/path/to/prompt.wav",           # 也接受 (numpy_array, sample_rate) 元组
    prompt_metadata=prompt_metadata,  # 关键：与参考音频匹配（留空则内部自动预处理）
    control="melody",                 # 默认；target 是乐谱/纯音乐时用 "score"
    cfg=3.0,                           # melody 吞字可提到 4-5
    use_fp16=False,                    # FP32 音质最稳；GPU 加速可设 True
    seed=12306,
)
```

其他核心模块：

```python
from core.edit_algorithm import replace_lyrics, extract_lyrics
from core.align_algorithm import align_track
from core.g2p import char_to_phoneme, word_to_phoneme
```

---

## 注意事项

- 所有模型都下载到 **ComfyUI 的 `models/` 目录**下（`folder_paths.models_dir`），不是插件目录内
- SoulX-Singer 模型需放到 `ComfyUI/models/Soul-AILab/` 下
- CT-Transformer 标点模型自动下载到 `ComfyUI/models/ct-transformer-punc/`
- NLTK 数据自动下载到 `ComfyUI/models/nltk/`
- `g2pM` 首次使用时自动下载模型到 `~/.g2pM/`
- 歌词替换支持**任意长度差异**的新旧歌词（Collapse / Collapse+Distribute / Expand 三种模式自动选择）
- `<SP>` 标记始终保留，f0（帧级数据）完全不做修改
- 推理默认 FP32（`use_fp16=OFF`）——与上游参考实现一致，音质最稳；GPU 显存紧张时可开 FP16
- `MIDI Synthesize Audio` 的 `prompt_metadata` 强烈推荐从参考音频预转写得到，可显著加速并避免 SVS 口齿不清

---

## 项目结构

```
ComfyUI-MIDI-Edit/
├── __init__.py          # ComfyUI 插件入口，导出节点映射
├── nodes.py             # ComfyUI 节点定义（6 个节点）+ 历史 API 再导出
├── core/                # 核心算法包（模块化）
│   ├── g2p.py           # G2P：char_to_phoneme / word_to_phoneme / normalize_digits
│   ├── ct_transformer.py # CT-Transformer 标点恢复模型（智能断句）
│   ├── midi_format.py   # Token / Track 数据结构 + JSON parse/serialize（FPS=50）
│   ├── text_utils.py    # clean_lyrics / split_lyrics_to_sentences / is_reduplication
│   ├── speed.py         # 变速：duration 缩放 + f0 插值重采样
│   ├── edit_algorithm.py # MIDIEditLyrics 实现：replace_lyrics / extract_lyrics 等
│   ├── align_algorithm.py # MidiLyricsAlignment v3：align_track / segment_sentences / calculate_spd
│   └── soulsx_singer.py # SoulX-Singer 集成的纯函数层：transcribe_audio / synthesize_audio
├── SoulX-Singer/        # git submodule — 歌声合成引擎 + 预处理管线
├── requirements.txt     # Python 依赖（含 SoulX-Singer 所需库）
├── pyproject.toml       # Comfy Registry 发布配置
├── CHANGELOG.md         # 更新日志
├── docs/
│   ├── REQUIREMENT.md   # 原始需求文档
│   ├── alignment-algorithm.md      # MidiLyricsAlignment 算法说明
│   ├── midi-json-format.md         # MIDI JSON 字段说明
│   ├── workflow.json               # 端到端"魔改歌词"工作流（转写→编辑→合成）
│   ├── midi-edit-lyrics.json       # 仅歌词编辑工作流文件
│   └── midi-edit-lyrics.json.png   # 工作流截图
├── tests/
│   ├── test_alignment.py            # 统一对齐算法测试套件
│   ├── conftest.py                  # pytest 配置（slow marker）
│   └── fixtures/
│       └── vocal_sample.json        # 真实人声 track 回归 fixture
├── models/
│   └── nltk/            # NLTK 数据（自动下载）
└── README.md
```

---

## 节点速查表

| 节点名 | 类名 | 分类 | 主要输入 | 主要输出 |
|--------|------|------|----------|----------|
| MIDI Transcribe Audio | `MIDITranscribeAudio` | MIDI-SoulX | `audio` | `midi_json` |
| MIDI Synthesize Audio | `MIDISynthesizeAudio` | MIDI-SoulX | `midi_json` + `prompt_audio` | `audio` |
| MIDI Edit Lyrics | `MIDIEditLyrics` | MIDI-Edit | `midi_json` + `new_lyrics` | `midi_json` |
| MIDI Lyrics Alignment | `MidiLyricsAlignment` | MIDI | `midi_json` + `lyrics` | `midi_json` + `warnings` |
| MIDI Extract Lyrics | `MIDIExtractLyrics` | MIDI-Edit | `midi_json` | `lyrics_text` |
| MIDI Merge Repeated Chars | `MIDIMergeRepeatedChars` | MIDI-Edit | `text` | `text` |

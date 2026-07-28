# Changelog

All notable changes to ComfyUI-MIDI-Edit will be documented in this file.


## [2026-07-27] v3.4.2

### Added

- **Score 模式自动子音符拆分（contour preservation）**。score 模式用单个
  `note_pitch` 合成，丢失音符内部的 f0 走向（实测 63% 的音符内部变化 > 2 半音），
  导致合成旋律变平、听起来走调。现在 score 模式合成前会自动把 f0 走向大的音符
  （跨度 ≥ 2 半音）拆成 2 个子音符，每个子音符的 pitch 取该段 f0 中位数，形成
  阶梯近似。第二个子音符 `note_type=3`（续音）防止模型重新咬字。仅拆 target，
  不拆 prompt（音色参考保持原样）。仅对 score 模式生效，melody 不受影响。

  采用两遍扫描：第一遍找出所有拆分候选及 f0 跨度；第二遍按乐句（SP 之间的区域）
  分组，每句只保留跨度最大的 `max_splits_per_phrase=8` 个拆分，防止长句 token
  数暴增导致模型音色迁移失败（变男声）。

### Fixed

- **Score 模式"变男声"bug**。两个独立原因：
  1. **Broken token**：ROSVOT 偶尔在歌手实际在唱的位置检测到静音（pitch=0），
     preserve_sp 替换文字后变成"有真实 phoneme 但 pitch=0"的破损 token，
     模型看到音素却没有音高，导致音色异常。新增 `_fix_zero_pitch_notes`
     在合成前自动用 f0 中位数填补这些位置的 pitch。
  2. **Token 膨胀**：长句（如 23 字）拆分后 token 数暴增，稀释模型在
     prompt↔target 之间的 cross-attention，导致音色迁移失败。通过每句
     拆分数上限（`max_splits_per_phrase=8`，按 f0 跨度优先）解决。

### Changed

- **`MIDI Lyrics Alignment` 的 `preserve_sp` 默认改为 ON**。保留原曲 SP 结构
  （位置/时长/f0）是改歌词合成的推荐模式，改为默认勾选。

- **`hybrid` 控制模式从节点 UI 移除**。实测 hybrid 效果与 melody 几乎无区别
  （模型在 f0 存在时忽略 note_pitch），无实用价值。实现代码保留在
  `core/soulsx_singer.py` 供未来使用，但不再暴露为用户选项。


## [2026-07-27] v3.4.1

### Fixed

- **`MIDI Transcribe Audio` 修复 ROSVOT 偶发音高量化错误（score 模式走调）**。
  ROSVOT 在乐句末尾长音、颤音/滑音位置偶尔把 `note_pitch` 量化到错误的
  MIDI 数（实测最大偏差 +3.86 半音）。`MIDI Synthesize Audio (control=score)`
  直接读 `note_pitch` 合成，导致这些位置严重走调（用户报告的"的天涯" /
  "地开花" / "的泪花" 变调全部命中偏差 ≥ 1.9 半音的位置）。`control=melody`
  模式用 f0 真实音高合成，不受影响。

  新增 `_correct_pitch_from_f0` 后处理：对每个非 SP 音符，取其时长窗口内
  voiced f0 中位数转 MIDI 数，若与 `note_pitch` 偏差 ≥ 2 半音则替换。跳过
  SP（`note_pitch==0`）和 voiced 帧 < 3 的不可信位置。阈值硬编码 2 半音，
  FPS=50（与 SVS data_processor 一致）。

  在 segment 1 实测数据上验证：纠正 8/69 个非 SP 音符，纠正后剩余最大
  |Δsemi| = 1.408（< 1.5），且不引入任何新的 ≥ 2 半音偏差。该步骤在
  `convert_metadata`（f0 已写入）和 `_merge_invalid_repeated_chars`（note
  列表已定型）之后执行，ref / non-ref 两条流水线均覆盖。


## [2026-07-24] v3.4.0

### Added

- **`MidiLyricsAlignment` 新增 `preserve_sp` 可选输入**（BOOLEAN，默认 OFF）。
  开启时进入**保留原曲 SP 模式**：保留原曲所有 `<SP>` 的位置/时长/f0，
  保留原曲每个非 SP token 的 pitch/duration/f0，**只替换 text/phoneme**。

  解决了改词后合成的两个核心问题：
  1. **异常停顿**：标准模式丢弃原曲 SP 并重建，SP 数量和时长变化导致
     合成时出现莫名停顿。preserve_sp 模式保留原曲 SP，停顿与原曲一致。
  2. **melody 变调**：`MIDIEditLyrics` 的 Collapse 右对齐导致新字继承
     不同位置 token 的 pitch，melody 合成时变调（尤其尾音）。
     preserve_sp 模式同位置继承 pitch，旋律完全保留。

  **累计比例匹配**分配新歌词到原曲各 section（原 SP 之间的区域）：
  按各 section 的非 SP token 数比例分配新字数，自动处理新歌词与原曲
  字数/断句不一致的场景。section 内使用 Collapse（重复字复制）+
  Distribute（多字分配）+ Expand（拆分长 token）三模式自动适配。

  f0 和 time 字段完全保留原曲，不做任何重建。总时长守恒。

### Changed

- **`MIDI Transcribe Audio` 的 `merge_held_notes`**（原 `fix_held_note_sps`）
  重命名为更直观的名称。功能不变：合并 ROSVOT 在持续音中插入的 `<SP>`。

- **`MIDI Transcribe Audio` 新增 `merge_repeated_chars`**（BOOLEAN，默认 ON）。
  合并无效重复字（melisma 导致的连续相同字），duration 累加。
  参考歌词优先：有参考歌词时按歌词中该字的最大连续出现次数保留；
  无参考歌词时查 `models/reduplication-verbs.txt`（619 条 AA 形式叠词），
  在词库内保留 2 个、不在词库保留 1 个。

- **全局 DTW 参考分配**：`MIDI Transcribe Audio` 的两遍管线从比例分配
  改为全局 DTW 对齐分配参考歌词到各 segment。解决了 ASR 各段漏检率不
  均匀时参考歌词错位的问题（原曲有 132 字，ASR 检测 97 字 → 各段参考
  分配从 `[31,30,31,32,8]` 修正为 `[23,23,23,23,40]`）。

- **`models/reduplication-verbs.txt`** 整理为 619 条纯 AA 形式叠词
  （从原始 735 条混合格式提取，去重、合并 `_REDUP_WORDS` 硬编码表）。

### Fixed

- **`_merge_invalid_repeated_chars` 索引 bug**：合并函数使用相对索引
  而非绝对索引，导致 run 起始位置 `i > 0` 时删除错误位置的 token。

### Verified

- 沧海笑（30s 中文歌）：preserve_sp 模式下 SP 数量/时长/总时长与原曲
  完全一致。改词后 pitch 同位置继承，无移位。
- 人生不过一场体验（58s 中文歌）：全局 DTW 分配 132 字参考歌词到
  5 个 ASR segment（97 字），lyrics_text 100% 匹配参考歌词。
- 184 个现有测试全部通过。


## [2026-07-20] v3.3.0

### Added

- **`MIDI Transcribe Audio` 新增 `reference_lyrics` 可选输入**：当用户提供
  标准歌词时，ASR 解码会被强制对齐到该歌词，**文字识别率达到 100%**，且
  pitch / duration / f0 / SP 停顿 / ROSVOT 的 melisma 检测（一个字跨多个
  音符）全部保持由音频驱动的原行为不变——只替换每个 token 的*文字身份*。
  适用于：转写已知歌词的翻唱/原唱音频、消除 ASR 语音混淆
  （如 长→潮、几→记、之→知）。

  工作原理（不改 SoulX-Singer 子模块）：
  1. 第 1 次 ASR（无 hotword）→ 拿到 segment 字数估算
  2. 按字数比例把参考歌词切到各 segment
  3. 第 2 次 ASR `hotword=<segment 参考短语>` → 修正大部分语音混淆
  4. 字符级 DTW 兜底：把 ASR 字符映射到参考字，时间戳完全保留
  5. ROSVOT 拿对齐后的 `words` / `word_durs` 跑 note 检测，行为不变

  参考歌词里的换行/`，。！？；：` 等标点会被剥离（仅作语义切片提示用），
  SP 停顿仍由 `vocal_detector` + f0 后处理按音频驱动，与原管线一致。

  当前仅 Mandarin/Cantonese 路径走强制对齐（English 路径回退到默认 ASR）。

  **`core.soulsx_singer.py` 配套 API**：`transcribe_audio(...)` 和
  `_preprocess_audio_to_metadata(...)` 新增 `reference_lyrics: str | None`
  关键字参数，便于 HTTP API 服务器复用。新增 `_ForceAlignLyricTranscriber`
  内部类（包装原 `LyricTranscriber`，零侵入注入到 `pipeline.lyric_transcriber`）。

- **`MIDI Transcribe Audio` 新增 `lyrics_text` 输出**：节点现在返回
  `(midi_json, lyrics_text)` 二元组。`lyrics_text` 是 **ROSVOT 之前** 的歌词
  文本（即送入音符转录器之前的 ASR / 强制对齐结果），一字一音节，没有
  ROSVOT 的 melisma 重复。`<SP>` 停顿转换为换行符。便于：

  - 快速查看 ASR / 强制对齐的识别结果（无需解析 MIDI JSON）
  - 当 `reference_lyrics` 启用时，验证强制对齐是否成功（应与参考歌词一致）
  - 把识别结果接到下游文本节点（如 `MIDI Edit Lyrics`、`MIDI Merge Repeated
    Chars`）做进一步处理

  `core.soulsx_singer.py` 配套：`transcribe_audio(..., return_lyrics=True)`
  返回 `(midi_json, lyrics_text)` 元组；默认 `return_lyrics=False` 返回
  MIDI JSON 字符串（向后兼容）。

### Fixed

- **`MidiLyricsAlignment` 引入 MIDIEditLyrics 风格的 collapse 模式**：当新字数
  小于原曲非 SP token 数（`C < nonsl`）且原曲有连续重复字 slot 时
  （`C <= S`，S 为 slot 数），新字会被**复制到 slot 内每个 token**，而非
  丢弃多余 token。与 `MIDIEditLyrics` 的 `_build_collapsed_slots` 行为一致。

  **示例**：原曲 `好看的<女女>人`（6 token，slot 数=5）+ 新词
  `强壮的男人`（5 char）→ 输出 `强壮的<男男>人`（6 token，男 复制到 女
  slot），duration 完全守恒，`note_type=3` 标记第二个 男 为延续音。

  之前的行为是直接丢弃多余的 `女` token，输出 `强壮的男人`（5 token），
  duration 少 0.4s，丢失原曲的 melisma 结构。

  **算法细节**：
  - 新增 `_build_section_aware_slots` helper，按原曲连续相同字合并成 slot
    （不跨 SP 边界合并：`甲甲<SP>甲甲` 是 2 个独立 slot，不是 1 个 count=4 slot）
  - `align_track` 在 `C <= S 且 C < nonsl` 时走 collapse 模式
  - 每个 char 在 `char_to_token_indices` 映射下可指向多个原 token（复制）
  - 复制的 token 保留各自的 pitch / duration / f0 段，互不干扰
  - `C >= nonsl` 时走原贪心压缩（distribute/expand），不受影响
  - SPD 计算用 `N_effective = num_new_sp + output_nonsl`，正确反映复制后的
    实际输出 token 数

### Changed

- **`_preprocess_audio_to_metadata` 返回类型**：从 `list[dict]` 改为
  `tuple[list[dict], str]`（metadata, lyrics_text）。`_resolve_prompt_metadata`
  更新为只取 `[0]`。这是内部 API 变更，对外接口（`transcribe_audio` /
  `synthesize_audio`）通过 `return_lyrics` 参数保持向后兼容。

### Verified

- 沧海笑样本（董贞，30s 中文歌）：原 ASR 有 6 个字符错误（长→潮、几→记、
  场→潮、晓→笑、之→知），启用 `reference_lyrics` 后 100% 准确；
  `note_pitch` / `duration` 与原输出完全一致；ROSVOT 仍把"沧海笑"3 字
  扩展为 5 个 note（沧沧 / 海海 / 笑），melisma 结构完整保留。
- `lyrics_text` 输出：不带 reference 时显示 ASR 原始结果（含错误），带
  reference 时与参考歌词完全一致。
- **`MidiLyricsAlignment` collapse 复制**：原 `好看的<女女>人` + 新
  `强壮的男人` → 输出 `强壮的<男男>人`（6 token，男男替代女女 slot，
  pitch [65,67] 保留，note_type [2,3] 标记延续音，duration 守恒 3.2s）。
- **跨 SP 边界不合并**：原 `甲甲<SP>甲甲` → 2 个独立 slot，新词 `乙丙`
  → `乙乙<SP>丙丙`。
- **对照组**（`C >= nonsl`）：新词 `强壮的老男人`（6 char = 6 token）→
  输出 `强壮的老男人`（6 token，无复制）。
- 向后兼容：`transcribe_audio(return_lyrics=False)`（默认）与 v3.2.0 输出
  完全一致（text / pitch / note_type / duration 全等）。
- 全部 184 个测试通过（181 原有 + 3 新增 collapse 行为测试）。


## [2026-06-26] v3.2.0

### Fixed

- **`MIDI Synthesize Audio` SVS garbled pronunciation (root cause)**: the prompt
  metadata was previously taken from the first segment of the *target* MIDI JSON,
  which mismatches the reference (prompt) audio waveform. The SVS data processor
  truncates the prompt waveform to a frame count derived from the metadata's
  duration/f0, so a target-derived prompt produced misaligned/truncated audio and
  unintelligible singing. Prompt metadata now **always** comes from the reference
  voice: either from the new optional `prompt_metadata` input (recommended — feed
  the output of `MIDI Transcribe Audio` run on the prompt audio) or, when left
  empty, from preprocessing the prompt audio internally.

### Changed

- **`control` default `score` → `melody`**: aligns with the upstream SoulX-Singer
  CLI and the reference SVS implementation, which default to the F0 contour.
- **Inference now defaults to FP32** (previously FP16 on GPU). FP16 weights were a
  secondary suspect for degraded audio quality. A new `use_fp16` input lets you
  opt back into autocast mixed precision on CUDA for speed.
- **`MIDISynthesizeAudio` new optional inputs**:
  - `prompt_metadata` (STRING) — metadata describing the prompt audio's real
    acoustic content. Recommended: connect the `midi_json` output of
    `MIDI Transcribe Audio` run on the *reference* audio here, to avoid
    re-running preprocessing on every call. When empty, the node preprocesses the
    prompt audio internally.
  - `use_fp16` (BOOLEAN, default OFF/FP32) — enable FP16 autocast on GPU.

### Added

- **SoulX-Singer 集成**：以 git 子模块形式集成 SoulX-Singer，支持完整的歌声合成管线（音频转写 + 歌声合成）。
- **`MIDI Transcribe Audio` 节点**：AUDIO → MIDI JSON，调用 SoulX-Singer 预处理管线（人声分离 → F0 提取 → VAD → 歌词转录 → 音符转录）。
- **`MIDI Synthesize Audio` 节点**：MIDI JSON + 参考音色 AUDIO → 合成歌声 AUDIO（flow-matching 扩散模型）。
- **`core/soulsx_singer.py` 独立模块**：`transcribe_audio` / `synthesize_audio` 不依赖 ComfyUI，接受文件路径或 numpy 数组，供 HTTP API 服务器复用；`set_models_base(path)` / `get_models_base()` 自定义模型路径。
- **transformers 兼容补丁**：自动适配 transformers >= 4.53（LlamaAttention position_embeddings、LlamaConfig._attn_implementation、3-tuple 返回值）。
- **Internal SVS API** in `core/soulsx_singer.py`:
  - `_preprocess_audio_to_metadata` — runs the preprocess pipeline and returns the
    raw metadata list (shared core of `transcribe_audio`); raises `RuntimeError`
    on an empty result.
  - `_coerce_prompt_metadata` — coerces a user-supplied prompt_metadata value
    (None / str MIDI JSON / str raw metadata JSON / list / dict) into a validated
    metadata list; raises `ValueError` on a non-empty list with non-dict/empty-dict
    items.
  - `_resolve_prompt_metadata` — resolves prompt metadata from the provided value,
    falling back to auto-preprocessing the reference audio.
- **`MIDISynthesizeAudio` exposes `cfg` and `n_steps`**: two new optional inputs that
  pass through to SoulX-Singer inference. `cfg` (FLOAT, default 3.0) is the
  classifier-free guidance scale — raise toward 4-5 in `melody` mode if diction is
  muddy (stronger adherence to lyrics/phonemes); too high may over-saturate.
  `n_steps` (INT, default 32) is the flow-matching reverse-diffusion step count —
  higher slightly improves quality at the cost of speed. `core.synthesize_audio`
  gains matching `cfg` / `n_steps` keyword args (None = use config defaults 3 / 32);
  the SVS `config` is deep-copied per call so the global `_svs_config` singleton is
  never mutated.
- **`MIDI Extract Lyrics` new `resegment` switch**: when ON, the node ignores the
  original `<SP>` phrasing and re-segments the lyrics — digits are converted to
  Chinese number chars, spaces/newlines/punctuation are stripped, consecutive
  repeated characters are merged, CT-Transformer re-adds punctuation, and the result
  is emitted as one sentence per line (every punctuation-delimited fragment becomes
  its own line). Useful for re-flowing lyrics into natural sentence boundaries. When
  `resegment` is ON, `merge_repeated` has no extra effect (the merge step is already
  part of the pipeline).

### Docs

- **`control` selection strategy**: node description now documents when to pick
  each mode — `melody` when the target has a real vocal F0 (e.g. lyrics edited from
  a vocal) for closer timbre to the prompt; `score` when the target is a score /
  instrumental without a reliable F0, for clearer diction. `cfg` tuning guidance is
  also documented.

### Note

- **`rescale_cfg` is intentionally not exposed**: upstream `SoulXSinger.infer` does
  not accept it (it is hardcoded to 0.75 inside `flow_matching.reverse_diffusion`).
  Exposing it would require monkey-patching model internals, which violates the
  reuse principle (the SoulX-Singer submodule is treated as read-only).

## [2026-06-25] v3.1.0

### Changed

- **断句改用 CT-Transformer 标点模型**：不再用 jieba，不再参照原 SP 数
  - 先按标点切，超过 10 字用 CT-Transformer 加标点后按标点全切
- **SPLIT 分配改为按 duration 比例**：长 token 多分字，短 token 少分字
- **time 字段从 f0 帧数反算**：修复 SoulX-Singer "could not broadcast" 错误
- **移除无效参数**：w_pitch / w_duration / w_structure（DP 遗留）

### Refactored

- **代码模块化**：nodes.py 从 1764 行瘦身到 330 行
  - 新建 core/ 包（7 个模块）：g2p / ct_transformer / midi_format / text_utils / speed / edit_algorithm / align_algorithm
  - 删除 alignment/ 包

## [2026-06-24] v3.0.0

### Added

- **v3 对齐算法**：彻底重写 MidiLyricsAlignment，放弃 DP
  - 顺序映射 + 贪心压缩（jieba 辅助）
  - 原始 SP 移除，按新歌词断句重建 SP
  - f0 按 token duration 切段重建（50fps，源码确认）
  - SPD 公式计算 SP 时长
  - 断句：标点优先

### Removed

- DP 相关代码全部删除（cost.py / dp.py / rebuild.py / preprocess.py）

## [2026-06-23] v2.1.0

### Added

- **jieba 中文分词**：歌词先用 jieba 分词，多字词作为一个 Unit 映射到一 token
  - 多字词（全程/跟着/跑毒）不被拆散
  - section 分配按词数比例，不拆词
  - SPLIT 只发生在单字词上，大幅减少 SPLIT 次数
  - 14 字歌词从 ~14 SPLIT 降到 ~4 SPLIT
## [2026-06-23] v2.0.0

### Added

- **MidiLyricsAlignment 节点**：基于联合动态规划的统一歌词对齐算法
  - Section 级 DP：SP 硬保留，每段独立对齐，f0/pitch 绝对不动
  - 5 种原子操作：REPLACE / WORD_SPAN / SPLIT / DROP / SP_ALIGN
  - SPLIT 限制：每 token 最多 2 字（防止 0.03s/字灾难）
  - pitch 连贯性代价：DP 优先选 pitch 连贯的对齐路径
  - 字级歌词分配：按 section 容量比例切分，不再整行塞一段
  - 多 track 歌词按 duration 比例自动分配
  - 自包含 alignment/ 子包（不依赖外部 nodes.py）
  - warnings 作为第二个输出（SP_COUNT_REDUCED / MIN_DURATION_UNRESOLVED 等）
  - force_tone4 接线：高音（≥G5）中文音素强制四声
- **回归测试**：真实 42-token track + 150-token 性能测试
- **文档**：算法设计文档 + MIDI JSON 格式说明 + 设计 spec + 实现 plan
- **工作流**：ComfyUI 工作流文件（midi-lyrics-alignment.json）

### Fixed

- **note_type=3 叠词排除**：连续重复字标 type=3，但排除叠词（哥哥/妹妹等独立词汇）
- **note_type 语义化**：基于 token 在乐句/词中的角色判断，不继承原值不写死
- **SP 硬保留**：SP 位置/f0/pitch 绝对不动（SP 软约束导致 f0 错位 → 唱不出来）
- **f0 绝对不修改**：SP 硬保留保证 f0 与 token 对应关系不变
- **`_redistribute_drops` 均匀分配**：DROP duration 均匀分给已填 token（之前按比例，富者愈富）
- **`_uniform_sp_fill` 去重丢失 + SP 候选避开英文词内部**
- **`split_cost` off-by-one** + **`speed.py` 延迟 import**
- **ComfyUI nodes.py 命名冲突**：alignment 包完全自包含

## [2026-06-16] v1.8.0

### Added

- **英文单词级音素生成**：英文歌词不再逐字母处理，而是按完整单词生成 ARPAbet 音素（如 `wish` → `en_W-IH1-SH`），与 SoulX-Singer 原生格式完全一致
- **中英混合歌词支持**：同一句歌词可同时包含中文和英文，中文字逐字处理，英文词按词处理
- **英文单词比例分配**：当音符数多于英文单词数时，单词按词长比例分配到多个音符，首个音符 `note_type=2`，后续延续音符 `note_type=3`，共享同一音素
- **阿拉伯数字自动转中文**：歌词中的 `0-9` 自动转为 `零一二三四五六七八九`，再通过 g2pM 生成拼音

### Fixed

- **英文音素 `KeyError: 'en_a'`**：单字母不再直接作为音素，英文统一走 g2p_en 生成 ARPAbet 格式
- **英文合成噪声**：逐字母生成多音素"字母名"发音（如 `s` → `en_EH1-S`）导致短音符无法承载。改为单词级处理，每个音符承载完整单词音素
- **Collapse 模式 duration 篡改**（v1.7.1 已修复，此处补充说明）
- **`None` 输入崩溃**：ComfyUI 空输入/未连接节点导致 `expected string or bytes-like object` 错误。所有入口添加 None 防御
- **NLTK 数据路径**：`_NLTK_PACKAGES` 使用完整路径格式（`taggers/averaged_perceptron_tagger_eng`），避免本地数据查找失败导致下载超时

### Changed

- `clean_lyrics()` 保留空格（英文单词分词需要）
- `_split_lyrics_to_sentences()` 保留句内空格
- `_apply_char()` 支持 `preset_phoneme` 和 `is_continuation` 参数
- 英文单字母音素映射表 `_EN_LETTER_TO_PHONEME` 保留作为 fallback

## [2026-06-15] v1.7.1

### Fixed

- **Collapse 模式下 duration 不再被篡改**：当新歌词字数与原曲完全对应（N == S）时，duration 严格保持原值不变。此前 min-duration enforcement（0.30s 最低时长）会无条件把短 token 拉长并从长 token 借时间，破坏节奏。
- **浮点精度统一为 2 位小数**：输出 duration 不再出现 `0.32000000000000006` 等浮点伪影。新增 `_fmt_durs()` 校正函数，在四舍五入后调整末元素以确保总计不变。

### Changed

- Min-duration enforcement 现在仅在 Expand（拆分 token）时生效，Collapse 模式完全跳过。

## [2026-06-15] v1.7.0

### Added

- **变速功能**（`speed` 参数，默认 1.0）
  - 范围 0.1 ~ 3.0，步进 0.1
  - `duration` 按比例缩放（`*= speed`）
  - `f0` 按比例线性插值重采样（帧数 = `round(原帧数 × speed)`），音高轮廓不变，只拉伸/压缩时间轴
  - 浮点精度清理：duration 保留 4 位有效小数，f0 保留 1 位小数

## [2026-06-12] v1.6.0

### Added

- **字数分配模式选项**（`split_mode` 参数，默认 `token`）
  - `token`（默认）：按原曲每个 section 的 token 数比例分配字数（原有方式）
  - `duration`：按原曲每个 section 的时长比例分配字数（新方式，时长长的 section 分到更多字）
  - 仅在智能拆句触发时（句子数 ≠ 原 section 数）生效
  - 例如：总时长 20 秒，第一句 5 秒占 1/4 → 分配 1/4 的字数

## [2026-06-12] v1.5.0

### Added

- **固定停顿模式开关**（`fixed_pause` 参数，默认开启 Fixed）
  - **Fixed（默认）**：SP 时长保持不变，现有行为
  - **Flexible（关闭时）**：当检测到 token 拥挤或 SP 漫长时，自动将 SP 时间按节奏比例匀给句内 token
  - 触发条件（任一满足即可）：SP 时长 ≥ 2 倍 token 平均时长，或 token 平均时长 < 0.30s
  - 匀出后 SP 降至与一个普通 token 等长，释放的时间按现有 duration 比例分配给所有 filled token
  - 每个 section 独立判断，总 group 时长严格守恒

## [2026-06-12] v1.4.0

### Changed

- **智能拆句算法重写为比例分配模式**：当用户歌词句子数 ≠ 原曲 section 数时触发
  - 按原曲每个 section 的 token 数（每个 token 算 1 个字，重复字不合并）占总 token 数的比例，计算每个 section 的预期字数（四舍五入 + 最后 section 兜底）
  - 从第一个 section 开始，剩余歌词送 CT-Transformer 加标点，取第一个标点位置切分
  - 切点处字数与预期偏差 ≤ 30%（向上取整）→ 使用 AI 切点；否则按预期字数硬切
  - 切完后去掉所有标点，剩余歌词继续处理下一个 section；最后 section 收所有剩余
  - 句子数 = 原 section 数时不触发此逻辑，走原来的 collapse/expand 匹配

## [2026-06-12] v1.3.0

### Changed

- **智能拆句算法重写**：每次只切一刀，产出 2 份干净纯文字歌词
  - 切点优先级：句号/问号/叹号 > 逗号/顿号/分号/冒号
  - 切点位置：选靠近中间的标点，保证两半均衡
  - 切完后去掉所有标点，返回纯歌词文字
  - 不再合并多余的拆分结果 — 拆多少用多少
  - 下一轮继续选最长句子重复切分，直到句子数匹配原曲 section 数

### Added

- **CT-Transformer 智能拆句**：当用户歌词按标点/换行切分后少于原曲句子数时，使用 CT-Transformer 标点恢复模型（基于 FunASR）自动在自然语言边界处插入标点，将长句拆分为多个短句以匹配原曲结构
  - 模型自动从 ModelScope 下载到 ComfyUI `models/ct-transformer-punc/` 目录（~270MB，首次运行时下载）
  - 迭代拆句策略：优先拆分字数差异最大的句子，AI 无法断句时尝试下一个，全部无法拆分时从中间硬拆
  - 新增依赖：`modelscope`、`onnxruntime>=1.17.0`
- **Collapse+Distribute 模式**：替换了旧的 Token 模式。当新歌词字数多于 collapse slot 数但不超过原始 token 数时，多 count 的 slot 内会分配不同的字（如原曲 `天天` → 新歌词 `把它`），尊重原曲的去重结构
- **Right-align Collapse**：Collapse 模式改为右对齐映射，确保最后一个字映射到最后一个 slot（通常是结尾长音），避免结尾长音被均分导致歌曲戛然而止

### Changed

- **3 模式算法升级为 4 分支**：
  - Collapse（N ≤ S）：slot 映射，右对齐
  - Collapse+Distribute（S < N ≤ M）：slot 映射 + 多字 slot 内分配不同字
  - Expand（N > M）：拆分最长 token
- **最小 duration 下限**：重建阶段所有非 SP token 的 duration 不低于 0.30s，从同 section 最长 token 借时间

### Fixed

- **"飞"戛然而止**：Collapse 模式右对齐后，最后一个字（如"飞"）正确继承原曲结尾长音 slot 的 duration（~1.08s），不再被均匀分配
- **"它"没唱出来**：Collapse+Distribute 模式正确处理 `天天` 等重复字 slot，每个 token 分配独立的字而非使用旧的 1:1 Token 模式

## [2026-06-11]

### Added

- **Fallback sentence splitting**: when punctuation/newline split yields fewer sentences than original MIDI sections, lyrics are automatically redistributed to match the original section sizes (punctuation ignored, last sentence may be shorter)

### Changed

- **Empty token duration redistribution**: when a section has fewer new lyrics than original tokens, the freed duration is now evenly distributed among the filled tokens (remainder to last) instead of being absorbed into adjacent SPs

### Fixed

- **Consecutive SP merging**: empty tokens between SP markers no longer produce duplicate `<SP>` entries in the output
- **Empty remaining tokens**: tokens beyond the new lyrics length are properly cleared (text/phoneme emptied) instead of retaining original lyrics

### Added

- **Smart tone adjustment** (`force_tone4` toggle + `high_pitch_threshold` input)
  - When enabled, Chinese phonemes at very high pitch (MIDI note >= 79 / G5) are forced to tone 4 (去声)
  - Threshold is configurable (INT, range 0-127, default 79)
  - Only takes effect when the toggle is ON; English phonemes are not affected

### Changed

- **Default high-pitch threshold** changed from 72 (C5) to 79 (G5)
  - C5 is within normal vocal range for many singers and too low to force tone adjustment
  - G5 is the actual threshold where tone control becomes difficult

### Fixed

- Long-note repetition alignment now groups consecutive identical tokens into slots
  - MIDI text like `兄 兄` (long note spanning 2 syllables) is treated as 1 slot
  - User input chars are automatically expanded to match the original repetition pattern
  - SP tokens break consecutive groups (`兄 <SP> 兄` = 2 slots, not 1)

## [2026-06-05]

### Added

- **MIDI Edit Lyrics** node (`MIDIEditLyrics`)
  - Replace lyrics in MIDI JSON with new text
  - Auto-generate phonemes: Chinese chars → `zh_` prefixed pinyin, English → `en_` prefixed phonemes
  - `<SP>` markers and non-lyric fields (duration, note_pitch, note_type, f0) preserved
  - Position-based replacement: extras ignored, shortage keeps original

- **MIDI Extract Lyrics** node (`MIDIExtractLyrics`)
  - Extract lyrics text from MIDI JSON
  - Removes spaces, replaces `<SP>` with newlines

- **Dependencies**: `g2pM` (Chinese G2P), `g2p_en` (English G2P)
- **NLTK data**: auto-downloaded to local `models/nltk/` on first use
- **Project files**: `__init__.py`, `nodes.py`, `requirements.txt`, `README.md`

### Added

- MIT LICENSE
- `pyproject.toml` for Comfy Registry publishing
- Published to [Comfy Registry](https://registry.comfy.org) (publisher: ahkimkoo)

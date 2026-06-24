# Changelog

All notable changes to ComfyUI-MIDI-Edit will be documented in this file.


## [2026-06-24] v3.0.0

### Added

- **v3 对齐算法**：彻底重写 MidiLyricsAlignment，放弃 DP
  - 顺序映射 + 贪心压缩（jieba 辅助）
  - 原始 SP 移除，按新歌词断句重建 SP
  - f0 按 token duration 切段重建（50fps，源码确认）
  - SPD 公式计算 SP 时长
  - 断句：标点优先，不足参照原 SP 数切最长句

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

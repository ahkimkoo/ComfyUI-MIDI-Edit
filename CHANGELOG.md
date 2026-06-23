# Changelog

All notable changes to ComfyUI-MIDI-Edit will be documented in this file.

## [Unreleased]

### Added

- **MidiLyricsAlignment 节点**：基于联合动态规划的统一歌词对齐算法。
  - 单一 DP 处理所有字数匹配/不匹配情况，无 `if/else` 场景分支
  - 加权代价函数（`pitch` / `duration` / `structure`）求全局最优对齐
  - SP 软约束（数量守恒，位置可由 DP 最优放置）
  - 中英混合粒度（中文字 `max_occupy=1`，英文词 `max_occupy ≤ K=4`）
  - 5 种原子操作：`REPLACE` / `WORD_SPAN` / `SPLIT` / `DROP` / `SP_ALIGN`
  - 总 duration 与 SP 数量守恒（含多 section DROP 重分配）
  - 新增 `alignment/` 子包：`models` / `parser` / `cost` / `preprocess` / `dp` / `rebuild` / `speed`
- **回归测试**：基于真实人声 track（`docs/midi-edit-lyrics.json` 的 `vocal_0_15000`，42 token / 4 SP / 14.99s）的不变量测试
- **性能测试**：150 token track 在 3s 内完成对齐（纯 Python DP），`@pytest.mark.slow` 标记
- **`tests/conftest.py`**：注册 `slow` pytest marker
- **`docs/alignment-algorithm.md`** / **`docs/midi-json-format.md`**：算法与格式说明文档

### Fixed

- **多 track 歌词分配**：输入含多 track 时，整段歌词按各 track 的非 SP duration 比例自动分配（之前每个 track 都被塞入完整歌词，导致短 track 灾难性 SPLIT）
- **`_redistribute_drops` 多 section 守恒**：DROP 的 duration 改为全局按比例分配（之前每个 section 重复应用 `total_drop`，导致总时长膨胀）
- **`split_cost` off-by-one**：divisor 从 `current_share_count + 1` 改为 `+2`（SPLIT 的宿主已被前序操作消费，含 1 个初始消费者）
- **SP 守恒：`_uniform_sp_fill` 去重丢失**：当 `sp_target` 大于文本长度时，改为填满所有可用间隙并触发 `SP_COUNT_REDUCED` 警告（之前均匀填补产生重复位置，`set()` 去重后 SP 数静默减少）
- **SP 候选避开英文词内部**：新增 `_english_word_interiors(text)` 过滤，SP 候选位置不落在英文词内部（之前 tokenizer 的 en 分支词扫描吞掉词内 SP 候选）
- **`speed.py` 延迟 import**：移除模块级 `sys.path` 操纵，`from nodes import _apply_speed` 改为函数内延迟 import（避免模块加载副作用）
- **warnings 作为节点输出**：`RETURN_TYPES` 从 `("STRING",)` 改为 `("STRING", "STRING")`，增加 `warnings` 输出（之前仅 `print`，ComfyUI UI 看不到）
- **`force_tone4` 接线**：参数实际生效——高音（≥ G5）中文音素强制改四声（之前参数存在但 `align_lyrics` 内未使用）

### Changed

- README 节点列表从 3 个升级为 4 个，新增 `MidiLyricsAlignment (DP)` 章节（含参数表、与 `MIDIEditLyrics` 的差异对比、示例）
- 项目结构说明同步加入 `alignment/` 子包与 `tests/` 目录

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

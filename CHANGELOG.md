# Changelog

All notable changes to ComfyUI-MIDI-Edit will be documented in this file.

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

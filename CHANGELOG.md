# Changelog

All notable changes to ComfyUI-MIDI-Edit will be documented in this file.

## [2026-06-10]

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

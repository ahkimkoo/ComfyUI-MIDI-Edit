# ComfyUI-MIDI-Edit 2026 年 7 月发布说明 / July 2026 Release Notes

## 中文版

### 概览

2026 年 7 月的更新，核心目标是把 ComfyUI-MIDI-Edit 从一个“能用的
MIDI 编辑工具”，推进成一个更接近生产可用的 **音频转 MIDI、歌词替换、
歌声合成** 工作流。

这一轮版本重点改善了三个用户体验：

1. **转写更准**：从真实人声音频生成 MIDI 时，歌词更容易对齐到正确文本
2. **改词更稳**：替换歌词时，原曲停顿、时值和旋律结构更容易保留
3. **合成更自然**：score 模式下的跑调、音色漂移等问题明显减少

如果你的工作流是“从真实演唱音频出发，保留原曲节奏和情绪，只替换歌词内容”，
这是目前为止最重要的一次版本迭代。

### 主要亮点

#### 1. 参考歌词引导转写

`MIDI Transcribe Audio` 现在支持 **reference lyrics（参考歌词）引导 ASR**。

这意味着：

- 如果你已经知道原曲的标准歌词，可以在转写时直接提供参考文本
- 系统会用这份参考歌词修正常见的 ASR 误识别，例如同音字、漏字、段落错位
- 最终得到的 MIDI 文本内容会**更接近真实歌曲歌词**，同时 pitch / duration /
  f0 / 停顿等音乐信息仍然由原始音频决定

它带来的直接价值：

- 减少转写后再手工修歌词的工作量
- 降低后续改词时的对齐误差
- 让“我唱了什么”和“MIDI 写了什么”更加一致

算法上，核心不是只相信 ASR，而是把 **ASR + DTW 对齐** 组合起来：

- 先做识别
- 再把识别到的字序列和参考歌词做动态时间规整（DTW）
- 后续又进一步升级成 **全局 DTW 的参考歌词分配**，不再简单按比例切分参考歌词，
  所以面对某些段识别少、某些段识别多的情况时，更稳

#### 2. 新增 `lyrics_text` 输出

`MIDI Transcribe Audio` 现在会额外输出 **`lyrics_text`**。

这是一个很实用的产品级增强：你可以直接查看当前识别出来的歌词文本，不需要再去
手工阅读原始 MIDI JSON。

它的价值在于：

- 转写完就能快速做人工 QA
- 方便接到下游文本节点继续处理
- 更容易和原歌词、参考歌词直接对照

#### 3. `preserve_sp` 对齐模式

`MIDI Lyrics Alignment` 新增并强化了 **`preserve_sp`** 工作模式，并且现在已经是
推荐默认模式。

它的作用是：

- 保留原曲 `<SP>` 的位置
- 保留原曲停顿时长
- 保留原曲 `f0`
- 保留原曲音符的时间结构和 pitch 结构
- 只替换歌词内容（`text` / `phoneme`）

为什么重要：

在真实歌曲改词场景里，最容易毁掉听感的，往往不是字本身，而是把原曲的呼吸、停顿、
节奏结构打乱了。`preserve_sp` 的核心价值，就是**不重建停顿，而是继承原曲停顿**，
让改词结果更像原唱，而不是像重新排了一遍句子。

算法上，它会：

- 先按原曲 `<SP>` 把轨道切成多个 section
- 再按 section 比例把新歌词分配进去
- 非 SP 音符继承原始音乐结构，SP 保持不动

这也是为什么本轮之后，`preserve_sp` 被改成了默认勾选。

#### 4. 更好的持续音与重复字处理

这一轮也加强了两个很实用的转写后处理能力：

- 持续音中的伪停顿修复（`merge_held_notes`）
- melisma 导致的无效重复字合并（`merge_repeated_chars`）

对用户的意义是：

- 持续音里更少出现不合理的 `<SP>`
- 更少出现拖腔导致的重复字污染
- 下游歌词对齐和合成时，token 结构更干净

算法层面上，这里引入了一个整理过的叠词词表，用来区分“真正应该保留的叠词”
和“只是拖腔副产物的重复字”。

#### 5. score 模式音高修复

7 月里最关键的一批合成修复，集中在 **score 模式**。

原本的问题是：

- ROSVOT 有时会把某些音高量化错
- score 模式直接读这个 `note_pitch` 合成，就会明显跑调

现在增加了两个层次的修正：

1. **gross pitch correction**：对明显量化错误的 `note_pitch`，用该音符时窗内的
   voiced `f0` 中位数回推 MIDI 音高进行修正
2. **sub-note contour preservation**：如果一个音符内部的音高走向变化很大，就把它
   拆成两个“子音符”，让 score 模式不再只能看到一个平值 pitch，而是看到一个
   阶梯近似的轮廓

这件事对听感的意义非常直接：

- 原来某些句尾长音、滑音、颤音会被压平
- 现在这些位置在 score 模式下更接近原唱的音高走向

算法上：

- 先用 `f0` 中位数修正明显错误的 `note_pitch`
- 再对内部音高跨度大的音符做子音符拆分
- 每个子音符的 pitch 取对应片段的 `f0` 中位数，形成“阶梯近似”

#### 6. score 模式音色稳定性修复

在真实测试中，又发现并修复了两类 score 模式的稳定性问题：

1. **broken token 修复**：某些 token 有真实 phoneme，但 pitch 被转写成了 0，
   模型会看到“有字无音高”，引发异常输出
2. **长句 token 膨胀控制**：长句如果拆太多子音符，会让 token 数暴增，稀释模型在
   prompt 和 target 之间的音色迁移能力，造成整句音色漂移（例如“变男声”）

现在的处理方式是：

- 先修复这类 zero-pitch broken token
- 再对每句可拆分的子音符数量设置上限，并按音高跨度从大到小优先保留最重要的拆分

这让 score 模式在保留吐字清晰度的同时，显著降低了音色漂移风险。

### 推荐工作流

#### 场景一：真实歌曲改词

推荐链路：

1. `MIDI Transcribe Audio`
2. 可选：提供 `reference_lyrics`
3. `MIDI Lyrics Alignment`，并保持 `preserve_sp=ON`
4. `MIDI Synthesize Audio`

模式建议：

- 如果最优先保留原唱旋律和音高走向，用 **`melody`**
- 如果最优先保留吐字清晰度，用 **`score`**

#### 场景二：转写结果质检

推荐结合：

- `lyrics_text` 快速看识别结果
- `reference_lyrics` 提高歌词一致性

### 升级提示

- `preserve_sp` 现在已经是推荐默认工作流
- `hybrid` 控制模式在这一轮中被探索过，但真实测试下没有明显优于 `melody`，
  因此不再保留为默认用户入口

### 总结

2026 年 7 月这一轮更新的本质，是把 ComfyUI-MIDI-Edit 从“改 MIDI 文本”推进到了
“真正可用于真实人声歌曲改词与重合成”的阶段。

从产品角度看，这一轮最大的价值是：

- 转写更可信
- 改词更稳
- 合成更接近原曲
- 用户对系统当前识别/对齐/合成状态的可见性更强

一句话总结：**这次发布不是功能堆积，而是把整条工作流往可交付、可复用、可生产使用
的方向推进了一大步。**

---

## English Version

### Overview

The July 2026 release focused on one clear goal: moving ComfyUI-MIDI-Edit from
a useful MIDI editing tool toward a more production-ready workflow for
**audio-to-MIDI transcription, lyric replacement, and singing synthesis**.

This release improves three core user experiences:

1. **More accurate transcription** from real vocal recordings
2. **Safer lyric replacement** without breaking phrasing or melody
3. **More natural synthesis** with fewer pitch and timbre failures

If your workflow starts from a real sung performance and you want to preserve
the original timing, phrasing, and emotional shape while changing the lyrics,
this is the most important release so far.

### Highlights

#### 1. Reference-Lyrics Guided Transcription

`MIDI Transcribe Audio` now supports **reference-lyrics guided ASR**.

In practical terms:

- If you already know the correct lyrics of a song, you can provide them during
  transcription.
- The system uses the reference text to repair common ASR mistakes such as
  similar-sounding characters, missed syllables, and segment drift.
- The resulting MIDI text becomes **much closer to the real song lyrics**,
  while musical information such as pitch, duration, f0, and pauses is still
  driven by the source audio.

Why it matters:

- Less manual cleanup after transcription
- More stable downstream lyric replacement
- Better consistency between what was sung and what the MIDI says

Algorithmically, this is not just raw ASR. It combines **ASR + DTW alignment**:

- recognize first
- align the recognized syllable sequence against the reference lyrics using DTW
- then, in a later improvement, switch from naive proportional allocation to
  **global DTW-based reference distribution**, which is much more stable when
  some segments under-recognize more than others

#### 2. New `lyrics_text` Output

`MIDI Transcribe Audio` now also outputs **`lyrics_text`**.

This is a practical product feature: you can inspect the recognized lyrics as
plain text without manually reading raw MIDI JSON.

Why it matters:

- Faster QA after transcription
- Easier chaining into downstream text nodes
- Simpler comparison against original or reference lyrics

#### 3. `preserve_sp` Alignment Mode

`MIDI Lyrics Alignment` now includes a stronger **`preserve_sp`** workflow, and
it is now the recommended default mode.

What it does:

- Preserves original `<SP>` positions
- Preserves original pause durations
- Preserves original `f0`
- Preserves original note timing and pitch structure
- Replaces only the lyric content (`text` / `phoneme`)

Why it matters:

In real-song lyric replacement, the thing that most often breaks the result is
not the words themselves, but the loss of the original breathing, silence, and
phrasing structure. `preserve_sp` keeps the original pause structure instead of
rebuilding it from scratch, so edited output stays much closer to the source
performance.

Under the hood:

- the track is split into **sections between SP markers**
- new lyrics are distributed proportionally across those sections
- non-SP notes inherit the original musical structure while SP notes stay in place

This is why `preserve_sp` has now become the default recommended workflow.

#### 4. Better Handling of Held Notes and Repeated Characters

This release also improves two important transcription cleanup steps:

- held-note SP cleanup (`merge_held_notes`)
- invalid repeated-character merging (`merge_repeated_chars`)

Why it matters:

- Fewer fake pauses inside sustained notes
- Fewer duplicated characters caused by melisma artifacts
- Cleaner token structure for downstream alignment and synthesis

Algorithmically, this uses a curated reduplication lexicon to distinguish true
repeated words from repeated characters caused only by vocal stretching.

#### 5. Score-Mode Pitch Repair

One of the most important synthesis improvements in July focused on
**score mode**.

Previously:

- ROSVOT could occasionally quantize a note pitch incorrectly
- score-mode synthesis would directly use that incorrect `note_pitch`
- the result could sound obviously out of tune

This release adds two levels of repair:

1. **gross pitch correction**: clearly wrong `note_pitch` values are corrected
   from the local voiced `f0` median inside each note window
2. **sub-note contour preservation**: notes with large internal pitch movement
   are split into two sub-notes so score mode no longer sees only a single flat
   pitch value, but a staircase approximation of the original contour

Why it matters:

- long phrase endings, slides, and vibrato-heavy notes are no longer flattened
  as aggressively in score mode
- score mode now stays much closer to the original melodic movement

Algorithmically:

- first, obvious `note_pitch` errors are repaired from local `f0` statistics
- then, notes with large internal pitch spread are split into sub-notes
- each sub-note uses the `f0` median of its own segment, creating a staircase
  approximation of the original contour

#### 6. Score-Mode Timbre Stability Fixes

Real-world testing uncovered two additional stability issues in score mode:

1. **broken token repair**: some tokens had a real phoneme but `pitch=0`, so the
   model saw "there is a syllable, but no pitch" and could produce unstable output
2. **long-phrase token explosion control**: long phrases could accumulate too many
   sub-note tokens, weakening cross-attention between prompt and target and
   causing timbre drift (for example, an entire phrase sounding like a different voice)

The new handling is:

- repair zero-pitch broken tokens before synthesis
- cap the number of contour splits per phrase, while prioritizing the splits with
  the largest pitch movement

This significantly improves score-mode timbre stability while preserving its
main advantage: clearer diction.

### Recommended Workflows

#### Workflow A: real-song lyric replacement

Recommended chain:

1. `MIDI Transcribe Audio`
2. optionally provide `reference_lyrics`
3. `MIDI Lyrics Alignment` with `preserve_sp=ON`
4. `MIDI Synthesize Audio`

Mode guidance:

- Use **`melody`** when preserving the original vocal contour matters most.
- Use **`score`** when preserving diction clarity matters most.

#### Workflow B: transcription QA

Recommended combination:

- use `lyrics_text` for fast inspection
- use `reference_lyrics` when the source lyrics are already known

### Upgrade Notes

- `preserve_sp` is now the recommended default workflow
- the experimental `hybrid` control path was explored during this release cycle,
  but did not provide a practical improvement over `melody` in real tests, so it
  is no longer kept as a default user-facing option

### Summary

The July 2026 release moves ComfyUI-MIDI-Edit beyond “editing MIDI text” and
much closer to **real-world lyric replacement and vocal resynthesis from actual
performances**.

From a product perspective, the biggest gains are:

- more trustworthy transcription
- more stable lyric replacement
- synthesis that stays closer to the source performance
- better observability for users at each stage of the workflow

In short: **this release is not just a pile of new features — it is a serious
step toward a reusable, production-friendly workflow for real vocal songs.**

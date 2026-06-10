# ComfyUI-MIDI-Edit

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 自定义节点插件，搭配 [ComfyUI_RH_SoulX-Singer](https://github.com/HM-RunningHub/ComfyUI_RH_SoulX-Singer) 实现魔改歌词——替换 MIDI JSON 中的歌词文本并自动生成拼音/音素，也可从 MIDI JSON 中提取歌词。适用于 MIDI 歌曲生成工作流，支持中文和英文。

提供两个节点：

- **MIDI Edit Lyrics** — 替换歌词并自动生成音素
- **MIDI Extract Lyrics** — 提取歌词文本（去空格，`<SP>` 转换行）

---

## 功能特性

- 歌词替换 + 自动音素生成
- 歌词提取
- 支持中文（`zh_` 前缀拼音）和英文（`en_` 前缀音素）
- `<SP>` 标记自动保留，不影响音素对齐
- 按位置替换：不要求新旧歌词长度一致，多给忽略，少给保留原文
- **长音自动展开**：MIDI 中连续重复字（如 `兄 兄` 表示长音 2 拍）会自动将用户输入的字按相同次数展开

---

## 安装

### 依赖

- ComfyUI
- Python conda 环境 `comfyui`
- `g2pM>=0.1.2.5`
- `g2p_en>=2.1.0`
- NLTK 数据（插件首次运行自动下载到本地 `models/nltk/`）

### 安装步骤

```bash
# 1. 克隆到 ComfyUI custom_nodes 目录
cd ~/App/ComfyUI/custom_nodes
ln -s /path/to/ComfyUI-MIDI-Edit ComfyUI-MIDI-Edit

# 2. 安装 Python 依赖
conda activate comfyui
pip install g2pM g2p_en

# 3. 重启 ComfyUI
```

NLTK 数据会自动下载到项目内的 `models/nltk/` 目录，无需手动操作。

---

## 节点说明

### MIDI Edit Lyrics

替换 MIDI JSON 中的歌词并自动生成对应音素。

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING, multiline | MIDI JSON 字符串 |
| `new_lyrics` | STRING, multiline | 新歌词文本 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING | 修改后的 MIDI JSON 字符串 |

**处理逻辑：**

1. 清洗新歌词（去除标点、空格、换行，只保留中英文）
2. 将原始 text 中**连续相同的非 `<SP>` token** 合并为一个 slot（例如 `兄 兄` → 1 个 slot，重复数 2）
3. 用户输入的字数按 slot 数对齐（非 token 数），每个 slot 消耗 1 个用户字
4. 替换时，用户字按 slot 的重复数自动展开（例如 slot 重复 3 次，则输出 3 个相同的替换字）
5. 如果用户字比 slot 少，剩余 slot 保留原文
6. 如果用户字比 slot 多，多余忽略
7. 为每个替换的字自动生成音素

**分类：** `MIDI-Edit`

---

### MIDI Extract Lyrics

从 MIDI JSON 中提取歌词文本。

**输入：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `midi_json` | STRING, multiline | MIDI JSON 字符串 |

**输出：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `lyrics_text` | STRING | 提取的歌词文本 |

**处理逻辑：**

1. 遍历所有 track，拼接 `text` 字段
2. 去除所有空格
3. 将 `<SP>` 替换为换行符

**分类：** `MIDI-Edit`

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

### 示例 2：长音自动展开

MIDI 中连续重复字表示长音（占多个音节）。用户输入不需要手动写重复字，插件会自动展开匹配。

输入 MIDI JSON 的 text：`敌 敌 人 不 敢 靠 近 近 近 你`

（slots：敌×2, 人×1, 不×1, 敢×1, 靠×1, 近×3, 你×1 = 7 个 slot）

输入新歌词：`一二三四五六七`

输出 text：`一 一 二 三 四 五 近 近 近 你`

> "敌敌"（长音 2 拍）→ 用户写"一"自动展开为"一一"；"近近近"（长音 3 拍）→ 用户写"六"自动展开为"六六六"。

### 示例 3：提取歌词

输入 MIDI JSON 的 text：`<SP> 我 有 一 只 小 <SP> 毛 驴 我 从 来 都 不 <SP> 骑 有 一 天 <SP>`

输出：

```
我有一只小
毛驴我从来都不
骑有一天
```

---

## 工作流示例

完整 ComfyUI 工作流，展示 MIDI Edit Lyrics + MIDI Extract Lyrics 双路径并行处理：

![MIDI 歌词编辑工作流](docs/midi-edit-lyrics.json.png)

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

## 注意事项

- `g2pM` 首次使用时会自动下载模型（与 NLTK 数据分开）
- NLTK 数据存储在项目 `models/nltk/` 目录下，不影响系统环境
- 歌词替换是**按位置**而非按字数匹配，不要求新旧歌词长度一致
- `<SP>` 标记始终保留，不影响音素对齐

---

## 项目结构

```
ComfyUI-MIDI-Edit/
├── __init__.py          # ComfyUI 插件入口，导出节点映射
├── nodes.py             # 核心逻辑与节点定义
├── requirements.txt     # Python 依赖
├── docs/
│   ├── REQUIREMENT.md   # 原始需求文档
│   ├── midi-edit-lyrics.json       # ComfyUI 工作流文件
│   └── midi-edit-lyrics.json.png   # 工作流截图
├── models/
│   └── nltk/            # NLTK 数据（自动下载）
│       ├── taggers/
│       └── tokenizers/
└── README.md
```

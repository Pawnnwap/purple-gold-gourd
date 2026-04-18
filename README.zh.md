# 紫金葫芦

[![PyPI](https://img.shields.io/pypi/v/purple-gold-gourd)](https://pypi.org/project/purple-gold-gourd/)
[![Python](https://img.shields.io/pypi/pyversions/purple-gold-gourd)](https://pypi.org/project/purple-gold-gourd/)

一秒吸纳，顷刻炼化
把 B 站 UP 主或 YouTube 创作者转化为以转录为基础、支持声音复刻的对话人格。

名字取自《西游记》里的紫金红葫芦。正如银角大王那句 “我叫你一声，你敢答应吗？”，只要一声应答，便可瞬间吸纳；而本库则是在点名创作者之后，迅速吸纳其公开内容，转录、炼化为一个可对话、可检索、可复刻声音的人格分身。

## 功能概览

1. 通过名称、ID、handle 或主页 URL 解析 B 站或 YouTube 创作者。
2. 只下载所选视频的音频，不下载整段视频。
3. 使用 FunASR SenseVoice 转录，并保留时间戳与语言识别结果；也支持通过 CLI 导入本地音频或视频文件。
4. 导出 JSON 转录文件和对应 `.srt` 字幕。
5. 可在每个角色目录下的 `documents/` 文件夹中放入自定义 `.md` 文件；这些资料也会参与 skill 生成与 RAG。
6. 通过本地 OpenAI 兼容 LLM 提炼人格 `skill.md`。
7. 对转录片段和自定义资料建立 BM25 检索索引。
8. 当视频知识检索不足时，自动回退到网页搜索，并把结果作为外部 `Background info (背景信息)` 注入，而不是误当作角色自己的记忆。
9. 角色初始化时若检测到 `transcripts/` 或 `documents/` 有更新，会自动重生成 skill。
10. 可选使用 TTS 插件进行语音合成，并自动挑选较合适的参考音频。

## 安装

前置条件：

- Python 3.11 或更高版本
- `ffmpeg` 在 `PATH` 中可用，或通过 `FFMPEG_PATH` 指定路径
- 用于 skill 提炼和人格对话的 OpenAI 兼容聊天接口；默认目标为 LM Studio `http://127.0.0.1:1234/v1`

```powershell
# 推荐：完整本地体验
pip install "purple-gold-gourd[full]"

# 最小安装
pip install purple-gold-gourd

# 仅添加 FunASR 语音转录插件
pip install "purple-gold-gourd[speech]"

# 仅添加 Qwen3-TTS 语音合成与音频播放
pip install "purple-gold-gourd[tts]"

# 仅添加 B 站抓取支持
pip install "purple-gold-gourd[bilibili]"
```

主要的转录驱动构建流程需要 `speech` 扩展，因为内置的 STT 插件是 FunASR。
`tts` 扩展仅在需要语音回复或讨论音频输出时才需要安装。

安装 Bilibili 扩展后，还需要安装一次 Playwright 浏览器：

```powershell
playwright install chromium
```

本地开发时仍支持 editable 安装：

```powershell
pip install -e .
pip install -e ".[full]"

# 或使用便捷的 requirements 文件
pip install -r requirements.txt
pip install -r requirements-full.txt
```

`requirements.txt` 会安装本地包及核心依赖，`requirements-full.txt` 会安装 `.[full]`。
安装后可使用 `purple-gold-gourd`、`zijin-hulu` 命令入口。

## 包结构

```text
purple_gold_gourd/
  cli.py            命令行入口
  config.py         配置
  schema.py         数据结构
  utils.py          通用工具
  language.py       语言检测与归一化
  pipeline.py       构建流程编排

  plugins/
    stt/
      base.py       STT 插件接口
      registry.py   STT 插件注册与加载
      shared.py     字幕辅助
      funasr/
        plugin.py   FunASR STT 插件
        transcriber.py
    tts/
      base.py       TTS 插件接口
      registry.py   TTS 插件注册与加载
      shared.py     参考音频挑选、播放、文本预处理、验证
      qwen3/
        plugin.py   Qwen3-TTS 插件
        voice.py

  media/
    platforms.py    创作者解析
    downloader.py   纯音频下载
    transcribe.py   指向 STT 插件的兼容层

  synthesis/
    voice.py        指向 TTS 辅助逻辑的兼容层

  chat/
    llm.py          OpenAI 兼容补全封装
    retrieval.py    BM25 检索与弱 RAG 判断
    web_search.py   网页搜索回退
    skillgen.py     人格提炼
    persona.py      对话循环
```

## 快速开始

> **B 站说明：** B 站没有公开的创作者搜索接口，因此第一次构建 B 站角色时必须使用创作者的数字 UID（可在主页 URL 中找到，例如 `space.bilibili.com/208259`）。角色构建完成后，后续可以直接用名字从本地缓存中打开。

```powershell
# B 站——首次构建：必须使用数字 UID
purple-gold-gourd "208259" --platform bilibili

# B 站——后续运行：可直接用名字从本地缓存打开
purple-gold-gourd "敬汉卿"
purple-gold-gourd discuss "敬汉卿" "马督工" --topic "创作者是否应该依赖 AI 工具？" --rounds 3

# YouTube——handle 或 URL 可直接使用
purple-gold-gourd "@LinusTechTips" --platform youtube

# 只使用指定排名序号的视频做 RAG
purple-gold-gourd "@LinusTechTips" --series 1 3 8
purple-gold-gourd "@LinusTechTips" --series 2,5,9

# 向已有角色导入本地音频/视频
purple-gold-gourd "敬汉卿" --media D:\clips\interview.mp3 D:\clips\livestream.mp4

# 只构建，不进入对话（首次构建用 UID）
purple-gold-gourd "208259" --platform bilibili --build-only

# 启动时开启语音合成（已缓存后可用名字）
purple-gold-gourd "敬汉卿" --speak
```

也可以直接运行模块路径：

```powershell
python -m purple_gold_gourd.cli "208259" --platform bilibili
```

## 选取规则

- 如果传入 `--series`，就只使用这些从 1 开始计数的排名序号视频做 RAG。
- 如果所请求的视频本地还没有转录，会在初始化时当场下载并处理。
- 如果传入 `--media`，本地音频或视频会先统一转成音频，再并入该角色的转录资料。
- 如果不传 `--series`，则默认使用该创作者当前已经缓存的全部转录。
- 放在角色 `documents/` 目录下的 `.md` 文件也会参与 skill 生成与 RAG。
- 角色初始化时若发现 `transcripts/` 或 `documents/` 下文件有变化，会自动刷新 `skill.md`。
- 如果是第一次构建、还没有任何缓存转录，则先用 `--top` 指定的前几条视频完成首轮引导构建。

## 对话指令

| 指令 | 作用 |
|---|---|
| `/help` | 显示帮助 |
| `/speak on` | 开启语音合成 |
| `/speak off` | 关闭语音合成 |
| `/rebuild` | 重新下载、转录、提炼 |
| `/calibrate <路径> <起始-结束>` | 从任意音视频文件的指定时间段设置新的声音参考，例如 `/calibrate rec.mp4 00:10-00:20` |
| `/exit` | 退出 |

## 讨论模式控制命令

使用 `discuss` 时，不会进入普通的单角色聊天循环，而是直接按设定轮数运行讨论；但在开始前和每轮之间，仍可使用这些控制命令：

| 指令 | 作用 |
|---|---|
| `/help` | 显示讨论模式可用命令 |
| `/speak on` | 为后续轮次开启语音播放与音频保存 |
| `/speak off` | 为后续轮次关闭语音播放 |
| `/exit` | 提前停止，并保留已生成的部分记录 |

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--platform` | `auto` | `auto`、`youtube` 或 `bilibili` |
| `--top` | `10` | 首次引导时参与排名选择的视频数量 |
| `--scan-limit` | `30` | 排名前扫描的视频候选数量 |
| `--series` | 空 | 指定参与 RAG 的视频排名序号 |
| `--media` | 空 | 向该角色导入并转录本地音频/视频文件 |
| `--rebuild` | 关闭 | 忽略缓存并重建 |
| `--build-only` | 关闭 | 只构建人格，不进入对话 |
| `--speak` | 关闭 | 启动时开启语音合成 |

讨论模式参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `discuss` | 无 | 多角色讨论模式 |
| `--topic` | 必填 | 讨论主题 |
| `--rounds` | `3` | 完整讨论轮数；每轮每个角色各发言一次 |
| `--speak` | 关闭 | 讨论开始时即开启语音播放 |

## 环境变量

环境变量统一使用 `PURPLE_GOLD_GOURD_*` 前缀。

| 变量 | 用途 |
|---|---|
| `OPENAI_BASE_URL` | LLM 端点 |
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_MODEL` | 首选模型 |
| `OPENAI_MAX_CONTEXT_TOKENS` | 默认上下文预算 |
| `OPENAI_MAX_TOKENS` | 默认输出预算 |
| `OPENAI_MODEL_CONTEXT_TOKENS` | 按模型指定上下文预算 |
| `OPENAI_MODEL_MAX_TOKENS` | 按模型指定输出预算 |
| `PURPLE_GOLD_GOURD_STT_PLUGIN` | STT 插件，默认 `funasr` |
| `FUNASR_DEVICE` | `cuda:0` 或 `cpu` |
| `FUNASR_MODEL` | FunASR 模型 ID |
| `PURPLE_GOLD_GOURD_TTS_PLUGIN` | TTS 插件，默认 `qwen3` |
| `PURPLE_GOLD_GOURD_WEB_SEARCH` | 是否启用网页搜索回退 |
| `PURPLE_GOLD_GOURD_WEB_SEARCH_MAX_RESULTS` | 注入 Prompt 的网页结果上限 |
| `PURPLE_GOLD_GOURD_WEB_SEARCH_TIMEOUT_S` | 网页搜索超时秒数 |
| `PURPLE_GOLD_GOURD_VALIDATE_TTS` | 设置后会在合成后回转写验证音频 |
| `QWEN3_TTS_MODEL` | Qwen3-TTS 模型名或本地路径 |
| `QWEN3_TTS_DEVICE_MAP` | Qwen3-TTS 设备映射 |
| `QWEN3_TTS_DTYPE` | Qwen3-TTS 精度 |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | 可选 attention 后端 |
| `QWEN3_TTS_CHUNK_CHARS` | 每个 TTS 分段的大致字符数 |
| `QWEN3_TTS_DO_SAMPLE` | 是否启用 Qwen3-TTS 采样 |
| `QWEN3_TTS_MAX_NEW_TOKENS` | TTS 生成长度上限 |
| `FFMPEG_PATH` | ffmpeg 路径 |

## 数据目录

```text
data/creators/<platform>-<id>-<name>/
  manifest.json
  videos.json
  downloads/
  transcripts/
  documents/
  skill/
    skill.md
    notes/
  voice/
  outputs/
```

把你希望角色额外参考的自定义 markdown 文件直接放进 `documents/` 即可，不需要额外命令；下次初始化角色时会自动读入，并在需要时刷新 skill。

讨论记录会单独保存到 `data/discussions/<timestamp>-<topic>/` 下，其中包括 `discussion.json`、`discussion.md`、`discussion.txt`，以及在开启讨论语音时生成的 `audio/` 文件夹。

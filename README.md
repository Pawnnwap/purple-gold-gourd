# 紫金葫芦

[![PyPI](https://img.shields.io/pypi/v/purple-gold-gourd)](https://pypi.org/project/purple-gold-gourd/)
[![Python](https://img.shields.io/pypi/pyversions/purple-gold-gourd)](https://pypi.org/project/purple-gold-gourd/)

Named after the Purple-Gold Gourd from *Journey to the West*: "Dare you answer me when I call your name?" Once a creator answers the call, this library rapidly draws in their public voice, transcribes it, distills it, and refines it into a chat persona.

## What it does

1. Resolves a creator by name, ID, handle, or URL on Bilibili or YouTube.
2. Downloads audio only from selected videos.
3. Transcribes with FunASR SenseVoice, including timestamps and language detection, and can also ingest local audio or video files through the CLI.
4. Exports each transcript as JSON and `.srt`.
5. Lets you drop custom `.md` files into each character's `documents/` folder; those files also participate in skill generation and RAG.
6. Distills a persona `skill.md` with a local OpenAI-compatible LLM.
7. Builds BM25 retrieval over transcript chunks and custom documents.
8. Falls back to web search when transcript evidence is too thin, and injects that material as external `Background info (背景信息)` instead of persona memory.
9. Automatically refreshes the skill when transcripts or custom documents in the character data folder change.
10. Optionally synthesizes replies with a TTS plugin using an automatically selected voice prompt.

## Install

Prerequisites:

- Python 3.11 or newer
- `ffmpeg` available on your `PATH` or configured through `FFMPEG_PATH`
- An OpenAI-compatible chat endpoint for skill distillation and persona chat; the defaults target LM Studio at `http://127.0.0.1:1234/v1`

```powershell
# Recommended: full local experience
pip install "purple-gold-gourd[full]"

# Minimal package install
pip install purple-gold-gourd

# Add the bundled FunASR speech-transcription plugin
pip install "purple-gold-gourd[speech]"

# Add Qwen3-TTS synthesis and audio playback helpers
pip install "purple-gold-gourd[tts]"

# Add Bilibili scraping support
pip install "purple-gold-gourd[bilibili]"
```

The main transcript-backed build flow needs the `speech` extra because the bundled STT plugin is FunASR.
The `tts` extra is only needed when you want spoken replies or audio discussion output.

If you install the Bilibili extra, install a Playwright browser once:

```powershell
playwright install chromium
```

For local development, editable installs still work:

```powershell
pip install -e .
pip install -e ".[full]"

# Or use convenience requirements files
pip install -r requirements.txt
pip install -r requirements-full.txt
```

`requirements.txt` installs the local package with the core dependency set, and `requirements-full.txt` installs `.[full]`.
After installation, you can use `purple-gold-gourd` or `zijin-hulu` CLI entrypoint.

## Package layout

```text
purple_gold_gourd/
  cli.py            entry point
  config.py         AppConfig
  schema.py         data classes
  utils.py          shared helpers
  language.py       language detection and normalization
  pipeline.py       build orchestrator

  plugins/
    stt/
      base.py       STT plugin interfaces
      registry.py   internal STT plugin loader/registry
      shared.py     subtitle helpers
      funasr/
        plugin.py   FunASR STT plugin
        transcriber.py
    tts/
      base.py       TTS plugin interfaces
      registry.py   internal TTS plugin loader/registry
      shared.py     voice prompt selection, playback, text prep, validation
      qwen3/
        plugin.py   Qwen3-TTS plugin
        voice.py

  media/
    platforms.py    creator resolution
    downloader.py   audio-only media download
    transcribe.py   compatibility shim to the STT plugin

  synthesis/
    voice.py        compatibility shim to TTS helpers

  chat/
    llm.py          OpenAI-compatible completion helper
    retrieval.py    BM25 retrieval + weak-RAG assessment
    web_search.py   web search fallback
    skillgen.py     persona distillation
    persona.py      chat loop
```

## Quick start

> **Bilibili note:** Bilibili does not expose a public search API, so the first build of a Bilibili character requires the creator's numeric UID (visible in the profile URL, e.g. `space.bilibili.com/208259`). Once a character has been built once, you can reopen it by name from the local cache.

```powershell
# Bilibili — first build: must use the numeric UID
purple-gold-gourd "208259" --platform bilibili

# Bilibili — subsequent runs: name lookup works from local cache
purple-gold-gourd "敬汉卿"
purple-gold-gourd discuss "敬汉卿" "马督工" --topic "Should creators rely on AI tools?" --rounds 3

# YouTube — handle or URL works directly
purple-gold-gourd "@LinusTechTips" --platform youtube

# Use specific ranked videos only
purple-gold-gourd "@LinusTechTips" --series 1 3 8
purple-gold-gourd "@LinusTechTips" --series 2,5,9

# Import local audio/video into an existing character
purple-gold-gourd "敬汉卿" --media D:\clips\interview.mp3 D:\clips\livestream.mp4

# Build without opening chat
purple-gold-gourd "208259" --platform bilibili --build-only

# Start with speech synthesis enabled
purple-gold-gourd "敬汉卿" --speak
```

You can also run the current module path directly:

```powershell
python -m purple_gold_gourd.cli "208259" --platform bilibili
```

## Selection rules

- If you pass `--series`, only those 1-based ranked video numbers are used for RAG.
- If a requested video is missing locally, the library downloads and processes it immediately.
- If you pass `--media`, each local audio/video file is converted to audio first and then transcribed into the same character.
- If you do not pass `--series`, the library uses all cached transcripts for that creator.
- Any `.md` files you place under that character's `documents/` folder are also used for skill generation and RAG.
- During character initialization, if files under `transcripts/` or `documents/` changed, the library refreshes `skill.md` automatically.
- On a creator's first build, when no transcripts are cached yet, it bootstraps from the top `--top` videos.

## Chat commands

| Command | Effect |
|---|---|
| `/help` | Show available commands |
| `/speak on` | Enable reply synthesis |
| `/speak off` | Disable reply synthesis |
| `/rebuild` | Re-download, re-transcribe, and re-distill |
| `/calibrate <path> <start-end>` | Set a new voice reference from a time slice of any audio/video file, e.g. `/calibrate rec.mp4 00:10-00:20` |
| `/exit` | Quit |

## Discussion controls

When you use `discuss`, the CLI does not enter a normal one-character chat loop. Instead, it runs the requested rounds directly, while still letting you use these control commands before the start and between rounds:

| Command | Effect |
|---|---|
| `/help` | Show available discussion controls |
| `/speak on` | Enable discussion speech playback and audio saving for later turns |
| `/speak off` | Disable discussion speech playback for later turns |
| `/exit` | Stop early and keep the partial record |

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--platform` | `auto` | `auto`, `youtube`, or `bilibili` |
| `--top` | `10` | Videos to process during bootstrap ranking |
| `--scan-limit` | `30` | Candidate videos to inspect before ranking |
| `--series` | unset | 1-based ranked video numbers to use for RAG |
| `--media` | unset | Local audio/video files to import and transcribe for this character |
| `--rebuild` | off | Ignore cache and rebuild |
| `--build-only` | off | Stop after profile build |
| `--speak` | off | Enable voice synthesis from start |

Discussion mode flags:

| Flag | Default | Description |
|---|---|---|
| `discuss` | n/a | Multi-character discussion mode |
| `--topic` | required | Discussion topic |
| `--rounds` | `3` | Full discussion rounds; each character speaks once per round |
| `--speak` | off | Start the discussion with speech playback enabled |

## Environment overrides

Project-specific overrides use the `PURPLE_GOLD_GOURD_*` prefix.

| Variable | Purpose |
|---|---|
| `OPENAI_BASE_URL` | LLM endpoint |
| `OPENAI_API_KEY` | LLM API key |
| `OPENAI_MODEL` | Preferred model name |
| `OPENAI_MAX_CONTEXT_TOKENS` | Default prompt-context budget |
| `OPENAI_MAX_TOKENS` | Default completion budget |
| `OPENAI_MODEL_CONTEXT_TOKENS` | Per-model context limits |
| `OPENAI_MODEL_MAX_TOKENS` | Per-model completion limits |
| `PURPLE_GOLD_GOURD_STT_PLUGIN` | Active STT plugin, default `funasr` |
| `FUNASR_DEVICE` | `cuda:0` or `cpu` |
| `FUNASR_MODEL` | FunASR model ID |
| `PURPLE_GOLD_GOURD_TTS_PLUGIN` | Active TTS plugin, default `qwen3` |
| `PURPLE_GOLD_GOURD_WEB_SEARCH` | Enable guarded web search fallback |
| `PURPLE_GOLD_GOURD_WEB_SEARCH_MAX_RESULTS` | Max web results injected into prompts |
| `PURPLE_GOLD_GOURD_WEB_SEARCH_TIMEOUT_S` | Web-search timeout in seconds |
| `PURPLE_GOLD_GOURD_VALIDATE_TTS` | Validate synthesized speech by re-transcribing it when set |
| `QWEN3_TTS_MODEL` | Qwen3-TTS model id or local path |
| `QWEN3_TTS_DEVICE_MAP` | Qwen3-TTS device map |
| `QWEN3_TTS_DTYPE` | Qwen3-TTS dtype |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | Optional attention backend |
| `QWEN3_TTS_CHUNK_CHARS` | Approximate chars per TTS chunk |
| `QWEN3_TTS_DO_SAMPLE` | Enable or disable Qwen3-TTS sampling |
| `QWEN3_TTS_MAX_NEW_TOKENS` | Optional generation cap for TTS |
| `FFMPEG_PATH` | ffmpeg binary path |

## Data layout

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

Put any custom markdown files you want the persona to use into `documents/`. No extra command is needed; the next character initialization will pick them up automatically and refresh the skill when needed.

Discussion records are saved separately under `data/discussions/<timestamp>-<topic>/`, including `discussion.json`, `discussion.md`, `discussion.txt`, and an `audio/` folder when discussion speech is enabled.

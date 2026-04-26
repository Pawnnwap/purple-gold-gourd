from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from ..utils import USER_AGENT, ensure_dir, slugify, write_json

_TOPIC_MAX_CHARS = 15
_OPENING_DURATION_S = 4.5
_OPENING_FADE_S = 0.85
_AI_NAME_PREFIX = "AI · "
_AI_DISCLOSURE_WATERMARK = "AI生成，仅供娱乐，不代表相应UP主任何个人真实看法"
_OPENING_TITLE = "AI生成内容说明"
_OPENING_DISCLAIMER_LINES = (
    "本视频由AI基于公开资料自动生成。",
    "声音、头像、观点与措辞均为模拟合成。",
    "内容仅供娱乐，不代表相应UP主任何个人真实看法。",
    "请勿将其作为事实判断或本人立场引用。",
)
_OPENING_FOOTER = "即将进入AI模拟讨论"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = (hex_color or "").strip().lstrip("#")
    if raw.lower().startswith("0x"):
        raw = raw[2:]
    if len(raw) < 6:
        return (255, 255, 255)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


@dataclass(slots=True)
class DiscussionVideoResult:
    video_path: Path
    audio_path: Path
    subtitles_path: Path
    ass_path: Path
    metadata_path: Path
    avatar_paths: list[Path]


@dataclass(slots=True)
class _AudioTurn:
    round_number: int
    turn_number: int
    speaker: str
    character_slug: str
    text: str
    audio_path: Path
    start_s: float
    end_s: float
    spoken_text: str = ""

    @property
    def duration_s(self) -> float:
        return max(self.end_s - self.start_s, 0.0)


@dataclass(slots=True)
class _SubtitleSegment:
    start_s: float
    end_s: float
    speaker: str
    character_slug: str
    text: str


@dataclass(slots=True)
class _Panel:
    x: int
    y: int
    w: int
    h: int
    avatar_size: int
    avatar_x: int
    avatar_y: int
    name_x: int
    name_y: int
    meta_y: int
    name_max_w: int
    name_fontsize: int
    meta_fontsize: int
    color: str


_PALETTE = [
    "0xF6C453",
    "0x57C7FF",
    "0xFF6B8A",
    "0x8DFFB3",
    "0xB594FF",
    "0xFF9F45",
]


def render_discussion_video(
    *,
    ffmpeg_path: str,
    record_dir: Path,
    topic: str,
    participants: Sequence[object],
    turns: Sequence[object],
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> DiscussionVideoResult:
    """Render a narrated discussion into an MP4 with avatar panels and timed subtitles."""
    if not turns:
        raise ValueError("Cannot render discussion video without turns.")
    if not participants:
        raise ValueError("Cannot render discussion video without participants.")

    record_dir = Path(record_dir).resolve()
    video_dir = ensure_dir(record_dir / "video")
    display_topic = _compress_topic_if_long(topic, _TOPIC_MAX_CHARS)
    resolved_turns = _resolve_audio_turns(
        ffmpeg_path=ffmpeg_path,
        record_dir=record_dir,
        turns=turns,
    )
    segments = _build_subtitle_segments(resolved_turns)
    subtitles_path = video_dir / "discussion.srt"
    ass_path = video_dir / "discussion.ass"
    audio_path = video_dir / "discussion-audio.m4a"
    output_path = video_dir / "discussion.mp4"
    metadata_path = video_dir / "discussion-video.json"

    _write_srt(subtitles_path, segments, offset_s=_OPENING_DURATION_S)
    _write_ass(ass_path, segments, width=width, height=height, offset_s=_OPENING_DURATION_S)
    _concat_audio(ffmpeg_path, resolved_turns, audio_path)

    avatar_paths = [
        _prepare_avatar(ffmpeg_path, participant, video_dir, index)
        for index, participant in enumerate(participants)
    ]
    discussion_s = max((turn.end_s for turn in resolved_turns), default=0.1)
    total_s = discussion_s + _OPENING_DURATION_S
    _render_visuals(
        ffmpeg_path=ffmpeg_path,
        record_dir=record_dir,
        topic=display_topic,
        participants=participants,
        turns=resolved_turns,
        avatars=avatar_paths,
        audio_path=audio_path,
        ass_path=ass_path,
        output_path=output_path,
        duration_s=discussion_s,
        opening_duration_s=_OPENING_DURATION_S,
        width=width,
        height=height,
        fps=fps,
    )

    write_json(
        metadata_path,
        {
            "video_path": str(output_path),
            "audio_path": str(audio_path),
            "subtitles_path": str(subtitles_path),
            "ass_path": str(ass_path),
            "duration_s": round(total_s, 3),
            "discussion_duration_s": round(discussion_s, 3),
            "opening_duration_s": round(_OPENING_DURATION_S, 3),
            "width": width,
            "height": height,
            "fps": fps,
            "turns": [
                {
                    "round_number": turn.round_number,
                    "turn_number": turn.turn_number,
                    "speaker": turn.speaker,
                    "audio_path": str(turn.audio_path),
                    "start_s": round(turn.start_s, 3),
                    "end_s": round(turn.end_s, 3),
                }
                for turn in resolved_turns
            ],
        },
    )
    return DiscussionVideoResult(
        video_path=output_path,
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        ass_path=ass_path,
        metadata_path=metadata_path,
        avatar_paths=avatar_paths,
    )


def _resolve_audio_turns(
    *,
    ffmpeg_path: str,
    record_dir: Path,
    turns: Sequence[object],
) -> list[_AudioTurn]:
    resolved: list[_AudioTurn] = []
    cursor = 0.0
    missing: list[str] = []
    for turn in turns:
        audio_value = str(getattr(turn, "audio_path", "") or "").strip()
        speaker = str(getattr(turn, "speaker", "") or "speaker")
        if not audio_value:
            missing.append(speaker)
            continue
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = record_dir / audio_path
        if not audio_path.exists():
            missing.append(f"{speaker} ({audio_path})")
            continue
        duration_s = _audio_duration(ffmpeg_path, audio_path)
        if duration_s <= 0:
            missing.append(f"{speaker} ({audio_path})")
            continue
        start_s = cursor
        end_s = cursor + duration_s
        resolved.append(
            _AudioTurn(
                round_number=int(getattr(turn, "round_number", 0) or 0),
                turn_number=int(getattr(turn, "turn_number", len(resolved) + 1) or len(resolved) + 1),
                speaker=speaker,
                character_slug=str(getattr(turn, "character_slug", "") or slugify(speaker)),
                text=str(getattr(turn, "text", "") or ""),
                audio_path=audio_path,
                start_s=start_s,
                end_s=end_s,
                spoken_text=str(getattr(turn, "spoken_text", "") or ""),
            ),
        )
        cursor = end_s
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Discussion video needs speech audio for every turn. Missing: {names}")
    return resolved


def _audio_duration(ffmpeg_path: str, audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    ffprobe = _ffprobe_path(ffmpeg_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return 0.0
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return 0.0


def _ffprobe_path(ffmpeg_path: str) -> str:
    ffmpeg = Path(ffmpeg_path)
    probe_name = "ffprobe.exe" if ffmpeg.name.lower().endswith(".exe") else "ffprobe"
    sibling = ffmpeg.with_name(probe_name)
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


def _build_subtitle_segments(turns: list[_AudioTurn]) -> list[_SubtitleSegment]:
    segments: list[_SubtitleSegment] = []
    for turn in turns:
        subtitle_text = turn.spoken_text or turn.text
        pieces = _split_subtitle_text(subtitle_text)
        if not pieces:
            pieces = [turn.speaker]
        weights = [max(len(_subtitle_weight_text(piece)), 1) for piece in pieces]
        total_weight = sum(weights) or 1
        cursor = turn.start_s
        for index, piece in enumerate(pieces):
            if index == len(pieces) - 1:
                end_s = turn.end_s
            else:
                end_s = cursor + turn.duration_s * weights[index] / total_weight
                end_s = min(max(end_s, cursor + 0.35), turn.end_s)
            segments.append(
                _SubtitleSegment(
                    start_s=cursor,
                    end_s=end_s,
                    speaker=turn.speaker,
                    character_slug=turn.character_slug,
                    text=piece,
                ),
            )
            cursor = end_s
    return [segment for segment in segments if segment.end_s > segment.start_s]


def _split_subtitle_text(text: str, max_chars: int = 42) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    raw_sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?;\u3002\uff01\uff1f\uff1b])\s*", cleaned)
        if item.strip()
    ]
    pieces: list[str] = []
    buffer = ""
    for sentence in raw_sentences:
        if len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_chunk_long_subtitle(sentence, max_chars))
            continue
        if len(buffer) + len(sentence) + (1 if buffer else 0) <= max_chars:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def _chunk_long_subtitle(text: str, max_chars: int) -> list[str]:
    separators = r"([,\u3001\uff0c:])"
    tokens = [part for part in re.split(separators, text) if part]
    pieces: list[str] = []
    buffer = ""
    for token in tokens:
        candidate = buffer + token
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer.strip())
            buffer = token.strip()
        while len(buffer) > max_chars:
            pieces.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:].strip()
    if buffer:
        pieces.append(buffer.strip())
    return pieces


def _subtitle_weight_text(text: str) -> str:
    return re.sub(r"[\s,.;:!?'\"]+", "", text)


def _write_srt(path: Path, segments: list[_SubtitleSegment], *, offset_s: float = 0.0) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{_format_srt_time(segment.start_s + offset_s)} --> {_format_srt_time(segment.end_s + offset_s)}",
                _clean_subtitle_line_edges(_subtitle_display_text(segment)),
                "",
            ],
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ass(
    path: Path,
    segments: list[_SubtitleSegment],
    *,
    width: int,
    height: int,
    offset_s: float = 0.0,
) -> None:
    font_name = "Microsoft YaHei"
    fontsize = max(26, int(height * 0.040))
    margin_v = max(28, int(height * 0.045))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{font_name},{fontsize},&H00FFFFFF,&H000000FF,"
            f"&H4D000000,&H00000000,0,0,0,0,100,100,0.4,0,1,3,1,2,120,120,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for segment in segments:
        text = _clean_subtitle_line_edges(_wrap_ass_text(_subtitle_display_text(segment)))
        lines.append(
            "Dialogue: 0,"
            f"{_format_ass_time(segment.start_s + offset_s)},{_format_ass_time(segment.end_s + offset_s)},"
            f"Default,,0,0,0,,{_escape_ass_text(text)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subtitle_display_text(segment: _SubtitleSegment) -> str:
    speaker = segment.speaker.strip()
    text = segment.text.strip()
    is_host = segment.character_slug.strip().lower() == "host" or speaker in {"主持人", "Host"}
    if is_host:
        return f"{speaker}: {text}" if speaker and text else text or speaker
    return text or speaker


def _clean_subtitle_line_edges(text: str) -> str:
    ass_break = r"\N"
    marker = "\x00ASSLINEBREAK\x00"
    chunks = str(text or "").replace(ass_break, marker).split(marker)
    cleaned_chunks = [
        "\n".join(_strip_non_text_edges(line) for line in chunk.splitlines())
        for chunk in chunks
    ]
    return marker.join(cleaned_chunks).replace(marker, ass_break)


def _strip_non_text_edges(text: str) -> str:
    value = (text or "").strip()
    start = 0
    end = len(value)
    while start < end and not _is_subtitle_text_char(value[start]):
        start += 1
    while end > start and not _is_subtitle_text_char(value[end - 1]):
        end -= 1
    return value[start:end].strip()


def _is_subtitle_text_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[:1] in {"L", "M", "N"}


def _wrap_ass_text(text: str, limit: int = 32) -> str:
    if len(text) <= limit:
        return text
    lines: list[str] = []
    buffer = ""
    for token in re.split(r"(\s+)", text):
        if not token:
            continue
        candidate = buffer + token
        if len(candidate.strip()) <= limit:
            buffer = candidate
            continue
        if buffer.strip():
            lines.append(buffer.strip())
        buffer = token.strip()
    if buffer.strip():
        lines.append(buffer.strip())
    if len(lines) <= 1:
        return text
    return r"\N".join(lines)


def _format_srt_time(seconds: float) -> str:
    millis = max(int(round(seconds * 1000)), 0)
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    centis = max(int(round(seconds * 100)), 0)
    hours, rem = divmod(centis, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    line_break = "\x00LINEBREAK\x00"
    return (
        text.replace(r"\N", line_break)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
        .replace(line_break, r"\N")
    )


def _concat_audio(ffmpeg_path: str, turns: list[_AudioTurn], target: Path) -> None:
    concat_path = target.with_suffix(".concat.txt")
    concat_path.write_text(
        "\n".join(f"file '{_concat_file_path(turn.audio_path)}'" for turn in turns) + "\n",
        encoding="utf-8",
    )
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        str(target),
    ]
    _run(command)


def _concat_file_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def _prepare_avatar(ffmpeg_path: str, participant: object, video_dir: Path, index: int) -> Path:
    creator = getattr(getattr(participant, "manifest", None), "creator", None)
    name = str(getattr(creator, "name", "") or getattr(participant, "name", "") or "speaker")
    target = video_dir / f"avatar-{index + 1:02d}-{slugify(name)}.png"
    if target.exists() and target.stat().st_size > 0:
        return target
    source = _download_avatar_source(participant, video_dir, index)
    if source is not None:
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-vf",
            "scale=256:256:force_original_aspect_ratio=increase,crop=256:256,format=rgba",
            "-frames:v",
            "1",
            str(target),
        ]
        try:
            _run(command)
            return target
        except RuntimeError:
            pass
    _render_placeholder_avatar(ffmpeg_path, target, name, index)
    return target


def _download_avatar_source(participant: object, video_dir: Path, index: int) -> Path | None:
    for url in _participant_avatar_urls(participant):
        if url.startswith("//"):
            url = "https:" + url
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            suffix = ".img"
        target = video_dir / f"avatar-source-{index + 1:02d}{suffix}"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
                timeout=12,
            )
            response.raise_for_status()
            content = response.content
            if len(content) < 256:
                continue
            target.write_bytes(content)
            return target
        except Exception:
            continue
    return None


def _participant_avatar_urls(participant: object) -> list[str]:
    creator = getattr(getattr(participant, "manifest", None), "creator", None)
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        url = (url or "").strip()
        if not url or url in seen:
            return
        seen.add(url)
        urls.append(url)

    add(str(getattr(creator, "avatar_url", "") or ""))
    platform = str(getattr(creator, "platform", "") or "").lower()
    creator_id = str(getattr(creator, "creator_id", "") or "")
    if platform == "bilibili" and creator_id:
        for face in _fetch_bilibili_face_urls(creator_id):
            add(face)
    homepage = str(getattr(creator, "homepage_url", "") or "")
    if platform == "youtube" and homepage:
        for face in _fetch_youtube_face_urls(homepage):
            add(face)
    return urls


def _fetch_bilibili_face_urls(mid: str) -> list[str]:
    headers = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
    candidates: list[str] = []
    try:
        response = requests.get(
            "https://api.bilibili.com/x/web-interface/card",
            params={"mid": mid},
            headers=headers,
            timeout=10,
        )
        card = ((response.json() or {}).get("data") or {}).get("card") or {}
        face = str(card.get("face") or "")
        if face:
            candidates.append(face)
    except Exception:
        pass
    try:
        response = requests.get(
            "https://api.bilibili.com/x/space/acc/info",
            params={"mid": mid},
            headers=headers,
            timeout=10,
        )
        data = (response.json() or {}).get("data") or {}
        face = str(data.get("face") or "")
        if face:
            candidates.append(face)
    except Exception:
        pass
    return candidates


def _fetch_youtube_face_urls(homepage: str) -> list[str]:
    try:
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(homepage, download=False) or {}
    except Exception:
        return []
    candidates: list[str] = []
    thumbnails = info.get("thumbnails") or []
    if isinstance(thumbnails, list):
        for item in thumbnails:
            if isinstance(item, dict):
                url = str(item.get("url") or "")
                if url:
                    candidates.append(url)
    thumbnail = str(info.get("thumbnail") or "")
    if thumbnail:
        candidates.append(thumbnail)
    return list(reversed(candidates))


def _render_placeholder_avatar(ffmpeg_path: str, target: Path, name: str, index: int) -> None:
    color = _PALETTE[index % len(_PALETTE)]
    initials = _initials(name)
    drawtext = _drawtext(
        text=initials,
        x="(w-text_w)/2",
        y="(h-text_h)/2",
        fontsize=80,
        color="white",
    )
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=256x256",
        "-vf",
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.14:t=8,{drawtext}",
        "-frames:v",
        "1",
        str(target),
    ]
    try:
        _run(command)
    except RuntimeError:
        fallback = [
            ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=256x256",
            "-frames:v",
            "1",
            str(target),
        ]
        _run(fallback)


def _initials(name: str) -> str:
    text = re.sub(r"\s+", " ", name or "").strip()
    if not text:
        return "?"
    words = [word for word in re.split(r"[\s_\-]+", text) if word]
    latin = [word[0].upper() for word in words if word and word[0].isascii() and word[0].isalnum()]
    if len(latin) >= 2:
        return "".join(latin[:2])
    return "".join(ch for ch in text if not ch.isspace())[:2].upper()


def _render_visuals(
    *,
    ffmpeg_path: str,
    record_dir: Path,
    topic: str,
    participants: Sequence[object],
    turns: list[_AudioTurn],
    avatars: list[Path],
    audio_path: Path,
    ass_path: Path,
    output_path: Path,
    duration_s: float,
    opening_duration_s: float,
    width: int,
    height: int,
    fps: int,
) -> None:
    total_duration_s = duration_s + opening_duration_s
    filter_complex = _build_filter_complex(
        topic=topic,
        participants=participants,
        turns=turns,
        ass_path=ass_path,
        duration_s=total_duration_s,
        opening_duration_s=opening_duration_s,
        width=width,
        height=height,
        fps=fps,
    )
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x0A0E1F:s={width}x{height}:r={fps}",
        "-i",
        str(audio_path),
    ]
    for avatar in avatars:
        command.extend(["-loop", "1", "-i", str(avatar)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{total_duration_s:.3f}",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
    )
    _run(command, cwd=record_dir)


def _build_filter_complex(
    *,
    topic: str,
    participants: Sequence[object],
    turns: list[_AudioTurn],
    ass_path: Path,
    duration_s: float,
    opening_duration_s: float,
    width: int,
    height: int,
    fps: int,
) -> str:
    panels = _layout_panels(len(participants), width=width, height=height)
    speaker_panels = {
        _participant_name(participant): panels[index]
        for index, participant in enumerate(participants)
    }
    header_h = max(78, int(height * 0.105))
    title_fs = max(28, int(height * 0.042))
    accent_w = 4
    accent_h = max(24, int(title_fs * 0.85))
    accent_x = max(width // 24, 48)
    accent_y = (header_h - accent_h) // 2
    title_text = _truncate_for_pixels(topic or "Discussion", width - accent_x * 2 - 60, title_fs)
    base_filters = [
        "format=rgba",
        # Header band
        f"drawbox=x=0:y=0:w=iw:h={header_h}:color=0x10162B@0.94:t=fill",
        # Soft separator
        f"drawbox=x=0:y={header_h - 1}:w=iw:h=1:color=white@0.10:t=fill",
        # Accent vertical bar before title
        f"drawbox=x={accent_x}:y={accent_y}:w={accent_w}:h={accent_h}:color=0xF6C453@0.95:t=fill",
        _drawtext(
            text=title_text,
            x=accent_x + accent_w + 16,
            y=(header_h - title_fs) // 2,
            fontsize=title_fs,
            color="white",
        ),
        # Live badge: red dot + LIVE label, top right
        *_live_badge_filters(width=width, header_h=header_h),
    ]
    for index, participant in enumerate(participants):
        panel = panels[index]
        name = _participant_display_name(participant)
        platform = _participant_platform(participant)
        # Soft drop shadow
        base_filters.append(
            f"drawbox=x={panel.x + 2}:y={panel.y + 6}:w={panel.w}:h={panel.h}:color=0x000000@0.55:t=fill",
        )
        # Card body
        base_filters.append(
            f"drawbox=x={panel.x}:y={panel.y}:w={panel.w}:h={panel.h}:color=0x141B33@0.97:t=fill",
        )
        # Hairline border
        base_filters.append(
            f"drawbox=x={panel.x}:y={panel.y}:w={panel.w}:h={panel.h}:color=white@0.06:t=1",
        )
        # Left vertical accent in speaker color
        base_filters.append(
            f"drawbox=x={panel.x}:y={panel.y}:w=4:h={panel.h}:color={panel.color}@0.95:t=fill",
        )
        base_filters.append(
            _drawtext(
                text=_truncate_for_pixels(name, panel.name_max_w, panel.name_fontsize),
                x=panel.name_x,
                y=panel.name_y,
                fontsize=panel.name_fontsize,
                color="white",
            ),
        )
        # Small color dot before platform
        meta_dot_size = max(6, panel.meta_fontsize // 3)
        meta_dot_y = panel.meta_y + (panel.meta_fontsize - meta_dot_size) // 2 + 2
        base_filters.append(
            f"drawbox=x={panel.name_x}:y={meta_dot_y}:w={meta_dot_size}:h={meta_dot_size}:color={panel.color}@0.95:t=fill",
        )
        base_filters.append(
            _drawtext(
                text=_truncate_for_pixels(platform.upper(), panel.name_max_w - meta_dot_size - 8, panel.meta_fontsize),
                x=panel.name_x + meta_dot_size + 8,
                y=panel.meta_y,
                fontsize=panel.meta_fontsize,
                color="0xB7C0DA",
            ),
        )
    # Speaking indicator: brighten left card accent during turn
    panel_turns: dict[int, list[tuple[float, float]]] = {}
    for turn in turns:
        panel = speaker_panels.get(turn.speaker)
        if panel is None:
            continue
        idx = panels.index(panel)
        panel_turns.setdefault(idx, []).append((turn.start_s, turn.end_s))
        t0, t1 = f"{turn.start_s + opening_duration_s:.3f}", f"{turn.end_s + opening_duration_s:.3f}"
        base_filters.append(
            f"drawbox=x={panel.x}:y={panel.y}:w=8:h={panel.h}:color={panel.color}@1.0:t=fill:"
            f"enable='between(t,{t0},{t1})'"
        )
    chains = [f"[0:v]{','.join(base_filters)}[base]"]
    last_label = "base"
    for index, panel in enumerate(panels):
        input_index = index + 2
        radius = panel.avatar_size // 2
        chains.append(
            (
                f"[{input_index}:v]"
                f"scale={panel.avatar_size}:{panel.avatar_size}:force_original_aspect_ratio=increase,"
                f"crop={panel.avatar_size}:{panel.avatar_size},format=rgba,"
                f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='if(lte(hypot(X-{radius}\\,Y-{radius}),{radius - 2}),255,0)'"
                f"[avatar{index}]"
            ),
        )
        next_label = f"withavatar{index}"
        chains.append(
            f"[{last_label}][avatar{index}]overlay={panel.avatar_x}:{panel.avatar_y}[{next_label}]",
        )
        last_label = next_label
    badge_filters = []
    for panel in panels:
        badge_filters.extend(_avatar_ai_badge_filters(panel))
    if badge_filters:
        next_label = "withavatarbadges"
        chains.append(f"[{last_label}]{','.join(badge_filters)}[{next_label}]")
        last_label = next_label
    # Circular speaking ring: soft halo, radial main band, and inner shine.
    for index, panel in enumerate(panels):
        if index not in panel_turns:
            continue
        ring_pad = 24
        ring_size = panel.avatar_size + ring_pad * 2
        rc = ring_size / 2
        avatar_r = panel.avatar_size / 2
        halo_inner = avatar_r + 2
        halo_outer = avatar_r + 22
        halo_mid = (halo_inner + halo_outer) / 2
        halo_half = (halo_outer - halo_inner) / 2
        main_inner = avatar_r + 7
        main_outer = avatar_r + 15
        shine_inner = avatar_r + 4
        shine_outer = avatar_r + 6
        cr, cg, cb = _hex_to_rgb(panel.color)
        sr = int(cr + (255 - cr) * 0.72)
        sg = int(cg + (255 - cg) * 0.72)
        sb = int(cb + (255 - cb) * 0.72)
        period = 1.6
        dist = f"hypot(X-{rc:.1f}\\,Y-{rc:.1f})"
        pulse = f"abs(sin(2*PI*T/{period}))"
        radial = f"(({dist}-{main_inner:.1f})/{(main_outer - main_inner):.1f})"
        halo_chain = (
            f"color=c=black@0:s={ring_size}x{ring_size}:r=30:d=99999,"
            f"format=rgba,"
            f"geq="
            f"r='if(between({dist}\\,{halo_inner:.1f}\\,{halo_outer:.1f})\\,{cr}\\,0)':"
            f"g='if(between({dist}\\,{halo_inner:.1f}\\,{halo_outer:.1f})\\,{cg}\\,0)':"
            f"b='if(between({dist}\\,{halo_inner:.1f}\\,{halo_outer:.1f})\\,{cb}\\,0)':"
            f"a='if(between({dist}\\,{halo_inner:.1f}\\,{halo_outer:.1f})\\,"
            f"255*(0.10+0.24*{pulse})*(1-abs({dist}-{halo_mid:.1f})/{halo_half:.1f})\\,0)'"
            f"[ringhalo{index}]"
        )
        main_chain = (
            f"color=c=black@0:s={ring_size}x{ring_size}:r=30:d=99999,"
            f"format=rgba,"
            f"geq="
            f"r='if(between({dist}\\,{main_inner:.1f}\\,{main_outer:.1f})\\,{cr}+({sr}-{cr})*{radial}\\,0)':"
            f"g='if(between({dist}\\,{main_inner:.1f}\\,{main_outer:.1f})\\,{cg}+({sg}-{cg})*{radial}\\,0)':"
            f"b='if(between({dist}\\,{main_inner:.1f}\\,{main_outer:.1f})\\,{cb}+({sb}-{cb})*{radial}\\,0)':"
            f"a='if(between({dist}\\,{main_inner:.1f}\\,{main_outer:.1f})\\,255*(0.58+0.32*{pulse})\\,0)'"
            f"[ringmain{index}]"
        )
        shine_chain = (
            f"color=c=black@0:s={ring_size}x{ring_size}:r=30:d=99999,"
            f"format=rgba,"
            f"geq="
            f"r='if(between({dist}\\,{shine_inner:.1f}\\,{shine_outer:.1f})\\,{sr}\\,0)':"
            f"g='if(between({dist}\\,{shine_inner:.1f}\\,{shine_outer:.1f})\\,{sg}\\,0)':"
            f"b='if(between({dist}\\,{shine_inner:.1f}\\,{shine_outer:.1f})\\,{sb}\\,0)':"
            f"a='if(between({dist}\\,{shine_inner:.1f}\\,{shine_outer:.1f})\\,255*(0.28+0.36*{pulse})\\,0)'"
            f"[ringshine{index}]"
        )
        chains.extend([halo_chain, main_chain, shine_chain])
        if len(panel_turns[index]) == 1:
            t0, t1 = panel_turns[index][0]
            enable_expr = f"between(t,{t0 + opening_duration_s:.3f},{t1 + opening_duration_s:.3f})"
        else:
            parts = "+".join(
                f"between(t,{t0 + opening_duration_s:.3f},{t1 + opening_duration_s:.3f})"
                for t0, t1 in panel_turns[index]
            )
            enable_expr = f"gt({parts},0)"
        ring_x = panel.avatar_x - ring_pad
        ring_y = panel.avatar_y - ring_pad
        for layer in ("halo", "main", "shine"):
            next_label = f"withring{index}{layer}"
            chains.append(
                f"[{last_label}][ring{layer}{index}]"
                f"overlay=x={ring_x}:y={ring_y}:format=auto:eof_action=pass:shortest=0:"
                f"enable='{enable_expr}'"
                f"[{next_label}]"
            )
            last_label = next_label
    ass_filter_path = _filter_relative_path(ass_path.relative_to(ass_path.parents[1]))
    chains.append(f"[{last_label}]ass={ass_filter_path}[withsubs]")
    chains.append(f"[withsubs]{_bottom_ai_disclosure_filter(width=width, height=height)}[withwatermark]")
    chains.append(
        _opening_disclaimer_chain(
            width=width,
            height=height,
            fps=fps,
            opening_duration_s=opening_duration_s,
            fade_s=min(_OPENING_FADE_S, opening_duration_s),
        ),
    )
    chains.append("[withwatermark][opening]overlay=0:0:format=auto:eof_action=pass:shortest=0[v]")
    delay_ms = max(int(round(opening_duration_s * 1000)), 0)
    chains.append(f"[1:a]adelay={delay_ms}:all=1,apad,atrim=duration={duration_s:.3f},asetpts=N/SR/TB[a]")
    return ";".join(chains)


def _avatar_ai_badge_filters(panel: _Panel) -> list[str]:
    fontsize = max(15, panel.avatar_size // 7)
    text_x = panel.avatar_x + panel.avatar_size - int(fontsize * 1.75)
    text_y = panel.avatar_y + panel.avatar_size - fontsize - 8
    return [
        _drawtext(
            text="AI",
            x=text_x,
            y=text_y,
            fontsize=fontsize,
            color="white@0.38",
            bold=True,
            shadow=False,
        ),
    ]


def _bottom_ai_disclosure_filter(*, width: int, height: int) -> str:
    fontsize = max(14, int(height * 0.023))
    right_margin = max(36, width // 30)
    bottom_margin = max(58, height // 11)
    return _drawtext(
        text=_AI_DISCLOSURE_WATERMARK,
        x=f"w-text_w-{right_margin}",
        y=f"h-text_h-{bottom_margin}",
        fontsize=fontsize,
        color="white@0.58",
    )


def _opening_disclaimer_chain(
    *,
    width: int,
    height: int,
    fps: int,
    opening_duration_s: float,
    fade_s: float,
) -> str:
    margin_x = max(86, width // 11)
    title_fs = max(38, int(height * 0.070))
    body_fs = max(22, int(height * 0.033))
    footer_fs = max(18, int(height * 0.026))
    top_y = max(92, height // 7)
    line_gap = max(15, int(body_fs * 0.75))
    title_y = top_y
    body_y = title_y + title_fs + max(40, height // 18)
    fade_start = max(0.0, opening_duration_s - fade_s)
    filters = [
        "format=rgba",
        f"drawbox=x=0:y=0:w=iw:h=ih:color=0x070B18@0.98:t=fill",
        f"drawbox=x={margin_x}:y={title_y + 4}:w=6:h={title_fs}:color=0xF6C453@0.98:t=fill",
        _drawtext(
            text=_OPENING_TITLE,
            x=margin_x + 26,
            y=title_y,
            fontsize=title_fs,
            color="white",
        ),
    ]
    for index, line in enumerate(_OPENING_DISCLAIMER_LINES):
        filters.append(
            _drawtext(
                text=line,
                x=margin_x + 4,
                y=body_y + index * (body_fs + line_gap),
                fontsize=body_fs,
                color="0xD9E2F3",
            ),
        )
    filters.append(
        _drawtext(
            text=_OPENING_FOOTER,
            x=margin_x + 4,
            y=height - max(116, height // 6),
            fontsize=footer_fs,
            color="0x8EA0C4",
        ),
    )
    filters.append(f"fade=t=out:st={fade_start:.3f}:d={fade_s:.3f}:alpha=1")
    return (
        f"color=c=0x070B18:s={width}x{height}:r={fps}:d={opening_duration_s:.3f},"
        f"{','.join(filters)}[opening]"
    )


def _live_badge_filters(*, width: int, header_h: int) -> list[str]:
    badge_label_fs = max(14, int(header_h * 0.22))
    pill_h = max(22, badge_label_fs + 10)
    pill_w = max(70, int(badge_label_fs * 4.2))
    margin_right = max(28, width // 36)
    pill_x = width - margin_right - pill_w
    pill_y = (header_h - pill_h) // 2
    dot_size = max(8, badge_label_fs // 2)
    dot_x = pill_x + 12
    dot_y = pill_y + (pill_h - dot_size) // 2
    text_x = dot_x + dot_size + 8
    text_y = pill_y + (pill_h - badge_label_fs) // 2 - 1
    return [
        f"drawbox=x={pill_x}:y={pill_y}:w={pill_w}:h={pill_h}:color=0x1B0E18@0.85:t=fill",
        f"drawbox=x={pill_x}:y={pill_y}:w={pill_w}:h={pill_h}:color=0xFF4A6B@0.55:t=1",
        f"drawbox=x={dot_x}:y={dot_y}:w={dot_size}:h={dot_size}:color=0xFF4A6B@0.95:t=fill",
        _drawtext(
            text="LIVE",
            x=text_x,
            y=text_y,
            fontsize=badge_label_fs,
            color="0xFF8FA3",
        ),
    ]


def _layout_panels(count: int, *, width: int, height: int) -> list[_Panel]:
    if count <= 0:
        return []
    margin_x = max(56, width // 18)
    gap = max(20, width // 50)
    header_h = max(78, int(height * 0.105))
    footer_band = max(150, int(height * 0.24))
    top_pad = max(32, height // 22)
    available_h = height - header_h - footer_band - top_pad
    available_w = width - margin_x * 2
    if count <= 3:
        cols = count
        rows = 1
    elif count == 4:
        cols, rows = 2, 2
    else:
        cols = min(math.ceil(math.sqrt(count)), 4)
        rows = math.ceil(count / cols)
    panel_w = (available_w - gap * (cols - 1)) // cols
    panel_h = max(120, (available_h - gap * (rows - 1)) // max(rows, 1))
    y0 = header_h + top_pad
    panels: list[_Panel] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        x = margin_x + col * (panel_w + gap)
        y = y0 + row * (panel_h + gap)
        avatar_size = max(80, min(panel_h - 48, panel_w // 4, 132))
        pad = max(22, panel_h // 10)
        avatar_x = x + pad
        avatar_y = y + (panel_h - avatar_size) // 2
        text_left = avatar_x + avatar_size + max(20, pad)
        name_max_w = max(80, x + panel_w - pad - text_left)
        name_fontsize = max(24, min(panel_h // 5, 38))
        meta_fontsize = max(13, min(panel_h // 11, 18))
        text_block_h = name_fontsize + 10 + meta_fontsize
        name_y = y + (panel_h - text_block_h) // 2
        meta_y = name_y + name_fontsize + 10
        panels.append(
            _Panel(
                x=x,
                y=y,
                w=panel_w,
                h=panel_h,
                avatar_size=avatar_size,
                avatar_x=avatar_x,
                avatar_y=avatar_y,
                name_x=text_left,
                name_y=name_y,
                meta_y=meta_y,
                name_max_w=name_max_w,
                name_fontsize=name_fontsize,
                meta_fontsize=meta_fontsize,
                color=_PALETTE[index % len(_PALETTE)],
            ),
        )
    return panels


def _truncate_for_pixels(text: str, max_pixels: int, fontsize: int) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    avg_char_w = fontsize * 0.62
    max_chars = max(int(max_pixels / max(avg_char_w, 1)), 4)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(max_chars - 1, 1)] + "…"


def _participant_name(participant: object) -> str:
    creator = getattr(getattr(participant, "manifest", None), "creator", None)
    return str(getattr(creator, "name", "") or getattr(participant, "name", "") or "Speaker")


def _participant_display_name(participant: object) -> str:
    name = _participant_name(participant).strip()
    if name.startswith(_AI_NAME_PREFIX):
        return name
    return f"{_AI_NAME_PREFIX}{name}"


def _participant_platform(participant: object) -> str:
    """Short tag-line under the speaker's name. Never exposes platform IDs."""
    creator = getattr(getattr(participant, "manifest", None), "creator", None)
    platform = str(getattr(creator, "platform", "") or "creator").upper()
    tagline = _participant_tagline(participant)
    if tagline:
        return f"{platform} · {tagline}"
    return platform


_TAGLINE_CACHE: dict[str, str] = {}


def _participant_tagline(participant: object) -> str:
    manifest = getattr(participant, "manifest", None)
    skill_path = str(getattr(manifest, "skill_path", "") or "")
    if not skill_path:
        creator_dir = str(getattr(manifest, "creator_dir", "") or "")
        if creator_dir:
            skill_path = str(Path(creator_dir) / "skill" / "skill.md")
    if not skill_path or not Path(skill_path).exists():
        return ""
    cache_key = str(Path(skill_path).resolve())
    if cache_key in _TAGLINE_CACHE:
        return _TAGLINE_CACHE[cache_key]
    description = _read_skill_description(Path(skill_path))
    if not description:
        _TAGLINE_CACHE[cache_key] = ""
        return ""
    tagline = _llm_compress_tagline(description, max_chars=8)
    if not tagline:
        tagline = _fallback_tagline(description, max_chars=8)
    _TAGLINE_CACHE[cache_key] = tagline
    return tagline


def _read_skill_description(skill_path: Path) -> str:
    try:
        text = skill_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    match = re.search(r"^description:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return ""
    desc = match.group(1).strip()
    return re.sub(r"\s+", " ", desc)[:300]


def _fallback_tagline(text: str, max_chars: int) -> str:
    """Best-effort short tagline if no LLM. Skip generic boilerplate prefixes
    and parenthetical fragments so we never paste awkward truncations."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    cleaned = re.sub(r"[（(][^）)]*[）)]", "", cleaned)
    cleaned = re.sub(r"[\"'`「」《》【】]", "", cleaned)
    if not cleaned:
        return ""
    boilerplate = ("基于公开视频", "一个基于公开", "基于视频证据")
    sentences = [s.strip() for s in re.split(r"[。!?！？]", cleaned) if s.strip()]
    bad_chars = set("（）()[]{}<>\"'`")
    candidates: list[str] = []
    for sentence in sentences:
        if any(sentence.startswith(prefix) for prefix in boilerplate):
            continue
        for piece in re.split(r"[，、,；;]", sentence):
            piece = piece.strip().strip("·、，。,. ")
            if not piece or len(piece) > max_chars:
                continue
            if any(ch in bad_chars for ch in piece):
                continue
            if len(piece) < 3:
                continue
            candidates.append(piece)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: (-len(p), p))
    return candidates[0][:max_chars]


def _llm_compress_tagline(description: str, max_chars: int) -> str:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return ""
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = os.getenv("OPENAI_MODEL", "gemma-4-31b-it")
    try:
        from ..chat.llm import ManagedLLM

        llm = ManagedLLM(
            base_url=base_url,
            api_key=api_key,
            preferred_model=model,
            max_completion_tokens=48,
        )
        prompt = (
            f"为下面的人物描述生成一个不超过{max_chars}个汉字的极简标签，"
            "概括其核心定位（如：投资观察｜压力管理者｜历史科普）。"
            "只输出标签本身，不要加引号、标点或解释。\n\n"
            f"{description}"
        )
        result, _ = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=48,
        )
    except Exception:
        return ""
    cleaned = (result or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.sub(r"^[\"'`「」《》【】\[\(]+|[\"'`「」《》【】\]\)。.,，!?！？:：;；]+$", "", cleaned)
    return cleaned[:max_chars]


def _drawtext(
    *,
    text: str,
    x: int | str,
    y: int | str,
    fontsize: int,
    color: str,
    bold: bool = False,
    shadow: bool = True,
) -> str:
    font = _drawtext_font(bold=bold)
    escaped = _escape_drawtext_text(text)
    result = (
        f"drawtext={font}:text='{escaped}':"
        f"x={x}:y={y}:fontsize={fontsize}:fontcolor={color}"
    )
    if shadow:
        result += ":shadowcolor=black@0.45:shadowx=2:shadowy=2"
    return result


def _drawtext_font(*, bold: bool = False) -> str:
    bold_candidates = (
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
    )
    regular_candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for candidate in (bold_candidates + regular_candidates if bold else regular_candidates):
        if candidate.exists():
            return f"fontfile={_filter_absolute_path(candidate)}"
    return "font=Arial"


def _escape_drawtext_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
        .replace("\n", " ")
    )


def _filter_absolute_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace(":", r"\:")
    value = value.replace("'", r"\'")
    return f"'{value}'"


def _filter_relative_path(path: Path) -> str:
    value = path.as_posix().replace(":", r"\:").replace("'", r"\'")
    return f"'{value}'"


def _topic_char_count(text: str) -> int:
    return len((text or "").strip())


def _compress_topic_if_long(topic: str, max_chars: int) -> str:
    cleaned = (topic or "").strip()
    if _topic_char_count(cleaned) <= max_chars:
        return cleaned
    compressed = _llm_compress_topic(cleaned, max_chars)
    if compressed and _topic_char_count(compressed) <= max_chars:
        return compressed
    if compressed:
        return compressed[:max_chars]
    return cleaned[: max(max_chars - 1, 1)] + "…"


def _llm_compress_topic(topic: str, max_chars: int) -> str:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return ""
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = os.getenv("OPENAI_MODEL", "gemma-4-31b-it")
    try:
        from ..chat.llm import ManagedLLM

        llm = ManagedLLM(
            base_url=base_url,
            api_key=api_key,
            preferred_model=model,
            max_completion_tokens=64,
        )
        prompt = (
            f"将下面的讨论话题压缩为不超过{max_chars}个汉字/字符的精炼标题，"
            "保留核心含义；不要加引号、书名号、标点结尾或解释，只输出标题本身。\n\n"
            f"话题：{topic}"
        )
        result, _ = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=64,
        )
    except Exception:
        return ""
    if not result:
        return ""
    cleaned = result.strip()
    cleaned = re.sub(r"^[\"'`「」《》【】\[\(]+|[\"'`「」《》【】\]\)。,，.!?！？:：;；]+$", "", cleaned)
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    return cleaned


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 1800:
            stderr = stderr[-1800:]
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}: {stderr}")

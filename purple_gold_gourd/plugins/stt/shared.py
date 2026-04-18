from __future__ import annotations

from ...schema import TranscriptChunk


def build_srt(chunks: list[TranscriptChunk]) -> str:
    if not chunks:
        return ""
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(str(index))
        lines.append(f"{_srt_time(chunk.start_ms)} --> {_srt_time(chunk.end_ms)}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _srt_time(ms: int) -> str:
    total = max(int(ms), 0)
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

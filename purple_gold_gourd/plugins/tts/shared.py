from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..stt.base import SpeechTranscriber


def clip_audio(ffmpeg_path: str, source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    duration_sec = max((end_ms - start_ms) / 1000.0, 0.5)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start_ms / 1000.0:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class AudioPlayer:
    def __init__(self) -> None:
        self._winsound = None
        self._sd = None

    def play(self, audio_path: Path, wait: bool = False) -> None:
        try:
            self._play_with_winsound(audio_path, wait=wait)
            return
        except Exception:
            self._play_with_sounddevice(audio_path, wait=wait)

    def stop(self) -> None:
        if self._winsound is not None:
            self._winsound.PlaySound(None, 0)
        if self._sd is not None:
            self._sd.stop()

    def _play_with_winsound(self, audio_path: Path, wait: bool = False) -> None:
        if self._winsound is None:
            import winsound

            self._winsound = winsound
        self._winsound.PlaySound(None, 0)
        flags = self._winsound.SND_FILENAME | self._winsound.SND_NODEFAULT
        if wait:
            flags |= getattr(self._winsound, "SND_SYNC", 0)
        else:
            flags |= self._winsound.SND_ASYNC
        self._winsound.PlaySound(
            str(audio_path),
            flags,
        )

    def _play_with_sounddevice(self, audio_path: Path, wait: bool = False) -> None:
        if self._sd is None:
            import sounddevice as sd

            self._sd = sd
        import soundfile as sf

        self.stop()
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        self._sd.play(audio, sample_rate)
        if wait:
            self._sd.wait()


def split_for_synthesis(text: str, max_chars: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", text) if s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long_clause(sentence, max_chars))
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer = buffer + sentence if buffer else sentence
        else:
            chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def validate_synthesis(
    audio_path: Path,
    expected_text: str,
    transcriber: SpeechTranscriber,
) -> tuple[str, float]:
    from difflib import SequenceMatcher

    transcribed = transcriber.transcribe_text(audio_path)
    similarity = SequenceMatcher(None, _normalize_for_compare(transcribed), _normalize_for_compare(expected_text)).ratio()
    return transcribed, similarity


def prepare_tts_text(text: str, char_limit: int = 360) -> str:
    spoken = text.strip()
    spoken = re.sub(r"(?m)^\s*#{1,6}\s*", "", spoken)
    spoken = re.sub(r"(?m)^\s*>\s*", "", spoken)
    spoken = spoken.replace("**", "")
    spoken = spoken.replace("__", "")
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", spoken)
    spoken = re.sub(r"https?://\S+", "", spoken)
    spoken = re.sub(r"(?m)^\s*[-*]\s+", "", spoken)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "", spoken)
    spoken = re.sub(r"\s+", " ", spoken)
    spoken = spoken.replace("|", ", ")
    if len(spoken) <= char_limit:
        return spoken
    shortened = spoken[:char_limit]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ,.;:") + "..."


def _split_long_clause(sentence: str, max_chars: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[,、，；;:])\s*", sentence) if p.strip()]
    result: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > max_chars:
            if buffer:
                result.append(buffer)
                buffer = ""
            for index in range(0, len(part), max_chars):
                result.append(part[index: index + max_chars])
            continue
        if len(buffer) + len(part) <= max_chars:
            buffer = buffer + part if buffer else part
        else:
            result.append(buffer)
            buffer = part
    if buffer:
        result.append(buffer)
    return result or [sentence[:max_chars]]


def _normalize_for_compare(text: str) -> str:
    stripped = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return stripped.lower()

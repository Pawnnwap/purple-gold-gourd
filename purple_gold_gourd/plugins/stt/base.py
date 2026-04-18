from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Protocol

from ...config import AppConfig
from ...schema import TranscriptChunk, TranscriptFile, VideoInfo


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_path: Path, video: VideoInfo) -> TranscriptFile:
        ...

    def transcribe_text(self, audio_path: Path, batch_size_s: int = 60) -> str:
        ...

    def build_subtitles(self, chunks: list[TranscriptChunk]) -> str:
        ...


class STTPlugin(ABC):
    name: str

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._transcriber: SpeechTranscriber | None = None

    def create_transcriber(self) -> SpeechTranscriber:
        raise NotImplementedError("Subclass must implement create_transcriber")


SpeechPlugin = STTPlugin

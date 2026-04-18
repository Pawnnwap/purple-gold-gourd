from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Protocol

from ...config import AppConfig
from ...schema import TranscriptFile, VoiceSample
from ..stt.base import SpeechTranscriber


class VoiceSynthesizer(Protocol):
    def synthesize(self, text: str, prompt_text: str, prompt_audio: Path, target: Path) -> Path:
        ...


class AudioOutput(Protocol):
    def play(self, audio_path: Path, wait: bool = False) -> None:
        ...

    def stop(self) -> None:
        ...


class TTSPlugin(ABC):
    name: str

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._synthesizer: VoiceSynthesizer | None = None
        self._audio_output: AudioOutput | None = None

    def create_synthesizer(self) -> VoiceSynthesizer:
        raise NotImplementedError("Subclass must implement create_synthesizer")

    def create_audio_output(self) -> AudioOutput:
        raise NotImplementedError("Subclass must implement create_audio_output")

    def choose_voice_sample(
        self,
        transcriber: SpeechTranscriber,
        transcripts: list[TranscriptFile],
        output_dir: Path,
    ) -> VoiceSample | None:
        raise NotImplementedError("Subclass must implement choose_voice_sample")

    def prepare_spoken_text(self, text: str, char_limit: int = 360) -> str:
        raise NotImplementedError("Subclass must implement prepare_spoken_text")

    def validate_synthesis(
        self,
        audio_path: Path,
        expected_text: str,
        transcriber: SpeechTranscriber,
    ) -> tuple[str, float]:
        raise NotImplementedError("Subclass must implement validate_synthesis")


class BaseTTSPlugin(TTSPlugin):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)

    def create_audio_output(self) -> AudioOutput:
        from .shared import AudioPlayer

        if self._audio_output is None:
            self._audio_output = AudioPlayer()
        return self._audio_output

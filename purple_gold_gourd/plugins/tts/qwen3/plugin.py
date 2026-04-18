from __future__ import annotations

from pathlib import Path

from ....schema import TranscriptFile, VoiceSample
from ...stt.base import SpeechTranscriber
from ..base import BaseTTSPlugin, VoiceSynthesizer
from ..shared import choose_voice_sample, prepare_tts_text, validate_synthesis


class Qwen3TTSPlugin(BaseTTSPlugin):
    name = "qwen3"

    def create_synthesizer(self) -> VoiceSynthesizer:
        if self._synthesizer is None:
            from .voice import Qwen3TTSSynthesizer

            self._synthesizer = Qwen3TTSSynthesizer(self.config)
        return self._synthesizer

    def choose_voice_sample(
        self,
        transcriber: SpeechTranscriber,
        transcripts: list[TranscriptFile],
        output_dir: Path,
    ) -> VoiceSample | None:
        return choose_voice_sample(
            self.config.ffmpeg_path,
            transcriber,
            transcripts,
            output_dir,
            speaker_cache_dir=self.config.model_cache_dir / "speaker-id",
        )

    def prepare_spoken_text(self, text: str, char_limit: int = 360) -> str:
        return prepare_tts_text(text, char_limit=char_limit)

    def validate_synthesis(
        self,
        audio_path: Path,
        expected_text: str,
        transcriber: SpeechTranscriber,
    ) -> tuple[str, float]:
        return validate_synthesis(audio_path, expected_text, transcriber)

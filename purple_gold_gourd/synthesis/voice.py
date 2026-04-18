from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from ..plugins import get_stt_plugin
from ..plugins.stt import SpeechTranscriber
from ..plugins.tts.qwen3.voice import Qwen3TTSSynthesizer
from ..plugins.tts.shared import (
    AudioPlayer,
    clip_audio,
    prepare_tts_text,
    split_for_synthesis,
)
from ..plugins.tts.shared import choose_voice_sample as _choose_voice_sample
from ..plugins.tts.shared import validate_synthesis as _validate_synthesis
from ..schema import TranscriptFile, VoiceSample


def choose_voice_sample(config: AppConfig, transcripts: list[TranscriptFile], output_dir: Path) -> VoiceSample | None:
    transcriber: SpeechTranscriber = get_stt_plugin(config).create_transcriber()
    return _choose_voice_sample(
        config.ffmpeg_path,
        transcriber,
        transcripts,
        output_dir,
        speaker_cache_dir=config.model_cache_dir / "speaker-id",
    )


def validate_synthesis(audio_path: Path, expected_text: str, config: AppConfig) -> tuple[str, float]:
    transcriber: SpeechTranscriber = get_stt_plugin(config).create_transcriber()
    return _validate_synthesis(audio_path, expected_text, transcriber)


__all__ = [
    "AudioPlayer",
    "Qwen3TTSSynthesizer",
    "choose_voice_sample",
    "clip_audio",
    "prepare_tts_text",
    "split_for_synthesis",
    "validate_synthesis",
]

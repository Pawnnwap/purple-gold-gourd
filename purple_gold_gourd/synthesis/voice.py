from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from ..plugins import get_stt_plugin
from ..plugins.tts.qwen3.voice import Qwen3TTSSynthesizer
from ..plugins.tts.shared import (
    AudioPlayer,
    clip_audio,
    prepare_tts_text,
    split_for_synthesis,
    validate_synthesis as _validate_synthesis,
)
from ..schema import VoiceSample


def validate_synthesis(audio_path: Path, expected_text: str, config: AppConfig) -> tuple[str, float]:
    transcriber = get_stt_plugin(config).create_transcriber()
    return _validate_synthesis(audio_path, expected_text, transcriber)


__all__ = [
    "AudioPlayer",
    "Qwen3TTSSynthesizer",
    "VoiceSample",
    "clip_audio",
    "prepare_tts_text",
    "split_for_synthesis",
    "validate_synthesis",
]

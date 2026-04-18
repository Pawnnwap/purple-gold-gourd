from __future__ import annotations

from ..base import STTPlugin, SpeechTranscriber
from .transcriber import FunASRSpeechTranscriber


class FunASRSTTPlugin(STTPlugin):
    name = "funasr"

    def create_transcriber(self) -> SpeechTranscriber:
        if self._transcriber is None:
            self._transcriber = FunASRSpeechTranscriber(
                self.config.stt_setting("funasr_model"),
                self.config.stt_setting("funasr_device"),
                self.config.stt_path("funasr_cache_dir"),
            )
        return self._transcriber

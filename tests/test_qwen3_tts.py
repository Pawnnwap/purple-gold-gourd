from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class Qwen3TTSSynthesizerTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "soundfile is not installed")
    @unittest.skipUnless(importlib.util.find_spec("qwen_tts"), "qwen_tts is not installed")
    @unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa is not installed")
    def test_voice_clone_prompt_defaults_to_x_vector_only(self) -> None:
        import numpy as np
        import soundfile as sf

        from purple_gold_gourd.plugins.tts.qwen3.voice import Qwen3TTSSynthesizer

        class FakeConfig:
            def tts_setting(self, key: str, default: str | None = None) -> str:
                settings = {
                    "qwen3_tts_chunk_chars": "160",
                    "qwen3_tts_do_sample": "false",
                    "qwen3_tts_max_new_tokens": "420",
                    "qwen3_tts_adaptive_max_new_tokens": "true",
                }
                if key in settings:
                    return settings[key]
                if default is not None:
                    return default
                raise KeyError(key)

        class FakeModel:
            def __init__(self) -> None:
                self.model = self
                self.speaker_encoder_sample_rate = 24000
                self.embedding_calls = 0
                self.create_prompt_calls = 0
                self.generate_kwargs = []

            def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
                self.create_prompt_calls += 1
                return []

            def extract_speaker_embedding(self, *, audio, sr):
                self.embedding_calls += 1
                return np.array([0.1, 0.2], dtype=np.float32)

            def generate_voice_clone(self, **kwargs):
                self.generate_kwargs.append(kwargs)
                return [np.array([0.1, 0.2, 0.3], dtype=np.float32)], 24000

        class TestSynthesizer(Qwen3TTSSynthesizer):
            def __init__(self, config, model) -> None:
                super().__init__(config)
                self._fake_model = model

            def _ensure_loaded(self):
                return self._fake_model

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_audio = Path(tmpdir) / "prompt.wav"
            target = Path(tmpdir) / "target.wav"
            sf.write(str(prompt_audio), np.zeros(2400, dtype=np.float32), 24000)

            model = FakeModel()
            synth = TestSynthesizer(FakeConfig(), model)
            synth.synthesize("hello", "reference words", prompt_audio, target)

        self.assertEqual(model.embedding_calls, 1)
        self.assertEqual(model.create_prompt_calls, 0)
        self.assertEqual(model.generate_kwargs[0]["max_new_tokens"], 96)


if __name__ == "__main__":
    unittest.main()

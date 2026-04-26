from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from purple_gold_gourd.chat.llm import candidate_models
from purple_gold_gourd.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_project_specific_env_vars_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("purple_gold_gourd.config._resolve_lm_model", return_value="gemma-4-31b-it"):
                with patch.dict(
                    "os.environ",
                    {
                        "PURPLE_GOLD_GOURD_MODEL_CACHE_DIR": f"{tmpdir}/model-cache-root",
                        "PURPLE_GOLD_GOURD_STT_PLUGIN": "funasr",
                        "PURPLE_GOLD_GOURD_TTS_PLUGIN": "qwen3",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH": "false",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH_MAX_RESULTS": "7",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH_TIMEOUT_S": "2.5",
                        "QWEN3_TTS_X_VECTOR_ONLY": "false",
                    },
                    clear=False,
                ):
                    config = AppConfig.load(tmpdir)

        self.assertEqual(config.stt_plugin, "funasr")
        self.assertEqual(config.tts_plugin, "qwen3")
        self.assertFalse(config.web_search_enabled)
        self.assertEqual(config.web_search_max_results, 7)
        self.assertEqual(config.web_search_timeout_s, 2.5)
        self.assertEqual(config.model_cache_dir.name, "model-cache-root")
        self.assertEqual(config.stt_path("funasr_cache_dir").name, "modelscope")
        self.assertEqual(config.tts_path("qwen3_tts_cache_dir").name, "huggingface")
        self.assertEqual(config.tts_setting("qwen3_tts_x_vector_only"), "false")

    def test_default_lm_uses_google_gemma_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                config = AppConfig.load(tmpdir)

        self.assertEqual(config.lm_base_url, "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(config.lm_api_key, "")
        self.assertEqual(config.lm_model, "gemma-4-31b-it")
        self.assertEqual(config.tts_setting("qwen3_tts_do_sample"), "false")
        self.assertEqual(config.tts_setting("qwen3_tts_max_new_tokens"), "420")
        self.assertEqual(config.tts_setting("qwen3_tts_adaptive_max_new_tokens"), "true")

    def test_candidate_models_are_explicit_gemma_only(self) -> None:
        class ClientThatMustNotBeQueried:
            @property
            def models(self):
                raise AssertionError("model discovery should not run")

        with patch.dict("os.environ", {}, clear=True):
            models = candidate_models(ClientThatMustNotBeQueried(), "gemma-4-31b-it")

        self.assertEqual(models, ["gemma-4-31b-it", "gemma-4-26b-a4b-it"])


if __name__ == "__main__":
    unittest.main()

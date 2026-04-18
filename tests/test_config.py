from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from purple_gold_gourd.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_project_specific_env_vars_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("purple_gold_gourd.config._resolve_lm_model", return_value="qwen3-4b"):
                with patch.dict(
                    "os.environ",
                    {
                        "PURPLE_GOLD_GOURD_MODEL_CACHE_DIR": f"{tmpdir}/model-cache-root",
                        "PURPLE_GOLD_GOURD_STT_PLUGIN": "funasr",
                        "PURPLE_GOLD_GOURD_TTS_PLUGIN": "qwen3",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH": "false",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH_MAX_RESULTS": "7",
                        "PURPLE_GOLD_GOURD_WEB_SEARCH_TIMEOUT_S": "2.5",
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


if __name__ == "__main__":
    unittest.main()

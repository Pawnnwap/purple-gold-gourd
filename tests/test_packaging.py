from __future__ import annotations

import importlib
import sys
import unittest


class PackagingTests(unittest.TestCase):
    def test_qwen3_plugin_import_is_lazy_about_voice_runtime(self) -> None:
        sys.modules.pop("purple_gold_gourd.plugins.tts.qwen3.voice", None)
        sys.modules.pop("purple_gold_gourd.plugins.tts.qwen3.plugin", None)

        module = importlib.import_module("purple_gold_gourd.plugins.tts.qwen3.plugin")
        plugin = module.Qwen3TTSPlugin(config=object())

        self.assertNotIn("purple_gold_gourd.plugins.tts.qwen3.voice", sys.modules)
        self.assertIsNone(plugin._synthesizer)


if __name__ == "__main__":
    unittest.main()

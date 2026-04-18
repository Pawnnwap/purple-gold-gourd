from ..registry import register_tts_plugin
from .plugin import Qwen3TTSPlugin

register_tts_plugin(Qwen3TTSPlugin.name, Qwen3TTSPlugin)

__all__ = ["Qwen3TTSPlugin"]

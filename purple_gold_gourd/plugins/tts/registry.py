from __future__ import annotations

from collections.abc import Callable

from ...config import AppConfig
from .base import TTSPlugin

TTSPluginFactory = Callable[[AppConfig], TTSPlugin]

_TTS_PLUGINS: dict[str, TTSPluginFactory] = {}
_BUILTINS_LOADED = False


def register_tts_plugin(name: str, factory: TTSPluginFactory) -> None:
    plugin_name = name.strip().lower()
    if not plugin_name:
        raise ValueError("TTS plugin name cannot be empty.")
    _TTS_PLUGINS[plugin_name] = factory


def get_tts_plugin(config: AppConfig, name: str | None = None) -> TTSPlugin:
    _load_builtin_plugins()
    plugin_name = (name or config.tts_plugin).strip().lower()
    factory = _TTS_PLUGINS.get(plugin_name)
    if factory is None:
        available = ", ".join(sorted(_TTS_PLUGINS)) or "none"
        raise ValueError(f"Unknown TTS plugin '{plugin_name}'. Available plugins: {available}.")
    return factory(config)


def _load_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import qwen3  # noqa: F401

    _BUILTINS_LOADED = True

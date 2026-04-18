from __future__ import annotations

from collections.abc import Callable

from ...config import AppConfig
from .base import STTPlugin

STTPluginFactory = Callable[[AppConfig], STTPlugin]

_STT_PLUGINS: dict[str, STTPluginFactory] = {}
_BUILTINS_LOADED = False


def register_stt_plugin(name: str, factory: STTPluginFactory) -> None:
    plugin_name = name.strip().lower()
    if not plugin_name:
        raise ValueError("STT plugin name cannot be empty.")
    _STT_PLUGINS[plugin_name] = factory


def get_stt_plugin(config: AppConfig, name: str | None = None) -> STTPlugin:
    _load_builtin_plugins()
    plugin_name = (name or config.stt_plugin).strip().lower()
    factory = _STT_PLUGINS.get(plugin_name)
    if factory is None:
        available = ", ".join(sorted(_STT_PLUGINS)) or "none"
        raise ValueError(f"Unknown STT plugin '{plugin_name}'. Available plugins: {available}.")
    return factory(config)


def _load_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import funasr  # noqa: F401

    _BUILTINS_LOADED = True


SpeechPluginFactory = STTPluginFactory
register_speech_plugin = register_stt_plugin
get_speech_plugin = get_stt_plugin

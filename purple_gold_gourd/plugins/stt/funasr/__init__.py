from ..registry import register_stt_plugin
from .plugin import FunASRSTTPlugin

register_stt_plugin(FunASRSTTPlugin.name, FunASRSTTPlugin)

__all__ = ["FunASRSTTPlugin"]

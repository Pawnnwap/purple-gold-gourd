from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .utils import ensure_dir


@dataclass(slots=True)
class AppConfig:
    workspace_dir: Path
    data_dir: Path
    creators_dir: Path
    model_cache_dir: Path
    lm_base_url: str
    lm_api_key: str
    lm_model: str
    stt_plugin: str
    tts_plugin: str
    ffmpeg_path: str
    lm_settings: dict[str, str] = field(default_factory=dict)
    stt_settings: dict[str, str] = field(default_factory=dict)
    tts_settings: dict[str, str] = field(default_factory=dict)
    web_search_enabled: bool = True
    web_search_max_results: int = 4
    web_search_timeout_s: float = 8.0

    @classmethod
    def load(cls, workspace_dir: str | Path | None = None) -> AppConfig:
        root = Path(workspace_dir or Path.cwd()).resolve()
        data_dir = ensure_dir(root / "data")
        creators_dir = ensure_dir(data_dir / "creators")
        model_cache_dir = ensure_dir(
            Path(
                os.getenv(
                    "PURPLE_GOLD_GOURD_MODEL_CACHE_DIR",
                    str(data_dir / "model-cache"),
                ),
            ).expanduser().resolve(),
        )
        funasr_cache_dir = ensure_dir(model_cache_dir / "modelscope")
        qwen3_tts_cache_dir = ensure_dir(model_cache_dir / "huggingface")
        funasr_device = os.getenv("FUNASR_DEVICE")
        if not funasr_device:
            try:
                import torch

                funasr_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                funasr_device = "cpu"
        qwen3_tts_device_map = funasr_device
        qwen3_tts_dtype = "float32"
        if funasr_device.startswith("cuda"):
            try:
                import torch

                qwen3_tts_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            except Exception:
                qwen3_tts_dtype = "float16"
        stt_plugin = (
            os.getenv("PURPLE_GOLD_GOURD_STT_PLUGIN")
            or "funasr"
        ).strip().lower() or "funasr"
        tts_plugin = (
            os.getenv("PURPLE_GOLD_GOURD_TTS_PLUGIN")
            or "qwen3"
        ).strip().lower() or "qwen3"
        lm_base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        lm_api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        desired_model = os.getenv("OPENAI_MODEL", "qwen3-4b")
        lm_model = _resolve_lm_model(lm_base_url, lm_api_key, desired_model)
        return cls(
            workspace_dir=root,
            data_dir=data_dir,
            creators_dir=creators_dir,
            model_cache_dir=model_cache_dir,
            lm_base_url=lm_base_url,
            lm_api_key=lm_api_key,
            lm_model=lm_model,
            stt_plugin=stt_plugin,
            lm_settings={
                "lm_max_context_tokens": str(
                    _resolve_model_limit(
                        model_id=lm_model,
                        limit_env="OPENAI_MODEL_CONTEXT_TOKENS",
                        default_env="OPENAI_MAX_CONTEXT_TOKENS",
                        fallback=16_384,
                    ),
                ),
                "lm_max_completion_tokens": str(
                    _resolve_model_limit(
                        model_id=lm_model,
                        limit_env="OPENAI_MODEL_MAX_TOKENS",
                        default_env="OPENAI_MAX_TOKENS",
                        fallback=1_024,
                    ),
                ),
            },
            stt_settings={
                "funasr_model": os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall"),
                "funasr_device": funasr_device,
                "funasr_cache_dir": str(funasr_cache_dir),
            },
            tts_plugin=tts_plugin,
            tts_settings={
                "qwen3_tts_model": os.getenv("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                "qwen3_tts_cache_dir": str(qwen3_tts_cache_dir),
                "qwen3_tts_device_map": os.getenv("QWEN3_TTS_DEVICE_MAP", qwen3_tts_device_map),
                "qwen3_tts_dtype": os.getenv("QWEN3_TTS_DTYPE", qwen3_tts_dtype),
                "qwen3_tts_attn_implementation": os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION", ""),
                "qwen3_tts_chunk_chars": os.getenv("QWEN3_TTS_CHUNK_CHARS", "160"),
                "qwen3_tts_do_sample": os.getenv("QWEN3_TTS_DO_SAMPLE", "true"),
                "qwen3_tts_max_new_tokens": os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", ""),
            },
            ffmpeg_path=os.getenv("FFMPEG_PATH", shutil.which("ffmpeg") or "ffmpeg"),
            web_search_enabled=_parse_bool(os.getenv("PURPLE_GOLD_GOURD_WEB_SEARCH", "true"), True),
            web_search_max_results=_parse_positive_int(
                os.getenv("PURPLE_GOLD_GOURD_WEB_SEARCH_MAX_RESULTS", "4"),
                4,
            ),
            web_search_timeout_s=_parse_positive_float(
                os.getenv("PURPLE_GOLD_GOURD_WEB_SEARCH_TIMEOUT_S", "8"),
                8.0,
            ),
        )

    def stt_setting(self, key: str, default: str | None = None) -> str:
        value = self.stt_settings.get(key)
        if value is not None:
            return value
        if default is not None:
            return default
        raise KeyError(f"Missing STT setting '{key}'.")

    def tts_setting(self, key: str, default: str | None = None) -> str:
        value = self.tts_settings.get(key)
        if value is not None:
            return value
        if default is not None:
            return default
        raise KeyError(f"Missing TTS setting '{key}'.")

    def lm_setting(self, key: str, default: str | None = None) -> str:
        value = self.lm_settings.get(key)
        if value is not None:
            return value
        if default is not None:
            return default
        raise KeyError(f"Missing LM setting '{key}'.")

    def stt_path(self, key: str) -> Path:
        return Path(self.stt_setting(key)).resolve()

    def tts_path(self, key: str) -> Path:
        return Path(self.tts_setting(key)).resolve()

    @property
    def funasr_model(self) -> str:
        return self.stt_setting("funasr_model")

    @property
    def funasr_device(self) -> str:
        return self.stt_setting("funasr_device")

    @property
    def lm_max_context_tokens(self) -> int:
        return _parse_positive_int(self.lm_setting("lm_max_context_tokens", "16384"), 16_384)

    @property
    def lm_max_completion_tokens(self) -> int:
        return _parse_positive_int(self.lm_setting("lm_max_completion_tokens", "1024"), 1_024)


def _resolve_lm_model(base_url: str, api_key: str, desired_model: str) -> str:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        model_ids = [str(item.get("id") or "") for item in data if item.get("id")]
        if not model_ids:
            return desired_model
        if desired_model in model_ids:
            return desired_model
        desired_lower = desired_model.lower()
        desired_compact = desired_lower.replace(".", "-")
        for model_id in model_ids:
            lowered = model_id.lower()
            if desired_lower in lowered or desired_compact in lowered:
                return model_id
        for model_id in model_ids:
            if "embed" not in model_id.lower():
                return model_id
    except Exception:
        return desired_model
    return desired_model


def _resolve_model_limit(
    model_id: str,
    limit_env: str,
    default_env: str,
    fallback: int,
) -> int:
    default_limit = _parse_positive_int(os.getenv(default_env, str(fallback)), fallback)
    limits = _parse_model_limit_map(os.getenv(limit_env, ""))
    return _match_model_limit(model_id, limits, default_limit)


def _parse_model_limit_map(raw: str) -> dict[str, int]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        return {
            str(key).strip(): _parse_positive_int(str(value), 0)
            for key, value in payload.items()
            if _parse_positive_int(str(value), 0) > 0
        }
    result: dict[str, int] = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed = _parse_positive_int(value, 0)
        if parsed > 0:
            result[key.strip()] = parsed
    return result


def _match_model_limit(model_id: str, limits: dict[str, int], default: int) -> int:
    if not limits:
        return default
    aliases = _model_aliases(model_id)
    for key, value in limits.items():
        if aliases & _model_aliases(key):
            return value
    for key, value in limits.items():
        lowered = key.strip().lower()
        if any(alias in lowered or lowered in alias for alias in aliases):
            return value
    return default


def _model_aliases(model_id: str) -> set[str]:
    lowered = (model_id or "").strip().lower()
    if not lowered:
        return set()
    aliases = {
        lowered,
        lowered.replace(".", "-"),
        lowered.replace("_", "-"),
        lowered.replace(".", ""),
        lowered.replace("-", ""),
        lowered.replace("_", ""),
    }
    return {alias for alias in aliases if alias}


def _parse_positive_int(raw: str, default: int) -> int:
    try:
        parsed = int(str(raw).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(raw: str, default: float) -> float:
    try:
        parsed = float(str(raw).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _parse_bool(raw: str, default: bool) -> bool:
    value = str(raw or "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

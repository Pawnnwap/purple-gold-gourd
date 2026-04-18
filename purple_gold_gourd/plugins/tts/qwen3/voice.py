from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np

from ....config import AppConfig
from ....language import detect_text_language, language_label
from ....utils import ensure_dir
from ..shared import split_for_synthesis


class Qwen3TTSSynthesizer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._voice_clone_key: tuple[str, str] | None = None
        self._voice_clone_prompt = None

    def synthesize(self, text: str, prompt_text: str, prompt_audio: Path, target: Path) -> Path:
        import soundfile as sf

        model = self._ensure_loaded()
        voice_clone_prompt = self._ensure_voice_clone_prompt(prompt_audio, prompt_text)
        chunks = split_for_synthesis(
            text,
            max_chars=int(self.config.tts_setting("qwen3_tts_chunk_chars", "160")),
        )
        if not chunks:
            raise ValueError("Qwen3 TTS received empty text.")

        do_sample = self.config.tts_setting("qwen3_tts_do_sample", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        generation_kwargs = {"non_streaming_mode": True, "do_sample": do_sample}
        max_new_tokens = self.config.tts_setting("qwen3_tts_max_new_tokens", "").strip()
        if max_new_tokens:
            generation_kwargs["max_new_tokens"] = int(max_new_tokens)

        segments: list[np.ndarray] = []
        sample_rate: int | None = None
        for chunk in chunks:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=_qwen_language_name(chunk or prompt_text),
                    voice_clone_prompt=voice_clone_prompt,
                    **generation_kwargs,
                )
            if not wavs:
                continue
            sample_rate = sample_rate or int(sr)
            segments.append(np.asarray(wavs[0], dtype=np.float32).reshape(-1))

        if not segments or sample_rate is None:
            raise ValueError("Qwen3 TTS returned no audio.")

        ensure_dir(target.parent)
        sf.write(str(target), np.concatenate(segments), sample_rate)
        return target

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        cache_root = self.config.tts_path("qwen3_tts_cache_dir")
        hub_cache_dir = ensure_dir(cache_root / "hub")
        ensure_dir(cache_root / "assets")
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                import torch
                from qwen_tts import Qwen3TTSModel
                from transformers import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 TTS requires the 'qwen-tts' package. Install it with 'pip install -U qwen-tts'.",
            ) from exc
        transformers_logging.set_verbosity_error()

        model_name_or_path = self.config.tts_setting("qwen3_tts_model")
        kwargs = {
            "cache_dir": str(hub_cache_dir),
            "device_map": self.config.tts_setting("qwen3_tts_device_map"),
            "dtype": _resolve_torch_dtype(torch, self.config.tts_setting("qwen3_tts_dtype")),
        }
        attn_implementation = self.config.tts_setting("qwen3_tts_attn_implementation", "").strip()
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self._model = Qwen3TTSModel.from_pretrained(model_name_or_path, **kwargs)
        return self._model

    def _ensure_voice_clone_prompt(self, prompt_audio: Path, prompt_text: str):
        import pickle
        import soundfile as sf

        cache_key = (str(prompt_audio.resolve()), prompt_text.strip())
        if self._voice_clone_key == cache_key and self._voice_clone_prompt is not None:
            return self._voice_clone_prompt

        cache_path = prompt_audio.with_name(prompt_audio.stem + "-profile.pkl")
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)
                if cached.get("key") == cache_key:
                    self._voice_clone_key = cache_key
                    self._voice_clone_prompt = cached["data"]
                    return self._voice_clone_prompt
            except Exception:
                pass

        audio, sample_rate = sf.read(str(prompt_audio), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self._voice_clone_prompt = self._ensure_loaded().create_voice_clone_prompt(
                ref_audio=(np.asarray(audio, dtype=np.float32), int(sample_rate)),
                ref_text=prompt_text,
                x_vector_only_mode=False,
            )
        self._voice_clone_key = cache_key

        try:
            with open(cache_path, "wb") as fh:
                pickle.dump({"key": cache_key, "data": self._voice_clone_prompt}, fh, protocol=4)
        except Exception:
            pass

        return self._voice_clone_prompt


def _qwen_language_name(text: str) -> str:
    code = detect_text_language(text)
    if not code:
        return "Auto"
    return language_label(code)


def _resolve_torch_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.strip().lower()
    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Qwen3 TTS dtype '{dtype_name}'.")
    return mapping[normalized]

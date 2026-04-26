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
        self._voice_clone_key: tuple[str, str, bool] | None = None
        self._voice_clone_prompt = None

    def synthesize(self, text: str, prompt_text: str, prompt_audio: Path, target: Path) -> Path:
        import soundfile as sf

        model = self._ensure_loaded()
        x_vector_only = _tts_bool(self.config.tts_setting("qwen3_tts_x_vector_only", "true"), default=True)
        voice_clone_prompt = self._ensure_voice_clone_prompt(prompt_audio, prompt_text, x_vector_only)
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
        max_new_token_cap = int(max_new_tokens) if max_new_tokens else 0
        adaptive_max_new_tokens = _tts_bool(
            self.config.tts_setting("qwen3_tts_adaptive_max_new_tokens", "true"),
            default=True,
        )

        segments: list[np.ndarray] = []
        sample_rate: int | None = None
        for chunk in chunks:
            chunk_generation_kwargs = dict(generation_kwargs)
            if max_new_token_cap > 0:
                chunk_generation_kwargs["max_new_tokens"] = (
                    _estimate_max_new_tokens(chunk, max_new_token_cap)
                    if adaptive_max_new_tokens
                    else max_new_token_cap
                )
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=_qwen_language_name(chunk or prompt_text),
                    voice_clone_prompt=voice_clone_prompt,
                    **chunk_generation_kwargs,
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
                _install_noop_sox_module()
                import torch
                from qwen_tts import Qwen3TTSModel
                from transformers import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 TTS requires the 'qwen-tts' package. Install it with 'pip install -U qwen-tts'.",
            ) from exc
        _patch_qwen_sox_norm()
        transformers_logging.set_verbosity_error()

        model_name_or_path = self.config.tts_setting("qwen3_tts_model")
        device_map = self.config.tts_setting("qwen3_tts_device_map")
        dtype = _resolve_torch_dtype(torch, self.config.tts_setting("qwen3_tts_dtype"))
        kwargs: dict[str, object] = {
            "cache_dir": str(hub_cache_dir),
            "device_map": device_map,
            "dtype": dtype,
        }
        attn_implementation = self.config.tts_setting("qwen3_tts_attn_implementation", "").strip()
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        quantization = self._resolve_quantization(device_map)
        if quantization is not None:
            kwargs["quantization_config"] = quantization
            kwargs.pop("dtype", None)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            try:
                self._model = Qwen3TTSModel.from_pretrained(model_name_or_path, **kwargs)
            except Exception:
                if "quantization_config" in kwargs:
                    kwargs.pop("quantization_config", None)
                    kwargs["dtype"] = dtype
                    self._model = Qwen3TTSModel.from_pretrained(model_name_or_path, **kwargs)
                else:
                    raise
        return self._model

    def _resolve_quantization(self, device_map: str):
        mode = self.config.tts_setting("qwen3_tts_quantization", "auto").strip().lower()
        if mode in {"none", "off", "0", "false", ""}:
            return None
        on_cuda = "cuda" in (device_map or "").lower()
        if mode == "auto":
            if not on_cuda:
                return None
            mode = "4bit"
        try:
            import torch
            from transformers import BitsAndBytesConfig
            if mode in {"4bit", "int4", "nf4"}:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if on_cuda and torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            if mode in {"8bit", "int8"}:
                return BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            return None
        return None

    def _ensure_voice_clone_prompt(self, prompt_audio: Path, prompt_text: str, x_vector_only: bool):
        import pickle
        import soundfile as sf

        normalized_prompt_text = "" if x_vector_only else prompt_text.strip()
        cache_key = (str(prompt_audio.resolve()), normalized_prompt_text, x_vector_only)
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
            if x_vector_only:
                self._voice_clone_prompt = self._create_x_vector_prompt(np.asarray(audio, dtype=np.float32), int(sample_rate))
            else:
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

    def _create_x_vector_prompt(self, audio: np.ndarray, sample_rate: int):
        import librosa
        from qwen_tts import VoiceClonePromptItem

        model = self._ensure_loaded().model
        speaker_sample_rate = int(model.speaker_encoder_sample_rate)
        speaker_audio = audio
        if sample_rate != speaker_sample_rate:
            speaker_audio = librosa.resample(
                y=speaker_audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=speaker_sample_rate,
            )
        speaker_embedding = model.extract_speaker_embedding(
            audio=np.asarray(speaker_audio, dtype=np.float32),
            sr=speaker_sample_rate,
        )
        return [
            VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=speaker_embedding,
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None,
            ),
        ]


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


def _estimate_max_new_tokens(text: str, cap: int) -> int:
    # Qwen3 TTS 12Hz emits roughly 12 audio tokens per second. This keeps
    # short smoke/dialogue chunks from running to the full long-form cap.
    content_units = sum(2 if "\u4e00" <= char <= "\u9fff" else 1 for char in text.strip())
    estimated = int(content_units * 2.2 + 48)
    return max(96, min(cap, estimated))


def _tts_bool(raw: str, default: bool = False) -> bool:
    normalized = str(raw or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _install_noop_sox_module() -> None:
    import sys
    import types

    if "sox" in sys.modules:
        return

    class _NoopSoxTransformer:
        def norm(self, *args, **kwargs):
            return self

        def build_array(self, input_array, sample_rate_in):
            return np.asarray(input_array, dtype=np.float32)

    module = types.ModuleType("sox")
    module.Transformer = _NoopSoxTransformer
    sys.modules["sox"] = module


def _patch_qwen_sox_norm() -> None:
    try:
        from qwen_tts.core.tokenizer_25hz.vq import speech_vq
    except Exception:
        return

    def _numpy_peak_norm(self, audio):
        wav = np.asarray(audio, dtype=np.float32)
        if wav.size == 0:
            return wav
        peak = float(np.max(np.abs(wav)))
        target_peak = 10 ** (-6 / 20)
        if peak > target_peak:
            wav = wav * (target_peak / peak)
        return wav

    class _NoopSoxTransformer:
        def norm(self, *args, **kwargs):
            return self

        def build_array(self, input_array, sample_rate_in):
            return np.asarray(input_array, dtype=np.float32)

    speech_vq.sox.Transformer = _NoopSoxTransformer
    speech_vq.XVectorExtractor.sox_norm = _numpy_peak_norm

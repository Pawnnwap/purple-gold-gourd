from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

_THINK_BLOCK_RE = re.compile(r"<(?:think|thought)\b[^>]*>.*?</(?:think|thought)>", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?(?:think|thought)\b[^>]*>", flags=re.IGNORECASE)


def candidate_models(client: OpenAI, preferred_model: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(model_id: str) -> None:
        if not model_id or model_id in seen:
            return
        seen.add(model_id)
        ordered.append(model_id)

    add(preferred_model)
    for model_id in _backup_models():
        add(model_id)
    return ordered


def _backup_models() -> list[str]:
    raw = os.getenv("OPENAI_MODEL_BACKUPS") or os.getenv("OPENAI_MODEL_BACKUP") or "gemma-4-26b-a4b-it"
    return [item.strip() for item in raw.split(",") if item.strip()]


def strip_reasoning_blocks(text: str) -> str:
    if not text:
        return ""
    stripped = _THINK_BLOCK_RE.sub("", text)
    stripped = _THINK_TAG_RE.sub("", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


class ManagedLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        preferred_model: str,
        max_context_tokens: int = 0,
        max_completion_tokens: int = 0,
    ) -> None:
        timeout = _parse_timeout(os.getenv("OPENAI_TIMEOUT", ""))
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.preferred_model = preferred_model
        self.max_context_tokens = max(max_context_tokens, 0)
        self.max_completion_tokens = max(max_completion_tokens, 0)
        self.strict_model = os.getenv("OPENAI_STRICT_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        last_error: Exception | None = None
        effective_max_tokens = self.max_completion_tokens if max_tokens is None else max(max_tokens, 0)
        model_ids = [self.preferred_model] if self.strict_model else candidate_models(self.client, self.preferred_model)
        for model_id in model_ids:
            try:
                request: dict[str, Any] = {
                    "model": model_id,
                    "temperature": temperature,
                    "messages": messages,
                }
                if effective_max_tokens > 0:
                    request["max_tokens"] = effective_max_tokens
                response = self.client.chat.completions.create(**request)
                content = _message_text(response.choices[0].message)
                self.preferred_model = model_id
                return strip_reasoning_blocks(content), model_id
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if not self.strict_model and ("model is unloaded" in message or "currently not loaded" in message):
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No chat model candidates were available.")

    def input_token_budget(
        self,
        reserved_prompt_tokens: int = 0,
        reserved_output_tokens: int | None = None,
    ) -> int:
        if self.max_context_tokens <= 0:
            return 0
        output_tokens = self.max_completion_tokens if reserved_output_tokens is None else max(reserved_output_tokens, 0)
        return max(self.max_context_tokens - max(reserved_prompt_tokens, 0) - output_tokens, 0)


def complete_with_model_fallback(
    client: OpenAI,
    preferred_model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None = None,
) -> tuple[str, str]:
    llm = ManagedLLM(
        base_url=str(getattr(client, "base_url", "") or ""),
        api_key=str(getattr(client, "api_key", "") or ""),
        preferred_model=preferred_model,
        max_completion_tokens=max(max_tokens or 0, 0),
    )
    llm.client = client
    return llm.complete(messages=messages, temperature=temperature, max_tokens=max_tokens)


def _parse_timeout(raw: str) -> float:
    try:
        value = float(str(raw or "").strip())
    except Exception:
        return 180.0
    return value if value > 0 else 180.0


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
                continue
            if isinstance(item, dict):
                text = item.get("text") or ""
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content or "")

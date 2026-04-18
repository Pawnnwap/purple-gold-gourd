from __future__ import annotations

import json
import math
import re
import unicodedata
import hashlib
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

PUNCTUATION = set(".,!?;:。！？；：…，、")
T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(UTC)


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^\w\-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "creator"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def parse_human_number(text: str) -> int:
    if text is None:
        return 0
    if isinstance(text, (int, float)):
        return int(text)
    raw = clean_html(str(text)).replace(",", "").replace(" ", "").lower()
    raw = raw.replace("subscribers", "").replace("subscriber", "")
    raw = raw.replace("fans", "").replace("粉丝", "")
    raw = raw.replace("万", "w").replace("亿", "e")
    match = re.search(r"(\d+(?:\.\d+)?)([kmwe]?)", raw)
    if not match:
        digits = re.sub(r"\D", "", raw)
        return int(digits) if digits else 0
    value = float(match.group(1))
    suffix = match.group(2)
    multipliers = {
        "": 1,
        "k": 1_000,
        "m": 1_000_000,
        "w": 10_000,
        "e": 100_000_000,
    }
    return int(value * multipliers.get(suffix, 1))


def parse_upload_datetime(timestamp: int | None = None, upload_date: str | None = None) -> datetime | None:
    if timestamp:
        return datetime.fromtimestamp(int(timestamp), tz=UTC)
    if upload_date:
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(upload_date, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return None


def hotness_score(
    views: int,
    likes: int = 0,
    comments: int = 0,
    favorites: int = 0,
    shares: int = 0,
    published_at: datetime | None = None,
) -> float:
    engagement = likes + comments * 1.4 + favorites * 1.6 + shares * 1.8
    age_days = 30.0
    if published_at is not None:
        delta = utc_now() - published_at
        age_days = max(delta.total_seconds() / 86400.0, 0.5)
    freshness = math.pow(age_days + 2.0, 0.35)
    return (max(views, 0) + 3.0 * max(engagement, 0)) / freshness


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    cjk = [char for char in lowered if _is_cjk(char)]
    return latin + cjk


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def join_tokens(tokens: Iterable[str]) -> str:
    built: list[str] = []
    prev_tail = ""
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if built and _needs_space(prev_tail, token[0]):
            built.append(" ")
        built.append(token)
        prev_tail = token[-1]
    return "".join(built).strip()


def _needs_space(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def chunked_evenly(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[0]]
    result: list[Any] = []
    last_index = len(items) - 1
    for i in range(limit):
        idx = round(i * last_index / (limit - 1))
        result.append(items[idx])
    return result


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    lowered = text.lower()
    latin_words = re.findall(r"[a-z0-9_]+", lowered)
    latin_tokens = sum(max(1, math.ceil(len(word) / 4.0)) for word in latin_words)
    stripped_latin = re.sub(r"[a-z0-9_\s]+", "", lowered)
    other_tokens = len(stripped_latin)
    return max(latin_tokens + other_tokens, 1)


def chunked_by_budget(
    items: list[T],
    cost: Callable[[T], int],
    max_budget: int,
    overlap_items: int = 0,
) -> list[list[T]]:
    if not items:
        return []
    if max_budget <= 0:
        return [items]
    groups: list[list[T]] = []
    index = 0
    overlap_items = max(overlap_items, 0)
    while index < len(items):
        start_index = index
        group: list[T] = []
        used_budget = 0
        while index < len(items):
            item = items[index]
            item_cost = max(cost(item), 1)
            if group and used_budget + item_cost > max_budget:
                break
            group.append(item)
            used_budget += item_cost
            index += 1
        groups.append(group)
        if index >= len(items):
            break
        index = max(index - overlap_items, start_index + 1)
    return groups


def format_ms(ms: int) -> str:
    total = max(int(ms // 1000), 0)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

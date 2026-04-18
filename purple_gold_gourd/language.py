from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

_ALIASES = {
    "auto": "",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "cmn": "zh",
    "mandarin": "zh",
    "yue": "zh",
    "cantonese": "zh",
    "en": "en",
    "english": "en",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
    "mul": "",
    "mixed": "",
    "unknown": "",
}

_LABELS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}


def normalize_language_code(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("_", "-")
    if raw in _ALIASES:
        return _ALIASES[raw]
    if raw.startswith("zh"):
        return "zh"
    if raw.startswith("en"):
        return "en"
    if raw.startswith("ja") or raw.startswith("jp"):
        return "ja"
    if raw.startswith("ko") or raw.startswith("kr"):
        return "ko"
    return ""


def language_label(code: str) -> str:
    normalized = normalize_language_code(code)
    return _LABELS.get(normalized, "the detected language")


def detect_text_language(text: str) -> str:
    if not text:
        return ""
    counts = {"zh": 0, "ja": 0, "ko": 0, "en": 0}
    for char in text:
        codepoint = ord(char)
        if 0x3040 <= codepoint <= 0x30FF:
            counts["ja"] += 2
        elif 0xAC00 <= codepoint <= 0xD7AF:
            counts["ko"] += 2
        elif 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
            counts["zh"] += 1
        elif ("a" <= char <= "z") or ("A" <= char <= "Z"):
            counts["en"] += 1
    if counts["ja"] >= 4 and counts["ja"] >= counts["zh"] // 2:
        return "ja"
    if counts["ko"] >= 4 and counts["ko"] >= counts["en"] // 2:
        return "ko"
    if counts["zh"] >= 4 and counts["zh"] >= max(counts["en"] // 2, 1):
        return "zh"
    if counts["en"] >= 12:
        return "en"
    if counts["zh"] > counts["en"] and counts["zh"] >= 2:
        return "zh"
    if counts["en"] > 0:
        return "en"
    return ""


def dominant_language(codes: Iterable[str], fallback: str = "") -> str:
    normalized = [normalize_language_code(code) for code in codes]
    normalized = [code for code in normalized if code]
    if normalized:
        return Counter(normalized).most_common(1)[0][0]
    return normalize_language_code(fallback)


def detect_transcript_language(tag: str, text: str) -> str:
    normalized = normalize_language_code(tag)
    if normalized:
        return normalized
    detected = detect_text_language(text)
    return detected or "en"


def detect_creator_language(
    metadata_text: str,
    video_texts: Iterable[str],
    transcript_languages: Iterable[str],
) -> str:
    transcript_language = dominant_language(transcript_languages)
    if transcript_language:
        return transcript_language
    metadata_language = detect_text_language(metadata_text)
    if metadata_language:
        return metadata_language
    video_language = dominant_language(detect_text_language(text) for text in video_texts)
    return video_language or "en"


_OUTPUT_LANGUAGE_PATTERNS = {
    "zh": [
        r"(?i)\b(?:please\s+)?(?:answer|reply|respond|write|translate)\s+(?:only\s+)?in\s+chinese\b[\s,.:;/-]*",
        r"(?i)\b(?:please\s+)?translate\s+(?:this|that|it)?\s*(?:into|to)\s+chinese\b[\s,.:;/-]*",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u4e2d\u6587(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u4e2d\u6587",
    ],
    "en": [
        r"(?i)\b(?:please\s+)?(?:answer|reply|respond|write|translate)\s+(?:only\s+)?in\s+english\b[\s,.:;/-]*",
        r"(?i)\b(?:please\s+)?translate\s+(?:this|that|it)?\s*(?:into|to)\s+english\b[\s,.:;/-]*",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u82f1\u6587(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u82f1\u6587",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u82f1\u8bed",
    ],
    "ja": [
        r"(?i)\b(?:please\s+)?(?:answer|reply|respond|write|translate)\s+(?:only\s+)?in\s+japanese\b[\s,.:;/-]*",
        r"(?i)\b(?:please\s+)?translate\s+(?:this|that|it)?\s*(?:into|to)\s+japanese\b[\s,.:;/-]*",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u65e5\u6587(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u65e5\u8bed(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u65e5\u6587",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u65e5\u8bed",
    ],
    "ko": [
        r"(?i)\b(?:please\s+)?(?:answer|reply|respond|write|translate)\s+(?:only\s+)?in\s+korean\b[\s,.:;/-]*",
        r"(?i)\b(?:please\s+)?translate\s+(?:this|that|it)?\s*(?:into|to)\s+korean\b[\s,.:;/-]*",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u97e9\u6587(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u8bf7|\u9ebb\u70e6)?\u7528\u97e9\u8bed(?:\u56de\u7b54|\u56de\u590d|\u8bf4|\u8bb2|\u8f93\u51fa)?",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u97e9\u6587",
        "(?:\u7ffb\u8bd1|\u8bd1)(?:\u6210|\u4e3a)?\u97e9\u8bed",
    ],
}


def detect_output_language_request(text: str) -> str:
    for code, patterns in _OUTPUT_LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return code
    return ""


def strip_output_language_request(text: str) -> str:
    stripped = text
    for patterns in _OUTPUT_LANGUAGE_PATTERNS.values():
        for pattern in patterns:
            stripped = re.sub(pattern, "", stripped)
    stripped = re.sub(r"^[\s,.:;/-]+", "", stripped)
    stripped = re.sub(r"[\s,.:;/-]+$", "", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.strip()

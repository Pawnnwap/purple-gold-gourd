from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ..schema import TranscriptChunk, TranscriptFile
from ..utils import tokenize

_CJK_STOPCHARS = set("的了是我你他她它们也就都和与及或而但被把在着过吧呢吗啊呀哦嗯这那有不没")

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "say",
    "says",
    "she",
    "so",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(slots=True)
class RagAssessment:
    keyword_count: int
    matched_keyword_count: int
    keyword_coverage: float
    hit_count: int
    supporting_hit_count: int
    top_score: float
    avg_top3_score: float
    should_use_web_search: bool
    reason: str


class RagIndex:
    def __init__(self, chunks: list[TranscriptChunk]) -> None:
        self.chunks = chunks
        self.documents = [_index_tokens(chunk.text) for chunk in chunks]
        self.title_tokens = [set(_index_tokens(chunk.video_title or "")) for chunk in chunks]
        self.document_sets = [set(tokens) for tokens in self.documents]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.documents:
            self.doc_freq.update(set(tokens))
        lengths = [len(tokens) for tokens in self.documents]
        self.avg_len = sum(lengths) / max(len(lengths), 1)

    @classmethod
    def from_transcripts(cls, transcripts: list[TranscriptFile]) -> RagIndex:
        chunks: list[TranscriptChunk] = []
        for transcript in transcripts:
            chunks.extend(transcript.chunks)
        return cls(chunks)

    def search(self, query: str, top_k: int = 8) -> list[tuple[TranscriptChunk, float]]:
        query_tokens = _query_tokens(query)
        if not query_tokens:
            return []
        scores: list[tuple[TranscriptChunk, float]] = []
        total_docs = max(len(self.documents), 1)
        for index, (chunk, tokens) in enumerate(zip(self.chunks, self.documents)):
            score = 0.0
            token_counts = Counter(tokens)
            doc_len = max(len(tokens), 1)
            title_set = self.title_tokens[index] if index < len(self.title_tokens) else set()
            for token in query_tokens:
                tf = token_counts.get(token, 0)
                in_title = token in title_set
                if tf == 0 and not in_title:
                    continue
                df = self.doc_freq.get(token, 0) or 1
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                if tf > 0:
                    denom = tf + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(self.avg_len, 1))
                    score += idf * (tf * 2.5) / max(denom, 1e-9)
                if in_title:
                    score += idf * 1.6
                if len(token) >= 2 and not token.isascii():
                    score += idf * 0.4 * min(tf, 3)
            if score > 0:
                scores.append((chunk, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def assess(self, query: str, hits: list[tuple[TranscriptChunk, float]]) -> RagAssessment:
        keywords = list(dict.fromkeys(_query_tokens(query)))
        if not hits:
            return RagAssessment(
                keyword_count=len(keywords),
                matched_keyword_count=0,
                keyword_coverage=0.0,
                hit_count=0,
                supporting_hit_count=0,
                top_score=0.0,
                avg_top3_score=0.0,
                should_use_web_search=True,
                reason="no transcript chunks matched the question",
            )

        keyword_set = set(keywords)
        matched_keywords: set[str] = set()
        supporting_hit_count = 0
        top_score = hits[0][1]
        top_hits = hits[:3]
        for chunk, score in top_hits:
            overlap = keyword_set & set(tokenize(chunk.text))
            matched_keywords.update(overlap)
            enough_overlap = len(overlap) >= (1 if len(keyword_set) <= 2 else 2)
            enough_score = score >= max(0.9, top_score * 0.55)
            if enough_overlap or enough_score:
                supporting_hit_count += 1

        matched_keyword_count = len(matched_keywords)
        keyword_count = len(keyword_set)
        keyword_coverage = matched_keyword_count / max(keyword_count, 1)
        avg_top3_score = sum(score for _, score in top_hits) / max(len(top_hits), 1)

        should_use_web_search = False
        reason = "video retrieval looks sufficient"
        if keyword_count <= 2:
            if top_score < 0.6 or supporting_hit_count == 0:
                should_use_web_search = True
                reason = "short query only has a weak transcript match"
        else:
            # Trigger web search only when the transcript evidence is measurably thin:
            # low keyword coverage, a weak top hit, or too few supporting chunks.
            if keyword_coverage < 0.30:
                should_use_web_search = True
                reason = "query keyword coverage is too low"
            elif top_score < 0.75 and keyword_coverage < 0.50:
                should_use_web_search = True
                reason = "top retrieval score is weak for the matched keywords"
            elif supporting_hit_count < 2 and keyword_coverage < 0.40:
                should_use_web_search = True
                reason = "too few transcript chunks support the question"

        return RagAssessment(
            keyword_count=keyword_count,
            matched_keyword_count=matched_keyword_count,
            keyword_coverage=keyword_coverage,
            hit_count=len(hits),
            supporting_hit_count=supporting_hit_count,
            top_score=top_score,
            avg_top3_score=avg_top3_score,
            should_use_web_search=should_use_web_search,
            reason=reason,
        )


def _index_tokens(text: str) -> list[str]:
    base = tokenize(text)
    return base + _cjk_bigrams(base)


def _query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    filtered = [token for token in tokens if _is_query_keyword(token)]
    primary = filtered or tokens
    bigrams = _cjk_bigrams(primary)
    return primary + bigrams


def _cjk_bigrams(tokens: list[str]) -> list[str]:
    bigrams: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and not token.isascii() and token not in _CJK_STOPCHARS:
            run.append(token)
            continue
        bigrams.extend(_join_cjk_run(run))
        run = []
    bigrams.extend(_join_cjk_run(run))
    return bigrams


def _join_cjk_run(run: list[str]) -> list[str]:
    if len(run) < 2:
        return []
    return [run[i] + run[i + 1] for i in range(len(run) - 1)]


def _is_query_keyword(token: str) -> bool:
    if not token:
        return False
    if token.isascii():
        if token in _EN_STOPWORDS:
            return False
        if len(token) == 1 and not token.isdigit():
            return False
    elif len(token) == 1 and token in _CJK_STOPCHARS:
        return False
    return True

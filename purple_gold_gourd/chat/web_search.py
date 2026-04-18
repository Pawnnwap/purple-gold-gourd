from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


class WebSearchClient:
    def __init__(self, max_results: int = 4, timeout_s: float = 8.0) -> None:
        self.max_results = max(max_results, 1)
        self.timeout_s = max(timeout_s, 1.0)

    def search(self, query: str) -> list[WebSearchResult]:
        text = query.strip()
        if not text:
            return []
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "Web search requires the 'ddgs' package. Install dependencies with 'pip install -e .' again.",
            ) from exc

        rows = DDGS(timeout=self.timeout_s).text(text, max_results=self.max_results)
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for row in rows or []:
            url = _clean(str(row.get("href") or row.get("url") or ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                WebSearchResult(
                    title=_clean(str(row.get("title") or "")) or url,
                    url=url,
                    snippet=_clean(str(row.get("body") or row.get("snippet") or "")),
                ),
            )
            if len(results) >= self.max_results:
                break
        return results


def _clean(text: str) -> str:
    return " ".join(text.split())

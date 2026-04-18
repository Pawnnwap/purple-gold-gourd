from __future__ import annotations

import json
import re
import warnings
from difflib import SequenceMatcher
from hashlib import md5
from pathlib import Path
from urllib.parse import quote_plus, urlencode, urlparse

import requests
from yt_dlp import YoutubeDL

from ..schema import CreatorRef
from ..utils import USER_AGENT, clean_html, parse_human_number

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)


_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

_BILI_DM_PARAMS = {
    "dm_img_list": "[]",
    "dm_img_str": "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ",
    "dm_cover_img_str": (
        "QU5HTEUgKEludGVsLCBJbnRlbChSKSBVSEQgR3JhcGhpY3MgKDB4MDAwMEE3QTgpIERpcmVjdDNEMTEg"
        "dnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKEludGVsKQ"
    ),
    "dm_img_inter": '{"ds":[],"wh":[6574,3858,58],"of":[330,660,330]}',
}


class CreatorResolver:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Referer": "https://www.bilibili.com/",
            },
        )

    def resolve(self, query: str, platform: str = "auto") -> CreatorRef:
        query = query.strip()
        if not query:
            raise ValueError("Creator query is empty.")
        explicit = self._resolve_explicit(query)
        if explicit:
            return explicit
        if platform == "youtube":
            return self._resolve_youtube_name(query)
        if platform == "bilibili":
            return self._resolve_bilibili_name(query)
        if re.search(r"[\u4e00-\u9fff]", query):
            try:
                return self._resolve_bilibili_name(query)
            except Exception:
                return self._resolve_youtube_name(query)
        try:
            return self._resolve_youtube_name(query)
        except Exception:
            return self._resolve_bilibili_name(query)

    def _resolve_explicit(self, query: str) -> CreatorRef | None:
        lowered = query.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            if "youtube.com" in lowered or "youtu.be" in lowered:
                return self._resolve_youtube_url(query)
            if "bilibili.com" in lowered or "b23.tv" in lowered:
                return self._resolve_bilibili_url(query)
        if query.startswith("@") or query.startswith("UC"):
            return self._resolve_youtube_url(self._normalize_youtube_home(query))
        if query.isdigit() or query.lower().startswith("mid:"):
            mid = query.split(":", 1)[-1]
            return self._build_bilibili_creator(mid=mid, query=query)
        return None

    def _resolve_youtube_url(self, query: str) -> CreatorRef:
        url = self._normalize_youtube_home(query)
        info = self._extract_info(url)
        channel_id = str(info.get("channel_id") or info.get("uploader_id") or info.get("id") or "")
        handle = info.get("uploader_id") or info.get("channel") or ""
        channel_url = info.get("channel_url") or url
        name = info.get("channel") or info.get("uploader") or channel_id or handle or query
        return CreatorRef(
            platform="youtube",
            creator_id=channel_id or handle or name,
            name=name,
            homepage_url=channel_url.rstrip("/"),
            video_tab_url=self._ensure_suffix(channel_url, "videos"),
            query=query,
            followers=int(info.get("channel_follower_count") or 0),
            handle=str(handle),
            bio=str(info.get("description") or ""),
        )

    def _resolve_bilibili_url(self, query: str) -> CreatorRef:
        mid = self._extract_bilibili_mid(query)
        if not mid:
            info = self._extract_info(query)
            uploader_id = info.get("uploader_id") or ""
            mid = re.sub(r"\D", "", str(uploader_id))
        if not mid:
            raise ValueError(f"Could not resolve Bilibili mid from: {query}")
        return self._build_bilibili_creator(mid=mid, query=query)

    def _resolve_youtube_name(self, query: str) -> CreatorRef:
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        initial = self._extract_json_blob(response.text, "ytInitialData")
        candidates = list(self._walk_channel_renderers(initial))
        if not candidates:
            return self._resolve_youtube_name_fallback(query)
        picked = max(candidates, key=lambda item: self._score_youtube_channel(query, item))
        canonical = self._youtube_channel_url(picked)
        return CreatorRef(
            platform="youtube",
            creator_id=str(picked.get("channelId") or ""),
            name=self._youtube_text(picked.get("title")),
            homepage_url=canonical,
            video_tab_url=self._ensure_suffix(canonical, "videos"),
            query=query,
            followers=parse_human_number(self._youtube_text(picked.get("subscriberCountText"))),
            handle=self._youtube_handle(picked),
            bio=self._youtube_text(picked.get("descriptionSnippet")),
        )

    def _resolve_youtube_name_fallback(self, query: str) -> CreatorRef:
        with YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": True}) as ydl:
            data = ydl.extract_info(f"ytsearch10:{query}", download=False) or {}
        buckets: dict[str, dict[str, object]] = {}
        for entry in data.get("entries") or []:
            channel_id = str(entry.get("channel_id") or entry.get("uploader_id") or entry.get("channel") or "")
            if not channel_id:
                continue
            bucket = buckets.setdefault(
                channel_id,
                {
                    "name": entry.get("channel") or entry.get("uploader") or channel_id,
                    "channel_url": entry.get("channel_url") or "",
                    "score": 0.0,
                },
            )
            bucket["score"] = float(bucket["score"]) + SequenceMatcher(
                None,
                query.lower(),
                str(bucket["name"]).lower(),
            ).ratio()
        if not buckets:
            raise ValueError(f"Could not find YouTube creator for: {query}")
        channel_id, bucket = max(buckets.items(), key=lambda item: item[1]["score"])
        channel_url = str(bucket["channel_url"] or f"https://www.youtube.com/channel/{channel_id}")
        return CreatorRef(
            platform="youtube",
            creator_id=channel_id,
            name=str(bucket["name"]),
            homepage_url=channel_url,
            video_tab_url=self._ensure_suffix(channel_url, "videos"),
            query=query,
        )

    def _resolve_bilibili_name(self, query: str) -> CreatorRef:
        self._prime_bilibili_cookies()
        params = {"search_type": "bili_user", "keyword": query, "page": 1}
        response = self.session.get(
            "https://api.bilibili.com/x/web-interface/wbi/search/type",
            params=self._sign_wbi(params),
            timeout=20,
        )
        if response.status_code >= 400:
            fallback = self.session.get(
                "https://api.bilibili.com/x/web-interface/search/type",
                params=params,
                timeout=20,
            )
            fallback.raise_for_status()
            payload = fallback.json()
        else:
            payload = response.json()
        results = (payload.get("data") or {}).get("result") or []
        if not results:
            raise ValueError(f"Could not find Bilibili creator for: {query}")
        picked = max(results[:10], key=lambda item: self._score_bilibili_user(query, item))
        mid = str(picked.get("mid"))
        return self._build_bilibili_creator(
            mid=mid,
            query=query,
            name=clean_html(str(picked.get("uname") or mid)),
            fans=int(picked.get("fans") or 0),
            bio=str(picked.get("usign") or ""),
        )

    def _build_bilibili_creator(
        self,
        mid: str,
        query: str,
        name: str | None = None,
        fans: int = 0,
        bio: str = "",
    ) -> CreatorRef:
        base = f"https://space.bilibili.com/{mid}"
        if not name:
            card = self._fetch_bilibili_card(mid)
            if card:
                name = str(card.get("name") or mid)
                fans = int(card.get("fans") or fans)
                bio = str(card.get("sign") or bio)
            else:
                try:
                    self._prime_bilibili_cookies()
                    info = self.session.get(
                        "https://api.bilibili.com/x/space/wbi/acc/info",
                        params=self._sign_wbi(
                            {
                                "mid": mid,
                                "token": "",
                                "platform": "web",
                                "web_location": 1550101,
                                **_BILI_DM_PARAMS,
                            },
                        ),
                        timeout=20,
                    ).json()
                    data = info.get("data") or {}
                    name = str(data.get("name") or mid)
                    fans = int(data.get("fans") or fans)
                    bio = str(data.get("sign") or bio)
                except Exception:
                    name = mid
        return CreatorRef(
            platform="bilibili",
            creator_id=mid,
            name=name,
            homepage_url=base,
            video_tab_url=f"{base}/video",
            query=query,
            followers=fans,
            bio=bio,
        )

    def _prime_bilibili_cookies(self) -> None:
        self.session.get("https://www.bilibili.com", timeout=20)

    def _fetch_bilibili_card(self, mid: str) -> dict[str, object]:
        response = self.session.get(
            "https://api.bilibili.com/x/web-interface/card",
            params={"mid": mid},
            timeout=20,
        )
        payload = response.json()
        data = payload.get("data") or {}
        return data.get("card") or {}

    def _resolve_nav_wbi_keys(self) -> tuple[str, str]:
        nav = self.session.get("https://api.bilibili.com/x/web-interface/nav", timeout=20)
        nav.raise_for_status()
        payload = nav.json()
        data = payload.get("data") or {}
        wbi = data.get("wbi_img") or {}
        img_key = Path(urlparse(str(wbi.get("img_url") or "")).path).stem
        sub_key = Path(urlparse(str(wbi.get("sub_url") or "")).path).stem
        if not img_key or not sub_key:
            raise ValueError("Could not resolve Bilibili wbi keys.")
        return img_key, sub_key

    def _sign_wbi(self, params: dict[str, object]) -> dict[str, object]:
        import time

        img_key, sub_key = self._resolve_nav_wbi_keys()
        mixin = "".join((img_key + sub_key)[index] for index in _MIXIN_KEY_ENC_TAB)[:32]
        signed = dict(params)
        signed["wts"] = int(time.time())
        signed = {key: re.sub(r"[!'()*]", "", str(value)) for key, value in signed.items()}
        query = urlencode(sorted(signed.items()))
        signed["w_rid"] = md5(f"{query}{mixin}".encode()).hexdigest()
        return signed

    def _score_youtube_channel(self, query: str, renderer: dict[str, object]) -> float:
        title = self._youtube_text(renderer.get("title"))
        handle = self._youtube_handle(renderer)
        subscribers = parse_human_number(self._youtube_text(renderer.get("subscriberCountText")))
        similarity = max(
            SequenceMatcher(None, query.lower(), title.lower()).ratio(),
            SequenceMatcher(None, query.lower(), handle.lower()).ratio() if handle else 0.0,
        )
        return similarity * 10 + subscribers / 1_000_000

    def _score_bilibili_user(self, query: str, item: dict[str, object]) -> float:
        uname = clean_html(str(item.get("uname") or ""))
        fans = int(item.get("fans") or 0)
        similarity = SequenceMatcher(None, query.lower(), uname.lower()).ratio()
        return similarity * 10 + fans / 1_000_000

    def _extract_bilibili_mid(self, query: str) -> str:
        match = re.search(r"space\.bilibili\.com/(\d+)", query)
        return match.group(1) if match else ""

    def _normalize_youtube_home(self, query: str) -> str:
        if query.startswith("@"):
            return f"https://www.youtube.com/{query}"
        if query.startswith("UC"):
            return f"https://www.youtube.com/channel/{query}"
        return query

    def _ensure_suffix(self, url: str, suffix: str) -> str:
        url = url.rstrip("/")
        if url.endswith(f"/{suffix}"):
            return url
        return f"{url}/{suffix}"

    def _youtube_channel_url(self, renderer: dict[str, object]) -> str:
        endpoint = renderer.get("navigationEndpoint") or {}
        metadata = endpoint.get("commandMetadata") or {}
        web = metadata.get("webCommandMetadata") or {}
        path = str(web.get("url") or "")
        if path:
            return f"https://www.youtube.com{path}".rstrip("/")
        channel_id = renderer.get("channelId")
        return f"https://www.youtube.com/channel/{channel_id}"

    def _youtube_handle(self, renderer: dict[str, object]) -> str:
        handle_text = self._youtube_text(renderer.get("shortBylineText")) or self._youtube_text(renderer.get("ownerText"))
        if handle_text.startswith("@"):
            return handle_text
        url = self._youtube_channel_url(renderer)
        match = re.search(r"/(@[^/?#]+)", url)
        return match.group(1) if match else handle_text

    def _youtube_text(self, node: object) -> str:
        if not node:
            return ""
        if isinstance(node, str):
            return clean_html(node)
        if isinstance(node, dict):
            if "simpleText" in node:
                return clean_html(str(node["simpleText"]))
            if "runs" in node:
                return "".join(self._youtube_text(item) for item in node["runs"])
            if "text" in node:
                return clean_html(str(node["text"]))
        if isinstance(node, list):
            return "".join(self._youtube_text(item) for item in node)
        return clean_html(str(node))

    def _extract_json_blob(self, html: str, variable_name: str) -> dict[str, object]:
        patterns = [
            rf"{variable_name}\s*=\s*(\{{.*?\}});",
            rf"var\s+{variable_name}\s*=\s*(\{{.*?\}});",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.DOTALL)
            if match:
                return json.loads(match.group(1))
        raise ValueError(f"Could not extract {variable_name} from page.")

    def _walk_channel_renderers(self, data: object):
        if isinstance(data, dict):
            renderer = data.get("channelRenderer")
            if renderer:
                yield renderer
            for value in data.values():
                yield from self._walk_channel_renderers(value)
        elif isinstance(data, list):
            for item in data:
                yield from self._walk_channel_renderers(item)

    def _extract_info(self, url: str) -> dict[str, object]:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False) or {}

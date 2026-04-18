from __future__ import annotations

import math
import subprocess
from datetime import UTC
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from yt_dlp import YoutubeDL

from ..schema import CreatorRef, VideoInfo
from ..utils import USER_AGENT, ensure_dir, hotness_score, parse_upload_datetime
from .platforms import CreatorResolver


class MediaDownloader:
    def __init__(self, ffmpeg_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._resolver = CreatorResolver()
        self._bili_dm_params = {
            "dm_img_list": "[]",
            "dm_img_str": "V2ViR0wgMS4wIChPcGVuR0wgRVMgMi4wIENocm9taXVtKQ",
            "dm_cover_img_str": (
                "QU5HTEUgKEludGVsLCBJbnRlbChSKSBVSEQgR3JhcGhpY3MgKDB4MDAwMEE3QTgpIERpcmVjdDNEMTEg"
                "dnNfNV8wIHBzXzVfMCwgRDNEMTEpR29vZ2xlIEluYy4gKEludGVsKQ"
            ),
            "dm_img_inter": '{"ds":[],"wh":[6574,3858,58],"of":[330,660,330]}',
        }
        self._bili_headers_seeded = False

    def rank_creator_videos(self, creator: CreatorRef, scan_limit: int = 30) -> list[VideoInfo]:
        if creator.platform == "bilibili" and creator.creator_id.isdigit():
            return self._rank_bilibili_creator_videos(creator, scan_limit)
        entries = self._list_entries(creator.video_tab_url, creator.platform, scan_limit)
        videos: list[VideoInfo] = []
        seen: set[str] = set()
        for entry in entries:
            url = self._entry_url(entry, creator.platform)
            if not url:
                continue
            try:
                info = self._extract_video_info(url)
            except Exception:
                continue
            video = self._to_video_info(info, creator.platform)
            if not video.video_id or video.video_id in seen:
                continue
            seen.add(video.video_id)
            videos.append(video)
        videos.sort(key=lambda item: item.hotness, reverse=True)
        return videos

    def _rank_bilibili_creator_videos(self, creator: CreatorRef, scan_limit: int) -> list[VideoInfo]:
        seen: dict[str, VideoInfo] = {}
        page_size = min(max(scan_limit, 3), 40)
        page_count = max(math.ceil(max(scan_limit, 1) / page_size), 1)
        for order in ("click", "pubdate"):
            for page in range(1, page_count + 1):
                try:
                    items = self._fetch_bilibili_vlist(
                        creator.creator_id,
                        order=order,
                        page=page,
                        page_size=page_size,
                    )
                except Exception:
                    continue
                for item in items:
                    video = self._video_from_bilibili_arc(item, creator)
                    if not video.video_id:
                        continue
                    existing = seen.get(video.video_id)
                    if existing is None or video.hotness > existing.hotness:
                        seen[video.video_id] = video
        videos = list(seen.values())
        if not videos:
            videos = self._rank_bilibili_videos_ytdlp(creator, scan_limit)
        videos.sort(key=lambda item: item.hotness, reverse=True)
        return videos

    def _rank_bilibili_videos_ytdlp(self, creator: CreatorRef, scan_limit: int) -> list[VideoInfo]:
        try:
            entries = self._list_entries(creator.video_tab_url, "bilibili", scan_limit)
        except Exception:
            return []
        videos: list[VideoInfo] = []
        seen: set[str] = set()
        for entry in entries:
            video_id = str(entry.get("id") or "")
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            url = self._entry_url(entry, "bilibili")
            if not url:
                continue
            published = parse_upload_datetime(timestamp=entry.get("timestamp"))
            view_count = int(entry.get("view_count") or 0)
            videos.append(VideoInfo(
                platform="bilibili",
                video_id=video_id,
                title=str(entry.get("title") or video_id),
                url=url,
                uploader=creator.name,
                duration_sec=int(entry.get("duration") or 0),
                published_at=published.astimezone(UTC).isoformat() if published else "",
                view_count=view_count,
                hotness=hotness_score(views=view_count, published_at=published),
            ))
        return videos

    def download_audio(self, video: VideoInfo, output_dir: Path) -> Path:
        ensure_dir(output_dir)
        wav_path = output_dir / f"{video.video_id}.wav"
        if wav_path.exists():
            return wav_path
        info = self._extract_video_info(video.url)
        format_spec = self._pick_audio_only_format(info)
        opts = {
            "format": format_spec,
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "ffmpeg_location": self.ffmpeg_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                },
            ],
            "http_headers": {
                "User-Agent": USER_AGENT,
                "Referer": "https://www.bilibili.com/",
            },
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 30,
        }
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(video.url, download=True)
        if wav_path.exists():
            return wav_path
        candidates = sorted(output_dir.glob(f"{video.video_id}.*"))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(f"Audio download finished but no file was found for {video.video_id}.")

    def transcode_local_media(self, source_path: Path, output_dir: Path, target_stem: str) -> Path:
        ensure_dir(output_dir)
        resolved_source = source_path.resolve()
        if not resolved_source.exists() or not resolved_source.is_file():
            raise FileNotFoundError(f"Local media file was not found: {resolved_source}")
        wav_path = output_dir / f"{target_stem}.wav"
        if wav_path.exists():
            return wav_path
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(resolved_source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not wav_path.exists():
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(
                f"Failed to convert local media to audio: {resolved_source}\n{stderr}",
            )
        return wav_path

    def _pick_audio_only_format(self, info: dict[str, object]) -> str:
        formats = info.get("formats") or []
        audio_only = [
            item
            for item in formats
            if self._is_audio_only_format(item)
        ]
        if not audio_only:
            video_id = str(info.get("id") or "unknown")
            raise ValueError(
                "No audio-only stream is available for "
                f"{video_id}. Refusing to download a muxed video file.",
            )
        best = max(audio_only, key=self._audio_format_score)
        format_id = str(best.get("format_id") or "").strip()
        if not format_id:
            raise ValueError("Audio-only stream selection failed because the chosen format has no format_id.")
        return format_id

    def _is_audio_only_format(self, item: object) -> bool:
        if not isinstance(item, dict):
            return False
        vcodec = str(item.get("vcodec") or "").lower()
        acodec = str(item.get("acodec") or "").lower()
        return vcodec == "none" and acodec not in {"", "none"}

    def _audio_format_score(self, item: dict[str, object]) -> tuple[float, float, float, float, int]:
        ext = str(item.get("ext") or "").lower()
        ext_rank = {
            "m4a": 4,
            "webm": 3,
            "opus": 3,
            "mp3": 2,
            "aac": 2,
        }.get(ext, 1)
        abr = self._as_float(item.get("abr"))
        asr = self._as_float(item.get("asr"))
        tbr = self._as_float(item.get("tbr"))
        filesize = self._as_float(item.get("filesize") or item.get("filesize_approx"))
        return (abr, asr, tbr, filesize, ext_rank)

    def _as_float(self, value: object) -> float:
        try:
            return float(value or 0)
        except Exception:
            return 0.0

    def _list_entries(self, url: str, platform: str, scan_limit: int) -> list[dict[str, object]]:
        opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": True,
            "playlistend": scan_limit,
            "lazy_playlist": True,
            "http_headers": {
                "User-Agent": USER_AGENT,
                "Referer": "https://www.bilibili.com/",
            },
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        entries = list(info.get("entries") or [])
        if platform == "youtube" and not entries and not url.endswith("/streams"):
            with YoutubeDL({**opts, "playlistend": scan_limit}) as ydl:
                info = ydl.extract_info(url.replace("/videos", "/streams"), download=False) or {}
            entries = list(info.get("entries") or [])
        return entries

    def _entry_url(self, entry: dict[str, object], platform: str) -> str:
        raw = str(entry.get("webpage_url") or entry.get("url") or "")
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        video_id = str(entry.get("id") or raw)
        if not video_id:
            return ""
        if platform == "youtube":
            return f"https://www.youtube.com/watch?v={video_id}"
        if video_id.startswith("BV"):
            return f"https://www.bilibili.com/video/{video_id}"
        return raw

    def _extract_video_info(self, url: str) -> dict[str, object]:
        opts = {
            "quiet": True,
            "skip_download": True,
            "http_headers": {
                "User-Agent": USER_AGENT,
                "Referer": "https://www.bilibili.com/",
            },
        }
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    def _fetch_bilibili_vlist(self, mid: str, order: str, page: int, page_size: int) -> list[dict[str, object]]:
        params = {
            "mid": mid,
            "pn": page,
            "ps": page_size,
            "tid": 0,
            "special_type": "",
            "order": order,
            "index": 0,
            "keyword": "",
            "order_avoided": "true",
            "platform": "web",
            "web_location": "333.1387",
            **self._bili_dm_params,
        }
        payload = self._request_bilibili_arc_search(params)

        if (payload.get("code") or 0) != 0:
            payload = self._fetch_bilibili_vlist_unsigned(mid, order, page, page_size)

        if (payload.get("code") or 0) != 0:
            try:
                self._seed_bilibili_browser_session(mid)
                params.update(self._bili_dm_params)
                payload = self._request_bilibili_arc_search(params)
            except Exception:
                pass

        data = payload.get("data") or {}
        listing = data.get("list") or {}
        return listing.get("vlist") or []

    def _fetch_bilibili_vlist_unsigned(self, mid: str, order: str, page: int, page_size: int) -> dict[str, object]:
        params = {
            "mid": mid,
            "pn": page,
            "ps": page_size,
            "order": order,
            "tid": 0,
            "keyword": "",
            "jsonp": "jsonp",
        }
        try:
            self._resolver._prime_bilibili_cookies()
            response = self._resolver.session.get(
                "https://api.bilibili.com/x/space/arc/search",
                params=params,
                timeout=20,
            )
            return response.json()
        except Exception:
            return {"code": -1}

    def _request_bilibili_arc_search(self, params: dict[str, object]) -> dict[str, object]:
        self._resolver._prime_bilibili_cookies()
        response = self._resolver.session.get(
            "https://api.bilibili.com/x/space/wbi/arc/search",
            params=self._resolver._sign_wbi(params),
            timeout=20,
        )
        try:
            return response.json()
        except Exception:
            return {"code": response.status_code or -1, "message": response.text[:200]}

    def _seed_bilibili_browser_session(self, mid: str) -> None:
        if self._bili_headers_seeded:
            return
        from playwright.sync_api import sync_playwright

        executable_path = self._find_browser_executable()
        capture: dict[str, object] = {}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=executable_path)
            context = browser.new_context(
                viewport={"width": 1400, "height": 1800},
                user_agent=USER_AGENT,
                locale="zh-CN",
            )
            page = context.new_page()

            def on_response(response) -> None:
                if "/x/space/wbi/arc/search" not in response.url or capture:
                    return
                capture["url"] = response.url
                capture["headers"] = dict(response.request.headers)

            page.on("response", on_response)
            page.goto(f"https://space.bilibili.com/{mid}/upload/video", wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(4000)
            for cookie in context.cookies():
                self._resolver.session.cookies.set(
                    cookie["name"],
                    cookie["value"],
                    domain=cookie["domain"],
                    path=cookie["path"],
                )
            browser.close()

        url = str(capture.get("url") or "")
        if url:
            parsed = parse_qs(urlsplit(url).query)
            for key in ("dm_img_list", "dm_img_str", "dm_cover_img_str", "dm_img_inter"):
                if parsed.get(key):
                    self._bili_dm_params[key] = parsed[key][0]
        headers = capture.get("headers") or {}
        if headers:
            for key in ("referer", "accept-language", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform"):
                value = headers.get(key)
                if value:
                    self._resolver.session.headers[key] = value
        self._bili_headers_seeded = True

    def _find_browser_executable(self) -> str | None:
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    def _video_from_bilibili_arc(self, item: dict[str, object], creator: CreatorRef) -> VideoInfo:
        created = int(item.get("created") or 0)
        published = parse_upload_datetime(timestamp=created)
        duration = self._parse_duration(item.get("length"))
        return VideoInfo(
            platform="bilibili",
            video_id=str(item.get("bvid") or ""),
            title=str(item.get("title") or ""),
            url=f"https://www.bilibili.com/video/{item.get('bvid')}",
            uploader=str(item.get("author") or creator.name),
            duration_sec=duration,
            published_at=published.astimezone(UTC).isoformat() if published else "",
            description=str(item.get("description") or ""),
            view_count=int(item.get("play") or 0),
            like_count=0,
            comment_count=int(item.get("comment") or 0),
            favorite_count=0,
            share_count=0,
            hotness=hotness_score(
                views=int(item.get("play") or 0),
                comments=int(item.get("comment") or 0),
                published_at=published,
            ),
        )

    def _parse_duration(self, value: object) -> int:
        if isinstance(value, int):
            return value
        text = str(value or "").strip()
        if not text:
            return 0
        parts = [int(part) for part in text.split(":") if part.isdigit()]
        if not parts:
            return 0
        total = 0
        for part in parts:
            total = total * 60 + part
        return total

    def _to_video_info(self, info: dict[str, object], platform: str) -> VideoInfo:
        published = parse_upload_datetime(
            timestamp=info.get("timestamp"),
            upload_date=str(info.get("upload_date") or ""),
        )
        hotness = hotness_score(
            views=int(info.get("view_count") or 0),
            likes=int(info.get("like_count") or 0),
            comments=int(info.get("comment_count") or 0),
            favorites=int(info.get("favorite_count") or 0),
            shares=int(info.get("repost_count") or 0),
            published_at=published,
        )
        published_text = ""
        if published is not None:
            published_text = published.astimezone(UTC).isoformat()
        return VideoInfo(
            platform=platform,
            video_id=str(info.get("id") or ""),
            title=str(info.get("title") or ""),
            url=str(info.get("webpage_url") or info.get("original_url") or ""),
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            duration_sec=int(info.get("duration") or 0),
            published_at=published_text,
            description=str(info.get("description") or ""),
            view_count=int(info.get("view_count") or 0),
            like_count=int(info.get("like_count") or 0),
            comment_count=int(info.get("comment_count") or 0),
            favorite_count=int(info.get("favorite_count") or 0),
            share_count=int(info.get("repost_count") or 0),
            hotness=hotness,
        )

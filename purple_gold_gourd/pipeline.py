from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .chat.skillgen import SkillBuilder
from .config import AppConfig
from .language import (
    detect_creator_language,
    detect_text_language,
    detect_transcript_language,
)
from .media.downloader import MediaDownloader
from .media.platforms import CreatorResolver
from .plugins import get_stt_plugin, get_tts_plugin
from .schema import ProfileManifest, TranscriptChunk, TranscriptFile, VideoInfo
from .utils import ensure_dir, read_json, sha256_file, sha256_text, slugify, write_json


class BuildPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.resolver = CreatorResolver()
        self.downloader = MediaDownloader(config.ffmpeg_path)
        self.stt = get_stt_plugin(config)
        self.tts = get_tts_plugin(config)
        self.transcriber = self.stt.create_transcriber()
        self.skill_builder = SkillBuilder(
            config.lm_base_url,
            config.lm_api_key,
            config.lm_model,
            max_context_tokens=config.lm_max_context_tokens,
            max_completion_tokens=config.lm_max_completion_tokens,
        )

    def build(
        self,
        query: str,
        platform: str = "auto",
        top_n: int = 10,
        scan_limit: int = 30,
        series_numbers: list[int] | None = None,
        include_title_keywords: list[str] | None = None,
        limit_to_target_videos: bool = False,
        local_media_paths: list[str | Path] | None = None,
        rebuild: bool = False,
    ) -> ProfileManifest:
        selected_series_numbers = self._normalize_series_numbers(series_numbers)
        title_keywords = self._normalize_title_keywords(include_title_keywords)
        normalized_local_media_paths = self._normalize_local_media_paths(local_media_paths)
        if not rebuild:
            matched = self._match_cached(query, platform)
            if matched is not None:
                return self._refresh_cached(
                    matched,
                    top_n=top_n,
                    scan_limit=scan_limit,
                    series_numbers=selected_series_numbers,
                    include_title_keywords=title_keywords,
                    limit_to_target_videos=limit_to_target_videos,
                    local_media_paths=normalized_local_media_paths,
                    rebuild=False,
                )

        creator = self.resolver.resolve(query, platform=platform)
        creator_slug = slugify(f"{creator.platform}-{creator.creator_id}-{creator.name}")
        creator_dir = ensure_dir(self.config.creators_dir / creator_slug)
        manifest_path = creator_dir / "manifest.json"

        if manifest_path.exists() and not rebuild:
            return self._refresh_cached(
                manifest_path,
                top_n=top_n,
                scan_limit=scan_limit,
                series_numbers=selected_series_numbers,
                include_title_keywords=title_keywords,
                limit_to_target_videos=limit_to_target_videos,
                local_media_paths=normalized_local_media_paths,
                rebuild=False,
            )

        manifest = ProfileManifest(
            creator=creator,
            creator_slug=creator_slug,
            creator_dir=str(creator_dir),
            videos=[],
            transcript_paths=[],
            skill_path=str(creator_dir / "skill" / "skill.md"),
            selected_series_numbers=selected_series_numbers,
            source_state_signature="",
        )
        videos = self._resolve_target_videos(
            manifest=manifest,
            top_n=top_n,
            scan_limit=scan_limit,
            series_numbers=selected_series_numbers,
            include_title_keywords=title_keywords,
        )
        return self._materialize_profile(
            manifest=manifest,
            target_videos=videos,
            local_media_paths=normalized_local_media_paths,
            limit_to_target_videos=limit_to_target_videos,
            rebuild=rebuild,
        )

    def _match_cached(self, query: str, platform: str) -> Path | None:
        needle = query.strip().lower()
        if not needle:
            return None
        matches: list[tuple[Path, str]] = []
        creators_dir = self.config.creators_dir
        if not creators_dir.exists():
            return None
        for creator_dir in sorted(creators_dir.iterdir()):
            manifest_path = creator_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            data = read_json(manifest_path)
            if not data:
                continue
            creator_data = data.get("creator") or {}
            name = str(creator_data.get("name") or "").lower()
            creator_slug = str(data.get("creator_slug") or creator_dir.name).lower()
            creator_id = str(creator_data.get("creator_id") or "").lower()
            creator_query = str(creator_data.get("query") or "").lower()
            homepage_url = str(creator_data.get("homepage_url") or "").lower()
            creator_platform = str(creator_data.get("platform") or "")
            if platform != "auto" and creator_platform != platform:
                continue
            candidates = [name, creator_slug, creator_id, creator_query, homepage_url, creator_dir.name.lower()]
            if any(needle == item or needle in item for item in candidates if item):
                matches.append((manifest_path, str(creator_data.get("name") or creator_dir.name)))
        if len(matches) == 1:
            return matches[0][0]
        if len(matches) > 1:
            options = ", ".join(name for _, name in matches)
            raise ValueError(
                f"'{query}' matches multiple cached profiles: {options}. "
                "Please be more specific.",
            )
        return None

    def _refresh_cached(
        self,
        manifest_path: Path,
        top_n: int,
        scan_limit: int,
        series_numbers: list[int],
        include_title_keywords: list[str],
        limit_to_target_videos: bool,
        local_media_paths: list[Path],
        rebuild: bool,
    ) -> ProfileManifest:
        existing = ProfileManifest.from_dict(read_json(manifest_path))
        target_series_numbers = series_numbers
        target_videos = self._resolve_target_videos(
            manifest=existing,
            top_n=top_n,
            scan_limit=scan_limit,
            series_numbers=target_series_numbers,
            include_title_keywords=include_title_keywords,
        )
        refreshed = replace(existing, selected_series_numbers=target_series_numbers)
        return self._materialize_profile(
            manifest=refreshed,
            target_videos=target_videos,
            local_media_paths=local_media_paths,
            limit_to_target_videos=limit_to_target_videos,
            rebuild=rebuild,
        )

    def load_transcripts(self, manifest: ProfileManifest) -> list[TranscriptFile]:
        transcripts: list[TranscriptFile] = []
        for path in manifest.transcript_paths:
            transcripts.append(self._load_transcript(Path(path)))
        for path in self._document_paths(Path(manifest.creator_dir) / "documents"):
            transcripts.append(self._load_document(path))
        return transcripts

    def _load_transcript(self, path: Path) -> TranscriptFile:
        transcript = TranscriptFile.from_dict(read_json(path))
        detected = detect_transcript_language(transcript.language, "\n".join([transcript.raw_text, transcript.full_text]))
        changed = detected != transcript.language
        if changed:
            transcript.language = detected
        if not transcript.subtitle_text and transcript.chunks:
            transcript.subtitle_text = self.transcriber.build_subtitles(transcript.chunks)
            changed = True
        if changed:
            write_json(path, transcript.to_dict())
        self._write_srt(path, transcript.subtitle_text)
        return transcript

    def _write_srt(self, json_path: Path, subtitle_text: str) -> None:
        if not subtitle_text:
            return
        srt_path = json_path.with_suffix(".srt")
        srt_path.write_text(subtitle_text, encoding="utf-8")

    def _attach_languages(
        self,
        creator,
        videos,
        transcripts: list[TranscriptFile],
    ) -> bool:
        changed = False
        transcript_by_id = {transcript.video_id: transcript for transcript in transcripts}
        for video in videos:
            transcript = transcript_by_id.get(video.video_id)
            detected = transcript.language if transcript else detect_text_language("\n".join([video.title, video.description]))
            detected = detected or "en"
            if video.language != detected:
                video.language = detected
                changed = True
        creator_language = detect_creator_language(
            metadata_text="\n".join([creator.name, creator.bio]),
            video_texts=["\n".join([video.title, video.description]) for video in videos],
            transcript_languages=[transcript.language for transcript in transcripts],
        )
        if creator.language != creator_language:
            creator.language = creator_language
            changed = True
        return changed

    def _normalize_series_numbers(self, series_numbers: list[int] | None) -> list[int]:
        if not series_numbers:
            return []
        normalized: list[int] = []
        seen: set[int] = set()
        for raw in series_numbers:
            value = int(raw)
            if value <= 0:
                raise ValueError("Series numbers must be positive 1-based integers.")
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _normalize_title_keywords(self, keywords: list[str] | None) -> list[str]:
        if not keywords:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in keywords:
            text = str(raw).strip()
            if not text:
                continue
            folded = text.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            normalized.append(text)
        return normalized

    def _normalize_local_media_paths(self, local_media_paths: list[str | Path] | None) -> list[Path]:
        if not local_media_paths:
            return []
        normalized: list[Path] = []
        seen: set[Path] = set()
        for raw in local_media_paths:
            path = Path(raw).expanduser().resolve()
            if path in seen:
                continue
            seen.add(path)
            normalized.append(path)
        return normalized

    def _resolve_target_videos(
        self,
        manifest: ProfileManifest,
        top_n: int,
        scan_limit: int,
        series_numbers: list[int],
        include_title_keywords: list[str] | None = None,
    ) -> list:
        include_title_keywords = include_title_keywords or []
        cache_path = self._target_videos_cache_path(manifest, top_n, scan_limit, series_numbers, include_title_keywords)
        try:
            if series_numbers:
                videos = self._select_target_videos(
                    creator=manifest.creator,
                    top_n=top_n,
                    scan_limit=scan_limit,
                    series_numbers=series_numbers,
                    include_title_keywords=include_title_keywords,
                )
                self._write_target_videos_cache(cache_path, videos)
                return videos

            cached_videos = self._cached_videos_for_creator(manifest)
            if cached_videos and len(cached_videos) >= top_n:
                videos = cached_videos[:top_n]
                self._write_target_videos_cache(cache_path, videos)
                return videos

            videos = self._select_target_videos(
                creator=manifest.creator,
                top_n=top_n,
                scan_limit=scan_limit,
                series_numbers=[],
                include_title_keywords=include_title_keywords,
            )
            self._write_target_videos_cache(cache_path, videos)
            return videos
        except Exception:
            cached_targets = self._read_target_videos_cache(cache_path)
            expected_count = len(series_numbers) if series_numbers else top_n
            if len(cached_targets) >= expected_count:
                return cached_targets
            raise

    def _target_videos_cache_path(
        self,
        manifest: ProfileManifest,
        top_n: int,
        scan_limit: int,
        series_numbers: list[int],
        include_title_keywords: list[str],
    ) -> Path:
        key = sha256_text(
            "\n".join(
                [
                    f"top={top_n}",
                    f"scan={scan_limit}",
                    "series=" + ",".join(str(item) for item in series_numbers),
                    "keywords=" + "\0".join(include_title_keywords),
                ],
            ),
        )[:16]
        return Path(manifest.creator_dir) / "targets" / f"{key}.json"

    def _read_target_videos_cache(self, path: Path) -> list[VideoInfo]:
        data = read_json(path, default=[])
        if not isinstance(data, list):
            return []
        videos: list[VideoInfo] = []
        for item in data:
            try:
                videos.append(VideoInfo.from_dict(item))
            except Exception:
                continue
        return self._dedupe_video_infos(videos)

    def _write_target_videos_cache(self, path: Path, videos: list[VideoInfo]) -> None:
        ensure_dir(path.parent)
        write_json(path, [video.to_dict() for video in self._dedupe_video_infos(videos)])

    def _select_target_videos(
        self,
        creator,
        top_n: int,
        scan_limit: int,
        series_numbers: list[int],
        include_title_keywords: list[str] | None = None,
    ) -> list:
        required_count = max(series_numbers or [top_n, 1])
        listed_videos = self.downloader.list_creator_videos(creator, scan_limit=required_count)
        if not listed_videos:
            raise ValueError(f"No videos found for {creator.name}.")
        if not series_numbers:
            return listed_videos[:top_n]
        if max(series_numbers) > len(listed_videos):
            raise ValueError(
                f"Requested series number {max(series_numbers)} is out of range for {creator.name}. "
                f"Only {len(listed_videos)} videos were found.",
            )
        return [listed_videos[number - 1] for number in series_numbers]

    def _with_title_keyword_videos(
        self,
        creator,
        selected: list[VideoInfo],
        scan_limit: int,
        keywords: list[str],
    ) -> list[VideoInfo]:
        if not keywords:
            return selected
        result = list(selected)
        seen = {video.video_id for video in result}
        keyword_scan_limit = max(scan_limit, 120)
        for keyword in keywords:
            matches = self.downloader.find_creator_videos_by_title_keyword(
                creator,
                keyword,
                scan_limit=keyword_scan_limit,
            )
            for video in matches:
                if video.video_id in seen:
                    continue
                seen.add(video.video_id)
                result.append(video)
        return result

    def _cached_videos_for_creator(self, manifest: ProfileManifest) -> list:
        creator_dir = Path(manifest.creator_dir)
        transcripts_dir = creator_dir / "transcripts"
        if not transcripts_dir.exists():
            return []

        existing_videos = {video.video_id: video for video in manifest.videos}
        ordered_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for raw_path in manifest.transcript_paths:
            path = Path(raw_path)
            if path.exists() and path not in seen_paths:
                ordered_paths.append(path)
                seen_paths.add(path)
        for transcript_path in sorted(transcripts_dir.glob("*.json")):
            if transcript_path not in seen_paths:
                ordered_paths.append(transcript_path)
                seen_paths.add(transcript_path)

        videos: list = []
        for transcript_path in ordered_paths:
            data = read_json(transcript_path)
            if not data:
                continue
            transcript = TranscriptFile.from_dict(data)
            if transcript.source_type == "document":
                continue
            video = existing_videos.get(transcript.video_id)
            if video is None:
                video = self._video_info_from_transcript(manifest.creator, transcript)
            videos.append(video)
        return videos

    def _video_info_from_transcript(self, creator, transcript: TranscriptFile) -> VideoInfo:
        return VideoInfo(
            platform="local" if transcript.source_type == "local_media" else creator.platform,
            video_id=transcript.video_id,
            title=transcript.video_title,
            url=transcript.source_path or transcript.video_url,
            uploader=creator.name,
            language=transcript.language,
        )

    def _all_transcript_json_paths(self, transcripts_dir: Path) -> list[Path]:
        if not transcripts_dir.exists():
            return []
        return sorted(path for path in transcripts_dir.glob("*.json") if path.is_file())

    def _document_paths(self, documents_dir: Path) -> list[Path]:
        if not documents_dir.exists():
            return []
        return sorted(path for path in documents_dir.rglob("*.md") if path.is_file())

    def _load_document(self, path: Path) -> TranscriptFile:
        text = path.read_text(encoding="utf-8").strip()
        language = detect_text_language(text) or "en"
        document_id = self._document_id(path)
        title = path.stem or path.name
        chunks = self._document_chunks(
            document_id=document_id,
            title=title,
            source_path=path,
            text=text,
        )
        return TranscriptFile(
            video_id=document_id,
            video_title=title,
            video_url=str(path.resolve()),
            language=language,
            full_text=text,
            raw_text=text,
            audio_path="",
            chunks=chunks,
            subtitle_text="",
            source_type="document",
            source_path=str(path.resolve()),
        )

    def _document_id(self, path: Path) -> str:
        return f"document-{slugify(path.stem)}-{sha256_text(str(path.resolve()))[:10]}"

    def _document_chunks(
        self,
        document_id: str,
        title: str,
        source_path: Path,
        text: str,
    ) -> list[TranscriptChunk]:
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        groups: list[str] = []
        current: list[str] = []
        current_chars = 0
        for paragraph in paragraphs:
            paragraph_chars = len(paragraph)
            if current and current_chars + paragraph_chars > 1_200:
                groups.append("\n\n".join(current).strip())
                current = []
                current_chars = 0
            current.append(paragraph)
            current_chars += paragraph_chars + 2
        if current:
            groups.append("\n\n".join(current).strip())

        chunks: list[TranscriptChunk] = []
        for index, group in enumerate(groups):
            chunks.append(
                TranscriptChunk(
                    video_id=document_id,
                    video_title=title,
                    video_url=str(source_path.resolve()),
                    start_ms=index * 1000,
                    end_ms=(index + 1) * 1000,
                    text=group,
                    source_type="document",
                    source_path=str(source_path.resolve()),
                ),
            )
        return chunks

    def _materialize_local_media(
        self,
        creator,
        downloads_dir: Path,
        transcripts_dir: Path,
        local_media_paths: list[Path],
        rebuild: bool,
    ) -> tuple[list[VideoInfo], list[str]]:
        videos: list[VideoInfo] = []
        transcript_paths: list[str] = []
        for source_path in local_media_paths:
            media_id = self._local_media_id(source_path)
            transcript_path = transcripts_dir / f"{media_id}.json"
            if transcript_path.exists() and not rebuild:
                transcript = self._load_transcript(transcript_path)
                if transcript.audio_path and not Path(transcript.audio_path).exists():
                    audio_path = self.downloader.transcode_local_media(source_path, downloads_dir, media_id)
                    transcript.audio_path = str(audio_path)
                    write_json(transcript_path, transcript.to_dict())
            else:
                audio_path = self.downloader.transcode_local_media(source_path, downloads_dir, media_id)
                video = VideoInfo(
                    platform="local",
                    video_id=media_id,
                    title=source_path.stem,
                    url=str(source_path.resolve()),
                    uploader=creator.name,
                )
                transcript = self.transcriber.transcribe(audio_path, video)
                transcript.source_type = "local_media"
                transcript.source_path = str(source_path.resolve())
                transcript.video_url = str(source_path.resolve())
                write_json(transcript_path, transcript.to_dict())
                self._write_srt(transcript_path, transcript.subtitle_text)
            transcript_paths.append(str(transcript_path))
            videos.append(self._video_info_from_transcript(creator, transcript))
        return videos, transcript_paths

    def _local_media_id(self, source_path: Path) -> str:
        return f"local-media-{slugify(source_path.stem)}-{sha256_text(str(source_path.resolve()))[:10]}"

    def _existing_local_media(
        self,
        creator,
        transcripts_dir: Path,
        exclude_paths: set[Path] | None = None,
    ) -> tuple[list[VideoInfo], list[str]]:
        exclude_paths = exclude_paths or set()
        videos: list[VideoInfo] = []
        transcript_paths: list[str] = []
        for path in self._all_transcript_json_paths(transcripts_dir):
            if path in exclude_paths:
                continue
            data = read_json(path)
            if not data:
                continue
            transcript = TranscriptFile.from_dict(data)
            if transcript.source_type != "local_media":
                continue
            transcript_paths.append(str(path))
            videos.append(self._video_info_from_transcript(creator, transcript))
        return videos, transcript_paths

    def _source_state_signature(
        self,
        transcript_paths: list[Path],
        document_paths: list[Path],
    ) -> str:
        parts: list[str] = []
        for path in sorted(transcript_paths):
            if path.exists():
                parts.append(f"transcript::{path.resolve()}::{sha256_file(path)}")
        for path in sorted(document_paths):
            if path.exists():
                parts.append(f"document::{path.resolve()}::{sha256_file(path)}")
        return sha256_text("\n".join(parts))

    def _dedupe_video_infos(self, videos: list[VideoInfo]) -> list[VideoInfo]:
        deduped: list[VideoInfo] = []
        seen: set[str] = set()
        for video in videos:
            if video.video_id in seen:
                continue
            seen.add(video.video_id)
            deduped.append(video)
        return deduped

    def _dedupe_paths(self, paths: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            resolved = str(Path(raw_path))
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)
        return deduped

    def _materialize_profile(
        self,
        manifest: ProfileManifest,
        target_videos: list,
        local_media_paths: list[Path],
        limit_to_target_videos: bool,
        rebuild: bool,
    ) -> ProfileManifest:
        target_videos = self._dedupe_video_infos(target_videos)
        creator_dir = Path(manifest.creator_dir)
        downloads_dir = ensure_dir(creator_dir / "downloads")
        transcripts_dir = ensure_dir(creator_dir / "transcripts")
        documents_dir = ensure_dir(creator_dir / "documents")
        skill_dir = ensure_dir(creator_dir / "skill")
        voice_dir = ensure_dir(creator_dir / "voice")
        manifest_path = creator_dir / "manifest.json"
        videos_path = creator_dir / "videos.json"

        remote_transcript_paths: list[str] = []
        for video in target_videos:
            transcript_path = transcripts_dir / f"{video.video_id}.json"
            if transcript_path.exists() and not rebuild:
                self._load_transcript(transcript_path)
            else:
                audio_path = self.downloader.download_audio(video, downloads_dir)
                transcript = self.transcriber.transcribe(audio_path, video)
                write_json(transcript_path, transcript.to_dict())
                self._write_srt(transcript_path, transcript.subtitle_text)
            remote_transcript_paths.append(str(transcript_path))

        local_videos, explicit_local_transcript_paths = self._materialize_local_media(
            manifest.creator,
            downloads_dir,
            transcripts_dir,
            local_media_paths,
            rebuild,
        )
        existing_local_videos, existing_local_transcript_paths = self._existing_local_media(
            manifest.creator,
            transcripts_dir,
            exclude_paths={Path(path) for path in explicit_local_transcript_paths},
        )

        if manifest.selected_series_numbers or limit_to_target_videos:
            transcript_paths = self._dedupe_paths(
                remote_transcript_paths + existing_local_transcript_paths + explicit_local_transcript_paths,
            )
        else:
            transcript_paths = [str(path) for path in self._all_transcript_json_paths(transcripts_dir)]

        transcripts = [self._load_transcript(Path(path)) for path in transcript_paths]
        document_paths = self._document_paths(documents_dir)
        documents = [self._load_document(path) for path in document_paths]

        all_videos = self._dedupe_video_infos(target_videos + existing_local_videos + local_videos)
        self._attach_languages(manifest.creator, all_videos, transcripts)

        source_state_signature = self._source_state_signature(
            transcript_paths=[Path(path) for path in transcript_paths],
            document_paths=document_paths,
        )
        should_refresh_skill = (
            rebuild
            or source_state_signature != manifest.source_state_signature
            or not Path(manifest.skill_path).exists()
        )
        if should_refresh_skill:
            skill_path = str(
                self.skill_builder.build(
                    manifest.creator,
                    all_videos,
                    transcripts,
                    documents,
                    skill_dir,
                ),
            )
        else:
            skill_path = manifest.skill_path

        # Voice sample is set exclusively via `set-voice` CLI; never auto-selected.
        if manifest.voice_sample and Path(manifest.voice_sample.audio_path).exists():
            voice_sample = manifest.voice_sample
        else:
            voice_sample = None

        updated = replace(
            manifest,
            videos=all_videos,
            transcript_paths=transcript_paths,
            skill_path=skill_path,
            source_state_signature=source_state_signature,
            voice_sample=voice_sample,
        )
        write_json(videos_path, [video.to_dict() for video in all_videos])
        write_json(manifest_path, updated.to_dict())
        return updated

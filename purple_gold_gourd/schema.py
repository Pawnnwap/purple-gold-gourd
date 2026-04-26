from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CreatorRef:
    platform: str
    creator_id: str
    name: str
    homepage_url: str
    video_tab_url: str
    query: str
    video_count: int = 0
    handle: str = ""
    bio: str = ""
    language: str = ""
    followers: int = 0
    avatar_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreatorRef:
        payload = dict(data)
        payload.setdefault("avatar_url", "")
        return cls(**payload)


@dataclass(slots=True)
class VideoInfo:
    platform: str
    video_id: str
    title: str
    url: str
    uploader: str
    duration_sec: int = 0
    published_at: str = ""
    description: str = ""
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    favorite_count: int = 0
    share_count: int = 0
    hotness: float = 0.0
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInfo:
        return cls(**data)


@dataclass(slots=True)
class TranscriptChunk:
    video_id: str
    video_title: str
    video_url: str
    start_ms: int
    end_ms: int
    text: str
    source_type: str = "transcript"
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptChunk:
        return cls(**data)


@dataclass(slots=True)
class TranscriptFile:
    video_id: str
    video_title: str
    video_url: str
    language: str
    full_text: str
    raw_text: str
    audio_path: str
    chunks: list[TranscriptChunk] = field(default_factory=list)
    subtitle_text: str = ""
    source_type: str = "transcript"
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptFile:
        payload = dict(data)
        payload["chunks"] = [TranscriptChunk.from_dict(item) for item in payload.get("chunks", [])]
        payload.setdefault("subtitle_text", "")
        payload.setdefault("source_type", "transcript")
        payload.setdefault("source_path", "")
        return cls(**payload)


@dataclass(slots=True)
class VoiceSample:
    audio_path: str           # clipped wav used for TTS voice clone
    prompt_text: str          # text spoken in the clip
    start_ms: int
    end_ms: int
    source_audio_path: str = ""   # original file the clip was cut from
    video_id: str = ""            # legacy field, kept for compat

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceSample:
        return cls(
            audio_path=str(data["audio_path"]),
            prompt_text=str(data["prompt_text"]),
            start_ms=int(data.get("start_ms", 0)),
            end_ms=int(data.get("end_ms", 0)),
            source_audio_path=str(data.get("source_audio_path", "")),
            video_id=str(data.get("video_id", "")),
        )


@dataclass(slots=True)
class ProfileManifest:
    creator: CreatorRef
    creator_slug: str
    creator_dir: str
    videos: list[VideoInfo]
    transcript_paths: list[str]
    skill_path: str
    selected_series_numbers: list[int] = field(default_factory=list)
    source_state_signature: str = ""
    voice_sample: VoiceSample | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "creator": self.creator.to_dict(),
            "creator_slug": self.creator_slug,
            "creator_dir": self.creator_dir,
            "videos": [video.to_dict() for video in self.videos],
            "transcript_paths": self.transcript_paths,
            "skill_path": self.skill_path,
            "selected_series_numbers": self.selected_series_numbers,
            "source_state_signature": self.source_state_signature,
            "voice_sample": self.voice_sample.to_dict() if self.voice_sample else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileManifest:
        return cls(
            creator=CreatorRef.from_dict(data["creator"]),
            creator_slug=data["creator_slug"],
            creator_dir=data["creator_dir"],
            videos=[VideoInfo.from_dict(item) for item in data.get("videos", [])],
            transcript_paths=data.get("transcript_paths", []),
            skill_path=data["skill_path"],
            selected_series_numbers=[int(item) for item in data.get("selected_series_numbers", [])],
            source_state_signature=str(data.get("source_state_signature") or ""),
            voice_sample=VoiceSample.from_dict(data["voice_sample"]) if data.get("voice_sample") else None,
        )

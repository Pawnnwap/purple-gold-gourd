from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ....language import detect_transcript_language
from ....schema import TranscriptChunk, TranscriptFile, VideoInfo
from ....utils import PUNCTUATION, join_tokens
from ..shared import build_srt

_TAG_PATTERN = re.compile(r"<\|[^|>]*\|>")


def _strip_tags(text: str) -> str:
    return _TAG_PATTERN.sub("", text)


class FunASRSpeechTranscriber:
    def __init__(self, model_name: str, device: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self._model: Any | None = None
        self._rich_postprocess = None

    @property
    def model(self) -> Any:
        if self._model is None:
            cache_dir = self.cache_dir.expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir))
            try:
                from funasr import AutoModel
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
            except ImportError as exc:
                raise RuntimeError(
                    "Speech transcription requires the optional 'speech' dependencies. "
                    "Install them with 'pip install .[speech]' (or install funasr directly).",
                ) from exc

            self._model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                disable_update=True,
            )
            self._rich_postprocess = rich_transcription_postprocess
        return self._model

    def transcribe(self, audio_path: Path, video: VideoInfo) -> TranscriptFile:
        results = self._generate(audio_path, output_timestamp=True, batch_size_s=300)
        if not results:
            raise ValueError(f"FunASR returned no transcription for {audio_path}")
        chunks: list[TranscriptChunk] = []
        cleaned_segments: list[str] = []
        raw_segments: list[str] = []
        tags = [self._extract_language_tag(str(item.get("text") or "")) for item in results]
        for item in results:
            raw_text = str(item.get("text") or "")
            raw_segments.append(raw_text)
            cleaned = self._rich_postprocess(raw_text).strip()
            if cleaned:
                cleaned_segments.append(cleaned)
            words = [str(word) for word in item.get("words") or []]
            timestamps = item.get("timestamp") or []
            chunks.extend(self._build_chunks(words, timestamps, video))
        if not chunks and cleaned_segments:
            chunks.append(
                TranscriptChunk(
                    video_id=video.video_id,
                    video_title=video.title,
                    video_url=video.url,
                    start_ms=0,
                    end_ms=max(video.duration_sec * 1000, 0),
                    text=" ".join(cleaned_segments),
                ),
            )
        full_text = "\n".join(chunk.text for chunk in chunks) if chunks else "\n".join(cleaned_segments)
        subtitle_text = build_srt(chunks)
        raw_language = Counter(tag for tag in tags if tag).most_common(1)[0][0] if any(tags) else ""
        language = detect_transcript_language(raw_language, "\n".join(raw_segments + cleaned_segments + [full_text]))
        return TranscriptFile(
            video_id=video.video_id,
            video_title=video.title,
            video_url=video.url,
            language=language,
            full_text=full_text.strip(),
            raw_text="\n".join(raw_segments).strip(),
            audio_path=str(audio_path),
            chunks=chunks,
            subtitle_text=subtitle_text,
        )

    def transcribe_text(self, audio_path: Path, batch_size_s: int = 60) -> str:
        results = self._generate(audio_path, output_timestamp=False, batch_size_s=batch_size_s)
        pieces: list[str] = []
        for item in results:
            raw = str(item.get("text") or "")
            cleaned = self._rich_postprocess(raw).strip()
            if cleaned:
                pieces.append(cleaned)
        return "".join(pieces).strip()

    def build_subtitles(self, chunks: list[TranscriptChunk]) -> str:
        return build_srt(chunks)

    def _generate(self, audio_path: Path, output_timestamp: bool, batch_size_s: int) -> list[dict[str, Any]]:
        results = self.model.generate(
            input=str(audio_path),
            cache={},
            language="auto",
            use_itn=True,
            ban_emo_unk=True,
            output_timestamp=output_timestamp,
            batch_size_s=batch_size_s,
        )
        return list(results or [])

    def _extract_language_tag(self, raw_text: str) -> str:
        match = re.search(r"<\|([a-z]+)\|>", raw_text.lower())
        return match.group(1) if match else ""

    def _build_chunks(
        self,
        words: list[str],
        timestamps: list[list[int]],
        video: VideoInfo,
    ) -> list[TranscriptChunk]:
        if not words or not timestamps or len(words) != len(timestamps):
            return []
        words = [_strip_tags(word) for word in words]
        chunks: list[TranscriptChunk] = []
        current_words: list[str] = []
        start_ms = int(timestamps[0][0])
        previous_end = start_ms
        for index, (word, span) in enumerate(zip(words, timestamps)):
            begin_ms = int(span[0])
            end_ms = int(span[1])
            gap = begin_ms - previous_end
            current_words.append(word)
            text = join_tokens(current_words)
            should_split = False
            if word and word[-1] in PUNCTUATION and len(text) >= 24:
                should_split = True
            if gap >= 1200 and len(text) >= 18:
                should_split = True
            if end_ms - start_ms >= 15000 and len(text) >= 24:
                should_split = True
            if index == len(words) - 1:
                should_split = True
            if should_split:
                final_text = join_tokens(current_words).strip()
                if final_text:
                    chunks.append(
                        TranscriptChunk(
                            video_id=video.video_id,
                            video_title=video.title,
                            video_url=video.url,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            text=final_text,
                        ),
                    )
                current_words = []
                if index + 1 < len(timestamps):
                    start_ms = int(timestamps[index + 1][0])
            previous_end = end_ms
        return chunks

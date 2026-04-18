from __future__ import annotations

import contextlib
import io
import re
import subprocess
from pathlib import Path

from ...schema import TranscriptChunk, TranscriptFile, VoiceSample
from ...utils import PUNCTUATION, ensure_dir, join_tokens
from .base import SpeechTranscriber

VOICE_MIN_MS = 5_000
VOICE_MAX_MS = 10_000
VOICE_TARGET_MS = 8_000
VOICE_MIN_CHARS = 14
VOICE_MAX_CHARS = 160
VOICE_SPEAKER_PROBE_LIMIT = 24
VOICE_SPEAKER_MIN_CLIPS = 4
VOICE_SPEAKER_MIN_CLUSTER_MEMBERS = 3
VOICE_SPEAKER_SIMILARITY = 0.72
VOICE_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SENTENCE_TERMINATORS = set(".!?\u3002\uff01\uff1f")
_SPEAKER_ENCODERS: dict[str, object] = {}


def choose_voice_sample(
    ffmpeg_path: str,
    transcriber: SpeechTranscriber,
    transcripts: list[TranscriptFile],
    output_dir: Path,
    speaker_cache_dir: Path | None = None,
) -> VoiceSample | None:
    ensure_dir(output_dir)
    candidates: list[tuple[float, TranscriptFile, TranscriptChunk]] = []
    for transcript in transcripts:
        total = len(transcript.chunks)
        for position, chunk in enumerate(transcript.chunks):
            score = _score_chunk(chunk, position, total)
            if score > 0:
                candidates.append((score, transcript, chunk))
        candidates.extend(_merged_candidates(transcript))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    prioritized_groups = [
        _prefer_main_speaker_candidates(
            ffmpeg_path=ffmpeg_path,
            candidates=candidates,
            output_dir=output_dir,
            speaker_cache_dir=speaker_cache_dir,
        ),
        candidates,
    ]
    seen_keys: set[tuple[str, int, int]] = set()
    for candidate_group in prioritized_groups:
        if not candidate_group:
            continue
        for _, transcript, chunk in candidate_group[:8]:
            key = (chunk.video_id, chunk.start_ms, chunk.end_ms)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            output_path = _voice_prompt_path(output_dir, chunk)
            clip_audio(ffmpeg_path, Path(transcript.audio_path), output_path, chunk.start_ms, chunk.end_ms)
            verified_text = transcriber.transcribe_text(output_path)
            if not _prompt_text_acceptable(verified_text, chunk.text):
                continue
            return VoiceSample(
                audio_path=str(output_path),
                prompt_text=verified_text,
                video_id=chunk.video_id,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
    _, transcript, chunk = candidates[0]
    output_path = _voice_prompt_path(output_dir, chunk)
    clip_audio(ffmpeg_path, Path(transcript.audio_path), output_path, chunk.start_ms, chunk.end_ms)
    verified_text = transcriber.transcribe_text(output_path) or chunk.text
    return VoiceSample(
        audio_path=str(output_path),
        prompt_text=verified_text,
        video_id=chunk.video_id,
        start_ms=chunk.start_ms,
        end_ms=chunk.end_ms,
    )


def clip_audio(ffmpeg_path: str, source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    duration_sec = max((end_ms - start_ms) / 1000.0, 0.5)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start_ms / 1000.0:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class AudioPlayer:
    def __init__(self) -> None:
        self._winsound = None
        self._sd = None

    def play(self, audio_path: Path, wait: bool = False) -> None:
        try:
            self._play_with_winsound(audio_path, wait=wait)
            return
        except Exception:
            self._play_with_sounddevice(audio_path, wait=wait)

    def stop(self) -> None:
        if self._winsound is not None:
            self._winsound.PlaySound(None, 0)
        if self._sd is not None:
            self._sd.stop()

    def _play_with_winsound(self, audio_path: Path, wait: bool = False) -> None:
        if self._winsound is None:
            import winsound

            self._winsound = winsound
        self._winsound.PlaySound(None, 0)
        flags = self._winsound.SND_FILENAME | self._winsound.SND_NODEFAULT
        if wait:
            flags |= self._winsound.SND_SYNC
        else:
            flags |= self._winsound.SND_ASYNC
        self._winsound.PlaySound(
            str(audio_path),
            flags,
        )

    def _play_with_sounddevice(self, audio_path: Path, wait: bool = False) -> None:
        if self._sd is None:
            import sounddevice as sd

            self._sd = sd
        import soundfile as sf

        self.stop()
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        self._sd.play(audio, sample_rate)
        if wait:
            self._sd.wait()


def split_for_synthesis(text: str, max_chars: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[\u3002\uff01\uff1f.!?])\s*", text) if s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long_clause(sentence, max_chars))
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer = buffer + sentence if buffer else sentence
        else:
            chunks.append(buffer)
            buffer = sentence
    if buffer:
        chunks.append(buffer)
    return chunks


def validate_synthesis(
    audio_path: Path,
    expected_text: str,
    transcriber: SpeechTranscriber,
) -> tuple[str, float]:
    from difflib import SequenceMatcher

    transcribed = transcriber.transcribe_text(audio_path)
    similarity = SequenceMatcher(None, _normalize_for_compare(transcribed), _normalize_for_compare(expected_text)).ratio()
    return transcribed, similarity


def _prefer_main_speaker_candidates(
    ffmpeg_path: str,
    candidates: list[tuple[float, TranscriptFile, TranscriptChunk]],
    output_dir: Path,
    speaker_cache_dir: Path | None,
) -> list[tuple[float, TranscriptFile, TranscriptChunk]]:
    if speaker_cache_dir is None or len(candidates) < VOICE_SPEAKER_MIN_CLIPS:
        return []
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        return []

    encoder_dir = ensure_dir(Path(speaker_cache_dir).expanduser().resolve())
    cache_key = str(encoder_dir)
    encoder = _SPEAKER_ENCODERS.get(cache_key)
    if encoder is None:
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                encoder = EncoderClassifier.from_hparams(
                    source=VOICE_SPEAKER_MODEL,
                    savedir=str(encoder_dir),
                    huggingface_cache_dir=str(encoder_dir.parent),
                    run_opts={"device": "cpu"},
                )
        except Exception:
            return []
        _SPEAKER_ENCODERS[cache_key] = encoder

    probe_dir = ensure_dir(output_dir / "speaker-probes")
    probes: list[tuple[float, TranscriptFile, TranscriptChunk, Path, object]] = []
    for index, (score, transcript, chunk) in enumerate(candidates[:VOICE_SPEAKER_PROBE_LIMIT], start=1):
        clip_path = probe_dir / f"{index:03d}-{chunk.video_id}-{chunk.start_ms}-{chunk.end_ms}.wav"
        try:
            clip_audio(ffmpeg_path, Path(transcript.audio_path), clip_path, chunk.start_ms, chunk.end_ms)
            audio, sample_rate = sf.read(str(clip_path), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            if len(audio) == 0:
                continue
            wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            if sample_rate != 16000:
                wavs = torchaudio.functional.resample(wavs, sample_rate, 16000)
            with torch.no_grad():
                embedding = encoder.encode_batch(wavs, normalize=True)
            vector = embedding.squeeze().detach().cpu().numpy().reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            probes.append((score, transcript, chunk, clip_path, vector / norm))
        except Exception:
            continue
    if len(probes) < VOICE_SPEAKER_MIN_CLIPS:
        return []

    clusters: list[dict[str, object]] = []
    for probe in sorted(probes, key=lambda item: item[0], reverse=True):
        vector = probe[4]
        best_cluster = None
        best_similarity = -1.0
        for cluster in clusters:
            similarity = float(np.dot(vector, cluster["centroid"]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        if best_cluster is None or best_similarity < VOICE_SPEAKER_SIMILARITY:
            clusters.append({"members": [probe], "centroid": vector})
            continue
        members = best_cluster["members"]
        members.append(probe)
        centroid = np.mean(np.stack([item[4] for item in members], axis=0), axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 0:
            best_cluster["centroid"] = centroid / centroid_norm
    eligible_clusters = [
        cluster
        for cluster in clusters
        if len(cluster["members"]) >= VOICE_SPEAKER_MIN_CLUSTER_MEMBERS
    ]
    if not eligible_clusters:
        return []
    dominant_cluster = max(
        eligible_clusters,
        key=lambda cluster: (
            len({item[2].video_id for item in cluster["members"]}),
            len(cluster["members"]),
            sum(float(item[0]) for item in cluster["members"]),
            sum(int(item[2].end_ms - item[2].start_ms) for item in cluster["members"]),
        ),
    )
    cluster_videos = {item[2].video_id for item in dominant_cluster["members"]}
    if len(cluster_videos) < 2 and len(dominant_cluster["members"]) < VOICE_SPEAKER_MIN_CLIPS:
        return []
    cluster_keys = {
        (item[2].video_id, int(item[2].start_ms), int(item[2].end_ms))
        for item in dominant_cluster["members"]
    }
    return [
        candidate
        for candidate in candidates
        if (candidate[2].video_id, candidate[2].start_ms, candidate[2].end_ms) in cluster_keys
    ]


def prepare_tts_text(text: str, char_limit: int = 360) -> str:
    spoken = text.strip()
    spoken = re.sub(r"(?m)^\s*#{1,6}\s*", "", spoken)
    spoken = re.sub(r"(?m)^\s*>\s*", "", spoken)
    spoken = spoken.replace("**", "")
    spoken = spoken.replace("__", "")
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", spoken)
    spoken = re.sub(r"https?://\S+", "", spoken)
    spoken = re.sub(r"(?m)^\s*[-*]\s+", "", spoken)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "", spoken)
    spoken = re.sub(r"\s+", " ", spoken)
    spoken = spoken.replace("|", ", ")
    if len(spoken) <= char_limit:
        return spoken
    shortened = spoken[:char_limit]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ,.;:") + "..."


def _prompt_text_acceptable(verified: str, original: str) -> bool:
    from difflib import SequenceMatcher

    if not verified:
        return False
    if len(verified) < VOICE_MIN_CHARS // 2:
        return False
    ratio = SequenceMatcher(None, _normalize_for_compare(verified), _normalize_for_compare(original)).ratio()
    return ratio >= 0.55


def _voice_prompt_path(output_dir: Path, chunk: TranscriptChunk) -> Path:
    return output_dir / f"{chunk.video_id}-{chunk.start_ms}-{chunk.end_ms}-voice-prompt.wav"


def _score_chunk(chunk: TranscriptChunk, position: int, total: int) -> float:
    text = chunk.text.strip()
    duration = chunk.end_ms - chunk.start_ms
    if not text or duration < VOICE_MIN_MS or duration > VOICE_MAX_MS:
        return 0.0
    if len(text) < VOICE_MIN_CHARS or len(text) > VOICE_MAX_CHARS:
        return 0.0
    if _has_repetition(text):
        return 0.0
    seconds = duration / 1000.0
    chars_per_sec = len(text) / seconds
    if chars_per_sec < 2.0 or chars_per_sec > 14.0:
        return 0.0
    if total >= 6:
        position_ratio = (position + 0.5) / total
        if position_ratio < 0.1 or position_ratio > 0.9:
            return 0.0
    duration_score = 10.0 - abs(duration - VOICE_TARGET_MS) / 1000.0
    sentence_bonus = 1.25 if text[-1] in SENTENCE_TERMINATORS else 1.0
    punct_bonus = 1.0 + 0.05 * min(sum(1 for ch in text if ch in PUNCTUATION), 4)
    density_bonus = 1.0 - abs(chars_per_sec - 6.0) / 20.0
    return duration_score * sentence_bonus * punct_bonus * max(density_bonus, 0.5)


def _has_repetition(text: str) -> bool:
    run_char = ""
    run = 0
    for ch in text:
        if ch == run_char:
            run += 1
            if run >= 5:
                return True
        else:
            run_char = ch
            run = 1
    tokens = [token for token in text.split() if token]
    for index in range(len(tokens) - 3):
        if tokens[index] == tokens[index + 1] == tokens[index + 2] == tokens[index + 3]:
            return True
    return False


def _merged_candidates(transcript: TranscriptFile) -> list[tuple[float, TranscriptFile, TranscriptChunk]]:
    candidates: list[tuple[float, TranscriptFile, TranscriptChunk]] = []
    chunks = transcript.chunks
    total = len(chunks)
    for start_index in range(total):
        start = chunks[start_index]
        merged_words: list[str] = [start.text]
        merged_start = start.start_ms
        merged_end = start.end_ms
        for end_index in range(start_index + 1, min(start_index + 4, total)):
            current = chunks[end_index]
            if current.start_ms - merged_end > 1200:
                break
            merged_words.append(current.text)
            merged_end = current.end_ms
            duration = merged_end - merged_start
            if duration > VOICE_MAX_MS:
                break
            text = join_tokens(merged_words).strip()
            synthetic = TranscriptChunk(
                video_id=start.video_id,
                video_title=start.video_title,
                video_url=start.video_url,
                start_ms=merged_start,
                end_ms=merged_end,
                text=text,
            )
            score = _score_chunk(synthetic, start_index, total)
            if score > 0:
                candidates.append((score * 0.95, transcript, synthetic))
    return candidates


def _split_long_clause(sentence: str, max_chars: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[,\u3001\uff0c\uff1b;:])\s*", sentence) if p.strip()]
    result: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > max_chars:
            if buffer:
                result.append(buffer)
                buffer = ""
            for index in range(0, len(part), max_chars):
                result.append(part[index: index + max_chars])
            continue
        if len(buffer) + len(part) <= max_chars:
            buffer = buffer + part if buffer else part
        else:
            result.append(buffer)
            buffer = part
    if buffer:
        result.append(buffer)
    return result or [sentence[:max_chars]]


def _normalize_for_compare(text: str) -> str:
    stripped = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return stripped.lower()

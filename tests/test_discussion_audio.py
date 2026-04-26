from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from purple_gold_gourd.chat.discussion import (
    DiscussionParticipant,
    DiscussionSpeechSynthesizer,
    DiscussionTurnRecord,
)
from purple_gold_gourd.schema import CreatorRef, ProfileManifest, VoiceSample


class _FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path, Path]] = []

    def synthesize(self, text: str, prompt_text: str, prompt_audio: Path, target: Path) -> Path:
        self.calls.append((text, prompt_text, prompt_audio, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-wav")
        return target


class _FakeTTS:
    def __init__(self) -> None:
        self.created = 0
        self.synthesizer = _FakeSynthesizer()

    def create_synthesizer(self) -> _FakeSynthesizer:
        self.created += 1
        return self.synthesizer

    def prepare_spoken_text(self, text: str, char_limit: int = 360) -> str:
        return text[:char_limit]


class _FakeChat:
    def __init__(self, tts: _FakeTTS) -> None:
        self.tts = tts


def _participant(name: str, query: str, tts: _FakeTTS, voice_path: Path) -> DiscussionParticipant:
    voice = VoiceSample(
        audio_path=str(voice_path),
        prompt_text=f"{name} prompt",
        start_ms=0,
        end_ms=1000,
    )
    manifest = ProfileManifest(
        creator=CreatorRef(
            platform="test",
            creator_id=query,
            name=name,
            homepage_url="https://example.com",
            video_tab_url="https://example.com/videos",
            query=query,
        ),
        creator_slug=f"creator-{query}",
        creator_dir=str(voice_path.parent),
        videos=[],
        transcript_paths=[],
        skill_path=str(voice_path.parent / "skill.md"),
        voice_sample=voice,
    )
    return DiscussionParticipant(query=query, manifest=manifest, chat=_FakeChat(tts))  # type: ignore[arg-type]


class DiscussionSpeechSynthesizerTests(unittest.TestCase):
    def test_discussion_audio_reuses_one_synthesizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            voice_a = root / "a.wav"
            voice_b = root / "b.wav"
            voice_a.write_bytes(b"a")
            voice_b.write_bytes(b"b")
            tts_a = _FakeTTS()
            tts_b = _FakeTTS()
            participant_a = _participant("Alice", "a", tts_a, voice_a)
            participant_b = _participant("Bob", "b", tts_b, voice_b)
            synthesizer = DiscussionSpeechSynthesizer()
            audio_dir = root / "audio"

            try:
                synthesizer.submit(
                    participant=participant_a,
                    turn=DiscussionTurnRecord(1, 1, "Alice", "a", "creator-a", "hello", []),
                    spoken_answer="hello",
                    audio_dir=audio_dir,
                )
                synthesizer.submit(
                    participant=participant_b,
                    turn=DiscussionTurnRecord(1, 2, "Bob", "b", "creator-b", "world", []),
                    spoken_answer="world",
                    audio_dir=audio_dir,
                )
                synthesizer.wait_all()
            finally:
                synthesizer.shutdown(wait=False, cancel_futures=True)

        self.assertEqual(tts_a.created, 1)
        self.assertEqual(tts_b.created, 0)
        self.assertEqual(len(tts_a.synthesizer.calls), 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from purple_gold_gourd.media.video import render_discussion_video


class DiscussionVideoSmokeTests(unittest.TestCase):
    def test_renderer_makes_mp4_and_timed_subtitles(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            record_dir = Path(tmp)
            audio_dir = record_dir / "audio"
            audio_dir.mkdir()
            _write_tone(audio_dir / "alice.wav", seconds=0.7, frequency=440)
            _write_tone(audio_dir / "bob.wav", seconds=0.8, frequency=554)
            _write_tone(audio_dir / "host.wav", seconds=0.6, frequency=330)

            participants = [
                _participant("Alice", "youtube", "@alice"),
                _participant("Bob", "bilibili", ""),
            ]
            turns = [
                SimpleNamespace(
                    round_number=1,
                    turn_number=1,
                    speaker="Host",
                    character_slug="host",
                    text="Welcome to the discussion.",
                    audio_path="audio/host.wav",
                ),
                SimpleNamespace(
                    round_number=1,
                    turn_number=2,
                    speaker="Alice",
                    character_slug="alice",
                    text='***"Hello from Alice!!!"',
                    audio_path="audio/alice.wav",
                ),
                SimpleNamespace(
                    round_number=1,
                    turn_number=3,
                    speaker="Bob",
                    character_slug="bob",
                    text="Bob answers with another short line.",
                    audio_path="audio/bob.wav",
                ),
            ]

            result = render_discussion_video(
                ffmpeg_path=ffmpeg,
                record_dir=record_dir,
                topic="Smoke test discussion",
                participants=participants,
                turns=turns,
                width=854,
                height=480,
                fps=12,
            )

            self.assertTrue(result.video_path.exists())
            self.assertGreater(result.video_path.stat().st_size, 1_000)
            self.assertTrue(result.audio_path.exists())
            subtitles = result.subtitles_path.read_text(encoding="utf-8")
            self.assertIn("00:00:04,500 -->", subtitles)
            self.assertIn("Host:", subtitles)
            self.assertIn("Hello from Alice", subtitles)
            self.assertNotIn('***"Hello from Alice!!!"', subtitles)
            self.assertNotIn("Alice:", subtitles)
            self.assertNotIn("Bob:", subtitles)
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertGreater(metadata["duration_s"], metadata["discussion_duration_s"])
            self.assertEqual(metadata["opening_duration_s"], 4.5)


def _participant(name: str, platform: str, handle: str) -> SimpleNamespace:
    creator = SimpleNamespace(
        name=name,
        platform=platform,
        handle=handle,
        creator_id="",
        avatar_url="",
    )
    return SimpleNamespace(manifest=SimpleNamespace(creator=creator))


def _write_tone(path: Path, *, seconds: float, frequency: int) -> None:
    sample_rate = 16_000
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(frames):
            value = int(0.25 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
            handle.writeframes(struct.pack("<h", value))


if __name__ == "__main__":
    unittest.main()

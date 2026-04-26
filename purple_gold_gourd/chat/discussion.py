from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from ..plugins.tts.base import VoiceSynthesizer
from ..plugins.tts.shared import prepare_tts_text
from ..schema import ProfileManifest, VoiceSample
from ..utils import ensure_dir, slugify, write_json
from .persona import PersonaChat


@dataclass(slots=True)
class DiscussionParticipant:
    query: str
    manifest: ProfileManifest
    chat: PersonaChat

    @property
    def name(self) -> str:
        return self.manifest.creator.name

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def voice_available(self) -> bool:
        return self.manifest.voice_sample is not None


@dataclass(slots=True)
class DiscussionTurnRecord:
    round_number: int
    turn_number: int
    speaker: str
    character_query: str
    character_slug: str
    text: str
    citations: list[str]
    audio_path: str = ""
    spoken_text: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "speaker": self.speaker,
            "character_query": self.character_query,
            "character_slug": self.character_slug,
            "text": self.text,
            "citations": self.citations,
            "audio_path": self.audio_path,
            "created_at": self.created_at,
        }

    def to_prompt_dict(self) -> dict[str, str | int]:
        return {
            "round_number": self.round_number,
            "speaker": self.speaker,
            "text": self.text,
        }


@dataclass(slots=True)
class PreparedDiscussionTurn:
    record: DiscussionTurnRecord
    spoken_answer: str
    audio_file: Path | None = None


@dataclass(slots=True)
class SpeechSynthesisJob:
    turn: DiscussionTurnRecord
    audio_file: Path
    future: Future[Path]


class DiscussionPlaybackQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discussion-audio")
        self._tail: Future[None] | None = None
        self._shutdown = False

    def enqueue_future(self, participant: DiscussionParticipant, audio_future: Future[Path]) -> None:
        if self._shutdown:
            return

        def _play_when_ready() -> None:
            participant.chat.player.play(audio_future.result(), True)

        self._tail = self._executor.submit(_play_when_ready)

    def wait(self) -> None:
        if self._tail is not None:
            self._tail.result()

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


class DiscussionSpeechSynthesizer:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discussion-tts")
        self._jobs: list[SpeechSynthesisJob] = []
        self._synthesizer: VoiceSynthesizer | None = None
        self._shutdown = False

    @property
    def has_jobs(self) -> bool:
        return bool(self._jobs)

    def submit(
        self,
        participant: DiscussionParticipant,
        turn: DiscussionTurnRecord,
        spoken_answer: str,
        audio_dir: Path,
        voice_sample: VoiceSample | None = None,
    ) -> SpeechSynthesisJob:
        if self._shutdown:
            raise RuntimeError("Speech synthesizer has already been shut down.")
        slug = turn.character_slug or participant.slug
        audio_file = audio_dir / f"{turn.turn_number:03d}-round{turn.round_number:02d}-{slug}.wav"
        turn.audio_path = str(audio_file.relative_to(audio_dir.parent))
        sample = voice_sample or participant.manifest.voice_sample
        if sample is None:
            raise ValueError(f"No voice sample is available for {participant.name}.")
        future = self._executor.submit(
            self._synthesize_with_voice_sample,
            participant,
            spoken_answer,
            audio_file,
            sample,
        )
        job = SpeechSynthesisJob(turn=turn, audio_file=audio_file, future=future)
        self._jobs.append(job)
        return job

    def _synthesize_with_voice_sample(
        self,
        participant: DiscussionParticipant,
        spoken_answer: str,
        audio_file: Path,
        voice_sample: VoiceSample,
    ) -> Path:
        if self._synthesizer is None:
            self._synthesizer = participant.chat.tts.create_synthesizer()
        spoken_text = participant.chat.tts.prepare_spoken_text(spoken_answer, char_limit=4000)
        return self._synthesizer.synthesize(
            text=spoken_text,
            prompt_text=voice_sample.prompt_text,
            prompt_audio=Path(voice_sample.audio_path),
            target=audio_file,
        )

    def wait_all(self) -> None:
        for job in self._jobs:
            job.future.result()

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


def create_discussion_record_dir(config: AppConfig, topic: str) -> Path:
    discussions_dir = ensure_dir(config.data_dir / "discussions")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    first_line = next((line.strip() for line in topic.splitlines() if line.strip()), topic)
    words = first_line.split()
    short_topic = " ".join(words[:30]) if len(words) > 30 else first_line
    topic_slug = slugify(short_topic)[:80] or "discussion"
    return ensure_dir(discussions_dir / f"{stamp}-{topic_slug}")


def prepare_discussion_turn(
    participant: DiscussionParticipant,
    topic: str,
    participants: list[DiscussionParticipant],
    prior_turns: list[DiscussionTurnRecord],
    round_number: int,
    total_rounds: int,
) -> PreparedDiscussionTurn:
    response = participant.chat.discuss(
        topic=topic,
        participants=[item.name for item in participants],
        prior_turns=[turn.to_prompt_dict() for turn in prior_turns],
        round_number=round_number,
        total_rounds=total_rounds,
    )
    turn_number = len(prior_turns) + 1
    spoken_text = prepare_tts_text(response.spoken_answer, char_limit=4000)
    return PreparedDiscussionTurn(
        record=DiscussionTurnRecord(
            round_number=round_number,
            turn_number=turn_number,
            speaker=participant.name,
            character_query=participant.query,
            character_slug=participant.manifest.creator_slug,
            text=response.display_answer,
            citations=response.citations,
            audio_path="",
            spoken_text=spoken_text,
            created_at=datetime.now().isoformat(timespec="seconds"),
        ),
        spoken_answer=spoken_text,
    )

def stop_discussion_audio(participants: list[DiscussionParticipant]) -> None:
    for participant in participants:
        try:
            participant.chat.player.stop()
        except Exception:
            continue


def save_discussion_snapshot(
    record_dir: Path,
    topic: str,
    requested_rounds: int,
    participants: list[DiscussionParticipant],
    turns: list[DiscussionTurnRecord],
    started_at: str,
    finished_at: str = "",
) -> None:
    ensure_dir(record_dir)
    ensure_dir(record_dir / "audio")
    completed_rounds = max((turn.round_number for turn in turns), default=0)
    payload = {
        "topic": topic,
        "requested_rounds": requested_rounds,
        "completed_rounds": completed_rounds,
        "started_at": started_at,
        "finished_at": finished_at,
        "participants": [
            {
                "query": participant.query,
                "name": participant.name,
                "creator_slug": participant.manifest.creator_slug,
                "creator_dir": participant.manifest.creator_dir,
                "voice_available": participant.voice_available,
            }
            for participant in participants
        ],
        "turns": [turn.to_dict() for turn in turns],
    }
    write_json(record_dir / "discussion.json", payload)
    (record_dir / "discussion.md").write_text(
        _render_discussion_markdown(topic, requested_rounds, participants, turns, started_at, finished_at),
        encoding="utf-8",
    )
    (record_dir / "discussion.txt").write_text(_render_discussion_text(turns), encoding="utf-8")


def _render_discussion_markdown(
    topic: str,
    requested_rounds: int,
    participants: list[DiscussionParticipant],
    turns: list[DiscussionTurnRecord],
    started_at: str,
    finished_at: str,
) -> str:
    lines = [
        "# Discussion Record",
        "",
        f"Topic: {topic}",
        f"Rounds requested: {requested_rounds}",
        f"Rounds completed: {max((turn.round_number for turn in turns), default=0)}",
        f"Started at: {started_at}",
    ]
    if finished_at:
        lines.append(f"Finished at: {finished_at}")
    lines.extend(
        [
            "",
            "Participants:",
            *[f"- {participant.name} ({participant.query})" for participant in participants],
            "",
        ],
    )
    current_round = None
    for turn in turns:
        if turn.round_number != current_round:
            current_round = turn.round_number
            lines.extend([f"## Round {current_round}", ""])
        lines.extend(
            [
                f"### {turn.turn_number}. {turn.speaker}",
                "",
                turn.text,
                "",
            ],
        )
        if turn.citations:
            lines.append("Refs: " + " | ".join(turn.citations))
        if turn.audio_path:
            lines.append(f"Audio: {turn.audio_path}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def prepare_host_turn(
    topic: str,
    participants: list[DiscussionParticipant],
    prior_turns: list[DiscussionTurnRecord],
    round_number: int,
    total_rounds: int,
) -> DiscussionTurnRecord:
    """Generate a host/moderator intro turn at the start of each round."""
    ref = participants[0].chat
    llm = ref.llm
    language = ref.language
    participant_names = "、".join(p.name for p in participants) if language == "zh" else ", ".join(p.name for p in participants)

    non_host = [t for t in prior_turns if t.character_slug != "host"]
    recent = non_host[-8:]

    if language == "zh":
        sys_prompt = (
            "你是一位中立专业的讨论节目主持人。\n"
            "每轮开场须做到三点：\n"
            "①简要概括话题背景和已有讨论要点（首轮则介绍事件背景）\n"
            "②补充必要的背景资料或关键事实\n"
            "③点明本轮讨论的核心焦点或值得深挖的方向\n"
            "语言简练有力，不加主观立场。只输出纯文本，一段话，适合口播朗读。"
        )
        prior_text = ""
        if recent:
            prior_text = "\n前几轮发言摘要：\n" + "\n".join(
                f"第{t.round_number}轮 {t.speaker}：{t.text[:150]}{'...' if len(t.text) > 150 else ''}"
                for t in recent
            )
        user_prompt = (
            f"话题：\n{topic}\n"
            f"{prior_text}\n"
            f"本场嘉宾（发言顺序）：{participant_names}\n"
            f"请开始第 {round_number}/{total_rounds} 轮的主持开场。"
        )
        speaker = "主持人"
    else:
        sys_prompt = (
            "You are a neutral, professional discussion host.\n"
            "Each round opening must cover three things: "
            "①briefly summarize the topic and prior discussion points (or introduce the topic if round 1) "
            "②add necessary background or key facts "
            "③identify the core focus or angle worth digging into this round.\n"
            "Be concise and objective. Plain text only, one paragraph, suitable for reading aloud."
        )
        prior_text = ""
        if recent:
            prior_text = "\nPrior discussion summary:\n" + "\n".join(
                f"Round {t.round_number} {t.speaker}: {t.text[:150]}{'...' if len(t.text) > 150 else ''}"
                for t in recent
            )
        user_prompt = (
            f"Topic:\n{topic}\n"
            f"{prior_text}\n"
            f"Guests (speaking order): {participant_names}\n"
            f"Please open round {round_number}/{total_rounds}."
        )
        speaker = "Host"

    messages = [
        {"role": "system", "content": _with_brevity_prompt(sys_prompt, getattr(ref.config, "brevity", False), language)},
        {"role": "user", "content": user_prompt},
    ]
    raw, _ = llm.complete(messages=messages, temperature=0.5)
    raw = ref._plain_text(raw)
    spoken = prepare_tts_text(raw, char_limit=4000)
    turn_number = len(prior_turns) + 1

    return DiscussionTurnRecord(
        round_number=round_number,
        turn_number=turn_number,
        speaker=speaker,
        character_query="host",
        character_slug="host",
        text=raw,
        citations=[],
        audio_path="",
        spoken_text=spoken,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def _render_discussion_text(turns: list[DiscussionTurnRecord]) -> str:
    lines: list[str] = []
    current_round = None
    for turn in turns:
        if turn.round_number != current_round:
            current_round = turn.round_number
            lines.append(f"[Round {current_round}]")
        lines.append(f"{turn.speaker}: {turn.text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _with_brevity_prompt(prompt: str, enabled: bool, language: str) -> str:
    if not enabled:
        return prompt
    suffix = (
        "保持简洁。每次回复控制在120字以内，最多一段。"
        if language == "zh"
        else "Be brief. Keep each reply under 80 words, one paragraph max."
    )
    return f"{prompt.rstrip()}\n\n{suffix}"

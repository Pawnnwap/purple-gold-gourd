from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from ..schema import ProfileManifest
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
    audio_file: Path | None = None


class DiscussionPlaybackQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discussion-audio")
        self._tail: Future[None] | None = None
        self._shutdown = False

    def enqueue(self, participant: DiscussionParticipant, audio_file: Path) -> None:
        if self._shutdown:
            return
        self._tail = self._executor.submit(participant.chat.player.play, audio_file, True)

    def wait(self) -> None:
        if self._tail is not None:
            self._tail.result()

    def shutdown(self, wait: bool, cancel_futures: bool) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


def create_discussion_record_dir(config: AppConfig, topic: str) -> Path:
    discussions_dir = ensure_dir(config.data_dir / "discussions")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    topic_slug = slugify(topic)[:60] or "discussion"
    return ensure_dir(discussions_dir / f"{stamp}-{topic_slug}")


def prepare_discussion_turn(
    participant: DiscussionParticipant,
    topic: str,
    participants: list[DiscussionParticipant],
    prior_turns: list[DiscussionTurnRecord],
    round_number: int,
    total_rounds: int,
    speak_enabled: bool,
    audio_dir: Path,
) -> PreparedDiscussionTurn:
    response = participant.chat.discuss(
        topic=topic,
        participants=[item.name for item in participants],
        prior_turns=[turn.to_prompt_dict() for turn in prior_turns],
        round_number=round_number,
        total_rounds=total_rounds,
    )
    turn_number = len(prior_turns) + 1
    audio_path = ""
    audio_file = None
    if speak_enabled and participant.voice_available:
        audio_target = audio_dir / f"{turn_number:03d}-round{round_number:02d}-{participant.slug}.wav"
        audio_file = participant.chat.speak(
            response.spoken_answer,
            target=audio_target,
            play=False,
        )
        audio_path = str(audio_file.relative_to(audio_dir.parent))
    return PreparedDiscussionTurn(
        record=DiscussionTurnRecord(
            round_number=round_number,
            turn_number=turn_number,
            speaker=participant.name,
            character_query=participant.query,
            character_slug=participant.manifest.creator_slug,
            text=response.display_answer,
            citations=response.citations,
            audio_path=audio_path,
            created_at=datetime.now().isoformat(timespec="seconds"),
        ),
        audio_file=audio_file,
    )


def run_discussion_turn(
    participant: DiscussionParticipant,
    topic: str,
    participants: list[DiscussionParticipant],
    prior_turns: list[DiscussionTurnRecord],
    round_number: int,
    total_rounds: int,
    speak_enabled: bool,
    audio_dir: Path,
) -> DiscussionTurnRecord:
    prepared = prepare_discussion_turn(
        participant=participant,
        topic=topic,
        participants=participants,
        prior_turns=prior_turns,
        round_number=round_number,
        total_rounds=total_rounds,
        speak_enabled=speak_enabled,
        audio_dir=audio_dir,
    )
    if speak_enabled and prepared.audio_file is not None:
        participant.chat.player.play(prepared.audio_file, wait=True)
    return prepared.record


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

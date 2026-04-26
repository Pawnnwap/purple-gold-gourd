from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .chat.discussion import (
    DiscussionPlaybackQueue,
    DiscussionParticipant,
    DiscussionSpeechSynthesizer,
    create_discussion_record_dir,
    prepare_discussion_turn,
    prepare_host_turn,
    save_discussion_snapshot,
    stop_discussion_audio,
)
from .chat.persona import PersonaChat
from .config import AppConfig
from .media.video import render_discussion_video
from .pipeline import BuildPipeline
from .plugins.tts.shared import clip_audio
from .schema import VoiceSample
from .utils import ensure_dir, write_json


DEFAULT_HOST_VOICE_PROMPT_TEXT = "欢迎来到紫金葫芦，我们开始今天的讨论。"


def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="紫金葫芦 / Purple-Gold Gourd: turn a creator into a chat persona CLI.",
        epilog='Multi-character discussion mode: purple-gold-gourd discuss Alice Bob --topic "..." --rounds 3',
    )
    parser.add_argument("query", nargs="?", help="Creator name, id, handle, or homepage URL.")
    parser.add_argument("--platform", choices=["auto", "youtube", "bilibili"], default="auto")
    parser.add_argument("--top", type=int, default=10, help="How many top videos to process.")
    parser.add_argument("--scan-limit", type=int, default=30, help="How many creator videos to list before selecting.")
    parser.add_argument(
        "--series",
        nargs="+",
        default=[],
        help="1-based ranked video numbers to use for RAG, for example: --series 1 3 8 or --series 1,3,8",
    )
    parser.add_argument(
        "--media",
        nargs="+",
        default=[],
        help="Local audio/video files to import into this character before transcription and skill refresh.",
    )
    parser.add_argument(
        "--include-title-keyword",
        action="append",
        default=[],
        help="Always include creator videos whose title contains this keyword. Can be repeated.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached profile and rebuild.")
    parser.add_argument("--build-only", action="store_true", help="Prepare the persona but do not open chat.")
    parser.add_argument("--speak", action="store_true", help="Start chat with TTS reply synthesis enabled.")
    parser.add_argument("--brevity", action="store_true", help="Ask the persona to keep replies brief.")
    return parser


def build_discuss_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="purple-gold-gourd discuss",
        description="Run a multi-character in-character discussion and save the record.",
    )
    parser.add_argument("characters", nargs="+", help="Characters in speaking order.")
    parser.add_argument("--topic", required=True, help="Discussion topic.")
    parser.add_argument("--rounds", type=int, default=3, help="How many full rounds to run.")
    parser.add_argument("--platform", choices=["auto", "youtube", "bilibili"], default="auto")
    parser.add_argument("--top", type=int, default=10, help="How many top videos to process for each character.")
    parser.add_argument("--scan-limit", type=int, default=30, help="How many creator videos to list before selecting.")
    parser.add_argument(
        "--series",
        nargs="+",
        default=[],
        help="1-based ranked video numbers to use for RAG, for example: --series 1 3 8 or --series 1,3,8",
    )
    parser.add_argument(
        "--include-title-keyword",
        action="append",
        default=[],
        help="Always include creator videos whose title contains this keyword. Can be repeated.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached profiles and rebuild.")
    parser.add_argument("--speak", action="store_true", help="Start the discussion with speech playback enabled.")
    parser.add_argument("--brevity", action="store_true", help="Ask every discussion participant and host to be brief.")
    parser.add_argument(
        "--to-video",
        action="store_true",
        help="Render the discussion to an MP4 with speech, avatar panels, waveform, and timed subtitles. Implies --speak.",
    )
    return parser


def build_set_voice_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="purple-gold-gourd set-voice",
        description="Set the TTS voice reference for a creator from a wav/video file + timestamp range.",
    )
    parser.add_argument("query", help="Creator name, id, or URL.")
    parser.add_argument("audio", help="Path to the source audio or video file.")
    parser.add_argument("timestamps", help="Time range as start-end, e.g. 01:01-01:16 or 1:30:00-1:35:00")
    parser.add_argument("--text", default="", help="Spoken text in the clip (skips auto-transcription).")
    parser.add_argument("--platform", choices=["auto", "youtube", "bilibili"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if raw_args and raw_args[0] == "discuss":
        args = build_discuss_parser().parse_args(raw_args[1:])
        _run_discuss(args)
        return
    if raw_args and raw_args[0] == "set-voice":
        args = build_set_voice_parser().parse_args(raw_args[1:])
        _run_set_voice(args)
        return
    args = build_chat_parser().parse_args(raw_args)
    _run_chat(args)


def _run_chat(args: argparse.Namespace) -> None:
    query = args.query or input("Creator name/id/url: ").strip()
    selected_series_numbers = _parse_series_numbers(args.series)
    config = AppConfig.load()
    config.brevity = args.brevity
    pipeline = BuildPipeline(config)

    print(f"Resolving and building persona for: {query}")
    manifest = pipeline.build(
        query=query,
        platform=args.platform,
        top_n=args.top,
        scan_limit=args.scan_limit,
        series_numbers=selected_series_numbers,
        include_title_keywords=args.include_title_keyword,
        local_media_paths=args.media,
        rebuild=args.rebuild,
    )

    print(f"\nCreator: {manifest.creator.name}")
    print(f"Platform: {manifest.creator.platform}")
    print(f"Language: {manifest.creator.language or 'unknown'}")
    print(f"Profile dir: {manifest.creator_dir}")
    print(f"Documents dir: {Path(manifest.creator_dir) / 'documents'}")
    print(f"Skill: {manifest.skill_path}")
    if manifest.selected_series_numbers:
        print("RAG series numbers: " + ", ".join(str(number) for number in manifest.selected_series_numbers))
    if manifest.voice_sample:
        print(f"Voice prompt: {manifest.voice_sample.audio_path}")
    else:
        print("Voice prompt: not found")

    if args.build_only:
        return

    transcripts = pipeline.load_transcripts(manifest)
    chat = PersonaChat(config, manifest, transcripts)
    speak_enabled = args.speak and manifest.voice_sample is not None

    if manifest.voice_sample:
        voice_status = "on" if speak_enabled else "off"
        print(f"Voice chat available. Current speak mode: {voice_status}.")
    else:
        print("Voice chat unavailable for this profile because no usable voice prompt was found.")

    print("\nChat ready. Type /help for commands.")
    while True:
        try:
            question = input(f"\n{manifest.creator.name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            return
        if question == "/help":
            print("/help | /speak on | /speak off | /rebuild | /calibrate <audio_path> <mm:ss-mm:ss> | /exit")
            print(f"When speak mode is on, replies are synthesized with the '{config.tts_plugin}' TTS plugin and played automatically.")
            print("/calibrate: set a new voice reference from a time range of any audio/video file.")
            continue
        if question == "/speak on":
            if not manifest.voice_sample:
                print("No voice prompt is available.")
            else:
                speak_enabled = True
                print("Reply synthesis enabled.")
            continue
        if question == "/speak off":
            speak_enabled = False
            print("Reply synthesis disabled.")
            continue
        if question == "/rebuild":
            manifest = pipeline.build(
                query=query,
                platform=args.platform,
                top_n=args.top,
                scan_limit=args.scan_limit,
                series_numbers=selected_series_numbers,
                include_title_keywords=args.include_title_keyword,
                local_media_paths=args.media,
                rebuild=True,
            )
            transcripts = pipeline.load_transcripts(manifest)
            chat = PersonaChat(config, manifest, transcripts)
            if not manifest.voice_sample:
                speak_enabled = False
            print("Profile rebuilt.")
            continue

        if question.startswith("/calibrate "):
            result = _calibrate_voice(question, manifest, chat, pipeline, config)
            if result is not None:
                manifest = result
            continue

        response = chat.answer(question)
        print(f"\n{manifest.creator.name}: {response.display_answer}")
        if response.citations:
            print("Refs: " + " | ".join(response.citations))
        if speak_enabled:
            try:
                audio_path = chat.speak(response.spoken_answer)
                print(f"Speaking from: {audio_path}")
            except Exception as exc:
                print(f"Audio synthesis failed: {exc}")


def _run_set_voice(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        sys.exit(f"File not found: {audio_path}")
    try:
        start_ms, end_ms = _parse_time_range(args.timestamps)
    except ValueError as exc:
        sys.exit(f"Bad time range: {exc}")
    if end_ms <= start_ms:
        sys.exit("End time must be after start time.")

    config = AppConfig.load()
    pipeline = BuildPipeline(config)
    print(f"Loading persona for: {args.query}")
    manifest = pipeline.build(query=args.query, platform=args.platform)

    voice_dir = ensure_dir(Path(manifest.creator_dir) / "voice")
    clip_name = f"manual-{start_ms}-{end_ms}-voice-prompt.wav"
    clip_path = voice_dir / clip_name

    print(f"Extracting {args.timestamps} from {audio_path.name} ...")
    try:
        clip_audio(config.ffmpeg_path, audio_path, clip_path, start_ms, end_ms)
    except Exception as exc:
        sys.exit(f"Audio extraction failed: {exc}")

    prompt_text = args.text.strip()
    if not prompt_text:
        print("Transcribing clip ...")
        try:
            prompt_text = pipeline.transcriber.transcribe_text(clip_path).strip()
        except Exception:
            pass
    if not prompt_text:
        sys.exit("Could not transcribe clip. Pass --text with the spoken content.")

    new_sample = VoiceSample(
        audio_path=str(clip_path),
        prompt_text=prompt_text,
        start_ms=start_ms,
        end_ms=end_ms,
        source_audio_path=str(audio_path),
    )
    updated = replace(manifest, voice_sample=new_sample)
    write_json(Path(manifest.creator_dir) / "manifest.json", updated.to_dict())
    print(f"Voice set for {manifest.creator.name}")
    print(f"  Source : {audio_path}")
    print(f"  Clip   : {clip_path}")
    print(f"  Text   : {prompt_text}")


def _run_discuss(args: argparse.Namespace) -> None:
    if args.rounds <= 0:
        raise ValueError("Discussion rounds must be positive.")

    config = AppConfig.load()
    config.brevity = args.brevity
    pipeline = BuildPipeline(config)
    selected_series_numbers = _parse_series_numbers(args.series)
    participants = _build_discussion_participants(
        queries=args.characters,
        args=args,
        config=config,
        pipeline=pipeline,
        selected_series_numbers=selected_series_numbers,
    )
    host_voice_sample = _ensure_host_voice_sample(config) if args.speak or args.to_video else None
    record_dir = create_discussion_record_dir(config, args.topic)
    audio_dir = ensure_dir(record_dir / "audio")
    started_at = datetime.now().isoformat(timespec="seconds")
    turns: list = []
    if args.to_video:
        missing_voices = [participant.name for participant in participants if not participant.voice_available]
        if missing_voices:
            names = ", ".join(missing_voices)
            raise ValueError(f"--to-video requires a usable voice prompt for every participant. Missing: {names}")
    speak_enabled = (args.speak or args.to_video) and any(participant.voice_available for participant in participants)
    live_playback_enabled = args.speak and not args.to_video
    auto_continue_enabled = args.to_video or not sys.stdin.isatty()
    playback_queue = DiscussionPlaybackQueue()
    speech_synthesizer = DiscussionSpeechSynthesizer()

    print(f"Discussion topic: {args.topic}")
    print("Participants: " + " -> ".join(participant.name for participant in participants))
    print(f"Rounds: {args.rounds}")
    print(f"Record dir: {record_dir}")
    for participant in participants:
        status = "available" if participant.voice_available else "not available"
        print(f"Voice for {participant.name}: {status}")
    if args.speak or args.to_video:
        status = "available" if host_voice_sample is not None else "not available"
        suffix = f" ({host_voice_sample.audio_path})" if host_voice_sample is not None else ""
        print(f"Voice for 主持人: {status}{suffix}")
    if args.speak and not speak_enabled:
        print("Speak mode requested, but no participant currently has a usable voice prompt.")
    if args.to_video:
        print("Video mode: speech synthesis enabled; live playback skipped while the MP4 is rendered.")

    save_discussion_snapshot(
        record_dir=record_dir,
        topic=args.topic,
        requested_rounds=args.rounds,
        participants=participants,
        turns=turns,
        started_at=started_at,
    )

    try:
        if auto_continue_enabled:
            should_continue = True
        else:
            should_continue, speak_enabled = _discussion_control_prompt(
                participants=participants,
                speak_enabled=speak_enabled,
                prompt_text="Press Enter to start, or use /speak on, /speak off, /help, /exit: ",
            )
        if not should_continue:
            save_discussion_snapshot(
                record_dir=record_dir,
                topic=args.topic,
                requested_rounds=args.rounds,
                participants=participants,
                turns=turns,
                started_at=started_at,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
            print(f"Discussion exited. Partial record saved to: {record_dir}")
            return

        host_tts_participant = participants[0] if participants else None

        for round_number in range(1, args.rounds + 1):
            print(f"\n=== Round {round_number}/{args.rounds} ===")

            # Host intro turn
            host_turn = prepare_host_turn(
                topic=args.topic,
                participants=participants,
                prior_turns=turns,
                round_number=round_number,
                total_rounds=args.rounds,
            )
            host_job = None
            if speak_enabled and host_tts_participant and host_voice_sample:
                host_job = speech_synthesizer.submit(
                    participant=host_tts_participant,
                    turn=host_turn,
                    spoken_answer=host_turn.spoken_text,
                    audio_dir=audio_dir,
                    voice_sample=host_voice_sample,
                )
                if live_playback_enabled:
                    playback_queue.enqueue_future(host_tts_participant, host_job.future)
            turns.append(host_turn)
            save_discussion_snapshot(
                record_dir=record_dir,
                topic=args.topic,
                requested_rounds=args.rounds,
                participants=participants,
                turns=turns,
                started_at=started_at,
            )
            print(f"\n{host_turn.speaker}: {host_turn.text}")
            if host_job is not None:
                status = "synthesizing" if not host_job.future.done() else "ready"
                print(f"Host audio: {host_turn.audio_path} ({status})")

            for participant in participants:
                prepared = prepare_discussion_turn(
                    participant=participant,
                    topic=args.topic,
                    participants=participants,
                    prior_turns=turns,
                    round_number=round_number,
                    total_rounds=args.rounds,
                )
                turn = prepared.record
                job = None
                if speak_enabled and participant.voice_available:
                    job = speech_synthesizer.submit(
                        participant=participant,
                        turn=turn,
                        spoken_answer=prepared.spoken_answer,
                        audio_dir=audio_dir,
                    )
                    prepared.audio_file = job.audio_file
                    if live_playback_enabled:
                        playback_queue.enqueue_future(participant, job.future)
                turns.append(turn)
                save_discussion_snapshot(
                    record_dir=record_dir,
                    topic=args.topic,
                    requested_rounds=args.rounds,
                    participants=participants,
                    turns=turns,
                    started_at=started_at,
                )
                print(f"\n{participant.name}: {turn.text}")
                if turn.citations:
                    print("Refs: " + " | ".join(turn.citations))
                if speak_enabled:
                    if prepared.audio_file is not None:
                        status = "synthesizing" if not job.future.done() else "ready"
                        print(f"Audio: {turn.audio_path} ({status})")
                    elif not participant.voice_available:
                        print("Audio: skipped because this character has no voice prompt.")
            if speak_enabled and live_playback_enabled:
                playback_queue.wait()
            if round_number >= args.rounds:
                break
            if auto_continue_enabled:
                should_continue = True
            else:
                should_continue, speak_enabled = _discussion_control_prompt(
                    participants=participants,
                    speak_enabled=speak_enabled,
                    prompt_text="Press Enter for the next round, or use /speak on, /speak off, /help, /exit: ",
                )
            if not should_continue:
                finished_at = datetime.now().isoformat(timespec="seconds")
                save_discussion_snapshot(
                    record_dir=record_dir,
                    topic=args.topic,
                    requested_rounds=args.rounds,
                    participants=participants,
                    turns=turns,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                print(f"Discussion exited. Partial record saved to: {record_dir}")
                return

        if speech_synthesizer.has_jobs:
            print("Waiting for speech synthesis ...")
            speech_synthesizer.wait_all()
        finished_at = datetime.now().isoformat(timespec="seconds")
        save_discussion_snapshot(
            record_dir=record_dir,
            topic=args.topic,
            requested_rounds=args.rounds,
            participants=participants,
            turns=turns,
            started_at=started_at,
            finished_at=finished_at,
        )
        if args.to_video:
            print("Rendering discussion video ...")
            video = render_discussion_video(
                ffmpeg_path=config.ffmpeg_path,
                record_dir=record_dir,
                topic=args.topic,
                participants=participants,
                turns=turns,
            )
            print(f"Video: {video.video_path}")
            print(f"Subtitles: {video.subtitles_path}")
        print(f"\nDiscussion finished. Record saved to: {record_dir}")
    finally:
        stop_discussion_audio(participants)
        speech_synthesizer.shutdown(wait=False, cancel_futures=True)
        playback_queue.shutdown(wait=False, cancel_futures=True)


def _build_discussion_participants(
    queries: list[str],
    args: argparse.Namespace,
    config: AppConfig,
    pipeline: BuildPipeline,
    selected_series_numbers: list[int],
) -> list[DiscussionParticipant]:
    participants: list[DiscussionParticipant] = []
    for query in queries:
        print(f"Resolving and building persona for discussion: {query}")
        manifest = pipeline.build(
            query=query,
            platform=args.platform,
            top_n=args.top,
            scan_limit=args.scan_limit,
            series_numbers=selected_series_numbers,
            include_title_keywords=args.include_title_keyword,
            limit_to_target_videos=True,
            rebuild=args.rebuild,
        )
        transcripts = pipeline.load_transcripts(manifest)
        chat = PersonaChat(config, manifest, transcripts)
        participants.append(
            DiscussionParticipant(
                query=query,
                manifest=manifest,
                chat=chat,
            ),
        )
    return participants


def _ensure_host_voice_sample(config: AppConfig) -> VoiceSample:
    audio_path = _resolve_config_path(config.host_voice_audio_path, config.workspace_dir)
    source_audio_path = _resolve_config_path(config.host_voice_source_audio_path, config.workspace_dir)
    metadata_path = audio_path.with_suffix(".json")
    prompt_text = config.host_voice_prompt_text.strip() or DEFAULT_HOST_VOICE_PROMPT_TEXT

    if not source_audio_path.exists():
        raise ValueError(
            "Host voice reference source is missing. "
            f"Expected qwen3 TTS sample at: {source_audio_path}"
        )
    ensure_dir(audio_path.parent)
    needs_copy = True
    if audio_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            needs_copy = (
                str(metadata.get("source_audio_path") or "") != str(source_audio_path)
                or str(metadata.get("generated_by") or "") != "qwen3-tts-sample"
            )
        except Exception:
            needs_copy = True
    if needs_copy:
        shutil.copy2(source_audio_path, audio_path)
        cache_path = audio_path.with_name(audio_path.stem + "-profile.pkl")
        if cache_path.exists():
            cache_path.unlink()
    write_json(
        metadata_path,
        {
            "audio_path": str(audio_path),
            "prompt_text": prompt_text,
            "source_audio_path": str(source_audio_path),
            "generated_by": "qwen3-tts-sample",
        },
    )
    return VoiceSample(
        audio_path=str(audio_path),
        prompt_text=prompt_text,
        start_ms=0,
        end_ms=0,
        source_audio_path=str(source_audio_path),
    )


def _resolve_config_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _discussion_control_prompt(
    participants: list[DiscussionParticipant],
    speak_enabled: bool,
    prompt_text: str,
) -> tuple[bool, bool]:
    while True:
        try:
            command = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            stop_discussion_audio(participants)
            print()
            return False, speak_enabled
        if not command:
            return True, speak_enabled
        if command == "/help":
            print("/help | /speak on | /speak off | /exit")
            print("Press Enter to continue the discussion.")
            continue
        if command == "/speak on":
            if not any(participant.voice_available for participant in participants):
                print("No discussion participant currently has a usable voice prompt.")
                continue
            speak_enabled = True
            print("Discussion speech playback enabled.")
            continue
        if command == "/speak off":
            speak_enabled = False
            stop_discussion_audio(participants)
            print("Discussion speech playback disabled.")
            continue
        if command in {"/exit", "/quit"}:
            stop_discussion_audio(participants)
            return False, speak_enabled
        print("Unknown discussion command. Type /help for options.")


def _parse_series_numbers(raw_values: list[str]) -> list[int]:
    parsed: list[int] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            text = part.strip()
            if not text:
                continue
            value = int(text)
            if value <= 0:
                raise ValueError("Series numbers must be positive 1-based integers.")
            parsed.append(value)
    return parsed


def _calibrate_voice(
    command: str,
    manifest,
    chat: PersonaChat,
    pipeline: BuildPipeline,
    config: AppConfig,
):
    """Handle /calibrate <audio_path> <mm:ss-mm:ss>. Returns updated manifest or None on failure."""
    parts = command.split()
    if len(parts) < 3:
        print("Usage: /calibrate <audio_path> <start_time-end_time>  e.g. /calibrate rec.mp4 00:10-00:20")
        return None

    range_str = parts[-1]
    audio_path = Path(" ".join(parts[1:-1])).expanduser().resolve()

    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        return None

    try:
        start_ms, end_ms = _parse_time_range(range_str)
    except ValueError as exc:
        print(f"Bad time range: {exc}")
        return None

    if end_ms <= start_ms:
        print("End time must be after start time.")
        return None

    voice_dir = ensure_dir(Path(manifest.creator_dir) / "voice")
    clip_name = f"calibrated-{start_ms}-{end_ms}-voice-prompt.wav"
    clip_path = voice_dir / clip_name

    print(f"Extracting {range_str} from {audio_path.name} ...")
    try:
        clip_audio(config.ffmpeg_path, audio_path, clip_path, start_ms, end_ms)
    except Exception as exc:
        print(f"Audio extraction failed: {exc}")
        return None

    print("Transcribing voice clip ...")
    try:
        prompt_text = pipeline.transcriber.transcribe_text(clip_path).strip()
    except Exception:
        prompt_text = ""

    if not prompt_text:
        try:
            prompt_text = input("Could not auto-transcribe. Enter the spoken text for this clip: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCalibration cancelled.")
            return None
    if not prompt_text:
        print("No transcript provided. Calibration cancelled.")
        return None

    print(f"Transcript: {prompt_text}")

    new_sample = VoiceSample(
        audio_path=str(clip_path),
        prompt_text=prompt_text,
        start_ms=start_ms,
        end_ms=end_ms,
        source_audio_path=str(audio_path),
    )

    updated_manifest = replace(manifest, voice_sample=new_sample)
    manifest_path = Path(manifest.creator_dir) / "manifest.json"
    write_json(manifest_path, updated_manifest.to_dict())
    chat.manifest = updated_manifest

    if chat.synthesizer is not None:
        chat.synthesizer._voice_clone_key = None
        chat.synthesizer._voice_clone_prompt = None

    print(f"Voice profile updated. Clip saved to: {clip_path}")
    return updated_manifest


def _parse_time_range(range_str: str) -> tuple[int, int]:
    if "-" not in range_str:
        raise ValueError("expected format like 00:10-00:20 or 1:30:00-1:35:00")
    dash = range_str.index("-")
    # Find the dash that separates start and end: it must not be within a colon-group.
    # Strategy: find the first '-' that is preceded by a digit and followed by a digit or colon.
    for i, ch in enumerate(range_str):
        if ch == "-" and i > 0 and range_str[i - 1].isdigit() and i + 1 < len(range_str) and (range_str[i + 1].isdigit()):
            dash = i
            break
    start_ms = _parse_time_ms(range_str[:dash])
    end_ms = _parse_time_ms(range_str[dash + 1:])
    return start_ms, end_ms


def _parse_time_ms(token: str) -> int:
    token = token.strip()
    parts = token.split(":")
    if len(parts) == 1:
        return int(float(parts[0]) * 1000)
    if len(parts) == 2:
        return int((int(parts[0]) * 60 + float(parts[1])) * 1000)
    if len(parts) == 3:
        return int((int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])) * 1000)
    raise ValueError(f"cannot parse time: {token!r}")


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()

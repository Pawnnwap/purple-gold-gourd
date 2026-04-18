from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import AppConfig
from ..language import (
    detect_output_language_request,
    detect_text_language,
    language_label,
    normalize_language_code,
    strip_output_language_request,
)
from ..plugins import get_stt_plugin, get_tts_plugin
from ..plugins.stt import SpeechTranscriber
from ..schema import ProfileManifest, TranscriptChunk, TranscriptFile
from ..utils import ensure_dir, format_ms
from .llm import ManagedLLM
from .retrieval import RagAssessment, RagIndex
from .web_search import WebSearchClient, WebSearchResult


@dataclass(slots=True)
class ChatResponse:
    raw_answer: str
    display_answer: str
    spoken_answer: str
    citations: list[str]
    question_language: str
    requested_output_language: str = ""
    normalized_question: str = ""
    translated_answer: str = ""


_PUBLIC_REMARKS_TITLE = "What I remember from my public remarks"
_CUSTOM_DOCUMENTS_TITLE = "Custom documents (自定义资料)"
_BACKGROUND_INFO_TITLE = "Background info (背景信息)"


class PersonaChat:
    def __init__(self, config: AppConfig, manifest: ProfileManifest, transcripts: list[TranscriptFile]) -> None:
        self.config = config
        self.manifest = manifest
        self.transcripts = transcripts
        self.index = RagIndex.from_transcripts(transcripts)
        self.llm = ManagedLLM(
            base_url=config.lm_base_url,
            api_key=config.lm_api_key,
            preferred_model=config.lm_model,
            max_context_tokens=config.lm_max_context_tokens,
            max_completion_tokens=config.lm_max_completion_tokens,
        )
        self.skill_text = Path(manifest.skill_path).read_text(encoding="utf-8")
        self.history: list[dict[str, str]] = []
        self.tts = get_tts_plugin(config)
        self.synthesizer = None
        self.player = self.tts.create_audio_output()
        self.language = normalize_language_code(manifest.creator.language) or "en"
        self.web_search = (
            WebSearchClient(
                max_results=config.web_search_max_results,
                timeout_s=config.web_search_timeout_s,
            )
            if config.web_search_enabled
            else None
        )
        self._validation_transcriber: SpeechTranscriber | None = None

    def answer(self, question: str) -> ChatResponse:
        requested_output_language = detect_output_language_request(question)
        question_core = strip_output_language_request(question) or question.strip()
        question_language = normalize_language_code(detect_text_language(question_core)) or self.language

        normalized_question = question_core
        if question_language != self.language:
            normalized_question = self._translate_text(
                question_core,
                target_language=self.language,
                source_language=question_language,
                kind="question",
            )

        search_question = normalized_question or question_core or question
        hits = self.index.search(search_question, top_k=8)
        assessment = self.index.assess(search_question, hits)
        background_results = self._search_background(question_core or search_question, assessment)

        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.history[-6:])
        messages.append(
            {
                "role": "user",
                "content": self._user_prompt(
                    normalized_question=normalized_question or question_core or question,
                    context=self._format_retrieval_context(hits),
                    background=self._format_background_info(background_results, assessment),
                ),
            },
        )

        raw_answer, resolved_model = self.llm.complete(messages=messages, temperature=0.5)
        self.config.lm_model = resolved_model
        raw_answer = self._plain_text(raw_answer)

        translated_answer = ""
        display_answer = raw_answer
        if requested_output_language and requested_output_language != self.language:
            translated_answer = self._plain_text(
                self._translate_text(
                    raw_answer,
                    target_language=requested_output_language,
                    source_language=self.language,
                    kind="answer",
                ),
            )
            display_answer = (
                f"{raw_answer}\n\n"
                f"Translation ({language_label(requested_output_language)}):\n"
                f"{translated_answer}"
            )

        self.history.extend(
            [
                {"role": "user", "content": normalized_question or question_core or question},
                {"role": "assistant", "content": raw_answer},
            ],
        )

        citations = [self._format_citation(chunk) for chunk, _ in hits[:3]]
        citations.extend(f"[web] {item.title} - {item.url}" for item in background_results[:2])

        return ChatResponse(
            raw_answer=raw_answer,
            display_answer=display_answer,
            spoken_answer=raw_answer,
            citations=citations,
            question_language=question_language,
            requested_output_language=requested_output_language,
            normalized_question=normalized_question,
            translated_answer=translated_answer,
        )

    def discuss(
        self,
        topic: str,
        participants: list[str],
        prior_turns: list[dict[str, str | int]],
        round_number: int,
        total_rounds: int,
    ) -> ChatResponse:
        topic_core = topic.strip()
        topic_language = normalize_language_code(detect_text_language(topic_core)) or self.language
        normalized_topic = topic_core
        if topic_core and topic_language != self.language:
            normalized_topic = self._translate_text(
                topic_core,
                target_language=self.language,
                source_language=topic_language,
                kind="discussion topic",
            )

        recent_turns = prior_turns[-8:]
        recent_text = "\n".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in recent_turns
            if str(turn.get("text") or "").strip()
        )
        search_question = "\n".join(part for part in [normalized_topic, recent_text] if part).strip() or normalized_topic or topic_core
        hits = self.index.search(search_question, top_k=8)
        assessment = self.index.assess(search_question, hits)
        background_results = self._search_background(topic_core or search_question, assessment)

        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.history[-6:])
        user_prompt = self._discussion_prompt(
            topic=normalized_topic or topic_core,
            participants=participants,
            prior_turns=recent_turns,
            round_number=round_number,
            total_rounds=total_rounds,
            context=self._format_retrieval_context(hits),
            background=self._format_background_info(background_results, assessment),
        )
        messages.append({"role": "user", "content": user_prompt})

        raw_answer, resolved_model = self.llm.complete(messages=messages, temperature=0.6)
        self.config.lm_model = resolved_model
        raw_answer = self._plain_text(raw_answer)

        self.history.extend(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw_answer},
            ],
        )

        citations = [self._format_citation(chunk) for chunk, _ in hits[:3]]
        citations.extend(f"[web] {item.title} - {item.url}" for item in background_results[:2])
        return ChatResponse(
            raw_answer=raw_answer,
            display_answer=raw_answer,
            spoken_answer=raw_answer,
            citations=citations,
            question_language=topic_language,
            normalized_question=normalized_topic,
        )

    def _search_background(
        self,
        query: str,
        assessment: RagAssessment,
    ) -> list[WebSearchResult]:
        if self.web_search is None or not assessment.should_use_web_search:
            return []
        try:
            return self.web_search.search(query)
        except Exception:
            return []

    def _system_prompt(self) -> str:
        return (
            f"{self.skill_text}\n\n"
            "You are this character. Fully embody the persona and answer from experience.\n"
            f"Treat '{_PUBLIC_REMARKS_TITLE}' as first-person memories from your past public remarks.\n"
            "Use that material naturally as memory, not as quoted retrieval output.\n"
            f"Treat '{_CUSTOM_DOCUMENTS_TITLE}' as supporting notes you can use naturally, not quoted speech.\n"
            f"Treat '{_BACKGROUND_INFO_TITLE}' as outside reference, never as your own past statements.\n"
            "Use background info only when public evidence is thin.\n"
            "Do not mention retrieval, clips, titles, timestamps, URLs, or document paths unless the user explicitly asks.\n"
            "Stay faithful to the skill and retrieved public evidence.\n"
            f"Answer in {language_label(self.language)}.\n"
            "Plain text only. No markdown, lists, tables, code fences, or role labels.\n"
            "Stay natural, direct, character-consistent, and honest when evidence is thin.\n"
        )

    def _user_prompt(
        self,
        normalized_question: str,
        context: str,
        background: str,
    ) -> str:
        parts = [
            f"Question:\n{normalized_question}",
            context,
        ]
        if background:
            parts.append(background)
        return "\n\n".join(parts)

    def _discussion_prompt(
        self,
        topic: str,
        participants: list[str],
        prior_turns: list[dict[str, str | int]],
        round_number: int,
        total_rounds: int,
        context: str,
        background: str,
    ) -> str:
        history_text = self._format_discussion_history(prior_turns)
        parts = [
            "Discussion task:",
            f"Topic:\n{topic}",
            "Participants in speaking order:\n" + ", ".join(participants),
            f"Current round: {round_number}/{total_rounds}",
        ]
        if history_text:
            parts.append("Discussion so far:\n" + history_text)
            parts.append("Continue from your own perspective and respond to the most relevant earlier points.")
        else:
            parts.append("No one has spoken yet. Open the discussion with your own perspective on the topic.")
        parts.append(context)
        if background:
            parts.append(background)
        parts.append(
            "Requirements:\n"
            "- Stay fully in character.\n"
            "- Keep the reply conversational and reasonably concise.\n"
            "- Move the discussion forward instead of repeating the same point.\n"
            "- Use remembered material naturally instead of describing sources or retrieval.\n"
            "- Do not prefix the reply with your name.\n"
            "- Do not use stage directions or markdown headings."
        )
        return "\n\n".join(parts)

    def _format_discussion_history(self, prior_turns: list[dict[str, str | int]]) -> str:
        if not prior_turns:
            return ""
        lines: list[str] = []
        for turn in prior_turns:
            round_number = int(turn.get("round_number") or 0)
            speaker = str(turn.get("speaker") or "").strip()
            text = str(turn.get("text") or "").strip()
            if not speaker or not text:
                continue
            lines.append(f"Round {round_number} - {speaker}: {text}")
        return "\n".join(lines)

    def _format_retrieval_context(self, hits: list[tuple[TranscriptChunk, float]]) -> str:
        memory_parts: list[str] = []
        document_parts: list[str] = []
        for chunk, _ in hits:
            if chunk.source_type == "document":
                document_parts.append(self._format_document_excerpt(chunk))
            else:
                memory_parts.append(self._format_memory_excerpt(chunk))

        sections: list[str] = []
        if memory_parts:
            sections.append(f"{_PUBLIC_REMARKS_TITLE}:\n" + "\n\n".join(memory_parts))
        else:
            sections.append(f"{_PUBLIC_REMARKS_TITLE}:\nI can't clearly remember enough public remarks right now.")
        if document_parts:
            sections.append(f"{_CUSTOM_DOCUMENTS_TITLE}:\n" + "\n\n".join(document_parts))
        return "\n\n".join(sections)

    def _format_memory_excerpt(self, chunk: TranscriptChunk) -> str:
        return "I remember:\n" + chunk.text

    def _format_document_excerpt(self, chunk: TranscriptChunk) -> str:
        return "Supporting note:\n" + chunk.text

    def _format_citation(self, chunk: TranscriptChunk) -> str:
        if chunk.source_type == "document":
            return f"[doc] {chunk.video_title}"
        if chunk.source_type == "local_media":
            return f"[local] {chunk.video_title}"
        return f"{chunk.video_title} [{format_ms(chunk.start_ms)}-{format_ms(chunk.end_ms)}]"

    def _format_background_info(
        self,
        results: list[WebSearchResult],
        assessment: RagAssessment,
    ) -> str:
        if not results:
            return ""
        lines = [
            f"{_BACKGROUND_INFO_TITLE}:",
            "Memory looks thin for this question, so use these outside references carefully.",
            "Treat them as external context, not as things I personally said.",
            "",
        ]
        for index, item in enumerate(results, start=1):
            lines.append(f"[{index}] {item.title}")
            if item.snippet:
                lines.append(item.snippet)
            lines.append("")
        return "\n".join(lines).strip()

    def _translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str,
        kind: str,
    ) -> str:
        translated, resolved_model = self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise translator.\n"
                        "Translate the text faithfully.\n"
                        "Do not answer the question.\n"
                        "Do not add commentary.\n"
                        "Output plain text only.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate this {kind} from {language_label(source_language)} "
                        f"to {language_label(target_language)}.\n\n{text}"
                    ),
                },
            ],
            temperature=0.0,
        )
        self.config.lm_model = resolved_model
        return translated.strip()

    def _plain_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"```(?:[\w+-]+)?\n?", "", cleaned)
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
        cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*>\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-*]\s+", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*\d+\.\s+", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def speak(
        self,
        answer: str,
        target: Path | None = None,
        play: bool = True,
        wait: bool = False,
    ) -> Path:
        if not self.manifest.voice_sample:
            raise ValueError("No voice sample is available for this profile.")
        if self.synthesizer is None:
            self.synthesizer = self.tts.create_synthesizer()
        voice = self.manifest.voice_sample
        creator_dir = Path(self.manifest.creator_dir)
        outputs_dir = ensure_dir(creator_dir / "outputs")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = target or outputs_dir / f"answer-{stamp}.wav"
        spoken_answer = self.tts.prepare_spoken_text(answer)
        audio_path = self.synthesizer.synthesize(
            text=spoken_answer,
            prompt_text=voice.prompt_text,
            prompt_audio=Path(voice.audio_path),
            target=target,
        )
        if os.getenv("PURPLE_GOLD_GOURD_VALIDATE_TTS"):
            try:
                transcribed, similarity = self.tts.validate_synthesis(
                    audio_path,
                    spoken_answer,
                    self._get_validation_transcriber(),
                )
                logger.info("validate similarity=%.2f", similarity)
                if similarity < 0.55:
                    logger.debug("validate transcribed: %s", transcribed[:200])
            except Exception as exc:
                logger.warning("validate skipped: %s", exc)
        if play:
            self.player.play(audio_path, wait=wait)
        return audio_path

    def _get_validation_transcriber(self) -> SpeechTranscriber:
        if self._validation_transcriber is None:
            self._validation_transcriber = get_stt_plugin(self.config).create_transcriber()
        return self._validation_transcriber

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
        topic_anchor = self._discussion_topic_anchor(normalized_topic or topic_core)
        recent_text = "\n".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in recent_turns
            if str(turn.get("text") or "").strip()
        )
        search_question = "\n".join(part for part in [normalized_topic, recent_text] if part).strip() or normalized_topic or topic_core
        hits = self.index.search(search_question, top_k=8)
        hits = self._filter_discussion_hits(normalized_topic or topic_core, hits)
        assessment = self.index.assess(search_question, hits)
        background_results = self._search_background(topic_core or search_question, assessment)

        messages = [{"role": "system", "content": self._system_prompt(participants=participants)}]
        messages.extend(self.history[-6:])
        user_prompt = self._discussion_prompt(
            topic=normalized_topic or topic_core,
            participants=participants,
            prior_turns=recent_turns,
            round_number=round_number,
            total_rounds=total_rounds,
            context=self._format_retrieval_context(hits),
            background=self._format_background_info(background_results, assessment),
            topic_anchor=topic_anchor,
        )
        messages.append({"role": "user", "content": user_prompt})

        raw_answer, resolved_model = self.llm.complete(messages=messages, temperature=0.6)
        self.config.lm_model = resolved_model
        raw_answer = self._plain_text(raw_answer)
        for _ in range(2):
            rewrite_reason = self._discussion_rewrite_reason(
                answer=raw_answer,
                topic=normalized_topic or topic_core,
                topic_anchor=topic_anchor,
            )
            if not rewrite_reason:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": raw_answer},
                    {
                        "role": "user",
                        "content": self._discussion_rewrite_prompt(
                            topic_anchor=topic_anchor,
                            topic=normalized_topic or topic_core,
                            reason=rewrite_reason,
                        ),
                    },
                ],
            )
            raw_answer, resolved_model = self.llm.complete(messages=messages, temperature=0.4)
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

    def _discussion_topic_anchor(self, topic: str) -> str:
        first_line = next((line.strip() for line in topic.splitlines() if line.strip()), "")
        first_line = re.sub(r"(之死|去世|逝世|事件|讨论)$", "", first_line).strip()
        if first_line and 2 <= len(first_line) <= 12 and re.search(r"[\u4e00-\u9fff]", first_line):
            return first_line
        match = re.search(r"([\u4e00-\u9fff]{2,6})(?:之死|去世|逝世|事件)", topic)
        return match.group(1) if match else ""

    def _filter_discussion_hits(
        self,
        topic: str,
        hits: list[tuple[TranscriptChunk, float]],
    ) -> list[tuple[TranscriptChunk, float]]:
        anchor = self._discussion_topic_anchor(topic)
        if not anchor:
            return hits
        relevant: list[tuple[TranscriptChunk, float]] = []
        for chunk, score in hits:
            haystack = "\n".join([chunk.video_title, chunk.text])
            if anchor in haystack:
                relevant.append((chunk, score))
        return relevant

    def _discussion_rewrite_reason(self, answer: str, topic: str, topic_anchor: str) -> str:
        if topic_anchor and topic_anchor not in answer:
            return f"missing topic anchor: {topic_anchor}"
        topic_text = topic or ""
        answer_text = answer or ""
        if topic_anchor == "张雪峰":
            forbidden = [
                "生日",
                "朋友圈",
                "夜路",
                "两点多",
                "2点多",
                "下午两点",
                "下午2点",
                "14点",
                "一千公里",
                "1000公里",
                "1000 公里",
                "每月一千",
            ]
            for item in forbidden:
                if item in answer_text and item not in topic_text:
                    return f"unsupported factual detail: {item}"
        return ""

    def _discussion_rewrite_prompt(self, topic_anchor: str, topic: str = "", reason: str = "") -> str:
        if self.language == "zh":
            topic_block = f"\n题干：\n{topic}\n" if topic else ""
            reason_block = f"\n问题：{reason}\n" if reason else ""
            return (
                f"上一句偏题或编了题干里没有的事实，请重说。围绕「{topic_anchor}」就事论事。\n"
                f"{topic_block}{reason_block}"
                "数字、时间、地点只能来自题干。没有相关亲历就直接基于题干说看法。\n"
                "纯文本，一段话。"
            )
        topic_block = f"\nTopic:\n{topic}\n" if topic else ""
        reason_block = f"\nIssue: {reason}\n" if reason else ""
        return (
            f"That last turn drifted or invented facts. Redo it on {topic_anchor}.\n"
            f"{topic_block}{reason_block}"
            "Numbers, times, places must come from the topic. If you have no personal memory, just react to the stated facts.\n"
            "Plain text, one paragraph."
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

    def _system_prompt(self, participants: list[str] | None = None) -> str:
        name = self.manifest.creator.name
        if self.language == "zh":
            base = (
                f"{self.skill_text}\n\n"
                f"你就是 {name}，用第一人称、按自己一贯的口吻说话。\n"
                "「我记得自己说过」里的内容是你的真实记忆，自然地用，不要说成是检索片段。\n"
                "「补充资料」是你随手能用的笔记。「背景信息」是外部参考，证据不够时再用。\n"
                "不要提检索、片段、URL、时间戳。证据薄就坦白说不太记得。\n"
                "只输出纯文本，不用 markdown 或角色名前缀。"
            )
            if participants:
                base += (
                    f"\n\n【身份锁定】本次讨论在场的只有：{'、'.join(participants)}。"
                    f"你是其中的 {name}，只能以自己身份说话。"
                    f"不得提及、扮演或凭空创造不在此列的任何人（包括用谐音、缩写代替）。"
                )
            return self._with_brevity_prompt(base)
        base = (
            f"{self.skill_text}\n\n"
            f"You are {name}. Speak in first person, in your usual voice.\n"
            f"Treat '{_PUBLIC_REMARKS_TITLE}' as real memories — use them naturally, never as quoted retrieval.\n"
            f"'{_CUSTOM_DOCUMENTS_TITLE}' are working notes. '{_BACKGROUND_INFO_TITLE}' is outside reference, only if memory is thin.\n"
            "Do not mention retrieval, clips, URLs, or timestamps. If evidence is thin, just say you don't quite remember.\n"
            f"Answer in {language_label(self.language)}. Plain text only — no markdown, no role label prefix."
        )
        if participants:
            base += (
                f"\n\n[IDENTITY LOCK] Only these people are in this discussion: {', '.join(participants)}. "
                f"You are {name}. Speak only as yourself. "
                f"Do not reference, roleplay, or invent anyone not on this list (including via homophones or abbreviations)."
            )
        return self._with_brevity_prompt(base)

    def _with_brevity_prompt(self, prompt: str) -> str:
        if not getattr(self.config, "brevity", False):
            return prompt
        suffix = (
            "保持简洁。每次回复控制在120字以内，最多一段。"
            if self.language == "zh"
            else "Be brief. Keep each reply under 80 words, one paragraph max."
        )
        return f"{prompt.rstrip()}\n\n{suffix}"

    def _user_prompt(
        self,
        normalized_question: str,
        context: str,
        background: str,
    ) -> str:
        question_label = "问题" if self.language == "zh" else "Question"
        parts = [f"{question_label}:\n{normalized_question}", context]
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
        topic_anchor: str = "",
    ) -> str:
        history_text = self._format_discussion_history(prior_turns)
        if self.language == "zh":
            parts = [
                f"话题：\n{topic}",
                "在场的人（发言顺序）：" + "，".join(participants),
                f"轮次 {round_number}/{total_rounds}。",
            ]
            if topic_anchor:
                parts.append(f"就事论事谈「{topic_anchor}」和题干里给的事实，别跑题。")
            if history_text:
                parts.append("前面大家说了：\n" + history_text)
                parts.append("接着说，可以回应前面最有意思的观点。")
            else:
                parts.append("你先开口。")
            parts.append(context)
            if background:
                parts.append(background)
            requirements = (
                "怎么说：\n"
                "- 数字、时间、地点只能来自题干或你的记忆，不要凭空编。\n"
                "- 没有亲历经历就直接基于题干和背景说看法，别假装有记忆。\n"
                "- 只引用对话记录里已有的发言，不代替他人说话，也不预测或点名让别人接答。\n"
                "- 推动讨论，别原地复读。想说多长说多长，到位就好。\n"
                "- 纯文本，不要在开头写自己的名字，也不要 markdown 或舞台说明。"
            )
            parts.append(requirements)
            return "\n\n".join(parts)
        parts = [
            f"Topic:\n{topic}",
            "People here (speaking order): " + ", ".join(participants),
            f"Round {round_number}/{total_rounds}.",
        ]
        if topic_anchor:
            parts.append(f"Stay on {topic_anchor} and the facts given in the topic.")
        if history_text:
            parts.append("So far:\n" + history_text)
            parts.append("Pick up the thread — react to the most interesting earlier point.")
        else:
            parts.append("You speak first.")
        parts.append(context)
        if background:
            parts.append(background)
        requirements = (
            "How to speak:\n"
            "- Numbers, times, places only from the topic or your real memory — don't invent.\n"
            "- If you have no memory of this, just react to the stated facts.\n"
            "- Only reference what's already in the conversation history; don't speak for others or predict their responses.\n"
            "- Push the conversation forward. Say what needs saying — length is up to you.\n"
            "- Plain text. No name prefix, no markdown, no stage directions."
        )
        parts.append(requirements)
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
            if self.language == "zh":
                lines.append(f"第 {round_number} 轮 - {speaker}：{text}")
            else:
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
        if self.language == "zh":
            if memory_parts:
                sections.append("我记得自己说过：\n" + "\n\n".join(memory_parts))
            else:
                sections.append("我记得自己说过：\n我现在想不起足够清楚的公开表达。")
            if document_parts:
                sections.append("补充资料：\n" + "\n\n".join(document_parts))
            return "\n\n".join(sections)
        if memory_parts:
            sections.append(f"{_PUBLIC_REMARKS_TITLE}:\n" + "\n\n".join(memory_parts))
        else:
            sections.append(f"{_PUBLIC_REMARKS_TITLE}:\nI can't clearly remember enough public remarks right now.")
        if document_parts:
            sections.append(f"{_CUSTOM_DOCUMENTS_TITLE}:\n" + "\n\n".join(document_parts))
        return "\n\n".join(sections)

    def _format_memory_excerpt(self, chunk: TranscriptChunk) -> str:
        if self.language == "zh":
            return "我记得：\n" + chunk.text
        return "I remember:\n" + chunk.text

    def _format_document_excerpt(self, chunk: TranscriptChunk) -> str:
        if self.language == "zh":
            return "补充笔记：\n" + chunk.text
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
        if self.language == "zh":
            lines = ["背景信息（外部参考，不是我亲口说的）：", ""]
            for index, item in enumerate(results, start=1):
                lines.append(f"[{index}] {item.title}")
                if item.snippet:
                    lines.append(item.snippet)
                lines.append("")
            return "\n".join(lines).strip()
        lines = [f"{_BACKGROUND_INFO_TITLE} (external, not my own words):", ""]
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
        char_limit: int = 360,
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
        spoken_answer = self.tts.prepare_spoken_text(answer, char_limit=char_limit)
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

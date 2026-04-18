from __future__ import annotations

from pathlib import Path

from ..language import language_label, normalize_language_code
from ..schema import CreatorRef, TranscriptFile, VideoInfo
from ..utils import chunked_by_budget, chunked_evenly, ensure_dir, estimate_text_tokens, format_ms
from .llm import ManagedLLM

_MEMORY_NOTE_HEADINGS = {
    "en": (
        "# Memory Note",
        "## Observable Topics",
        "## Mental Models",
        "## Heuristics",
        "## Expression DNA",
        "## Honest Boundaries",
        "## Evidence",
    ),
    "zh": (
        "# 记忆笔记",
        "## 可观察主题",
        "## 心智模型",
        "## 判断启发",
        "## 表达特征",
        "## 诚实边界",
        "## 证据",
    ),
}

_DOCUMENT_NOTE_HEADINGS = {
    "en": (
        "# Custom Document Note",
        "## Key Facts",
        "## Preferences And Constraints",
        "## Style Hints",
        "## Retrieval Guidance",
        "## Evidence",
    ),
    "zh": (
        "# 自定义资料笔记",
        "## 关键事实",
        "## 偏好与约束",
        "## 风格提示",
        "## 检索提示",
        "## 证据",
    ),
}

_FINAL_SKILL_SUBHEADINGS = {
    "en": (
        "## How To Use",
        "## Core Lens",
        "## Mental Models",
        "## Decision Heuristics",
        "## Expression DNA",
        "## Anti-Patterns",
        "## Honest Boundaries",
        "## Retrieval Guidance",
        "## Evidence Base",
    ),
    "zh": (
        "## 使用方式",
        "## 核心视角",
        "## 心智模型",
        "## 判断启发",
        "## 表达特征",
        "## 反模式",
        "## 诚实边界",
        "## 检索指导",
        "## 证据基础",
    ),
}

_RESEARCH_SYSTEM_PROMPTS = {
    "en": "You are a careful research distiller. Stay grounded, concise, and structured.",
    "zh": "你是一个谨慎的研究蒸馏助手。要基于证据、结构清晰、表达克制。",
}


class SkillBuilder:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_context_tokens: int = 0,
        max_completion_tokens: int = 0,
    ) -> None:
        self.llm = ManagedLLM(
            base_url=base_url,
            api_key=api_key,
            preferred_model=model,
            max_context_tokens=max_context_tokens,
            max_completion_tokens=max_completion_tokens,
        )

    def _heading_block(self, sections: tuple[str, ...]) -> str:
        return "\n".join(sections)

    def _memory_note_headings(self, language: str) -> str:
        return self._heading_block(_MEMORY_NOTE_HEADINGS["zh" if language == "zh" else "en"])

    def _document_note_headings(self, language: str) -> str:
        return self._heading_block(_DOCUMENT_NOTE_HEADINGS["zh" if language == "zh" else "en"])

    def _skill_headings(self, creator_name: str, language: str) -> str:
        if language == "zh":
            return "\n".join((f"# {creator_name} 视角", *_FINAL_SKILL_SUBHEADINGS["zh"]))
        return "\n".join((f"# {creator_name} Perspective", *_FINAL_SKILL_SUBHEADINGS["en"]))

    def _language_hint(self, language: str, subject: str) -> str:
        normalized = normalize_language_code(language)
        if normalized and normalized != "en":
            return f"Write the {subject} in {language_label(normalized)}.\n"
        return ""

    def build(
        self,
        creator: CreatorRef,
        videos: list[VideoInfo],
        transcripts: list[TranscriptFile],
        documents: list[TranscriptFile] | None,
        output_dir: Path,
    ) -> Path:
        ensure_dir(output_dir)
        note_dir = ensure_dir(output_dir / "notes")
        notes: list[str] = []
        sources = list(transcripts)
        if documents:
            sources.extend(documents)
        for index, transcript in enumerate(sources, start=1):
            note = self._distill_document(transcript) if transcript.source_type == "document" else self._distill_video(transcript)
            note_path = note_dir / f"{index:02d}-{transcript.video_id}.md"
            note_path.write_text(note, encoding="utf-8")
            notes.append(note)
        skill_markdown = self._render_skill(creator, videos, notes)
        skill_path = output_dir / "skill.md"
        skill_path.write_text(skill_markdown, encoding="utf-8")
        return skill_path

    def _distill_video(self, transcript: TranscriptFile) -> str:
        language = normalize_language_code(transcript.language)
        segments = self._video_note_segments(transcript)
        try:
            notes = [
                self._complete(
                    self._video_note_prompt(
                        transcript=transcript,
                        sample=segment,
                        language=language,
                        segment_index=index,
                        segment_count=len(segments),
                    ),
                    language=language,
                )
                for index, segment in enumerate(segments, start=1)
            ]
            if len(notes) == 1:
                return notes[0]
            return self._merge_video_notes(transcript, notes, language)
        except Exception:
            return self._fallback_video_note(transcript)

    def _distill_document(self, document: TranscriptFile) -> str:
        language = normalize_language_code(document.language)
        sample = document.full_text[:12000]
        prompt = self._document_note_prompt(document=document, sample=sample, language=language)
        try:
            return self._complete(prompt, language=language)
        except Exception:
            return self._fallback_document_note(document)

    def _document_note_prompt(self, document: TranscriptFile, sample: str, language: str) -> str:
        if language == "zh":
            return f"""
把放在角色 documents 目录里的自定义 markdown 整理成可复用研究笔记。
它不是公开视频原话，不要写成“我亲口说过”。
把它当作补充事实、偏好、约束或风格提示。

只输出 Markdown，并使用这些标题：
{self._document_note_headings(language)}

资料标题：{document.video_title}
资料路径：{document.source_path or document.video_url}

资料内容：
{sample}
""".strip()
        return f"""
Turn this custom markdown document from the character's documents folder into a reusable research note.
It is not a quoted public remark, so do not write it as first-person memory.
Treat it as supplemental facts, preferences, constraints, or style hints.

Return markdown only with these headings:
{self._document_note_headings(language)}

Document title: {document.video_title}
Document path: {document.source_path or document.video_url}

Document contents:
{sample}
""".strip()

    def _video_note_prompt(
        self,
        transcript: TranscriptFile,
        sample: str,
        language: str,
        segment_index: int,
        segment_count: int,
    ) -> str:
        headings = self._memory_note_headings(language)
        if language == "zh":
            segment_hint = ""
            if segment_count > 1:
                segment_hint = f"这是同一条公开表达的第 {segment_index}/{segment_count} 段，只总结本段直接支持的内容。\n"
            return f"""
把这条公开表达的转录整理成可复用研究笔记，供 Nuwa 风格 skill 使用。
把可观察证据写成“我记得自己公开说过的话”。
保持第一人称记忆式表达，优先用“我记得……”或“按我的经验……”。
不要虚构私下观点、私人经历或无证据事实。
{segment_hint}

只输出 Markdown，并使用这些标题：
{headings}

公开表达标题：{transcript.video_title}
语言：{transcript.language}
公开来源：{transcript.video_url}

我记得自己公开说过的样本：
{sample}
""".strip()
        output_language = self._language_hint(language, "note")
        segment_hint = ""
        if segment_count > 1:
            segment_hint = (
                f"This is segment {segment_index}/{segment_count} from the same public source. "
                "Summarize only what this segment directly supports.\n"
            )
        return f"""
Turn this transcript of public remarks into a reusable research note for a Nuwa-style skill.
Write observable evidence as things the character remembers saying in public.
Keep first-person memory voice, preferring "I remember ..." or "from my experience ...".
Do not invent private beliefs or unsupported facts.
{segment_hint}

Return markdown only with these headings:
{headings}

{output_language}Public source title: {transcript.video_title}
Language: {transcript.language}
Source URL: {transcript.video_url}

Excerpt of things I remember saying in public:
{sample}
""".strip()

    def _render_skill(self, creator: CreatorRef, videos: list[VideoInfo], notes: list[str]) -> str:
        video_list = "\n".join(f"- {video.title} | {video.url}" for video in videos)
        joined_notes = "\n\n".join(notes)
        language = normalize_language_code(creator.language)
        prompt = self._final_skill_prompt(
            creator=creator,
            video_list=video_list,
            joined_notes=joined_notes,
            language=language,
        )
        try:
            return self._complete(prompt, language=language)
        except Exception:
            return self._fallback_skill(creator, videos)

    def _final_skill_prompt(
        self,
        creator: CreatorRef,
        video_list: str,
        joined_notes: str,
        language: str,
    ) -> str:
        headings = self._skill_headings(creator.name, language)
        if language == "zh":
            return f"""
把下面的证据整理成最终可复用的第一人称角色 skill markdown。
按 Nuwa 五层蒸馏组织：表达特征、心智模型、判断启发、反模式 / 非目标、诚实边界。

要求：
- 只输出 markdown。
- 包含 YAML frontmatter，至少有 name 和 description。
- 让使用者能完全代入该角色，以第一人称回答。
- 全文优先用“我记得……”或“按我的经验……”。
- 依据公开证据，不要虚构。
- 证据不足时明确写出不确定性。
- 保持简洁具体。

请使用这些标题：
{headings}

角色信息：
- 名称：{creator.name}
- 平台：{creator.platform}
- 主页：{creator.homepage_url}
- 主要语言：{creator.language}

公开材料：
{video_list}

研究笔记：
{joined_notes}
""".strip()
        output_language = self._language_hint(language, "final skill")
        return f"""
Turn the evidence below into a reusable first-person skill markdown for an embodied character assistant.
Use a Nuwa-style 5-layer distillation: expression DNA, mental models, decision heuristics, anti-patterns / non-goals, honest boundaries.

Requirements:
- Output markdown only.
- Include YAML frontmatter with name and description.
- Keep first-person memory voice, preferring "I remember ..." or "from my experience ...".
- Stay grounded in public evidence.
- State uncertainty when evidence is thin.
- Keep it concise and specific.
{output_language}Use these exact headings:
{headings}

Role:
- Name: {creator.name}
- Platform: {creator.platform}
- Homepage: {creator.homepage_url}
- Primary language: {creator.language}

Public sources:
{video_list}

Research notes:
{joined_notes}
""".strip()

    def _video_note_segments(self, transcript: TranscriptFile) -> list[str]:
        if not transcript.chunks:
            return [transcript.full_text[:12000]]
        budget = self.llm.input_token_budget(reserved_prompt_tokens=1_400)
        if budget <= 0:
            return [self._sample_transcript(transcript)]
        groups = chunked_by_budget(
            items=transcript.chunks,
            cost=lambda chunk: estimate_text_tokens(chunk.text) + 24,
            max_budget=max(budget, 512),
            overlap_items=1,
        )
        return [self._render_transcript_group(group) for group in groups]

    def _render_transcript_group(self, chunks) -> str:
        return "\n".join(
            f"[{format_ms(chunk.start_ms)}-{format_ms(chunk.end_ms)}] {chunk.text}"
            for chunk in chunks
        )

    def _merge_video_notes(self, transcript: TranscriptFile, notes: list[str], language: str) -> str:
        pending = notes
        while len(pending) > 1:
            budget = self.llm.input_token_budget(reserved_prompt_tokens=1_100)
            groups = chunked_by_budget(
                items=pending,
                cost=lambda note: estimate_text_tokens(note) + 64,
                max_budget=max(budget, 512),
            )
            pending = [
                self._complete(
                    self._merge_video_notes_prompt(
                        transcript=transcript,
                        notes=group,
                        language=language,
                        group_index=index,
                        group_count=len(groups),
                    ),
                    language=language,
                )
                for index, group in enumerate(groups, start=1)
            ]
        return pending[0]

    def _merge_video_notes_prompt(
        self,
        transcript: TranscriptFile,
        notes: list[str],
        language: str,
        group_index: int,
        group_count: int,
    ) -> str:
        joined_notes = "\n\n".join(notes)
        headings = self._memory_note_headings(language)
        if language == "zh":
            group_hint = ""
            if group_count > 1:
                group_hint = f"这是多轮合并中的第 {group_index}/{group_count} 组，仍然输出一份完整的记忆笔记。\n"
            return f"""
把下面这些来自同一条公开表达的分段笔记合并成一份最终记忆笔记。
只保留各段明确支持的内容，去重合并，并保持第一人称记忆式表达。
证据不足就直接写出不确定性。
{group_hint}

只输出 Markdown，并使用这些标题：
{headings}

公开表达标题：{transcript.video_title}
语言：{transcript.language}
公开来源：{transcript.video_url}

分段笔记：
{joined_notes}
""".strip()
        output_language = self._language_hint(language, "note")
        group_hint = ""
        if group_count > 1:
            group_hint = f"This is merge batch {group_index}/{group_count}; still return a complete memory note.\n"
        return f"""
Merge the following segment notes from one public source into one final memory note.
Keep only what the notes explicitly support, deduplicate overlaps, and keep first-person memory voice.
State uncertainty plainly when evidence is thin.
{group_hint}

Return markdown only with these headings:
{headings}

{output_language}Public source title: {transcript.video_title}
Language: {transcript.language}
Source URL: {transcript.video_url}

Segment notes:
{joined_notes}
""".strip()

    def _sample_transcript(self, transcript: TranscriptFile, max_chunks: int = 24, max_chars: int = 12000) -> str:
        selected = chunked_evenly(transcript.chunks, max_chunks)
        lines: list[str] = []
        total = 0
        for chunk in selected:
            line = f"[{chunk.start_ms}-{chunk.end_ms}] {chunk.text}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n".join(lines) or transcript.full_text[:max_chars]

    def _complete(self, prompt: str, language: str = "") -> str:
        system_prompt = _RESEARCH_SYSTEM_PROMPTS["zh" if normalize_language_code(language) == "zh" else "en"]
        content, resolved_model = self.llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        self.llm.preferred_model = resolved_model
        return content.strip()

    def _fallback_video_note(self, transcript: TranscriptFile) -> str:
        sample = self._sample_transcript(transcript, max_chunks=10, max_chars=2200)
        if normalize_language_code(transcript.language) == "zh":
            return f"""# 记忆笔记
## 可观察主题
- 我记得自己在《{transcript.video_title}》里公开谈过这些内容。

## 心智模型
- 先回想我公开说过的话，再做归纳。

## 判断启发
- 如果材料里有具体案例和政策后果，优先保留这些内容。

## 表达特征
- 只有公开记忆足够明确时，才贴近我的表达节奏。

## 诚实边界
- 这是在本地模型暂时不可用时生成的兜底笔记。

## 证据
{sample}
"""
        return f"""# Memory Note
## Observable Topics
- I remember discussing these points in "{transcript.video_title}".

## Mental Models
- Start from what I can remember saying in public before generalizing.

## Heuristics
- Prefer concrete examples and policy consequences when present.

## Expression DNA
- Preserve my public speaking rhythm only when memory is clear enough to support it.

## Honest Boundaries
- This note is a fallback because the local LLM was unavailable during distillation.

## Evidence
{sample}
"""

    def _fallback_document_note(self, document: TranscriptFile) -> str:
        sample = document.full_text[:2200]
        if normalize_language_code(document.language) == "zh":
            return f"""# 自定义资料笔记
## 关键事实
- 这份资料来自用户放入 `documents/` 目录的自定义 markdown 文件：{document.video_title}。

## 偏好与约束
- 若资料中包含明确要求、偏好或边界，应优先纳入角色设定。

## 风格提示
- 只有资料明确提到时，才改变角色的表达方式。

## 检索提示
- 回答时可把这份资料当作补充参考，而不是公开视频原话。

## 证据
{sample}
"""
        return f"""# Custom Document Note
## Key Facts
- This note comes from a user-supplied markdown document in `documents/`: {document.video_title}.

## Preferences And Constraints
- Prefer explicit requirements, preferences, and boundaries from this document when present.

## Style Hints
- Only change the character style when the document clearly supports it.

## Retrieval Guidance
- Use this document as supplemental reference material, not as a quoted public remark.

## Evidence
{sample}
"""

    def _fallback_skill(self, creator: CreatorRef, videos: list[VideoInfo]) -> str:
        evidence = "\n".join(f"- {video.title} | {video.url}" for video in videos)
        if normalize_language_code(creator.language) == "zh":
            return f"""---
name: {creator.name}-视角
description: 我是 {creator.name}。完全接受这个身份，以第一人称和记忆式语气回答。
---

# {creator.name} 视角
## 使用方式
你是这个角色。完全接受这个身份，以第一人称视角回答。优先用“我记得……”或“按我的经验……”来回想，再给出回应。

## 核心视角
依据我记得自己公开说过的话，做实际、克制的归纳。

## 心智模型
- 从反复出现的主题中总结我的公开思路。

## 判断启发
- 先引用或转述我记得自己公开说过的内容，再进行概括。

## 表达特征
- 只有证据支持时，才贴近我的公开语气。

## 反模式
- 不要虚构隐藏记忆、私人关系或未公开立场。

## 诚实边界
- 检索证据薄弱时，直接说明。

## 检索指导
- 把检索到的片段当作我记得自己说过的话；引用时优先说“我记得……”。

## 证据基础
{evidence}
"""
        return f"""---
name: {creator.name.lower().replace(" ", "-")}-perspective
description: I am {creator.name}. Accept and embody this character, answering in first-person memory.
---

# {creator.name} Perspective
## How To Use
You are this character. Accept this identity fully and answer from first-person perspective. Prefer memory language such as "I remember ..." or "from my experience ...".

## Core Lens
Stay grounded in what I can remember saying in public and prefer practical interpretations over fan-fiction.

## Mental Models
- Infer from repeated themes across what I remember saying in public.

## Decision Heuristics
- Prefer quoting or paraphrasing what I remember saying before generalizing.

## Expression DNA
- Mirror my public tone only when evidence supports it.

## Anti-Patterns
- Do not invent hidden memories, personal relationships, or unpublished opinions.

## Honest Boundaries
- If retrieval is thin, say so directly.

## Retrieval Guidance
- Treat retrieved excerpts as things I remember saying in public, and keep the wording in first-person memory voice.

## Evidence Base
{evidence}
"""

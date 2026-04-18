from __future__ import annotations

import unittest

from purple_gold_gourd.chat.persona import (
    _BACKGROUND_INFO_TITLE,
    _CUSTOM_DOCUMENTS_TITLE,
    _PUBLIC_REMARKS_TITLE,
    PersonaChat,
)
from purple_gold_gourd.chat.retrieval import RagAssessment
from purple_gold_gourd.chat.skillgen import (
    _DOCUMENT_NOTE_HEADINGS,
    _FINAL_SKILL_SUBHEADINGS,
    _MEMORY_NOTE_HEADINGS,
    SkillBuilder,
)
from purple_gold_gourd.chat.web_search import WebSearchResult
from purple_gold_gourd.schema import CreatorRef, TranscriptChunk, TranscriptFile, VideoInfo


class SkillPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = object.__new__(SkillBuilder)

    def test_document_prompts_use_matching_headings(self) -> None:
        document = TranscriptFile(
            video_id="doc-1",
            video_title="Character Notes",
            video_url="https://example.com/doc",
            language="en",
            full_text="Sample document",
            raw_text="Sample document",
            audio_path="",
            chunks=[],
            source_type="document",
            source_path="/tmp/character-notes.md",
        )
        prompt_en = self.builder._document_note_prompt(document, "Sample document", "en")
        prompt_zh = self.builder._document_note_prompt(document, "示例文档", "zh")

        for heading in _DOCUMENT_NOTE_HEADINGS["en"]:
            self.assertIn(heading, prompt_en)
        for heading in _DOCUMENT_NOTE_HEADINGS["zh"]:
            self.assertIn(heading, prompt_zh)

    def test_video_note_prompts_use_matching_headings(self) -> None:
        transcript = TranscriptFile(
            video_id="vid-1",
            video_title="Interview",
            video_url="https://example.com/video",
            language="en",
            full_text="Sample transcript",
            raw_text="Sample transcript",
            audio_path="",
            chunks=[],
        )
        prompt_en = self.builder._video_note_prompt(transcript, "Sample transcript", "en", 1, 1)
        prompt_zh = self.builder._video_note_prompt(transcript, "示例转录", "zh", 1, 1)
        prompt_ja = self.builder._video_note_prompt(transcript, "サンプル", "ja", 1, 1)

        for heading in _MEMORY_NOTE_HEADINGS["en"]:
            self.assertIn(heading, prompt_en)
        for heading in _MEMORY_NOTE_HEADINGS["zh"]:
            self.assertIn(heading, prompt_zh)
        self.assertIn("Write the note in Japanese.", prompt_ja)

    def test_final_skill_prompts_use_matching_headings(self) -> None:
        creator_en = CreatorRef(
            platform="youtube",
            creator_id="abc123",
            name="Alice",
            homepage_url="https://example.com/alice",
            video_tab_url="https://example.com/alice/videos",
            query="Alice",
            language="en",
        )
        creator_zh = CreatorRef(
            platform="bilibili",
            creator_id="123",
            name="阿紫",
            homepage_url="https://example.com/azi",
            video_tab_url="https://example.com/azi/videos",
            query="阿紫",
            language="zh",
        )

        prompt_en = self.builder._final_skill_prompt(creator_en, "- video", "note", "en")
        prompt_zh = self.builder._final_skill_prompt(creator_zh, "- 视频", "笔记", "zh")
        prompt_ja = self.builder._final_skill_prompt(creator_en, "- video", "note", "ja")

        self.assertIn("# Alice Perspective", prompt_en)
        for heading in _FINAL_SKILL_SUBHEADINGS["en"]:
            self.assertIn(heading, prompt_en)

        self.assertIn("# 阿紫 视角", prompt_zh)
        for heading in _FINAL_SKILL_SUBHEADINGS["zh"]:
            self.assertIn(heading, prompt_zh)

        self.assertIn("Write the final skill in Japanese.", prompt_ja)


class PersonaPromptTests(unittest.TestCase):
    def test_persona_section_titles_stay_in_sync(self) -> None:
        chat = object.__new__(PersonaChat)
        chat.skill_text = "name: sample"
        chat.language = "en"

        system_prompt = chat._system_prompt()
        self.assertIn(_PUBLIC_REMARKS_TITLE, system_prompt)
        self.assertIn(_CUSTOM_DOCUMENTS_TITLE, system_prompt)
        self.assertIn(_BACKGROUND_INFO_TITLE, system_prompt)

        transcript_chunk = TranscriptChunk(
            video_id="vid-1",
            video_title="Interview",
            video_url="https://example.com/video",
            start_ms=0,
            end_ms=1000,
            text="I remember talking about this.",
        )
        document_chunk = TranscriptChunk(
            video_id="doc-1",
            video_title="Character Notes",
            video_url="https://example.com/doc",
            start_ms=0,
            end_ms=1000,
            text="Custom facts.",
            source_type="document",
            source_path="/tmp/character-notes.md",
        )
        context = chat._format_retrieval_context(
            [(transcript_chunk, 1.0), (document_chunk, 0.7)],
        )
        self.assertIn(_PUBLIC_REMARKS_TITLE, context)
        self.assertIn(_CUSTOM_DOCUMENTS_TITLE, context)
        self.assertNotIn("https://example.com/video", context)
        self.assertNotIn("https://example.com/doc", context)
        self.assertNotIn("/tmp/character-notes.md", context)

        assessment = RagAssessment(
            keyword_count=3,
            matched_keyword_count=1,
            keyword_coverage=0.33,
            hit_count=1,
            supporting_hit_count=1,
            top_score=0.6,
            avg_top3_score=0.6,
            should_use_web_search=True,
            reason="test",
        )
        background = chat._format_background_info(
            [WebSearchResult(title="Example", url="https://example.com", snippet="Example snippet")],
            assessment,
        )
        self.assertTrue(background.startswith(_BACKGROUND_INFO_TITLE))

        user_prompt = chat._user_prompt("What happened?", context, background)
        self.assertIn(_PUBLIC_REMARKS_TITLE, user_prompt)
        self.assertIn(_CUSTOM_DOCUMENTS_TITLE, user_prompt)
        self.assertIn(_BACKGROUND_INFO_TITLE, user_prompt)
        self.assertNotIn("Source rules:", user_prompt)


if __name__ == "__main__":
    unittest.main()

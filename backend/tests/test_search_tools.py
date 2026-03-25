"""Tests for CourseSearchTool in search_tools.py"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_store(documents=None, metadata=None, error=None, course_link=None, lesson_link=None):
    """Helper: build a mock VectorStore"""
    store = MagicMock()
    if error:
        results = SearchResults(documents=[], metadata=[], distances=[], error=error)
    else:
        docs = documents or []
        meta = metadata or []
        results = SearchResults(documents=docs, metadata=meta, distances=[0.1] * len(docs))
    store.search.return_value = results
    store.get_course_link.return_value = course_link
    store.get_lesson_link.return_value = lesson_link
    return store


class TestCourseSearchToolDefinition:
    def test_tool_name(self):
        tool = CourseSearchTool(MagicMock())
        assert tool.get_tool_definition()["name"] == "search_course_content"

    def test_required_fields(self):
        tool = CourseSearchTool(MagicMock())
        schema = tool.get_tool_definition()["input_schema"]
        assert schema["required"] == ["query"]
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]


class TestCourseSearchToolExecute:
    def test_successful_search_returns_formatted_text(self):
        store = make_store(
            documents=["Lesson content here"],
            metadata=[{"course_title": "Intro to AI", "lesson_number": 1}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="neural networks")
        assert "[Intro to AI - Lesson 1]" in result
        assert "Lesson content here" in result

    def test_passes_filters_to_store(self):
        store = make_store(documents=["text"], metadata=[{"course_title": "ML", "lesson_number": 2}])
        tool = CourseSearchTool(store)
        tool.execute(query="transformers", course_name="ML", lesson_number=2)
        store.search.assert_called_once_with(query="transformers", course_name="ML", lesson_number=2)

    def test_empty_results_returns_no_content_message(self):
        store = make_store(documents=[], metadata=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="unknown topic")
        assert "No relevant content found" in result

    def test_empty_results_with_filters_mentions_filters(self):
        store = make_store(documents=[], metadata=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="topic", course_name="Python", lesson_number=3)
        assert "Python" in result
        assert "3" in result

    def test_search_error_returns_error_message(self):
        store = make_store(error="ChromaDB connection failed")
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")
        assert "ChromaDB connection failed" in result

    def test_header_without_lesson_number(self):
        store = make_store(
            documents=["Course overview text"],
            metadata=[{"course_title": "Deep Learning", "lesson_number": None}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="overview")
        assert "[Deep Learning]" in result
        assert "Lesson" not in result

    def test_multiple_results_joined(self):
        store = make_store(
            documents=["text A", "text B"],
            metadata=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="topic")
        assert "text A" in result
        assert "text B" in result


class TestCourseSearchToolSources:
    def test_sources_stored_after_search(self):
        store = make_store(
            documents=["content"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            lesson_link="https://example.com/lesson1",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="AI")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "AI Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_sources_deduplicated(self):
        """Same course+lesson from multiple chunks should only appear once in sources"""
        store = make_store(
            documents=["chunk 1", "chunk 2"],
            metadata=[
                {"course_title": "AI Course", "lesson_number": 1},
                {"course_title": "AI Course", "lesson_number": 1},
            ],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="AI")
        assert len(tool.last_sources) == 1

    def test_falls_back_to_course_link_when_no_lesson_link(self):
        store = make_store(
            documents=["content"],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            lesson_link=None,
            course_link="https://example.com/course",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="AI")
        assert tool.last_sources[0]["url"] == "https://example.com/course"

    def test_url_is_none_when_no_links_available(self):
        store = make_store(
            documents=["content"],
            metadata=[{"course_title": "AI Course", "lesson_number": None}],
            lesson_link=None,
            course_link=None,
        )
        tool = CourseSearchTool(store)
        tool.execute(query="AI")
        assert tool.last_sources[0]["url"] is None

    def test_sources_empty_on_no_results(self):
        store = make_store(documents=[], metadata=[])
        tool = CourseSearchTool(store)
        tool.execute(query="anything")
        assert tool.last_sources == []


class TestToolManager:
    def test_register_and_retrieve_definitions(self):
        manager = ToolManager()
        store = make_store()
        tool = CourseSearchTool(store)
        manager.register_tool(tool)
        defs = manager.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_execute_tool_by_name(self):
        manager = ToolManager()
        store = make_store(documents=["result"], metadata=[{"course_title": "X", "lesson_number": 1}])
        manager.register_tool(CourseSearchTool(store))
        result = manager.execute_tool("search_course_content", query="test")
        assert "result" in result

    def test_execute_unknown_tool(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_last_sources_after_search(self):
        manager = ToolManager()
        store = make_store(
            documents=["content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
        )
        manager.register_tool(CourseSearchTool(store))
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources(self):
        manager = ToolManager()
        store = make_store(
            documents=["content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
        )
        manager.register_tool(CourseSearchTool(store))
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        assert manager.get_last_sources() == []

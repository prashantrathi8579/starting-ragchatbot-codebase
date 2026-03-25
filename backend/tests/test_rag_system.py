"""Tests for RAGSystem query flow and component integration"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_system import RAGSystem
from vector_store import SearchResults


def make_config(api_key="test-key"):
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = api_key
    cfg.ANTHROPIC_MODEL = "claude-test"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "/tmp/test_chroma"
    return cfg


@pytest.fixture
def rag(tmp_path):
    """RAGSystem with all heavy components mocked out"""
    with (
        patch("rag_system.DocumentProcessor") as MockDP,
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.SessionManager") as MockSM,
    ):
        mock_vs = MagicMock()
        mock_ai = MagicMock()
        mock_sm = MagicMock()
        mock_dp = MagicMock()
        MockVS.return_value = mock_vs
        MockAI.return_value = mock_ai
        MockSM.return_value = mock_sm
        MockDP.return_value = mock_dp

        system = RAGSystem(make_config())
        yield system, mock_vs, mock_ai, mock_sm


class TestRAGQuery:
    def test_query_returns_response_and_sources(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "AI answer"
        # Simulate tool having run and stored a source
        system.search_tool.last_sources = [{"label": "Course A - Lesson 1", "url": "https://x.com"}]

        response, sources = system.query("What is ML?")
        assert response == "AI answer"
        assert sources == [{"label": "Course A - Lesson 1", "url": "https://x.com"}]

    def test_sources_reset_after_query(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "answer"
        system.search_tool.last_sources = [{"label": "X", "url": None}]

        system.query("test")
        assert system.search_tool.last_sources == []

    def test_query_with_session_fetches_history(self, rag):
        system, vs, ai, sm = rag
        sm.get_conversation_history.return_value = "User: hi\nAssistant: hello"
        ai.generate_response.return_value = "answer"

        system.query("follow up question", session_id="session_1")
        sm.get_conversation_history.assert_called_once_with("session_1")
        call_kwargs = ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_query_without_session_no_history(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "answer"

        system.query("standalone question", session_id=None)
        sm.get_conversation_history.assert_not_called()
        call_kwargs = ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None

    def test_session_updated_after_query(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "AI says this"

        system.query("user question", session_id="session_1")
        sm.add_exchange.assert_called_once_with("session_1", "user question", "AI says this")

    def test_no_session_update_without_session_id(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "answer"

        system.query("test", session_id=None)
        sm.add_exchange.assert_not_called()

    def test_tools_passed_to_ai_generator(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "answer"

        system.query("test")
        call_kwargs = ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert isinstance(call_kwargs["tools"], list)
        assert len(call_kwargs["tools"]) > 0

    def test_empty_sources_when_no_search_triggered(self, rag):
        system, vs, ai, sm = rag
        ai.generate_response.return_value = "General knowledge answer"
        system.search_tool.last_sources = []

        _, sources = system.query("What is the capital of France?")
        assert sources == []


class TestRAGCourseAnalytics:
    def test_get_course_analytics(self, rag):
        system, vs, ai, sm = rag
        vs.get_course_count.return_value = 3
        vs.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]

        analytics = system.get_course_analytics()
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3


class TestRAGAddDocument:
    def test_add_course_document_calls_processor_and_store(self, rag):
        system, vs, ai, sm = rag
        from models import Course, Lesson, CourseChunk
        mock_course = Course(title="Test Course", course_link="https://x.com", instructor="Prof X")
        mock_chunks = [
            CourseChunk(content="chunk text", course_title="Test Course", lesson_number=1, chunk_index=0)
        ]
        system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)

        course, n_chunks = system.add_course_document("/path/to/doc.txt")
        assert course.title == "Test Course"
        assert n_chunks == 1
        vs.add_course_metadata.assert_called_once_with(mock_course)
        vs.add_course_content.assert_called_once_with(mock_chunks)

    def test_add_course_document_handles_error(self, rag):
        system, vs, ai, sm = rag
        system.document_processor.process_course_document.side_effect = Exception("parse error")

        course, n_chunks = system.add_course_document("/bad/path.txt")
        assert course is None
        assert n_chunks == 0

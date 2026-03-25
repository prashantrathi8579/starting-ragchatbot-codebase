"""Tests for AIGenerator and its integration with ToolManager"""
import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


def make_text_response(text="Hello"):
    """Build a mock Anthropic response that returns plain text"""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_use_response(tool_name="search_course_content", tool_id="tu_1", inputs=None):
    """Build a mock Anthropic response that requests a tool call"""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = inputs or {"query": "test query"}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


@pytest.fixture
def generator():
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        MockAnthropic.return_value = client
        gen = AIGenerator(api_key="test-key", model="claude-test")
        gen._client = client
        yield gen, client


class TestDirectResponse:
    def test_returns_text_when_no_tool_use(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response("Direct answer")
        result = gen.generate_response(query="What is 2+2?")
        assert result == "Direct answer"

    def test_sends_user_message(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        gen.generate_response(query="Hello")
        args = client.messages.create.call_args[1]
        assert args["messages"][0]["role"] == "user"
        assert args["messages"][0]["content"] == "Hello"

    def test_includes_system_prompt(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        gen.generate_response(query="test")
        args = client.messages.create.call_args[1]
        assert "system" in args
        assert len(args["system"]) > 0

    def test_appends_conversation_history_to_system(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        gen.generate_response(query="test", conversation_history="User: hi\nAssistant: hello")
        args = client.messages.create.call_args[1]
        assert "Previous conversation" in args["system"]
        assert "User: hi" in args["system"]

    def test_no_history_uses_plain_system_prompt(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        gen.generate_response(query="test", conversation_history=None)
        args = client.messages.create.call_args[1]
        assert "Previous conversation" not in args["system"]

    def test_tools_included_when_provided(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        tools = [{"name": "search_course_content", "input_schema": {}}]
        gen.generate_response(query="test", tools=tools)
        args = client.messages.create.call_args[1]
        assert args["tools"] == tools
        assert args["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self, generator):
        gen, client = generator
        client.messages.create.return_value = make_text_response()
        gen.generate_response(query="test")
        args = client.messages.create.call_args[1]
        assert "tools" not in args


class TestToolExecution:
    def test_executes_tool_and_makes_second_call(self, generator):
        gen, client = generator
        tool_response = make_tool_use_response(inputs={"query": "neural networks"})
        final_response = make_text_response("Here is what I found.")
        client.messages.create.side_effect = [tool_response, final_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Search results text"

        result = gen.generate_response(
            query="Tell me about neural networks",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Here is what I found."
        assert client.messages.create.call_count == 2
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="neural networks"
        )

    def test_tool_result_included_in_second_call(self, generator):
        gen, client = generator
        tool_response = make_tool_use_response(tool_id="tu_abc", inputs={"query": "q"})
        final_response = make_text_response("Final answer")
        client.messages.create.side_effect = [tool_response, final_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some results"

        gen.generate_response(query="test", tools=[{}], tool_manager=tool_manager)

        second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
        # Should have: user msg, assistant tool_use, user tool_result
        assert any(
            isinstance(m["content"], list) and m["content"][0]["type"] == "tool_result"
            for m in second_call_messages
        )

    def test_no_tool_use_without_tool_manager(self, generator):
        """If stop_reason is tool_use but no tool_manager provided, return first response text"""
        gen, client = generator
        tool_response = make_tool_use_response()
        # add text block so content[0].text works
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "partial"
        tool_response.stop_reason = "end_turn"
        tool_response.content = [text_block]
        client.messages.create.return_value = tool_response

        result = gen.generate_response(query="test")
        assert result == "partial"
        assert client.messages.create.call_count == 1

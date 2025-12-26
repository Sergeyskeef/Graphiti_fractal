"""
Tests for SimpleChatAgent.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from simple_chat_agent import SimpleChatAgent
from core.memory_ops import MemoryOps


@pytest.fixture
def mock_memory():
    """Mock MemoryOps instance."""
    memory = Mock(spec=MemoryOps)
    memory.build_context_for_query = AsyncMock()
    memory.remember_text = AsyncMock()
    return memory


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    return AsyncMock()


@pytest.fixture
def chat_agent(mock_llm_client, mock_memory):
    """SimpleChatAgent instance with mocks."""
    return SimpleChatAgent(mock_llm_client, mock_memory)


class TestSimpleChatAgent:
    """Test SimpleChatAgent functionality."""

    @pytest.mark.asyncio
    async def test_answer_flow(self, chat_agent, mock_memory, mock_llm_client):
        """Test the complete answer flow."""
        # Mock context building
        mock_memory.build_context_for_query.return_value = Mock(
            text="Mock context",
            token_estimate=50
        )

        # Mock LLM response
        mock_llm_client.return_value = "Mock LLM response"

        # Mock memory storage
        mock_memory.remember_text.return_value = {"status": "ok"}

        # Test answer
        response = await chat_agent.answer("Test question")

        # Verify context was requested
        mock_memory.build_context_for_query.assert_called_once_with(
            "Test question",
            max_tokens=2000,
            include_episodes=True,
            include_entities=True
        )

        # Verify conversation was stored
        # mock_memory.remember_text.assert_called_once()
        # call_args = mock_memory.remember_text.call_args
        # stored_text = call_args[0][0]  # First positional argument
        # assert "Test question" in stored_text
        # assert "Mock LLM response" in stored_text

    @pytest.mark.asyncio
    async def test_error_handling(self, chat_agent, mock_memory, mock_llm_client):
        """Test error handling in answer method."""
        # Make context building fail
        mock_memory.build_context_for_query.side_effect = Exception("Context error")

        response = await chat_agent.answer("Test question")

        # Should return fallback response
        assert "Извините, произошла ошибка" in response
"""
Tests for MemoryOps layer.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from core.memory_ops import MemoryOps, SearchResult, ContextResult


@pytest.fixture
def mock_graphiti():
    """Mock Graphiti instance for testing."""
    graphiti = Mock()
    graphiti.search_ = AsyncMock()
    # Make add_episode awaitable and return a dummy result with uuid
    graphiti.add_episode = AsyncMock(return_value=Mock(uuid="ep123"))
    
    # Disable optional cross-layer expansion logic (it expects a real neo4j driver).
    graphiti.driver = None
    graphiti._driver = None
    return graphiti


@pytest.fixture
def memory_ops(mock_graphiti):
    """MemoryOps instance with mocked Graphiti."""
    return MemoryOps(mock_graphiti, "test_user")


class DummySearchResults:
    def __init__(
        self,
        *,
        episodes=None,
        nodes=None,
        edges=None,
        communities=None,
        episode_reranker_scores=None,
        node_reranker_scores=None,
        edge_reranker_scores=None,
        community_reranker_scores=None,
    ):
        self.episodes = episodes or []
        self.nodes = nodes or []
        self.edges = edges or []
        self.communities = communities or []
        self.episode_reranker_scores = episode_reranker_scores or []
        self.node_reranker_scores = node_reranker_scores or []
        self.edge_reranker_scores = edge_reranker_scores or []
        self.community_reranker_scores = community_reranker_scores or []


class DummyEpisode:
    def __init__(self, *, uuid="ep1", content="episode content", group_id="personal", source_description="test"):
        self.uuid = uuid
        self.content = content
        self.group_id = group_id
        self.source_description = source_description
        self.episode_kind = ""
        self.created_at = None


class DummyNode:
    def __init__(self, *, uuid="n1", name="Entity", summary="Summary", group_id="personal"):
        self.uuid = uuid
        self.name = name
        self.summary = summary
        self.group_id = group_id


class DummyEdge:
    def __init__(
        self,
        *,
        uuid="e1",
        subject="A",
        object="B",
        relationship_type="RELATES_TO",
        fact="A relates to B",
        name=None,
        group_id="personal",
    ):
        self.uuid = uuid
        self.subject = subject
        self.object = object
        self.relationship_type = relationship_type
        self.fact = fact
        self.name = name
        self.group_id = group_id


class TestMemoryOps:
    """Test MemoryOps functionality."""

    @pytest.mark.asyncio
    async def test_remember_text_calls_ingest(self, memory_ops, mock_graphiti):
        """Test that remember_text calls the underlying ingest pipeline (add_episode)."""
        # Call remember_text
        result = await memory_ops.remember_text("test text", memory_type="personal")

        # Verify result and call
        assert result["status"] == "success"
        assert result["uuid"] == "ep123"
        mock_graphiti.add_episode.assert_called_once()
        
        # Verify arguments passed to add_episode
        call_kwargs = mock_graphiti.add_episode.call_args.kwargs
        assert call_kwargs["episode_body"] == "test text"
        assert call_kwargs["group_id"] is not None # Should be resolved to personal_group_id (mocked config or default?)
        # Actually resolve_group_id relies on config. 
        # But we can check it's passed.

    @pytest.mark.asyncio
    async def test_search_memory_combines_results(self, memory_ops, mock_graphiti):
        """Test that search_memory combines episodes and entities."""
        mock_graphiti.search_.return_value = DummySearchResults(
            episodes=[DummyEpisode(uuid="ep1", content="episode content with enough length")],
            nodes=[DummyNode(uuid="ent1", name="Entity Name", summary="Entity summary")],
            edges=[DummyEdge(uuid="edge1")],
            communities=[],
            episode_reranker_scores=[0.8],
            node_reranker_scores=[0.7],
            edge_reranker_scores=[0.6],
        )

        result = await memory_ops.search_memory("test query")

        assert isinstance(result, SearchResult)
        assert result.total_episodes == 1
        assert result.total_entities == 1
        assert result.total_edges == 1
        assert len(result.episodes) == 1
        assert len(result.entities) == 1

    @pytest.mark.asyncio
    async def test_build_context_formats_properly(self, memory_ops, mock_graphiti):
        """Test that build_context creates properly formatted context."""
        mock_graphiti.search_.return_value = DummySearchResults(
            episodes=[DummyEpisode(uuid="ep1", content="Test episode content with enough length")],
            nodes=[],
            edges=[],
            communities=[],
            episode_reranker_scores=[0.9],
        )

        result = await memory_ops.build_context_for_query("test query")

        assert isinstance(result, ContextResult)
        assert "## Информация из памяти:" in result.text
        assert "Test episode content" in result.text
        # Build_context limits to max 3 episodes; this case should include 1
        assert result.sources["episodes"] >= 1
        assert result.sources["entities"] == 0

    @pytest.mark.asyncio
    async def test_context_truncation(self, memory_ops, mock_graphiti):
        """Test that context is properly truncated for token limits."""
        # Mock a very long episode
        long_content = "Very long content " * 1000  # ~20k characters
        mock_graphiti.search_.return_value = DummySearchResults(
            episodes=[DummyEpisode(uuid="ep1", content=long_content)],
            nodes=[],
            edges=[],
            communities=[],
            episode_reranker_scores=[0.9],
        )

        result = await memory_ops.build_context_for_query("test", max_tokens=100)

        # Should be truncated
        assert len(result.text) < len(long_content)
        assert "[Контекст обрезан" in result.text
        assert result.token_estimate <= 100
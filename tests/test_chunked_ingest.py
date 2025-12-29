import pytest
from unittest.mock import AsyncMock, Mock, patch
from knowledge.ingest import ingest_text_document

@pytest.fixture
def mock_graphiti():
    g = Mock()
    # Mock add_episode to return an object with a uuid attribute and extracted_lists
    ep_mock = Mock()
    ep_mock.uuid = "test-uuid"
    ep_mock.episode.uuid = "test-uuid" # Handle both return styles
    ep_mock.extracted_entities = []
    ep_mock.extracted_edges = []
    
    g.add_episode = AsyncMock(return_value=ep_mock)
    g.driver = Mock()
    g.driver.execute_query = AsyncMock(return_value=Mock(records=[]))
    return g

@pytest.fixture
def mock_jobs():
    with patch("api.jobs.update_upload_job") as mock:
        yield mock

@pytest.mark.asyncio
async def test_ingest_chunks_text(mock_graphiti, mock_jobs):
    # Create a text that is longer than the chunk size (1500)
    # 2000 chars should split into at least 2 chunks
    long_text = "A" * 1600 + " " + "B" * 400
    
    result = await ingest_text_document(
        mock_graphiti,
        long_text,
        source_description="test_doc",
        user_id="user1",
        job_id="job123"
    )
    
    # Verify result
    assert result["status"] == "ok"
    assert result["chunks"] >= 2
    assert result["added"] >= 2
    
    # Verify add_episode called multiple times
    assert mock_graphiti.add_episode.call_count >= 2
    
    # Verify job updates
    # Should update for start (0/N), each chunk (i/N), and done (N/N)
    assert mock_jobs.call_count >= 3
    
    # Check calls arguments
    calls = mock_jobs.call_args_list
    # First call: processed_chunks=0
    assert calls[0].kwargs['processed_chunks'] == 0
    assert calls[0].kwargs['stage'] == 'ingest'
    
    # Last call: stage='done'
    assert calls[-1].kwargs['stage'] == 'done'
    assert calls[-1].kwargs['processed_chunks'] == result["chunks"]

@pytest.mark.asyncio
async def test_ingest_short_text(mock_graphiti, mock_jobs):
    short_text = "Short text"
    
    result = await ingest_text_document(
        mock_graphiti,
        short_text,
        source_description="short_doc"
    )
    
    assert result["status"] == "ok"
    assert result["chunks"] == 1
    assert result["added"] == 1
    mock_graphiti.add_episode.assert_called_once()


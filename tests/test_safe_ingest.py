import pytest
from unittest.mock import AsyncMock, Mock
from pydantic import ValidationError
from knowledge.ingest import ingest_text_document
from core.safe_graphiti import filter_graphiti_results

@pytest.mark.asyncio
async def test_filter_graphiti_results_with_malformed_data():
    # Mock Graphiti result object with some valid and some invalid entities
    class MockEntity:
        def __init__(self, uuid, name, summary=""):
            self.uuid = uuid
            self.name = name
            self.summary = summary
            self.node_type = "Entity"

    class MockEdge:
        def __init__(self, source, target, rel_type):
            self.source_node_uuid = source
            self.target_node_uuid = target
            self.relationship_type = rel_type

    mock_results = Mock()
    mock_results.extracted_entities = [
        MockEntity("uuid1", "Valid Entity"),
        MockEntity(None, "Invalid - No UUID"),
        MockEntity("uuid2", None),
        None
    ]
    mock_results.extracted_edges = [
        MockEdge("uuid1", "uuid2", "RELATES_TO"),
        MockEdge(None, "uuid2", "NO_SOURCE"),
        None
    ]

    filtered = filter_graphiti_results(mock_results)
    
    assert len(filtered["entities"]) == 1
    assert filtered["entities"][0]["uuid"] == "uuid1"
    assert filtered["dropped_entities"] == 3
    
    assert len(filtered["edges"]) == 1
    assert filtered["edges"][0]["relationship_type"] == "RELATES_TO"
    assert filtered["dropped_edges"] == 2

@pytest.mark.asyncio
async def test_ingest_document_validation_recovery():
    # Mock graphiti and its driver
    mock_graphiti = AsyncMock()
    mock_driver = AsyncMock()
    mock_graphiti.driver = mock_driver
    
    # Mock add_episode to raise ValidationError
    # We use a real Pydantic ValidationError for authenticity if possible, 
    # but it's hard to instantiate without a model. So we mock it or use a dummy model.
    from pydantic import BaseModel
    class DummyModel(BaseModel):
        x: int
    
    try:
        DummyModel(x="not an int")
    except ValidationError as e:
        ve = e

    mock_graphiti.add_episode.side_effect = ve
    
    # Mock driver.execute_query to find the episode during recovery
    mock_driver.execute_query.return_value = Mock(records=[{"uuid": "recovered-uuid"}])
    
    # Run ingest
    result = await ingest_text_document(
        mock_graphiti,
        "Test content",
        source_description="Test Source",
        user_id="test-user"
    )
    
    assert result["status"] == "ok"
    assert "recovered-uuid" in str(mock_graphiti.mock_calls) # Should have been used in subsequent calls
    assert len(result["warnings"]) > 0
    assert "Graphiti returned malformed entities/edges" in result["warnings"][0]

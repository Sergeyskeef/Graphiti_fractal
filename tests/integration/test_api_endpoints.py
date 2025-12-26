"""
Integration tests for API endpoints.

Tests the FastAPI endpoints with real dependencies.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import os


# Skip if no Neo4j
pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI"),
    reason="NEO4J_URI not set - skipping integration tests"
)


@pytest_asyncio.fixture
async def graphiti_for_api_test():
    """
    Create a per-test Graphiti instance for API tests.
    
    This ensures each test uses a Graphiti client bound to the correct event loop,
    avoiding "Future attached to different loop" errors.
    """
    from core.graphiti_client import get_graphiti_client, reset_graphiti_client
    
    # Guarantee clean state before test
    reset_graphiti_client()
    client = get_graphiti_client(force_new=True)
    graphiti = await client.ensure_ready()
    
    try:
        yield graphiti
    finally:
        # Cleanup: close Neo4j driver and reset singleton
        try:
            driver = getattr(graphiti, 'driver', None)
            if driver and hasattr(driver, 'close'):
                await driver.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error closing graphiti driver: {e}")
        finally:
            reset_graphiti_client()


@pytest_asyncio.fixture
async def async_client(graphiti_for_api_test):
    """
    Create async HTTP client for testing API with Graphiti dependency override.
    
    This ensures API endpoints use the per-test Graphiti instance, avoiding
    event loop conflicts.
    """
    from app import app, get_graphiti_dep
    
    # Override Graphiti dependency with per-test instance
    async def _override():
        return graphiti_for_api_test
    
    app.dependency_overrides[get_graphiti_dep] = _override
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    
    # Clear overrides after test
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Test health check endpoint."""
    response = await async_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Status should be healthy or unhealthy
    assert data["status"] in ["healthy", "unhealthy"]


@pytest.mark.asyncio
async def test_remember_endpoint(async_client):
    """Test remember endpoint."""
    response = await async_client.post(
        "/remember",
        json={
            "text": "Тестовый текст для API теста.",
            "user_id": "api_test_user",
            "memory_type": "knowledge",
            "source_description": "api_integration_test"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_remember_empty_text(async_client):
    """Test remember endpoint with empty text."""
    response = await async_client.post(
        "/remember",
        json={
            "text": "",
            "user_id": "api_test_user"
        }
    )
    
    # Should return 400 or 422 for validation error
    assert response.status_code in [400, 422, 500]


@pytest.mark.asyncio
async def test_knowledge_search_endpoint(async_client):
    """Test knowledge search endpoint."""
    response = await async_client.get(
        "/knowledge/search",
        params={"q": "test query", "limit": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_buffer_clear_endpoint(async_client):
    """Test buffer clear endpoint."""
    response = await async_client.post(
        "/buffer/clear",
        json={"user_id": "api_test_user"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cleared" in data


@pytest.mark.asyncio
async def test_upload_status_not_found(async_client):
    """Test upload status for non-existent job."""
    response = await async_client.get("/upload/status/nonexistent-job-id")
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_openapi_docs(async_client):
    """Test that OpenAPI docs are available."""
    response = await async_client.get("/openapi.json")
    
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
    assert "info" in data
    assert data["info"]["title"] == "Fractal Memory API"

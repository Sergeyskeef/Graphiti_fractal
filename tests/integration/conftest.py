"""
Pytest fixtures for integration tests.
"""

import pytest
import os
import pytest_asyncio

# Skip all integration tests if NEO4J_URI is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("NEO4J_URI"),
    reason="NEO4J_URI not set - skipping integration tests"
)


@pytest_asyncio.fixture(scope="session")
async def graphiti_client():
    """
    Get Graphiti client for integration tests.
    
    Session-scoped to reuse the same Neo4j driver across all tests.
    In strict mode, pytest-asyncio manages the event loop automatically.
    """
    from core.graphiti_client import get_graphiti_client
    
    client = get_graphiti_client()
    graphiti = await client.ensure_ready()
    yield graphiti
    
    # Cleanup: close Neo4j driver to avoid event loop conflicts
    try:
        driver = getattr(graphiti, 'driver', None)
        if driver and hasattr(driver, 'close'):
            await driver.close()
    except Exception as e:
        # Log but don't fail on cleanup errors
        import logging
        logging.getLogger(__name__).warning(f"Error closing graphiti driver: {e}")


@pytest_asyncio.fixture
async def clean_test_data(graphiti_client):
    """
    Clean up test data before and after each test.
    
    Uses a specific test group_id to isolate test data.
    """
    test_group_id = "integration_test"
    driver = graphiti_client.driver
    
    # Clean before test
    await driver.execute_query(
        "MATCH (n) WHERE n.group_id = $gid DETACH DELETE n",
        gid=test_group_id
    )
    
    yield test_group_id
    
    # Clean after test
    await driver.execute_query(
        "MATCH (n) WHERE n.group_id = $gid DETACH DELETE n",
        gid=test_group_id
    )


@pytest_asyncio.fixture
async def memory_ops(graphiti_client):
    """Create MemoryOps instance for testing."""
    from core.memory_ops import MemoryOps
    
    return MemoryOps(graphiti_client, "test_user")


@pytest_asyncio.fixture
async def llm_client():
    """Get LLM client for testing."""
    from core.llm import get_async_client
    
    client = get_async_client()
    if not client:
        pytest.skip("OPENAI_API_KEY not set")
    return client

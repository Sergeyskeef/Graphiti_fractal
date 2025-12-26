"""
Integration tests for chat persistence:
- No duplicate saves
- Race condition safety for turn_index
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from core.graphiti_client import get_graphiti_client
from core.memory_ops import MemoryOps
from simple_chat_agent import SimpleChatAgent
from core.llm import get_async_client


async def wait_until(predicate, timeout=5.0, interval=0.1, description="condition"):
    """
    Poll until predicate returns True or timeout is reached.
    
    Args:
        predicate: Async callable that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Polling interval in seconds
        description: Description for error messages
        
    Returns:
        True if condition was met, False if timeout
    """
    start = asyncio.get_event_loop().time()
    while True:
        if await predicate():
            return True
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            return False
        await asyncio.sleep(interval)


@pytest.mark.asyncio
async def test_no_duplicate_chat_turns(graphiti_client):
    """
    Test that a single chat request creates exactly one chat_turn episode.
    """
    graphiti = graphiti_client
    user_id = f"test_no_dup_{datetime.now(timezone.utc).timestamp()}"
    marker_text = f"DUPLICATE_TEST_{datetime.now(timezone.utc).isoformat()}"
    
    # Create agent and send one message with unique marker
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    agent = SimpleChatAgent(llm_client, memory)
    
    await agent.answer_core(f"Test message for duplicate check: {marker_text}")
    
    # Wait for background task to complete using polling
    driver = graphiti.driver
    async def check_turn_exists():
        query = """
        MATCH (e:Episodic {episode_kind: "chat_turn"})
        WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
          AND e.content CONTAINS $marker
        RETURN count(e) AS count
        """
        result = await driver.execute_query(query, user_id=user_id, marker=marker_text)
        count = result.records[0]["count"] if result.records else 0
        return count >= 1
    
    # Poll until turn appears or timeout
    found = await wait_until(check_turn_exists, timeout=10.0, description="chat_turn with marker")
    assert found, f"Chat turn with marker '{marker_text}' was not created within timeout"
    
    # Check that there's exactly 1 turn with this marker
    final_query = """
    MATCH (e:Episodic {episode_kind: "chat_turn"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
      AND e.content CONTAINS $marker
    RETURN count(e) AS count
    """
    final_result = await driver.execute_query(final_query, user_id=user_id, marker=marker_text)
    final_count = final_result.records[0]["count"] if final_result.records else 0
    
    # Should have exactly 1 chat_turn with this marker
    assert final_count == 1, f"Expected exactly 1 chat_turn with marker, got {final_count}"


@pytest.mark.asyncio
async def test_turn_index_race_condition(graphiti_client):
    """
    Test that concurrent chat requests get unique turn_index values.
    """
    graphiti = graphiti_client
    user_id = f"test_race_{datetime.now(timezone.utc).timestamp()}"
    marker_prefix = f"RACE_TEST_{datetime.now(timezone.utc).isoformat()}"
    
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    
    # Send 5 concurrent requests with unique markers
    async def send_chat(i: int):
        agent = SimpleChatAgent(llm_client, memory)
        marker = f"{marker_prefix}_MSG_{i}"
        return await agent.answer_core(f"Concurrent message {i}: {marker}")
    
    results = await asyncio.gather(*[send_chat(i) for i in range(5)])
    
    # Wait for all background tasks to complete using polling
    driver = graphiti.driver
    async def check_all_turns_exist():
        query = """
        MATCH (e:Episodic {episode_kind: "chat_turn"})
        WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
          AND e.content CONTAINS $marker_prefix
        RETURN count(e) AS count
        """
        result = await driver.execute_query(query, user_id=user_id, marker_prefix=marker_prefix)
        count = result.records[0]["count"] if result.records else 0
        return count >= 5
    
    # Poll until all turns appear or timeout
    found = await wait_until(check_all_turns_exist, timeout=15.0, description="5 chat_turns with markers")
    assert found, f"Not all 5 chat turns with marker prefix '{marker_prefix}' were created within timeout"
    
    # Check that all turn_index values are unique
    query = """
    MATCH (e:Episodic {episode_kind: "chat_turn"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
      AND e.content CONTAINS $marker_prefix
    RETURN e.turn_index AS turn_index, e.conversation_id AS conversation_id
    ORDER BY e.turn_index
    """
    result = await driver.execute_query(query, user_id=user_id, marker_prefix=marker_prefix)
    
    turn_indices = [record["turn_index"] for record in result.records]
    conversation_ids = set(record["conversation_id"] for record in result.records)
    
    # All should be in the same conversation
    assert len(conversation_ids) == 1, f"Expected 1 conversation_id, got {len(conversation_ids)}"
    
    # All turn_index values should be unique
    assert len(turn_indices) == len(set(turn_indices)), f"Duplicate turn_index values found: {turn_indices}"
    
    # Should have 5 turns
    assert len(turn_indices) == 5, f"Expected 5 turns, got {len(turn_indices)}"
    
    # Turn indices should be sequential (1, 2, 3, 4, 5)
    assert turn_indices == list(range(1, 6)), f"Expected [1,2,3,4,5], got {turn_indices}"


@pytest.mark.asyncio
async def test_chat_summary_count(graphiti_client):
    """
    Test that summaries are created correctly (every 10 turns).
    """
    graphiti = graphiti_client
    user_id = f"test_summary_{datetime.now(timezone.utc).timestamp()}"
    marker_prefix = f"SUMMARY_TEST_{datetime.now(timezone.utc).isoformat()}"
    
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    
    # Send 12 messages with markers (should create 1 summary at turn 10)
    for i in range(12):
        agent = SimpleChatAgent(llm_client, memory)
        marker = f"{marker_prefix}_MSG_{i}"
        await agent.answer_core(f"Message {i}: {marker}")
        await asyncio.sleep(0.2)  # Small delay between messages
    
    # Wait for summary to be created using polling
    driver = graphiti.driver
    async def check_summary_exists():
        query = """
        MATCH (e:Episodic {episode_kind: "chat_summary"})
        WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
        RETURN count(e) AS count
        """
        result = await driver.execute_query(query, user_id=user_id)
        count = result.records[0]["count"] if result.records else 0
        return count >= 1
    
    # Poll until summary appears or timeout
    found = await wait_until(check_summary_exists, timeout=20.0, description="chat_summary")
    
    # Check summary count
    summary_query = """
    MATCH (e:Episodic {episode_kind: "chat_summary"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
    RETURN count(e) AS count
    """
    summary_result = await driver.execute_query(summary_query, user_id=user_id)
    summary_count = summary_result.records[0]["count"] if summary_result.records else 0
    
    # Should have at least 1 summary (at turn 10)
    assert summary_count >= 1, f"Expected at least 1 summary, got {summary_count}"


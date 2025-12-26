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


@pytest.mark.asyncio
async def test_no_duplicate_chat_turns(graphiti_client):
    """
    Test that a single chat request creates exactly one chat_turn episode.
    """
    graphiti = graphiti_client
    user_id = f"test_no_dup_{datetime.now(timezone.utc).timestamp()}"
    
    # Get initial count
    driver = graphiti.driver
    initial_query = """
    MATCH (e:Episodic {episode_kind: "chat_turn"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
    RETURN count(e) AS count
    """
    initial_result = await driver.execute_query(initial_query, user_id=user_id)
    initial_count = initial_result.records[0]["count"] if initial_result.records else 0
    
    # Create agent and send one message
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    agent = SimpleChatAgent(llm_client, memory)
    
    await agent.answer_core("Test message for duplicate check")
    
    # Wait for background tasks to complete (chat turn storage is async)
    await asyncio.sleep(5)
    
    # Check final count
    final_result = await driver.execute_query(initial_query, user_id=user_id)
    final_count = final_result.records[0]["count"] if final_result.records else 0
    
    # Should have exactly 1 new chat_turn
    assert final_count == initial_count + 1, f"Expected {initial_count + 1} chat_turns, got {final_count}"


@pytest.mark.asyncio
async def test_turn_index_race_condition(graphiti_client):
    """
    Test that concurrent chat requests get unique turn_index values.
    """
    graphiti = graphiti_client
    user_id = f"test_race_{datetime.now(timezone.utc).timestamp()}"
    
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    
    # Send 5 concurrent requests
    async def send_chat(message: str):
        agent = SimpleChatAgent(llm_client, memory)
        return await agent.answer_core(message)
    
    messages = [f"Concurrent message {i}" for i in range(5)]
    results = await asyncio.gather(*[send_chat(msg) for msg in messages])
    
    # Wait for background tasks to complete (chat turn storage is async)
    await asyncio.sleep(5)
    
    # Check that all turn_index values are unique
    driver = graphiti.driver
    query = """
    MATCH (e:Episodic {episode_kind: "chat_turn"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
    RETURN e.turn_index AS turn_index, e.conversation_id AS conversation_id
    ORDER BY e.turn_index
    """
    result = await driver.execute_query(query, user_id=user_id)
    
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
    
    memory = MemoryOps(graphiti, user_id)
    llm_client = get_async_client()
    
    # Send 12 messages (should create 1 summary at turn 10)
    for i in range(12):
        agent = SimpleChatAgent(llm_client, memory)
        await agent.answer_core(f"Message {i}")
        await asyncio.sleep(0.5)  # Small delay between messages
    
    # Wait for background tasks
    await asyncio.sleep(3)
    
    # Check summary count
    driver = graphiti.driver
    summary_query = """
    MATCH (e:Episodic {episode_kind: "chat_summary"})
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
    RETURN count(e) AS count
    """
    summary_result = await driver.execute_query(summary_query, user_id=user_id)
    summary_count = summary_result.records[0]["count"] if summary_result.records else 0
    
    # Should have at least 1 summary (at turn 10)
    assert summary_count >= 1, f"Expected at least 1 summary, got {summary_count}"


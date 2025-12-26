"""
Chat persistence utilities for atomic turn index allocation and safe chat storage.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def allocate_turn_index(graphiti, user_id: str, conversation_id: str) -> int:
    """
    Atomically allocate next turn index for a conversation in Neo4j.
    
    This ensures unique turn indices even under concurrent requests from
    multiple processes/containers.
    
    Args:
        graphiti: Graphiti instance
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Next turn index (1-based)
    """
    driver = graphiti.driver
    
    query = """
    MERGE (c:ChatTurnCounter {
        user_id: $user_id,
        conversation_id: $conversation_id
    })
    ON CREATE SET c.value = 0
    SET c.value = c.value + 1
    RETURN c.value AS turn_index
    """
    
    try:
        result = await driver.execute_query(
            query,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        if result.records:
            turn_index = result.records[0]["turn_index"]
            logger.debug(
                f"Allocated turn_index={turn_index} for user={user_id}, conversation={conversation_id}"
            )
            return turn_index
        else:
            logger.error("Failed to allocate turn_index - no records returned")
            return 1  # Fallback
    except Exception as e:
        logger.error(f"Error allocating turn_index: {e}", exc_info=e)
        return 1  # Fallback


async def get_conversation_turn_count(graphiti, user_id: str, conversation_id: str) -> int:
    """
    Get current turn count for a conversation (for summary logic).
    
    Args:
        graphiti: Graphiti instance
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Number of chat_turn episodes for this conversation
    """
    driver = graphiti.driver
    
    query = """
    MATCH (e:Episodic {
        episode_kind: "chat_turn",
        conversation_id: $conversation_id
    })
    WHERE EXISTS((:User {user_id: $user_id})-[:AUTHORED]->(e))
    RETURN count(e) AS turn_count
    """
    
    try:
        result = await driver.execute_query(
            query,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        if result.records:
            return result.records[0]["turn_count"]
        return 0
    except Exception as e:
        logger.error(f"Error getting conversation turn count: {e}", exc_info=e)
        return 0


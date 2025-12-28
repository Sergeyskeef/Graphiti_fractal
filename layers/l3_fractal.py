import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum

from core import get_graphiti_client
from core.llm import llm_chat_response
from core.memory_ops import MemoryOps
from layers.l1_consolidation import get_l1_context
from layers.l2_semantic import get_l2_semantic_context

logger = logging.getLogger(__name__)

class AbstractionLevel(Enum):
    L1_EPISODE = "episode"
    L2_SEMANTIC = "semantic_pattern"
    L3_FRACTAL = "fractal_abstraction"


async def build_l3_profile(graphiti, entity_name: str, user_id: str = "system"):
    """
    Generates a high-level L3 Fractal Profile for an entity based on L2 community structures.
    Saves the result as a new episode in the graph (Synthesis).
    """
    logger.info(f"Building L3 profile for '{entity_name}'...")
    
    # 1. Get L2 Context (Community Summaries)
    l2_ctx = await get_l2_semantic_context(graphiti, entity_name)
    if not l2_ctx:
        logger.warning(f"Not enough L2 data to build profile for '{entity_name}'")
        return None

    # 2. Synthesize Profile using LLM
    prompt = f"""
    You are the 'Fractal Memory' system analyst.
    Analyze the following semantic structures (communities) related to the entity '{entity_name}':
    
    {l2_ctx}
    
    Task:
    Synthesize a high-level "L3 Fractal Profile" for '{entity_name}'.
    The profile should define:
    1. System Role: What function does this entity perform in the larger system?
    2. Responsibilities: Key areas of influence.
    3. Trajectory: How has this entity evolved? (Deduce from community levels/types).
    4. Key Relationships: Who are the main collaborators or dependencies?
    
    Format:
    Use a structured, analytical tone. Use Markdown headers.
    """
    
    try:
        messages = [{"role": "user", "content": prompt}]
        profile_text = await llm_chat_response(messages, context="l3_build")
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        return None

    # 3. Save as Episode (Artifact)
    # We use a specific source_description to easily retrieve it later.
    source_desc = f"l3_profile:{entity_name}"
    
    ops = MemoryOps(graphiti, user_id)
    result = await ops.ingest_pipeline(
        profile_text,
        source_description=source_desc,
        memory_type="knowledge"  # Store in knowledge or project layer
    )
    
    logger.info(f"L3 profile built and saved: {result.get('uuid')}")
    return profile_text


async def get_l3_fractal_context(graphiti, entity_name: str):
    """
    L3: Retrieve the latest Fractal Profile for the entity.
    If no profile exists, falls back to a message.
    """
    driver = getattr(graphiti, "driver", None) or getattr(graphiti, "_driver", None)
    if not driver:
        return None

    source_desc = f"l3_profile:{entity_name}"
    
    # Find the most recent profile
    query = """
    MATCH (e:Episodic)
    WHERE e.source_description = $source
    RETURN e.content as content, e.created_at as created_at
    ORDER BY e.created_at DESC
    LIMIT 1
    """
    
    try:
        if hasattr(driver, 'execute_query'):
            res = await driver.execute_query(query, source=source_desc)
            records = res.records
        else:
            async with driver.session() as session:
                res = await session.run(query, source=source_desc)
                records = await res.list()
                
        if records:
            content = records[0]["content"]
            created = records[0]["created_at"]
            return f"🌀 L3 FRACTAL PROFILE (Generated {created}):\n\n{content}"
            
    except Exception as e:
        logger.error(f"Error fetching L3 profile: {e}")

    return f"No L3 profile found for '{entity_name}'. Run build_l3_profile() to generate."


async def test_l3():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    
    # Optional: trigger build
    # await build_l3_profile(graphiti, "Sergey")
    
    context = await get_l3_fractal_context(graphiti, "Sergey")
    print(context)


if __name__ == "__main__":
    asyncio.run(test_l3())

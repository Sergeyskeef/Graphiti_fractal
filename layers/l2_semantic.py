import asyncio
import logging
from collections import defaultdict

from core import get_graphiti_client

logger = logging.getLogger(__name__)

async def trigger_community_build(graphiti):
    """
    Background task to rebuild communities (Graphiti native clustering).
    This should be called periodically, not on every request.
    """
    try:
        logger.info("Starting L2 community build...")
        await graphiti.build_communities()
        logger.info("L2 community build completed.")
    except Exception as e:
        logger.error(f"Failed to build communities: {e}")


async def get_l2_semantic_context(graphiti, entity_name: str) -> str:
    """
    L2: Retrieve community summaries (Graphiti Native).
    Uses pre-calculated community summaries to provide structural context.
    """
    driver = getattr(graphiti, "driver", None) or getattr(graphiti, "_driver", None)
    if not driver:
        return "Graphiti driver not found for L2 context."

    # 1. Find communities relevant to the entity (via membership or text match)
    # Graphiti structure: (Entity)-[:IN_COMMUNITY]->(Community) or similar.
    # Checking docs/schema: usually (:Entity)-[:MEMBER_OF]->(:Community) or similar.
    # Let's assume standard Graphiti schema for communities.
    # If not sure, we can search for nodes with label 'Community'.
    
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($name)
    MATCH (e)-[:IN_COMMUNITY]->(c:Community)
    RETURN DISTINCT c.uuid as uuid, c.name as name, c.summary as summary, c.level as level
    ORDER BY c.level ASC
    LIMIT 5
    """
    
    try:
        if hasattr(driver, 'execute_query'):
            res = await driver.execute_query(query, name=entity_name)
            records = res.records
        else:
            async with driver.session() as session:
                res = await session.run(query, name=entity_name)
                records = await res.list()
    except Exception as e:
        logger.warning(f"L2 community query failed: {e}. Are communities built?")
        return "L2 Context: No community structure found (try running build_communities)."

    if not records:
        return None

    summary_text = f"🧠 L2 Semantic Context (Communities) for '{entity_name}':\n\n"
    
    for rec in records:
        c_name = rec['name'] or "Unnamed Community"
        c_sum = rec['summary'] or "No summary available."
        c_level = rec.get('level', '?')
        summary_text += f"=== Community: {c_name} (Level {c_level}) ===\n{c_sum}\n\n"

    return summary_text


async def test_l2():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    
    # Optional: trigger build if needed for test (commented out for speed)
    # await trigger_community_build(graphiti)
    
    context = await get_l2_semantic_context(graphiti, "Sergey")
    print(context)


if __name__ == "__main__":
    asyncio.run(test_l2())

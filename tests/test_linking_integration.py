import asyncio
import uuid
import logging
import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graphiti_client import get_graphiti_client
from core.memory_ops import MemoryOps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_linking_int")

async def test_integration():
    client = get_graphiti_client()
    graphiti = await client.ensure_ready()
    
    unique_name = f"LinkTest_{uuid.uuid4().hex[:6]}"
    logger.info(f"Testing with unique entity name: {unique_name}")
    
    # 1. Create Episode in 'group_a'
    logger.info("Adding episode to group_a...")
    await graphiti.add_episode(
        name="test_ep_a",
        episode_body=f"{unique_name} is a test entity in group A. It has property A.",
        source_description="test",
        reference_time=datetime.now(timezone.utc),
        group_id="group_a"
    )
    
    # 2. Create Episode in 'group_b'
    logger.info("Adding episode to group_b...")
    await graphiti.add_episode(
        name="test_ep_b",
        episode_body=f"{unique_name} is also in group B. It has property B.",
        source_description="test",
        reference_time=datetime.now(timezone.utc),
        group_id="group_b"
    )
    
    # 3. Verify SAME_AS
    logger.info("Verifying SAME_AS bridge...")
    driver = graphiti.driver
    query = """
    MATCH (e1:Entity {name: $name, group_id: 'group_a'})
    MATCH (e2:Entity {name: $name, group_id: 'group_b'})
    MATCH (e1)-[r:SAME_AS]-(e2)
    RETURN count(r) as links
    """
    
    # Retry a few times as extraction/linking might be async or slow? 
    # Actually linking is awaited in add_episode, but extraction is LLM based.
    # LLM might fail to extract the exact name. This is the flaky part.
    # We assume LLM works for simple sentences.
    
    links = 0
    for _ in range(3):
        res = await driver.execute_query(query, name=unique_name)
        links = res.records[0]['links']
        if links > 0:
            break
        await asyncio.sleep(1)
        
    if links > 0:
        logger.info("✅ SAME_AS bridge created successfully.")
    else:
        logger.error("❌ SAME_AS bridge NOT found.")
        # Debug: check if entities exist
        debug_q = "MATCH (e:Entity) WHERE e.name CONTAINS 'LinkTest_' RETURN e.name, e.group_id"
        d_res = await driver.execute_query(debug_q)
        logger.info(f"Entities found: {[r.data() for r in d_res.records]}")
        return

    # 4. Verify Retrieval Expansion
    logger.info("Verifying Retrieval Expansion...")
    memory = MemoryOps(graphiti, "tester")
    
    # Search in group_a only
    # Should find 'property B' fact via expansion if logic works
    results = await memory.search_memory(f"What about {unique_name}?", scopes=["group_a"])
    
    found_expanded = False
    for edge in results.edges:
        if edge.get('is_expanded'):
            logger.info(f"Found expanded edge: {edge}")
            found_expanded = True
    
    for entity in results.entities:
        if entity.get('is_expanded'):
            logger.info(f"Found expanded entity: {entity['name']} ({entity['group_id']})")
            found_expanded = True
            
    if found_expanded:
        logger.info("✅ Retrieval expansion verified.")
    else:
        logger.warning("⚠️ No expanded items found (could be due to no facts extracted yet).")

if __name__ == "__main__":
    asyncio.run(test_integration())

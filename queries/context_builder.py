from typing import Optional, Dict, Any, List
import re
import logging

# Import filters for robust temporal retrieval
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

logger = logging.getLogger(__name__)

async def build_agent_context(graphiti, entity_name: str, context_size: str = "full") -> Optional[str]:
    """
    Build context window for LLM agent using optimized bulk fetching.
    
    context_size: "minimal" (5 nodes) | "medium" (15 nodes) | "full" (50 nodes)
    """

    size_map = {
        "minimal": 5,
        "medium": 15,
        "full": 50,
    }
    limit = size_map.get(context_size, 15)

    # 1. Search for relevant edges using Graphiti's hybrid search
    # CRITICAL: Use search_() with explicit temporal filter to avoid invalidated facts.
    try:
        # Current facts only (invalid_at IS NULL)
        search_filter = SearchFilters(
            invalid_at=[[DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)]]
        )
        
        # We use search_ which returns a SearchResult object
        search_result = await graphiti.search_(
            query=entity_name,
            config=COMBINED_HYBRID_SEARCH_RRF,
            search_filter=search_filter
        )
        
        # We are primarily interested in edges for context, but search_result also has nodes/episodes
        # We filter edges manually to limit count if needed, though search_ usually respects config limits.
        # But here we just take what returned.
        edges = search_result.edges if search_result else []
        
        # Slice to limit if needed (though bulk fetch controls size too)
        if len(edges) > limit:
            edges = edges[:limit]

    except Exception as e:
        logger.error(f"Error searching graphiti for context: {e}")
        return None

    if not edges:
        return None

    # 2. Collect unique UUIDs to fetch
    uuids = set()
    valid_edges = []
    
    for edge in edges:
        src_uuid = getattr(edge, "source_node_uuid", None)
        tgt_uuid = getattr(edge, "target_node_uuid", None)
        
        if src_uuid and tgt_uuid:
            uuids.add(src_uuid)
            uuids.add(tgt_uuid)
            valid_edges.append(edge)
            
    if not uuids:
        return None

    # 3. Bulk fetch node data in one Cypher query (N+1 fix)
    driver = getattr(graphiti, "driver", None) or getattr(graphiti, "_driver", None)
    if not driver:
        logger.error("Graphiti driver not found")
        return None

    node_map: Dict[str, Dict[str, Any]] = {}
    
    fetch_query = """
    MATCH (n)
    WHERE n.uuid IN $uuids
    RETURN n.uuid as uuid, 
           labels(n) as labels, 
           n.name as name, 
           n.summary as summary, 
           n.content as content, 
           n.episode_body as episode_body, 
           n.source_description as source_description,
           n.deleted as deleted
    """
    
    try:
        if hasattr(driver, 'execute_query'):
            res = await driver.execute_query(fetch_query, uuids=list(uuids))
            records = res.records
        else:
            async with driver.session() as session:
                res = await session.run(fetch_query, uuids=list(uuids))
                records = await res.list()

        for record in records:
            node_data = {
                "uuid": record["uuid"],
                "labels": record["labels"],
                "name": record["name"],
                "summary": record["summary"],
                "content": record["content"],
                "episode_body": record["episode_body"],
                "source_description": record["source_description"],
                "deleted": record["deleted"]
            }
            node_map[record["uuid"]] = node_data
            
    except Exception as e:
        logger.error(f"Error bulk fetching nodes: {e}")
        return None

    # 4. Helper functions for text formatting
    def is_hashy(val: str) -> bool:
        return bool(val and re.fullmatch(r"[0-9a-fA-F-]{8,}", val))

    def clean(val):
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() == "unknown":
            return None
        return s

    def get_node_text(node_data: Dict[str, Any]) -> str:
        if not node_data:
            return "unknown"
            
        labels = node_data.get("labels", []) or []
        label = labels[0] if labels else ""
        
        if label == "Episodic":
            for attr in ("summary", "content", "episode_body"):
                val = clean(node_data.get(attr))
                if val:
                    if len(val) > 240:
                        val = val[:240].strip() + "..."
                    return val
                    
        for attr in ("summary", "name", "source_description"):
            val = clean(node_data.get(attr))
            if not val:
                continue
            if attr == "name" and is_hashy(val):
                continue
            return val
            
        return node_data.get("uuid", "unknown")

    # 5. Build facts from edges and fetched nodes
    facts = []
    seen_facts = set()
    
    for edge in valid_edges[:limit]:
        src_uuid = getattr(edge, "source_node_uuid", None)
        tgt_uuid = getattr(edge, "target_node_uuid", None)
        rel = getattr(edge, "relationship_type", "RELATES_TO")
        
        src_node = node_map.get(src_uuid)
        tgt_node = node_map.get(tgt_uuid)
        
        if not src_node or not tgt_node:
            continue
            
        if src_node.get("deleted") or tgt_node.get("deleted"):
            continue
            
        src_txt = get_node_text(src_node)
        tgt_txt = get_node_text(tgt_node)
        
        if is_hashy(src_txt) and is_hashy(tgt_txt):
            continue
            
        fact_str = f"- {src_txt} {rel} {tgt_txt}"
        
        if fact_str not in seen_facts:
            facts.append(fact_str)
            seen_facts.add(fact_str)

    if not facts:
        return None

    context = f"You have the following context about '{entity_name}':\n\n"
    context += "\n".join(facts)
    return context

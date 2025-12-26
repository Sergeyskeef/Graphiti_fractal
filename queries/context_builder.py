from typing import Optional
import re


async def build_agent_context(graphiti, entity_name: str, context_size: str = "full") -> Optional[str]:
    """
    Build context window for LLM agent.

    context_size: "minimal" (5 nodes) | "medium" (15 nodes) | "full" (50 nodes)
    """

    size_map = {
        "minimal": 5,
        "medium": 15,
        "full": 50,
    }
    limit = size_map.get(context_size, 15)

    edges = await graphiti.search(entity_name, num_results=limit)
    if not edges:
        return None

    def is_hashy(val: str) -> bool:
        return bool(val and re.fullmatch(r"[0-9a-fA-F-]{8,}", val))

    def clean(val):
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() == "unknown":
            return None
        return s

    def node_text(node):
        labels = getattr(node, "labels", []) or []
        label = labels[0] if labels else ""
        if label == "Episodic":
            for attr in ("summary", "content", "episode_body"):
                val = clean(getattr(node, attr, None))
                if val:
                    if len(val) > 240:
                        val = val[:240].strip() + "..."
                    return val
        for attr in ("summary", "name", "source_description"):
            val = clean(getattr(node, attr, None))
            if not val:
                continue
            if attr == "name" and is_hashy(val):
                continue
            return val
        return getattr(node, "uuid", "unknown")

    facts = []
    for edge in edges[:limit]:
        src_uuid = getattr(edge, "source_node_uuid", None)
        tgt_uuid = getattr(edge, "target_node_uuid", None)
        rel = getattr(edge, "relationship_type", "RELATES_TO")
        if not src_uuid or not tgt_uuid:
            continue
        try:
            src_node = await graphiti.get_node_by_uuid(src_uuid)
            tgt_node = await graphiti.get_node_by_uuid(tgt_uuid)
        except Exception:
            continue
        if getattr(src_node, "deleted", False) or getattr(tgt_node, "deleted", False):
            continue
        src_txt = node_text(src_node)
        tgt_txt = node_text(tgt_node)
        if is_hashy(src_txt) and is_hashy(tgt_txt):
            continue
        facts.append(f"- {src_txt} {rel} {tgt_txt}")

    if not facts:
        return None

    context = f"You have the following context about '{entity_name}':\n\n"
    context += "\n".join(facts)
    return context


# Usage example (in a script):
# from core import get_graphiti_client
# graphiti = await get_graphiti_client().ensure_ready()
# context = await build_agent_context(graphiti, "Sergey", "full")


import asyncio
from collections import defaultdict

from core import get_graphiti_client


async def get_l2_semantic_context(graphiti, entity_name: str):
    """
    L2: Extract semantic patterns from relationships.
    Показывает структуру взаимодействий сущности.
    """

    edges = await graphiti.search(entity_name, num_results=10)
    if not edges:
        return None

    relationship_patterns = defaultdict(list)

    for edge in edges:
        rel_type = getattr(edge, "relationship_type", "RELATES_TO")
        relationship_patterns[rel_type].append(
            {
                "source": getattr(edge, "source_node_uuid", "?"),
                "target": getattr(edge, "target_node_uuid", "?"),
                "confidence": getattr(edge, "confidence", 0.95),
            }
        )

    summary = f"🧠 L2 Semantic Context for '{entity_name}':\n\n"
    summary += "Relationship Patterns:\n"

    for rel_type, instances in relationship_patterns.items():
        summary += f"\n  {rel_type} ({len(instances)} instances):\n"
        for instance in instances[:3]:
            summary += (
                f"    • {instance['source']} → {instance['target']} "
                f"(confidence: {instance['confidence']:.0%})\n"
            )

    return summary


async def test_l2():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    context = await get_l2_semantic_context(graphiti, "Sergey")
    print(context)


if __name__ == "__main__":
    asyncio.run(test_l2())


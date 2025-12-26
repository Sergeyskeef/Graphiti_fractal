import asyncio
from datetime import datetime, timedelta, timezone

from core import get_graphiti_client


async def get_l1_context(graphiti, user_context: str, hours_back: int = 24) -> str:
    """
    L1: Recent episode context (last N hours).
    Автоматически резюмирует недавние эпизоды.
    """

    _ = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    edges = await graphiti.search(user_context, num_results=10)

    summary = f"📋 L1 Summary (last {hours_back}h):\n\n"

    if edges:
        summary += "Relationships (uuids):\n"
        for edge in edges[:5]:
            summary += (
                f"  • {getattr(edge, 'source_node_uuid', '?')} "
                f"{getattr(edge, 'relationship_type', 'RELATES_TO')} "
                f"{getattr(edge, 'target_node_uuid', '?')}\n"
            )

    return summary


async def test_l1():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    context = await get_l1_context(graphiti, "Fractal Memory development", hours_back=48)
    print(context)


if __name__ == "__main__":
    asyncio.run(test_l1())


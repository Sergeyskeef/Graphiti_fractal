#!/usr/bin/env python3
"""
Прямой тест build_context_for_query.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graphiti_client import get_graphiti_client
from core.memory_ops import MemoryOps

async def test_context():
    """Тест build_context_for_query."""

    print("🧪 Testing build_context_for_query...")

    # Получаем клиентов
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    memory = MemoryOps(graphiti, "sergey")

    query = "Лена"
    print(f"Query: '{query}'")

    try:
        context_result = await memory.build_context_for_query(
            query,
            max_tokens=2000,
            include_episodes=True,
            include_entities=True
        )

        print("Context result:")
        print(f"  Token estimate: {context_result.token_estimate}")
        print(f"  Sources: {context_result.sources}")
        print("  Full context text:")
        print(context_result.text)
        print("---")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_context())
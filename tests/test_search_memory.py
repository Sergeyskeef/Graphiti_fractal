#!/usr/bin/env python3
"""
Прямой тест search_memory после rebuild'а.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graphiti_client import get_graphiti_client
from core.memory_ops import MemoryOps

async def test_search_memory():
    """Тест нового search_memory с Graphiti."""

    print("🧪 Testing search_memory with Graphiti search_()...")

    # Получаем клиентов
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    memory = MemoryOps(graphiti, "sergey")

    # Тестовые запросы
    queries = [
        "Лена",
        "Женя",
        "архетипы Марка",
        "дизайнер",
        "разработчик"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            result = await memory.search_memory(query, limit=5)

            print(f"  Episodes: {result.total_episodes}")
            print(f"  Entities: {result.total_entities}")
            print(f"  Edges: {result.total_edges}")
            print(f"  Communities: {result.total_communities}")

            # Показываем топ результатов
            if result.entities:
                print("  Top entities:")
                for entity in result.entities[:2]:
                    print(f"    - {entity.get('name', '')}: {entity.get('summary', '')[:50]}...")

            if result.episodes:
                print("  Top episodes:")
                for episode in result.episodes[:2]:
                    content = episode.get('content', '')[:50]
                    score = episode.get('score', 0)
                    print(f"    - Score {score:.2f}: {content}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_memory())
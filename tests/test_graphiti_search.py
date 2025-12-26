#!/usr/bin/env python3
"""
Прямой тест Graphiti search_() для диагностики проблемы с group_ids
"""

import asyncio
import sys
import os

# Add project root to path so `import core.*` works when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from core.graphiti_client import get_graphiti_client

async def test_graphiti_search():
    """Тест Graphiti search_() с разными group_ids."""

    print("🧪 Testing Graphiti search_() directly...")

    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()

    query = "Лена"

    # Тест 1: без group_ids
    print(f"\n1. Search without group_ids:")
    results1 = await graphiti.search_(query=query, config=COMBINED_HYBRID_SEARCH_RRF)
    print(f"   Episodes: {len(results1.episodes)}")
    print(f"   Nodes: {len(results1.nodes)}")
    print(f"   Edges: {len(results1.edges)}")

    # Тест 2: с group_ids=['personal', 'knowledge', 'project']
    print(f"\n2. Search with group_ids=['personal', 'knowledge', 'project']:")
    results2 = await graphiti.search_(
        query=query,
        config=COMBINED_HYBRID_SEARCH_RRF,
        group_ids=['personal', 'knowledge', 'project']
    )
    print(f"   Episodes: {len(results2.episodes)}")
    print(f"   Nodes: {len(results2.nodes)}")
    print(f"   Edges: {len(results2.edges)}")

    # Тест 3: с group_ids=['personal']
    print(f"\n3. Search with group_ids=['personal']:")
    results3 = await graphiti.search_(
        query=query,
        config=COMBINED_HYBRID_SEARCH_RRF,
        group_ids=['personal']
    )
    print(f"   Episodes: {len(results3.episodes)}")
    print(f"   Nodes: {len(results3.nodes)}")
    print(f"   Edges: {len(results3.edges)}")

    # Тест 4: проверим episodes в результатах
    if results2.episodes:
        print("\n4. Episodes in results2 (should be 0):")
        for i, ep in enumerate(results2.episodes[:3]):
            uuid = getattr(ep, 'uuid', 'no-uuid')
            content = getattr(ep, 'content', '')[:50]
            print(f"   {i+1}. {uuid}: {content}...")
    else:
        print("\n4. No episodes in results2")

    if results3.episodes:
        print("\n5. Episodes in results3:")
        for i, ep in enumerate(results3.episodes[:3]):
            uuid = getattr(ep, 'uuid', 'no-uuid')
            content = getattr(ep, 'content', '')[:50]
            print(f"   {i+1}. {uuid}: {content}...")
    else:
        print("\n5. No episodes in results3")

if __name__ == "__main__":
    asyncio.run(test_graphiti_search())
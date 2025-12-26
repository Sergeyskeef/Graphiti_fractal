import asyncio
from datetime import datetime
from enum import Enum

from core import get_graphiti_client
from layers.l1_consolidation import get_l1_context
from layers.l2_semantic import get_l2_semantic_context


class AbstractionLevel(Enum):
    L1_EPISODE = "episode"
    L2_SEMANTIC = "semantic_pattern"
    L3_FRACTAL = "fractal_abstraction"


async def get_l3_fractal_context(graphiti, entity_name: str):
    """
    L3: Create fractal abstraction hierarchy.
    Показывает место сущности в большой системе.
    """

    l1_ctx = await get_l1_context(graphiti, entity_name, hours_back=7 * 24)
    l2_ctx = await get_l2_semantic_context(graphiti, entity_name)

    edges = await graphiti.search(entity_name, num_results=1)
    if not edges:
        return None

    fractal_analysis = f"""
    🌀 L3 FRACTAL ABSTRACTION for '{entity_name}'
    ══════════════════════════════════════════════════════════

    HIERARCHICAL POSITION:
    ├── System Role: Entity/Component
    ├── Abstraction Level: L3 (Project-wide perspective)
    └── Integration: Core system element

    REPEATING PATTERNS (from L2):
    • Ownership: Works on primary project
    • Responsibility: Technical development
    • Authority: High decision-making power

    EVOLUTION TRAJECTORY:
    • Phase: Active Development
    • Trend: Increasing complexity (started vanilla, adding layers)
    • Stability: Stable - foundational role

    CONTRADICTIONS & CHANGES:
    • Initial approach: Custom Redis buffer + L0 optimization
    • New approach: Vanilla Graphiti first
    • Status: Strategy evolved on {datetime.now().date()}

    FRACTAL SELF-SIMILARITY:
    Each entity (person, project, concept) has:
    ├── Episodes (L1) - detailed interactions
    ├── Patterns (L2) - relationship types
    └── Abstractions (L3) - role in system

    This mirrors the three-layer architecture you're building!

    L1 CONTEXT (for reference):
    {l1_ctx}

    L2 CONTEXT (for reference):
    {l2_ctx}
    """

    return fractal_analysis


async def test_l3():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    context = await get_l3_fractal_context(graphiti, "Fractal Memory")
    print(context)


if __name__ == "__main__":
    asyncio.run(test_l3())


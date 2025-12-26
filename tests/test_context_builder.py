import asyncio

import pytest

from queries.context_builder import build_agent_context


class DummySearchResults:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or []
        self.edges = edges or []


class DummyNode:
    def __init__(self, name, node_type="Dummy", uuid="1", labels=None, summary=None, content=None):
        self.name = name
        self.node_type = node_type
        self.uuid = uuid
        self.labels = labels or ["Entity"]
        self.summary = summary
        self.content = content


class DummyEdge:
    def __init__(self, source_node_uuid, target_node_uuid, relationship_type="RELATES_TO"):
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.relationship_type = relationship_type


class DummyGraphiti:
    def __init__(self):
        self._calls = []
        self._nodes = {}

    async def _search(self, query, limit=1, **kwargs):
        self._calls.append((query, limit, kwargs))
        if query == "missing":
            return []
        src = DummyNode(name=query.title(), uuid="src")
        tgt = DummyNode(name="Other", uuid="tgt")
        self._nodes[src.uuid] = src
        self._nodes[tgt.uuid] = tgt
        edge = DummyEdge(source_node_uuid=src.uuid, target_node_uuid=tgt.uuid)
        return [edge]

    async def search(self, query, num_results=1, **kwargs):
        # адаптер под текущий интерфейс Graphiti
        return await self._search(query, limit=num_results, **kwargs)

    async def get_node_by_uuid(self, uuid):
        return self._nodes[uuid]


@pytest.mark.asyncio
async def test_build_agent_context_returns_none_when_not_found():
    graphiti = DummyGraphiti()
    result = await build_agent_context(graphiti, "missing")
    assert result is None


@pytest.mark.asyncio
async def test_build_agent_context_builds_list():
    graphiti = DummyGraphiti()
    result = await build_agent_context(graphiti, "sergey", context_size="minimal")
    assert "Sergey" in result
    assert "context" in result.lower()


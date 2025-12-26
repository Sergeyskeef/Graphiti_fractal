import asyncio
import json

from core import get_graphiti_client


async def export_graph_for_vis(graphiti, depth: int = 2, limit: int = 50):
    """
    Export graph structure for D3.js/Cytoscape visualization.
    Returns: {nodes: [...], edges: [...]}
    """

    search_results = await graphiti._search("*", limit=limit)

    nodes_data = []
    edges_list = []

    for node in search_results.nodes:
        nodes_data.append(
            {
                "id": str(node.uuid),
                "label": node.name,
                "title": node.node_type,
                "type": node.node_type,
                "size": 20 if "Person" in node.node_type else 30,
            }
        )

    for edge in search_results.edges:
        edges_list.append(
            {
                "from": str(edge.source_id),
                "to": str(edge.target_id),
                "label": edge.relationship_type,
                "arrows": "to",
            }
        )

    return {
        "nodes": nodes_data,
        "edges": edges_list,
        "statistics": {
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_list),
            "node_types": list(set(n["type"] for n in nodes_data)),
        },
    }


async def export_to_file(graphiti, filename: str = "visualization/graph_data.json"):
    data = await export_graph_for_vis(graphiti)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Exported to {filename}")
    print(f"   Nodes: {data['statistics']['total_nodes']}")
    print(f"   Edges: {data['statistics']['total_edges']}")


async def test_export():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    await export_to_file(graphiti)


if __name__ == "__main__":
    asyncio.run(test_export())


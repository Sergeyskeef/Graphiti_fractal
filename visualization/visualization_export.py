import asyncio
import json
import logging

from core import get_graphiti_client

logger = logging.getLogger(__name__)

async def export_graph_for_vis(graphiti, limit: int = 500):
    """
    Export graph structure for D3.js/Cytoscape visualization using direct Cypher.
    
    Fetches:
    - Nodes: Entity, Episodic, Community
    - Edges: RELATES_TO, SAME_AS, MENTIONS, BELONGS_TO
    """
    driver = getattr(graphiti, "driver", None) or getattr(graphiti, "_driver", None)
    if not driver:
        logger.error("Graphiti driver not found for export")
        return {"nodes": [], "edges": [], "error": "No driver"}

    nodes_map = {}
    edges_list = []

    # 1. Fetch Nodes (Entities & Communities)
    # We limit to Entities and Communities to keep visualization clean, 
    # optionally Episodic if needed (but usually too many).
    query_nodes = """
    MATCH (n)
    WHERE (n:Entity OR n:Community) AND n.uuid IS NOT NULL
    RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.group_id as group_id, n.summary as summary
    LIMIT $limit
    """

    try:
        if hasattr(driver, 'execute_query'):
            res_nodes = await driver.execute_query(query_nodes, limit=limit)
            records_nodes = res_nodes.records
        else:
            async with driver.session() as session:
                res_nodes = await session.run(query_nodes, limit=limit)
                records_nodes = await res_nodes.list()

        for rec in records_nodes:
            uuid = rec['uuid']
            labels = rec['labels']
            node_type = "Entity"
            if "Community" in labels:
                node_type = "Community"
            elif "Episodic" in labels:
                node_type = "Episodic"
            elif "User" in labels:
                node_type = "User"
            
            nodes_map[uuid] = {
                "id": str(uuid),
                "label": rec['name'] or f"{node_type}:{uuid[:4]}",
                "title": rec['summary'] or "",
                "type": node_type,
                "group": rec['group_id'] or "default",
                "size": 30 if node_type == "Community" else 20
            }

    except Exception as e:
        logger.error(f"Error exporting nodes: {e}")

    # 2. Fetch Edges
    # We only fetch edges where both source and target are in our fetched nodes map
    query_edges = """
    MATCH (n)-[r]->(m)
    WHERE n.uuid IS NOT NULL AND m.uuid IS NOT NULL
    RETURN n.uuid as source, m.uuid as target, type(r) as type, r.fact as fact
    LIMIT $limit
    """

    try:
        if hasattr(driver, 'execute_query'):
            res_edges = await driver.execute_query(query_edges, limit=limit * 2)
            records_edges = res_edges.records
        else:
            async with driver.session() as session:
                res_edges = await session.run(query_edges, limit=limit * 2)
                records_edges = await res_edges.list()

        for rec in records_edges:
            src = rec['source']
            tgt = rec['target']
            
            # Filter edges to only those connecting nodes we have
            if src in nodes_map and tgt in nodes_map:
                edges_list.append({
                    "from": str(src),
                    "to": str(tgt),
                    "label": rec['type'],
                    "title": rec['fact'] or "",
                    "arrows": "to"
                })

    except Exception as e:
        logger.error(f"Error exporting edges: {e}")

    nodes_data = list(nodes_map.values())

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

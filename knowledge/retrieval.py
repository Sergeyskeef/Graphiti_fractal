from __future__ import annotations

from core.settings import settings


async def search_knowledge(
    graphiti,
    query: str,
    *,
    limit: int = 10,
    group_id: str | None = None,
) -> list[dict]:
    """
    Структурный поиск знаний без LLM/embeddings.
    Основа — fulltext индексы Graphiti:
      - node_name_and_summary (Entity)
      - episode_content (Episodic)
    """
    q = (query or "").strip()
    if not q:
        return []
    driver = graphiti.driver
    res = await driver.execute_query(
        """
        CALL {
          CALL db.index.fulltext.queryNodes('node_name_and_summary', $q) YIELD node, score
          RETURN node, score, 'Entity' AS kind
          UNION
          CALL db.index.fulltext.queryNodes('episode_content', $q) YIELD node, score
          RETURN node, score, 'Episodic' AS kind
        }
        WITH node, score, kind
        WHERE coalesce(node.deleted,false) = false
          AND (node.group_id IS NULL OR node.group_id <> $egid)
          AND ($gid IS NULL OR node.group_id = $gid)
          AND (
            kind <> 'Episodic'
            OR NOT (coalesce(node.source_description,'') IN ['chat_user','chat_bot'])
          )
          AND (
            kind <> 'Entity'
            OR toLower(coalesce(node.name,'')) <> 'unknown'
          )
        RETURN kind, node.uuid AS uuid, node.name AS name, node.summary AS summary, node.content AS content, score
        ORDER BY score DESC
        LIMIT $limit
        """,
        q=q,
        egid=settings.EXPERIENCE_GROUP_ID,
        gid=group_id,
        limit=max(1, min(limit, 50)),
    )
    items = []
    for rec in res.records:
        kind = rec["kind"]
        name = rec.get("name")
        summary = rec.get("summary")
        content = rec.get("content")
        text = summary or content or name
        if not text:
            continue
        t = str(text).strip()
        if not t:
            continue
        if len(t) > 500:
            t = t[:500].strip() + "..."
        items.append(
            {
                "kind": kind,
                "uuid": rec.get("uuid"),
                "score": rec.get("score"),
                "text": t,
            }
        )
    return items



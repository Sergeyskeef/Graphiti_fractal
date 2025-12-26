from __future__ import annotations

from core.settings import settings


async def get_success_patterns(graphiti, *, task_type: str | None, context_hash: str | None, limit: int = 5):
    """
    Базовый retrieval по опыту: возвращаем последние успешные TaskRun в похожем контексте.
    """
    driver = graphiti.driver
    res = await driver.execute_query(
        """
        MATCH (tr:TaskRun)
        WHERE tr.group_id = $gid
          AND tr.status = 'success'
          AND ($task_type IS NULL OR tr.task_type = $task_type)
          AND ($ctx IS NULL OR tr.context_hash = $ctx)
        OPTIONAL MATCH (tr)-[:HAS_TOOLCALL]->(tc:ToolCall)
        WITH tr, collect(DISTINCT tc.tool)[0..10] AS tools
        RETURN tr.uuid AS run_id,
               tr.task_type AS task_type,
               tr.goal AS goal,
               tr.repo AS repo,
               tr.project AS project,
               tr.context_hash AS context_hash,
               tr.ended_at AS ended_at,
               tr.duration_ms AS duration_ms,
               tr.quality_score AS quality_score,
               tr.tool_chain AS tool_chain,
               tools AS tools
        ORDER BY tr.ended_at DESC
        LIMIT $limit
        """,
        gid=settings.EXPERIENCE_GROUP_ID,
        task_type=task_type,
        ctx=context_hash,
        limit=max(1, min(limit, 50)),
    )
    return [dict(rec) for rec in res.records]


async def get_antipatterns(graphiti, *, task_type: str | None, context_hash: str | None, limit: int = 5):
    """
    Антипримеры: группируем по error_type и tool_chain_hash в похожем контексте.
    """
    driver = graphiti.driver
    res = await driver.execute_query(
        """
        MATCH (tr:TaskRun)
        WHERE tr.group_id = $gid
          AND tr.status IN ['failure','timeout','aborted']
          AND ($task_type IS NULL OR tr.task_type = $task_type)
          AND ($ctx IS NULL OR tr.context_hash = $ctx)
        WITH tr.error_type AS error_type,
             tr.tool_chain_hash AS chain_hash,
             count(*) AS c,
             collect(DISTINCT tr.tool_chain)[0] AS example_chain,
             max(tr.ended_at) AS last_seen
        RETURN error_type, chain_hash, c, example_chain, last_seen
        ORDER BY c DESC, last_seen DESC
        LIMIT $limit
        """,
        gid=settings.EXPERIENCE_GROUP_ID,
        task_type=task_type,
        ctx=context_hash,
        limit=max(1, min(limit, 50)),
    )
    return [dict(rec) for rec in res.records]



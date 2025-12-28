"""
Memory Operations Layer

Provides high-level operations for working with memory (episodic and semantic).
Uses existing Graphiti functionality without modifying its behavior.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from time import perf_counter

from .graphiti_client import get_graphiti_client, get_write_semaphore
from .datetime_utils import normalize_dt, dt_to_iso, calculate_recency_days
from .text_utils import is_correction_text, fingerprint
from .types import SearchResult, ContextResult, EpisodeDict, EntityDict, EdgeDict, CommunityDict
from .config import get_config
from .rate_limit_retry import with_rate_limit_retry
from .authorship import attach_author

from knowledge.ingest import resolve_group_id, _infer_memory_type # Reuse helpers
from queries.context_builder import build_agent_context
from experience.writer import ingest_experience

# Graphiti search imports
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator

logger = logging.getLogger(__name__)

# Feature Flags
ENABLE_SAME_AS_2HOP = False  # Optional 2-hop expansion through SAME_AS


# Global storage for recent memories by user (limited to prevent memory leaks)
_recent_memories = {}  # user_id -> deque


def clear_recent_memories(user_id: str) -> int:
    """
    Clear recent memory cache for a specific user.
    Returns number of cleared items.
    """
    if user_id in _recent_memories:
        count = len(_recent_memories[user_id])
        del _recent_memories[user_id]
        return count
    return 0


class MemoryOps:
    """
    High-level memory operations using existing Graphiti infrastructure.

    Provides unified interface for:
    - Storing text memories with automatic classification
    - Searching across episodic and semantic memory
    - Building context for LLM queries
    """

    def __init__(self, graphiti, user_id: str):
        """
        Initialize MemoryOps with Graphiti instance.

        Args:
            graphiti: Graphiti client instance
            user_id: User identifier for personal memory isolation
        """
        self.graphiti = graphiti
        self.user_id = user_id

    async def ingest_pipeline(
        self,
        text: str,
        *,
        source_description: str = "ingest_pipeline",
        memory_type: str = "knowledge",
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Atomic ingestion pipeline ensuring idempotency and reliability.
        
        1. Calculates fingerprint (idempotency check).
        2. Uses single add_episode call (temporal validity).
        3. Attaches authorship and metadata atomically (or best effort with retry).
        """
        start_time = perf_counter()
        
        if not text or not text.strip():
            return {"status": "error", "reason": "empty_text"}

        # 1. Fingerprint & Idempotency
        fp = fingerprint(text)
        driver = getattr(self.graphiti, "driver", None) or getattr(self.graphiti, "_driver", None)
        
        if driver:
            # Check if exactly this content already exists
            res = await driver.execute_query(
                "MATCH (e:Episodic) WHERE e.fingerprint = $fp RETURN e.uuid as uuid LIMIT 1", 
                fp=fp
            )
            if res.records:
                existing_uuid = res.records[0]["uuid"]
                logger.info(f"Ingest skipped (duplicate): {fp} -> {existing_uuid}")
                return {"status": "skipped", "reason": "duplicate", "uuid": existing_uuid}

        # 2. Resolve Group ID
        # If group_id is not provided, resolve from memory_type
        target_group_id = group_id or resolve_group_id(memory_type)

        # 3. Add Episode with Retry
        async def _add_op():
            return await self.graphiti.add_episode(
                name=source_description[:100] or "pipeline_ingest",
                episode_body=text,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
                group_id=target_group_id
            )

        try:
            # We use the global write semaphore implicit in call chains or if needed here
            # But add_episode in graphiti client usually handles connection.
            # We wrap in retry for reliability.
            result = await with_rate_limit_retry(_add_op, op_name="ingest_pipeline")
        except Exception as e:
            logger.error(f"Ingest pipeline failed during add_episode: {e}")
            raise

        # 4. Extract UUID
        episode_uuid = None
        if hasattr(result, 'uuid'):
            episode_uuid = result.uuid
        elif isinstance(result, dict):
            episode_uuid = result.get('uuid')
        elif hasattr(result, 'episode') and hasattr(result.episode, 'uuid'):
             episode_uuid = result.episode.uuid
        
        if not episode_uuid:
            logger.error(f"Ingest pipeline: No UUID returned. Result: {result}")
            return {"status": "error", "reason": "no_uuid_returned"}

        # 5. Post-processing (Fingerprint, Author)
        if driver:
            try:
                # Set fingerprint for future idempotency
                await driver.execute_query(
                    "MATCH (e:Episodic {uuid: $uuid}) SET e.fingerprint = $fp",
                    uuid=episode_uuid, fp=fp
                )
                
                # Attach Author
                if self.user_id:
                    await attach_author(episode_uuid, self.user_id)
                    
            except Exception as e:
                logger.warning(f"Ingest pipeline post-processing partial fail: {e}")
                # We don't fail the whole operation if just metadata update fails, 
                # but we log it. The episode is already safely in DB.

        elapsed = perf_counter() - start_time
        logger.info(f"Ingest pipeline success: {episode_uuid} ({elapsed:.3f}s)")
        
        return {
            "status": "success",
            "uuid": episode_uuid,
            "group_id": target_group_id,
            "elapsed": elapsed
        }

    async def remember_text(
        self,
        text: str,
        *,
        memory_type: Optional[str] = None,
        source_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store text in memory with automatic classification.

        Args:
            text: Text to remember
            memory_type: Optional memory type (personal/project/knowledge/experience)
            source_description: Optional source description

        Returns:
            Dict with operation result
        """
        # Store in global recent memories for immediate access
        from collections import deque
        if self.user_id not in _recent_memories:
            _recent_memories[self.user_id] = deque(maxlen=20)

        _recent_memories[self.user_id].append({
            "text": text,
            "memory_type": memory_type,
            "source_description": source_description,
            "timestamp": datetime.now().isoformat()
        })
        
        # Inference if needed
        if not memory_type:
            memory_type = _infer_memory_type(text, source_description or "")

        # Use atomic pipeline
        return await self.ingest_pipeline(
            text,
            source_description=source_description or "memory_ops",
            memory_type=memory_type
        )

    async def remember_experience(
        self,
        experience_data: Dict[str, Any],
        source_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store experience using existing experience layer.

        Args:
            experience_data: Experience data dict
            source_description: Optional source description

        Returns:
            Dict with operation result
        """
        from experience.models import ExperienceIngestRequest

        # Convert dict to ExperienceIngestRequest if needed
        if isinstance(experience_data, dict):
            request = ExperienceIngestRequest(**experience_data)
        else:
            request = experience_data

        return await ingest_experience(self.graphiti, request)

    async def search_memory(
        self,
        query: str,
        *,
        scopes: Optional[List[str]] = None,
        limit: int = 10,
        include_episodes: bool = True,
        include_entities: bool = True,
        as_of: Optional[datetime] = None
    ) -> SearchResult:
        """
        Search across episodic and semantic memory using Graphiti search_().

        Args:
            query: Search query
            scopes: Optional list of memory scopes (personal/project/knowledge/experience)
            limit: Maximum results per type
            include_episodes: Whether to search episodes
            include_entities: Whether to search entities
            as_of: Optional datetime for point-in-time queries (defaults to current time)

        Returns:
            Structured search results from Graphiti
        """
        episodes = []
        entities = []
        edges = []
        communities = []

        logger.debug(
            f"Memory search started | query='{query[:50]}' user_id={self.user_id} scopes={scopes} limit={limit}",
            extra={
                "query": query,
                "user_id": self.user_id,
                "scopes": scopes,
                "limit": limit
            }
        )

        try:
            # Use Graphiti's advanced search with COMBINED_HYBRID_SEARCH_RRF
            # No lock needed for read operations
            # IMPORTANT: graphiti_core (Neo4j) fulltext query builder has a precedence bug for multi group_ids:
            # it generates `group_id:"a" OR group_id:"b" AND (query)` without parentheses. On Neo4j fulltext
            # this can yield 0 episodic hits, causing major recall loss.
            #
            # Workaround: for multi-scope searches, run without group_ids and filter results post-hoc by group_id.
            group_ids = scopes if (scopes and len(scopes) <= 1) else None

            # Create temporal filters: current-by-default or point-in-time
            search_filter = None
            if as_of is None:
                # Current facts only (invalid_at IS NULL)
                search_filter = SearchFilters(
                    invalid_at=[[DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)]]
                )
            else:
                # Point-in-time: facts valid at the specified time
                search_filter = SearchFilters(
                    valid_at=[[DateFilter(date=as_of, comparison_operator=ComparisonOperator.less_than_equal)]],
                    invalid_at=[
                        [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
                        [DateFilter(date=as_of, comparison_operator=ComparisonOperator.greater_than)]
                    ]
                )

            search_results = await self.graphiti.search_(
                query=query,
                config=COMBINED_HYBRID_SEARCH_RRF,
                group_ids=group_ids,
                search_filter=search_filter,
            )

            if scopes and len(scopes) > 1:
                allowed = set(scopes)

                def _filter_with_scores(items, scores):
                    kept_items = []
                    kept_scores = []
                    for i, item in enumerate(items):
                        gid = getattr(item, "group_id", None)
                        if gid in allowed:
                            kept_items.append(item)
                            if i < len(scores):
                                kept_scores.append(scores[i])
                    return kept_items, kept_scores

                eps, eps_scores = _filter_with_scores(
                    getattr(search_results, "episodes", []) or [],
                    getattr(search_results, "episode_reranker_scores", []) or [],
                )
                nodes, node_scores = _filter_with_scores(
                    getattr(search_results, "nodes", []) or [],
                    getattr(search_results, "node_reranker_scores", []) or [],
                )
                edges_raw, edge_scores = _filter_with_scores(
                    getattr(search_results, "edges", []) or [],
                    getattr(search_results, "edge_reranker_scores", []) or [],
                )
                comms, comm_scores = _filter_with_scores(
                    getattr(search_results, "communities", []) or [],
                    getattr(search_results, "community_reranker_scores", []) or [],
                )

                search_results.episodes = eps
                search_results.episode_reranker_scores = eps_scores
                search_results.nodes = nodes
                search_results.node_reranker_scores = node_scores
                search_results.edges = edges_raw
                search_results.edge_reranker_scores = edge_scores
                search_results.communities = comms
                search_results.community_reranker_scores = comm_scores

            # Process episodes
            for i, episode_node in enumerate(search_results.episodes):
                if include_episodes and len(episodes) < limit:
                    content = getattr(episode_node, 'content', '') or getattr(episode_node, 'summary', '')
                    if content and len(content.strip()) > 20:
                        # Keep more content so context builder can extract relevant snippets (not only the header).
                        content_short = content[:8000]
                        is_correction = is_correction_text(content_short)
                        
                        # Get episode_kind from metadata
                        episode_kind = getattr(episode_node, 'episode_kind', '')
                        source_description = getattr(episode_node, 'source_description', '')

                        # Use reranker score if available, otherwise default
                        score = (search_results.episode_reranker_scores[i]
                                if i < len(search_results.episode_reranker_scores)
                                else 0.6)

                        # Chat retrieval policy: apply score adjustments
                        # Reduce score for regular chat_turn episodes (they shouldn't dominate)
                        if episode_kind == 'chat_turn' or source_description == "chat":
                            score *= 0.3
                        
                        # Boost score for chat_summary episodes (they're valuable summaries)
                        if episode_kind == 'chat_summary':
                            score *= 1.3
                        
                        # Add correction bonus (applied after other adjustments)
                        if is_correction:
                            score += 2.0

                        episodes.append({
                            "uuid": getattr(episode_node, 'uuid', f'episode-{i}'),
                            "content": content_short,
                            "name": getattr(episode_node, 'name', 'Episode'),
                            "score": score,
                            "type": "episode",
                            "group_id": getattr(episode_node, 'group_id', ''),
                            "is_correction": is_correction,
                            "episode_kind": episode_kind,
                            "source_description": source_description,
                            "created_at": normalize_dt(getattr(episode_node, 'created_at', None))
                        })

            # Process entities (nodes)
            for i, entity_node in enumerate(search_results.nodes):
                if include_entities and len(entities) < limit:
                    score = (search_results.node_reranker_scores[i]
                            if i < len(search_results.node_reranker_scores)
                            else 0.7)

                    entities.append({
                        "uuid": getattr(entity_node, 'uuid', f'entity-{i}'),
                        "name": getattr(entity_node, 'name', ''),
                        "summary": getattr(entity_node, 'summary', ''),
                        "score": score,
                        "type": "entity",
                        "group_id": getattr(entity_node, 'group_id', '')
                    })

            # Process edges (facts/relationships)
            for i, entity_edge in enumerate(search_results.edges):
                score = (search_results.edge_reranker_scores[i]
                        if i < len(search_results.edge_reranker_scores)
                        else 0.5)

                # Extract structured fields if available
                subject = getattr(entity_edge, 'subject', None) or getattr(entity_edge, 'source_name', None)
                object_val = getattr(entity_edge, 'object', None) or getattr(entity_edge, 'target_name', None)
                relationship_type = getattr(entity_edge, 'relationship_type', None) or getattr(entity_edge, 'type', None) or getattr(entity_edge, 'rel_type', None)
                fact = getattr(entity_edge, 'fact', '')
                edge_name = getattr(entity_edge, 'name', None)

                edges.append({
                    "uuid": getattr(entity_edge, 'uuid', f'edge-{i}'),
                    "fact": fact,
                    "subject": subject,
                    "object": object_val,
                    "relationship_type": relationship_type,
                    "name": edge_name,
                    "score": score,
                    "type": "edge",
                    "group_id": getattr(entity_edge, 'group_id', '')
                })

            # Process communities
            for i, community_node in enumerate(search_results.communities):
                score = (search_results.community_reranker_scores[i]
                        if i < len(search_results.community_reranker_scores)
                        else 0.4)

                communities.append({
                    "uuid": getattr(community_node, 'uuid', f'community-{i}'),
                    "name": getattr(community_node, 'name', ''),
                    "summary": getattr(community_node, 'summary', ''),
                    "score": score,
                    "type": "community",
                    "group_id": getattr(community_node, 'group_id', '')
                })

            logger.debug("Memory search completed", extra={
                "query": query,
                "user_id": self.user_id,
                "episodes_found": len(episodes),
                "entities_found": len(entities),
                "edges_found": len(edges),
                "communities_found": len(communities)
            })

            # --- CROSS-LAYER EXPANSION (1-HOP & 2-HOP) ---
            # If we found entities, look for connected entities in other layers via SAME_AS
            if entities:
                found_uuids = [e["uuid"] for e in entities]
                allowed_scopes = list(scopes) if scopes else None
                
                # Get driver from graphiti instance
                driver = getattr(self.graphiti, 'driver', None) or getattr(self.graphiti, '_driver', None)
                
                if driver:
                    # 1-HOP EXPANSION with strict limits
                    # Max 20 neighbors, Max 50 edges total
                    expansion_query = """
                    MATCH (e:Entity)
                    WHERE e.uuid IN $uuids
                    MATCH (e)-[:SAME_AS]-(neighbor:Entity)
                    WHERE neighbor.group_id <> e.group_id
                      AND ($allowed_scopes IS NULL OR neighbor.group_id IN $allowed_scopes)
                    
                    WITH collect(DISTINCT neighbor) as all_neighbors
                    WITH all_neighbors[0..20] as limited_neighbors  // Limit to 20 neighbors
                    
                    UNWIND limited_neighbors as neighbor
                    OPTIONAL MATCH (neighbor)-[r:RELATES_TO]->(target:Entity)
                    
                    WITH neighbor, r, target
                    LIMIT 50  // Global limit on edges returned from expansion
                    
                    RETURN 
                        neighbor.uuid as neighbor_uuid,
                        neighbor.name as neighbor_name,
                        neighbor.summary as neighbor_summary,
                        neighbor.group_id as neighbor_group_id,
                        elementId(r) as edge_id,
                        type(r) as rel_type,
                        target.name as target_name,
                        r.fact as fact
                    """
                    
                    try:
                        exp_records = []
                        if hasattr(driver, 'execute_query'):
                            res = await driver.execute_query(expansion_query, uuids=found_uuids, allowed_scopes=allowed_scopes)
                            exp_records = res.records
                        else:
                            async with driver.session() as session:
                                res = await session.run(expansion_query, uuids=found_uuids, allowed_scopes=allowed_scopes)
                                exp_records = await res.list()
                                
                        existing_entity_uuids = set(found_uuids)
                        existing_edge_keys = set() # crude dedup for edges
                        neighbor_uuids = [] # Track for 2-hop

                        for rec in exp_records:
                            n_uuid = rec['neighbor_uuid']
                            neighbor_uuids.append(n_uuid)
                            
                            # Add neighbor entity if new
                            if n_uuid not in existing_entity_uuids:
                                entities.append({
                                    "uuid": n_uuid,
                                    "name": rec['neighbor_name'],
                                    "summary": rec['neighbor_summary'] or "",
                                    "score": 0.5, # Lower score for indirect
                                    "type": "entity",
                                    "group_id": rec['neighbor_group_id'] or "",
                                    "is_expanded": True,
                                    "hop": 1
                                })
                                existing_entity_uuids.add(n_uuid)
                            
                            # Add edge if exists
                            if rec['edge_id']:
                                edge_key = f"{n_uuid}-{rec['rel_type']}-{rec['target_name']}"
                                if edge_key not in existing_edge_keys:
                                    edges.append({
                                        "uuid": f"exp-{rec['edge_id']}",
                                        "fact": rec['fact'] or "",
                                        "subject": rec['neighbor_name'],
                                        "object": rec['target_name'],
                                        "relationship_type": rec['rel_type'],
                                        "name": None,
                                        "score": 0.4,
                                        "type": "edge",
                                        "group_id": rec['neighbor_group_id'] or "",
                                        "is_expanded": True,
                                        "hop": 1
                                    })
                                    existing_edge_keys.add(edge_key)
                        
                        # 2-HOP EXPANSION (Optional)
                        if ENABLE_SAME_AS_2HOP and neighbor_uuids:
                            distinct_neighbors = list(set(neighbor_uuids))
                            
                            query_2hop = """
                            MATCH (n:Entity)
                            WHERE n.uuid IN $uuids
                            MATCH (n)-[:SAME_AS]-(neighbor_2:Entity)
                            WHERE neighbor_2.group_id <> n.group_id
                              AND ($allowed_scopes IS NULL OR neighbor_2.group_id IN $allowed_scopes)
                            
                            WITH collect(DISTINCT neighbor_2) as neighbors_2
                            WITH neighbors_2[0..10] as limited_neighbors_2 // Stricter limit for 2-hop
                            
                            UNWIND limited_neighbors_2 as n2
                            RETURN 
                                n2.uuid as uuid,
                                n2.name as name, 
                                n2.summary as summary, 
                                n2.group_id as group_id
                            """
                            
                            recs_2 = []
                            if hasattr(driver, 'execute_query'):
                                r2 = await driver.execute_query(query_2hop, uuids=distinct_neighbors, allowed_scopes=allowed_scopes)
                                recs_2 = r2.records
                            else:
                                async with driver.session() as session:
                                    r2 = await session.run(query_2hop, uuids=distinct_neighbors, allowed_scopes=allowed_scopes)
                                    recs_2 = await r2.list()
                            
                            for r2 in recs_2:
                                if r2['uuid'] not in existing_entity_uuids:
                                    entities.append({
                                        "uuid": r2['uuid'],
                                        "name": r2['name'],
                                        "summary": r2['summary'] or "",
                                        "score": 0.3, # Even lower score
                                        "type": "entity",
                                        "group_id": r2['group_id'] or "",
                                        "is_expanded": True,
                                        "hop": 2
                                    })
                                    existing_entity_uuids.add(r2['uuid'])
                                    
                    except Exception as ex:
                        logger.warning(f"Cross-layer expansion failed: {ex}")

        except Exception as e:
            logger.exception("Graphiti search error", extra={
                "query": query,
                "user_id": self.user_id,
                "scopes": scopes
            })

        return SearchResult(
            episodes=episodes,
            entities=entities,
            edges=edges,
            communities=communities,
            total_episodes=len(episodes),
            total_entities=len(entities),
            total_edges=len(edges),
            total_communities=len(communities)
        )

    async def build_context_for_query(
        self,
        query: str,
        *,
        scopes: Optional[List[str]] = None,
        max_tokens: int = 4000,
        include_episodes: bool = True,
        include_entities: bool = True
    ) -> ContextResult:
        """
        Build formatted context for LLM queries.

        Args:
            query: Query to build context for
            scopes: Memory scopes to include
            max_tokens: Maximum token limit (approximated by text length)
            include_episodes: Whether to include episodic memories
            include_entities: Whether to include semantic entities

        Returns:
            Formatted context with metadata
        """
        # Set default scopes if none provided
        if scopes is None:
            # Default: include all memory layers.
            scopes = ["personal", "knowledge", "project", "experience"]

        logger.debug(
            f"Building context for query | query='{query[:50]}' scopes={scopes} max_tokens={max_tokens}",
            extra={
                "query": query,
                "scopes": scopes,
                "max_tokens": max_tokens
            }
        )

        # Determine temporal mode based on query content
        as_of = None
        query_lower = query.lower()

        # Simple heuristics for historical queries
        historical_keywords = [
            'раньше', 'до', 'когда', 'история', 'в прошлом', 'as of', 'на дату',
            'в 2024', 'в 2023', 'в 2022', 'в 2021', 'в 2020',
            'месяц назад', 'год назад', 'дней назад', 'недель назад'
        ]

        if any(keyword in query_lower for keyword in historical_keywords):
            # For historical queries, search across all time (no as_of filter)
            # This allows finding historical facts that may still be relevant
            as_of = None  # Keep current-by-default, but could be extended to parse specific dates
        # else: as_of remains None, so we get current facts only

        # Search memory using Graphiti
        search_result = await self.search_memory(
            query,
            scopes=scopes,
            limit=8,  # More results for better context
            include_episodes=include_episodes,
            include_entities=include_entities,
            as_of=as_of
        )

        # Retrieval explain logging
        request_id = f"ctx_{int(datetime.now(timezone.utc).timestamp())}"
        logger.debug("Retrieval explain", extra={
            "request_id": request_id,
            "query": query,
            "scopes": scopes,
            "entities_found": search_result.total_entities,
            "edges_found": search_result.total_edges,
            "episodes_found": search_result.total_episodes,
            "communities_found": search_result.total_communities,
            "top_entity": sorted(search_result.entities, key=lambda e: e.get("score", 0), reverse=True)[0].get("name", "") if search_result.entities else None,
            "top_entity_score": sorted(search_result.entities, key=lambda e: e.get("score", 0), reverse=True)[0].get("score", 0) if search_result.entities else 0,
            "top_episode_correction": sorted(search_result.episodes, key=lambda e: e.get("score", 0), reverse=True)[0].get("is_correction", False) if search_result.episodes else False,
            "top_episode_score": sorted(search_result.episodes, key=lambda e: e.get("score", 0), reverse=True)[0].get("score", 0) if search_result.episodes else 0
        })

        # Build formatted context
        context_parts = []
        sources = {"episodes": 0, "entities": 0, "edges": 0, "communities": 0}

        def _best_snippet(text: str, q: str, *, max_len: int = 520, window: int = 240) -> str:
            """
            Pick a snippet from potentially long episode text, biased towards query term matches.
            Falls back to head slice.
            """
            t = (text or "").strip()
            if not t:
                return ""
            q = (q or "").strip()
            if not q:
                return t[:max_len]

            import re
            tokens = [w for w in re.findall(r"[\\wА-Яа-яЁё]{3,}", q.lower()) if w]
            hay = t.lower()
            hit = None
            for tok in tokens[:8]:
                idx = hay.find(tok)
                if idx != -1:
                    hit = idx
                    break
            if hit is None:
                return t[:max_len]

            start = max(0, hit - window)
            end = min(len(t), start + max_len)
            snippet = t[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(t):
                snippet = snippet + "…"
            return snippet

        # Build a more coherent context for LLM
        # Start with direct episodes (most readable)
        all_episodes = search_result.episodes
        if all_episodes:
            context_parts.append("## Информация из памяти:")
            # Sort all episodes by score
            sorted_episodes = sorted(all_episodes, key=lambda e: e.get("score", 0), reverse=True)

            episodes_added = 0
            # Scan deeper than the final cap, because many top hits can be chat logs that we later filter out.
            for episode in sorted_episodes[:30]:
                content = episode.get("content", "")
                episode_kind = episode.get("episode_kind", "")

                # Include chat_summary episodes (high priority)
                if episode_kind == 'chat_summary':
                    context_parts.append(f"Предыдущие обсуждения: {content}")
                    episodes_added += 1
                    if episodes_added >= 3:
                        break
                    continue

                # Skip regular chat_turn episodes unless they contain corrections
                if episode_kind == 'chat_turn':
                    is_correction = episode.get("is_correction", False)
                    if not is_correction:
                        continue  # Skip regular chat turns
                    # Include chat corrections
                    content = content[:400]
                    context_parts.append(f"- ОБНОВЛЕНИЕ В РАЗГОВОРЕ: {content}")
                    episodes_added += 1
                    if episodes_added >= 3:
                        break
                    continue

                # Skip chat conversation logs - they don't provide useful context
                if (content.startswith("User: ") or
                    content.startswith("Assistant: ") or
                    "К сожалению, в предоставленном контексте нет информации" in content or
                    "Если у вас есть конкретные вопросы" in content):
                    continue

                content = _best_snippet(content, query, max_len=520, window=240)
                is_correction = episode.get("is_correction", False)

                if is_correction:
                    context_parts.append(f"- ОБНОВЛЕНИЕ: {content}")
                else:
                    context_parts.append(f"- {content}")

                episodes_added += 1
                if episodes_added >= 3:  # Limit to 3 useful episodes
                    break

            sources["episodes"] = episodes_added
            logger.debug("Context build episodes", extra={
                "episodes_added": episodes_added,
                "episodes_considered": min(30, len(sorted_episodes)),
                "episodes_filtered": min(30, len(sorted_episodes)) - episodes_added,
                "query": query
            })

        # Add key entities when episodic context is sparse (helps recover facts from entity summaries).
        if search_result.entities and sources["episodes"] == 0:
            context_parts.append("\n## Ключевые сущности:")
            sorted_entities = sorted(search_result.entities, key=lambda e: e.get("score", 0), reverse=True)
            added = 0
            for ent in sorted_entities[:5]:
                name = (ent.get("name") or "").strip()
                summary = (ent.get("summary") or "").strip()
                if not name:
                    continue
                if summary:
                    context_parts.append(f"- {name}: {summary[:240]}")
                else:
                    context_parts.append(f"- {name}")
                added += 1
            sources["entities"] = added

        # Add edges (facts/relationships) - top 8
        if search_result.edges:
            context_parts.append("\n## Связи/факты:")
            # Sort by score
            sorted_edges = sorted(search_result.edges, key=lambda e: e.get("score", 0), reverse=True)

            edges_added_count = 0
            for edge in sorted_edges[:8]:  # Top 8 edges
                subject = edge.get("subject")
                object_val = edge.get("object")
                relationship_type = edge.get("relationship_type")
                fact = edge.get("fact", "")
                edge_name = edge.get("name")
                
                # Format edge: prefer structured format if available
                if subject and object_val and relationship_type:
                    # Structured format: "Сергей — развивает → проект Марк"
                    formatted = f"{subject} — {relationship_type} → {object_val}"
                    context_parts.append(f"- {formatted}")
                    edges_added_count += 1
                elif fact and len(fact.strip()) > 0:
                    # Fallback to fact text
                    fact_short = fact[:100]  # Keep it short
                    # Add edge name/type if available
                    if edge_name:
                        context_parts.append(f"- [{edge_name}] {fact_short}")
                    else:
                        context_parts.append(f"- {fact_short}")
                    edges_added_count += 1
                elif edge_name:
                    # Only name available
                    context_parts.append(f"- {edge_name}")
                    edges_added_count += 1
            
            sources["edges"] = edges_added_count
            logger.debug(
                f"Context build edges | added={edges_added_count} total={len(sorted_edges[:8])} query='{query[:30]}'",
                extra={
                    "edges_added": edges_added_count,
                    "edges_total": len(sorted_edges[:8]),
                    "query": query
                }
            )

        # Add communities (if available and relevant) - top 3
        # NOTE: Communities may have empty summaries. Monitor on tests - if 80% are empty/useless,
        # consider temporarily disabling or reducing their influence.
        if search_result.communities:
            context_parts.append("\n## Сообщества:")
            # Sort by score
            sorted_communities = sorted(search_result.communities, key=lambda c: c.get("score", 0), reverse=True)

            communities_added_count = 0
            for community in sorted_communities[:3]:  # Top 3 communities
                name = community.get("name", "")
                summary = community.get("summary", "")[:80]  # Very short
                if summary and len(summary.strip()) > 0:
                    context_parts.append(f"- {name}: {summary}")
                    communities_added_count += 1
                elif name:
                    # Only name available (no summary)
                    context_parts.append(f"- {name}")
                    communities_added_count += 1
            
            sources["communities"] = communities_added_count
            logger.debug(
                f"Context build communities | added={communities_added_count} total={len(sorted_communities[:3])} query='{query[:30]}'",
                extra={
                    "communities_added": communities_added_count,
                    "communities_total": len(sorted_communities[:3]),
                    "query": query
                }
            )

        # Combine and truncate if needed
        full_text = "\n".join(context_parts)
        token_estimate = len(full_text) // 4  # Rough approximation

        if token_estimate > max_tokens:
            # Truncate to fit token limit
            max_chars = max_tokens * 4
            full_text = full_text[:max_chars] + "\n[Контекст обрезан для соответствия лимиту токенов]"
            token_estimate = max_tokens

        logger.debug(
            f"Context build complete | query='{query[:30]}' tokens={token_estimate} sections={sources}",
            extra={
                "query": query,
                "final_tokens": token_estimate,
                "sections": sources
            }
        )

        return ContextResult(
            text=full_text,
            token_estimate=token_estimate,
            sources=sources
        )


# Temporary test function
async def test_memory_diagnostic():
    """Diagnostic test for memory operations."""
    from .graphiti_client import get_graphiti_client

    print("=== MEMORY DIAGNOSTIC TEST ===")

    try:
        graphiti_client = get_graphiti_client()
        graphiti = await graphiti_client.ensure_ready()

        memory = MemoryOps(graphiti, "sergey")

        # Test build_context_for_query
        print("\n--- Testing build_context_for_query ---")
        ctx = await memory.build_context_for_query("Кто такой Марк?", max_tokens=4000)
        print(f"Context result: {ctx}")

        # Test search_memory directly
        print("\n--- Testing search_memory ---")
        res = await memory.search_memory("Кто такой Марк?", scopes=None)
        print("EPISODES:", [ep["name"][:50] for ep in res.episodes[:3]])
        print("ENTITIES:", [e["name"] for e in res.entities[:10]])

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_memory_diagnostic())

import os
import asyncio
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from neo4j.exceptions import ClientError

from core.migrations import apply_migrations
from core.embeddings import get_embedding
from graphiti_core.embedder.client import EmbedderClient
import logging

logger = logging.getLogger(__name__)

class CustomEmbedder(EmbedderClient):
    """Custom embedder that uses our get_embedding function with caching."""
    async def create(self, input_data):
        if isinstance(input_data, str):
            result = await get_embedding(input_data)
            if not result:
                logger.warning(f"Empty embedding for string input '{input_data[:50]}...', returning zeros")
                return [0.0] * 1536
            return result
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            # Average embeddings for list of strings
            vecs = []
            for text in input_data:
                vec = await get_embedding(text)
                if vec:
                    vecs.append(vec)

            if not vecs:
                logger.warning(f"No valid embeddings for list input {len(input_data)} strings, returning zeros")
                return [0.0] * 1536

            # Element-wise average
            return [sum(vals) / len(vals) for vals in zip(*vecs)]
        else:
            logger.error(f"Unsupported input type for CustomEmbedder: {type(input_data)} (value: {input_data!r})")
            return [0.0] * 1536

    async def create_batch(self, input_data_list):
        results = []
        for text in input_data_list:
            result = await self.create(text)
            results.append(result)
        return results

load_dotenv()


class ExtendedGraphiti(Graphiti):
    """
    Расширенная версия Graphiti, которая добавляет логику склейки сущностей (Entity Gluing)
    после добавления эпизода.
    """
    
    STOP_WORDS = {
        "project", "system", "data", "memory", "graph", "ai", "model", 
        "user", "assistant", "chat", "summary", "context", "fact",
        "проект", "система", "данные", "память", "граф", "ии", "модель", 
        "пользователь", "ассистент", "чат", "саммари", "контекст", "факт",
        "unknown", "none", "null"
    }

    async def add_episode(self, *args, **kwargs):
        # Вызов оригинального метода
        result = await super().add_episode(*args, **kwargs)
        
        # Пост-процессинг: склейка сущностей
        try:
            episode_uuid = None
            if hasattr(result, 'uuid'):
                episode_uuid = result.uuid
            elif hasattr(result, 'episode') and hasattr(result.episode, 'uuid'):
                episode_uuid = result.episode.uuid
            elif isinstance(result, dict):
                episode_uuid = result.get('uuid')
            
            if episode_uuid:
                await self._link_cross_layer_entities(episode_uuid)
            else:
                logger.warning(f"Could not determine episode UUID from result type {type(result)}")
        except Exception as e:
            logger.error(f"Error in cross-layer linking: {e}", exc_info=True)
            
        return result

    def _normalize_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        
        # 1. Lowercase and trim
        norm = name.lower().strip()
        
        # 2. Cyrillic normalization
        norm = norm.replace('ё', 'е')
        
        # 3. Remove punctuation (keep alphanumeric and spaces)
        import re
        norm = re.sub(r'[^\w\s]', '', norm)
        
        # 4. Collapse whitespace
        norm = re.sub(r'\s+', ' ', norm).strip()
        
        # 5. Check length and stop words
        if len(norm) < 3:  # Too short (require at least 3 chars)
            return None
            
        if norm in self.STOP_WORDS:
            return None
            
        return norm

    async def _link_cross_layer_entities(self, episode_uuid: str):
        driver = getattr(self, 'driver', None) or getattr(self, '_driver', None)
        
        if not driver:
            logger.warning("Graphiti driver not found, skipping cross-layer linking")
            return

        # 1. Fetch entities from the episode
        fetch_query = """
        MATCH (ep:Episodic {uuid: $episode_uuid})-[:MENTIONS]->(e:Entity)
        WHERE e.name IS NOT NULL
        RETURN e.uuid as uuid, e.name as name, e.group_id as group_id
        """
        
        entities_to_update = []
        
        try:
            records = []
            if hasattr(driver, 'execute_query'):
                res = await driver.execute_query(fetch_query, episode_uuid=episode_uuid)
                records = res.records
            else:
                async with driver.session() as session:
                    res = await session.run(fetch_query, episode_uuid=episode_uuid)
                    records = await res.list() # safe iteration if async
            
            # 2. Normalize names in Python
            for record in records:
                uuid = record['uuid']
                name = record['name']
                norm = self._normalize_name(name)
                
                if norm:
                    entities_to_update.append({"uuid": uuid, "name_norm": norm})
            
            if not entities_to_update:
                logger.info(f"No valid entities to link for episode {episode_uuid}")
                return

            # 3. Batch update name_norm and Link
            link_query = """
            UNWIND $updates AS update
            MATCH (e:Entity {uuid: update.uuid})
            SET e.name_norm = update.name_norm
            
            WITH e
            MATCH (other:Entity {name_norm: e.name_norm})
            WHERE other.group_id <> e.group_id AND other.uuid <> e.uuid
            
            // Lock order to prevent deadlocks/dupes
            WITH e, other
            WHERE e.uuid < other.uuid
            
            MERGE (e)-[r:SAME_AS]->(other)
            RETURN count(r) as created_count, collect(other.name) as linked_names
            """
            
            if hasattr(driver, 'execute_query'):
                res = await driver.execute_query(link_query, updates=entities_to_update)
                records = res.records
            else:
                async with driver.session() as session:
                    res = await session.run(link_query, updates=entities_to_update)
                    records = await res.list()
            
            total_created = sum(r['created_count'] for r in records)
            linked_names = [name for r in records for name in r['linked_names']]
            
            logger.info(f"Cross-layer linking stats for episode {episode_uuid}", extra={
                "candidates_processed": len(entities_to_update),
                "bridges_created": total_created,
                "linked_entities": linked_names[:10]
            })
            
        except Exception as e:
            logger.error(f"Failed to execute cross-layer linking: {e}")


class GraphitiClient:
    """Обёртка над Graphiti с ленивой инициализацией и созданием индексов."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        # graphiti_core 0.24.x не принимает database/openai_api_key в __init__
        # Используем custom embedder с нашим кэшированием
        custom_embedder = CustomEmbedder()
        self._graphiti = ExtendedGraphiti(
            uri=uri,
            user=user,
            password=password,
            embedder=custom_embedder,
        )
        self._ready = False
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> Graphiti:
        """Создаёт индексы/констрейнты один раз перед использованием."""
        if self._ready:
            return self._graphiti

        async with self._lock:
            if not self._ready:
                try:
                    await self._graphiti.build_indices_and_constraints()
                except ClientError as exc:
                    # Игнорируем дублирующиеся индексы, чтобы не падать на уже существующей схеме
                    if "EquivalentSchemaRuleAlreadyExists" not in str(exc):
                        raise
                # Наши миграции поверх схемы Graphiti (идемпотентно)
                await apply_migrations(self._graphiti)
                self._ready = True

        return self._graphiti

    @property
    def raw(self) -> Graphiti:
        """Доступ к исходному клиенту Graphiti (при необходимости)."""
        return self._graphiti


# Write semaphore for Graphiti operations (limit concurrent writes)
WRITE_SEMAPHORE = asyncio.Semaphore(2)  # Максимум 2 одновременные записи

# Global singleton cache (for production use)
_graphiti_singleton: GraphitiClient | None = None


def reset_graphiti_client() -> None:
    """Reset the global Graphiti client singleton (useful for tests)."""
    global _graphiti_singleton
    _graphiti_singleton = None
    # Also clear lru_cache if it exists
    if hasattr(get_graphiti_client, 'cache_clear'):
        get_graphiti_client.cache_clear()


def get_graphiti_client(*, force_new: bool = False) -> GraphitiClient:
    """
    Get Graphiti client instance.
    
    Args:
        force_new: If True, create a new client instance instead of reusing singleton.
                   Useful for tests to avoid event loop conflicts.
    
    Returns:
        GraphitiClient instance
    """
    global _graphiti_singleton
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise RuntimeError("ENV vars NEO4J_URI/USER/PASSWORD are required")

    # If force_new or no singleton exists, create new client
    if force_new or _graphiti_singleton is None:
        _graphiti_singleton = GraphitiClient(
            uri=uri,
            user=user,
            password=password,
        )
    
    return _graphiti_singleton

def get_write_semaphore() -> asyncio.Semaphore:
    """Получить семафор для операций записи."""
    return WRITE_SEMAPHORE


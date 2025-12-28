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
        
        # Используем стандартный класс Graphiti без надстроек (Native Graphiti Way)
        self._graphiti = Graphiti(
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

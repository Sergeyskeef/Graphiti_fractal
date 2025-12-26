from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from time import perf_counter

from fastapi import HTTPException

from core.config import get_config
from core.text_utils import normalize_text, fingerprint, split_into_paragraphs
from core.embeddings import get_embedding
from core.graphiti_client import get_write_semaphore

logger = logging.getLogger(__name__)


async def episode_exists(graphiti, fp: str, content: str) -> bool:
    # 1. Exact match fallback
    driver = graphiti.driver
    res = await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.fingerprint = $fp OR e.content = $content
        RETURN e.uuid AS uuid
        LIMIT 1
        """,
        fp=fp,
        content=content,
    )
    if len(res.records) > 0:
        return True
    return False


async def find_similar_episode(graphiti, vector: list[float], threshold: float = 0.95) -> str | None:
    """
    Search for semantically similar episode.
    Returns UUID if found, None otherwise.
    """
    if not vector:
        return None
        
    driver = graphiti.driver
    try:
        # Check if index exists first to avoid error during init
        # But usually we just try query. If index missing, it might fail.
        # We rely on migration 002.
        res = await driver.execute_query(
            """
            CALL db.index.vector.queryNodes('fractal_episodic_vector', 1, $vec)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.uuid AS uuid, score
            """,
            vec=vector,
            threshold=threshold
        )
        if res.records:
            rec = res.records[0]
            # print(f"DEBUG: Found similar episode {rec['uuid']} score={rec['score']}")
            return rec["uuid"]
    except Exception as e:
        # Index might not exist yet or vector dimension mismatch
        pass
    return None


async def update_last_seen(graphiti, uuid: str, group_id: str):
    driver = graphiti.driver
    await driver.execute_query(
        """
        MATCH (e:Episodic {uuid: $uuid})
        SET e.last_seen_at = $ts, e.group_id = $gid
        """,
        uuid=uuid,
        ts=datetime.now(timezone.utc).isoformat(),
        gid=group_id
    )


async def set_fingerprint(graphiti, fp: str, content: str):
    driver = graphiti.driver
    await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.content = $content AND (e.fingerprint IS NULL)
        SET e.fingerprint = $fp
        """,
        content=content,
        fp=fp,
    )


async def set_embedding(graphiti, content: str, vector: list[float]):
    if not vector:
        return
    driver = graphiti.driver
    await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.content = $content
        SET e.embedding = $vec
        """,
        content=content,
        vec=vector,
    )


async def set_group_id(graphiti, content: str, group_id: str):
    driver = graphiti.driver
    await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.content = $content
        SET e.group_id = $gid
        """,
        content=content,
        gid=group_id,
    )


async def link_user(graphiti, fp: str, user_id: str):
    driver = graphiti.driver
    await driver.execute_query(
        """
        MERGE (u:User {user_id:$uid})
        WITH u
        MATCH (e:Episodic {fingerprint:$fp})
        MERGE (u)-[:AUTHORED]->(e)
        """,
        uid=user_id,
        fp=fp,
    )


def _infer_memory_type(text: str, source_description: str = "") -> str:
    """
    Автоматически определяет тип памяти на основе анализа текста и источника.

    Returns:
        "personal" | "project" | "experience" | "knowledge"
    """
    text_lower = text.lower()
    source_lower = source_description.lower()

    # Личные данные (о людях, отношениях, характеристиках)
    personal_keywords = [
        "я ", "мне ", "мой ", "моя ", "мои ", "меня ", "мной ",
        "человек", "личность", "характер", "отношения", "друзья",
        "семья", "родители", "дети", "любим", "интерес",
        "воспитание", "привычки", "здоровье", "эмоции"
    ]

    # Данные о проектах (технические, рабочие)
    project_keywords = [
        "проект", "задача", "разработка", "код", "программа",
        "алгоритм", "система", "архитектура", "дизайн",
        "тестирование", "деплой", "продакшн", "баги", "фичи",
        "коммит", "репозиторий", "билд", "конфиг", "документация"
    ]

    # Опыт работы (ошибки, успехи, паттерны)
    experience_keywords = [
        "ошибка", "проблема", "решение", "успех", "паттерн",
        "урок", "опыт", "практика", "метод", "подход",
        "техника", "стратегия", "результат", "итог"
    ]

    # Проверяем по источнику
    if "personal" in source_lower or "личн" in source_lower:
        logger.debug(f"Inferred 'personal' from source: {source_description}")
        return "personal"
    elif "project" in source_lower or "проект" in source_lower:
        logger.debug(f"Inferred 'project' from source: {source_description}")
        return "project"
    elif "experience" in source_lower or "опыт" in source_lower:
        logger.debug(f"Inferred 'experience' from source: {source_description}")
        return "experience"

    # Анализируем текст по ключевым словам
    personal_score = sum(1 for keyword in personal_keywords if keyword in text_lower)
    project_score = sum(1 for keyword in project_keywords if keyword in text_lower)
    experience_score = sum(1 for keyword in experience_keywords if keyword in text_lower)

    # Определяем тип по максимальному счету
    max_score = max(personal_score, project_score, experience_score)

    if max_score == 0:
        inferred = "knowledge"  # Общие знания по умолчанию
    elif personal_score == max_score:
        inferred = "personal"
    elif project_score == max_score:
        inferred = "project"
    else:
        inferred = "experience"

    logger.debug(
        f"Inferred '{inferred}' from text analysis "
        f"(scores: p={personal_score}, r={project_score}, e={experience_score})"
    )
    return inferred


def _get_group_id(memory_type: str) -> str:
    """Возвращает group_id по типу памяти."""
    config = get_config()
    if memory_type == "personal":
        return config.memory.personal_group_id
    elif memory_type == "project":
        return config.memory.project_group_id
    elif memory_type == "experience":
        return config.memory.experience_group_id
    else:
        return config.memory.knowledge_group_id


def resolve_group_id(memory_type: str) -> str:
    """Возвращает group_id по типу памяти (для API)."""
    return _get_group_id(memory_type)


async def remember_text(
    graphiti,
    text: str,
    *,
    source_description: str = "user_chat",
    user_id: str | None = None,
    memory_type: str | None = None,
) -> dict:
    txt = text.strip()
    if not txt:
        raise ValueError("text is empty")

    # Определяем тип памяти: явный имеет приоритет над автоматическим
    if memory_type is None:
        memory_type = _infer_memory_type(txt, source_description)
        routing_mode = "auto"
    else:
        routing_mode = "explicit"

    group_id = resolve_group_id(memory_type)

    # Логируем решение маршрутизации
    logger.info(f"[memory] Routed text (mode={routing_mode}) to group '{memory_type}' (group_id: {group_id})")

    # Используем единую функцию ingest
    return await ingest_text_document(
        graphiti,
        txt,
        source_description=source_description,
        user_id=user_id,
        group_id=group_id,
    )



async def ingest_text_document(
    graphiti,
    text: str,
    *,
    source_description: str = "uploaded_text",
    user_id: str | None = None,
    job_id: str | None = None,
    group_id: str | None = None,
) -> dict:
    """
    Единая точка входа для загрузки текста (и из /remember, и из /upload).
    Никакого локального чанкинга, один episode_body = весь текст.
    """
    # Import job functions from api.jobs (no circular import)
    from api.jobs import update_upload_job
    
    config = get_config()
    start = perf_counter()

    # 1) Базовый профиль/обновление job
    on_rate_limit_cb = None
    warnings = []
    if job_id:
        update_upload_job(job_id,
                          stage="ingest",
                          total_chunks=1,
                          processed_chunks=0)
        
        def _on_rate_limit(sleep_s: float, attempt: int):
            update_upload_job(
                job_id, 
                stage="rate_limited", 
                message=f"Rate limited. Retrying in {sleep_s:.1f}s (Attempt {attempt})",
                retry_in_seconds=sleep_s,
                attempt=attempt
            )
        on_rate_limit_cb = _on_rate_limit

    # 2) Вызов Graphiti одним эпизодом
    ref_time = datetime.now(timezone.utc)

    write_semaphore = get_write_semaphore()
    async with write_semaphore:
        from core.rate_limit_retry import with_rate_limit_retry
        from pydantic import ValidationError
        from core.safe_graphiti import filter_graphiti_results

        try:
            episode_result = await with_rate_limit_retry(
                lambda: graphiti.add_episode(
                    name=source_description or "Uploaded document",
                    episode_body=text,
                    source_description=source_description,
                    reference_time=ref_time,
                    group_id=group_id,  # Передаем group_id сразу в Graphiti
                ),
                op_name="add_episode:upload",
                on_rate_limit=on_rate_limit_cb
            )
            
            # Filter results for safety (especially for logging or further processing)
            safe_results = filter_graphiti_results(episode_result)
            
            # Handle warnings for job status
            warnings = []
            if safe_results["dropped_entities"] > 0 or safe_results["dropped_edges"] > 0:
                warn_msg = f"Dropped {safe_results['dropped_entities']} entities and {safe_results['dropped_edges']} edges due to validation errors"
                warnings.append(warn_msg)
                logger.warning(f"[INGEST] {warn_msg}", extra={
                    "job_id": job_id,
                    "dropped_entities": safe_results["dropped_entities"],
                    "dropped_edges": safe_results["dropped_edges"]
                })

            actual_episode = episode_result.episode if hasattr(episode_result, 'episode') else episode_result
        
        except ValidationError as ve:
            # If the library fails to validate its own output, we log it but try to continue 
            # by finding the episodic node we just (likely) created.
            logger.error(f"[INGEST] Validation error during Graphiti ingestion: {ve}")
            warnings = ["Graphiti returned malformed entities/edges, but the episode was likely created."]
            
            # Fallback: find the episode by source and reference_time (approximate)
            # or by content matching if we had a fingerprint (but we don't have it here yet)
            driver = graphiti.driver
            find_query = """
            MATCH (e:Episodic)
            WHERE e.content = $content AND e.source_description = $source
            RETURN e.uuid AS uuid
            ORDER BY e.created_at DESC
            LIMIT 1
            """
            find_res = await driver.execute_query(find_query, content=text, source=source_description or "Uploaded document")
            if find_res.records:
                # We found the episode! We can continue.
                from collections import namedtuple
                ActualEpisode = namedtuple('ActualEpisode', ['uuid'])
                actual_episode = ActualEpisode(uuid=find_res.records[0]['uuid'])
                logger.info(f"[INGEST] Recovered episode UUID after ValidationError: {actual_episode.uuid}")
            else:
                # Truly failed
                raise HTTPException(status_code=500, detail=f"Graphiti validation error and could not recover: {ve}")
        except Exception as e:
            logger.error(f"[INGEST] Unexpected error during Graphiti ingestion: {e}", exc_info=True)
            raise

        if actual_episode and actual_episode.uuid:
            try:
                # Truncate for embedding if too long
                max_embed_chars = config.app.max_embedding_chars
                embed_text = text
                if len(text) > max_embed_chars:
                    logger.warning(
                        f"Text too long ({len(text)} chars), "
                        f"truncating to {max_embed_chars} for embedding"
                    )
                    embed_text = text[:max_embed_chars]

                vec = await get_embedding(embed_text)
                if vec:
                    await set_embedding(graphiti, text, vec)
                    # Verify it stuck using UUID
                    verify_query = "MATCH (e:Episodic {uuid: $uuid}) SET e.embedding = $vec RETURN size(e.embedding) as sz"
                    verify_res = await graphiti.driver.execute_query(verify_query, uuid=actual_episode.uuid, vec=vec)
                    if verify_res.records and verify_res.records[0]['sz'] == 1536:
                        logger.info(f"Enforced embedding for episode {actual_episode.uuid}")
                    else:
                        logger.warning(f"Failed to verify embedding write for {actual_episode.uuid}")
                else:
                    logger.error(f"Failed to generate embedding for episode {actual_episode.uuid}")
            except Exception as e:
                logger.error(f"Error enforcing embedding for episode {actual_episode.uuid}: {e}")

    # 3) Линковка с User, если есть user_id (как уже было в проекте)
    if user_id and actual_episode and actual_episode.uuid:
        from core.authorship import attach_author
        await attach_author(actual_episode.uuid, user_id)

    # 4) Установка group_id для эпизода (fallback, если Graphiti не установил)
    if group_id:
        try:
            await set_group_id(graphiti, text, group_id)
        except Exception as e:
            logger.warning(f"Failed to set group_id via Cypher: {e}")

    # 5) Обновление job и профиля
    elapsed = perf_counter() - start
    if job_id:
        update_upload_job(job_id,
                          stage="done",
                          total_chunks=1,
                          processed_chunks=1,
                          profile={"total_time": elapsed})

    logger.info(
        f"Document ingested: source='{source_description}' len={len(text)} elapsed={elapsed:.3f}s"
    )

    return {
        "status": "ok",
        "added": 1,
        "chunks": 1,
        "elapsed": elapsed,
        "warnings": warnings if 'warnings' in locals() else []
    }


async def ingest_text_document_simple(
    graphiti,
    text: str,
    *,
    source_description: str = "uploaded_text",
    user_id: str | None = None,
) -> dict:
    """
    Резервная простая версия ingest без Graphiti.add_episode.
    Используется только для отладки при проблемах с Graphiti.
    """
    from datetime import datetime, timezone
    import uuid
    # Используем нашу новую функцию разбиения
    from core.text_utils import split_into_paragraphs
    parts = split_into_paragraphs(text, max_len=1800, overlap=0)
    logger.info(f"[INGEST-SIMPLE] Split text into {len(parts)} chunks using semantic splitting")

    added = 0
    ref_time = datetime.now(timezone.utc)
    uid = user_id or os.getenv("USER_ID", "sergey")

    driver = graphiti.driver

    for idx, part in enumerate(parts, start=1):
        fp = fingerprint(part)
        logger.info(f"[INGEST-SIMPLE] Processing chunk {idx}/{len(parts)}, len={len(part)}")

        # Проверяем существование
        exists_query = """
        MATCH (e:Episodic)
        WHERE e.fingerprint = $fp
        RETURN e.uuid AS uuid
        LIMIT 1
        """
        result = await driver.execute_query(exists_query, params={"fp": fp})
        if result.records:
            logger.info(f"[INGEST-SIMPLE] Chunk {idx} already exists, skipping")
            continue

        try:
            # Создаем episode напрямую
            episode_uuid = str(uuid.uuid4())
            create_query = """
            CREATE (e:Episodic {
                uuid: $uuid,
                name: $name,
                content: $content,
                source_description: $source_desc,
                created_at: $created_at,
                valid_at: $valid_at,
                fingerprint: $fp,
                group_id: $group_id
            })
            """
            params = {
                "uuid": episode_uuid,
                "name": f"Upload chunk {idx}",
                "content": part,
                "source_desc": source_description,
                "created_at": ref_time.isoformat(),
                "valid_at": ref_time.isoformat(),
                "fp": fp,
                "group_id": "knowledge"
            }
            await driver.execute_query(create_query, params=params)

            # Создаем или связываем пользователя
            user_query = """
            MERGE (u:User {user_id: $uid})
            WITH u
            MATCH (e:Episodic {uuid: $episode_uuid})
            MERGE (u)-[:AUTHORED]->(e)
            """
            await driver.execute_query(user_query, params={
                "uid": uid,
                "episode_uuid": episode_uuid
            })

            logger.info(f"[INGEST-SIMPLE] Episode added for chunk {idx}")
            added += 1

        except Exception as e:
            logger.error(f"[INGEST-SIMPLE] Failed to add episode for chunk {idx}: {type(e).__name__}: {e}")
            raise

    return {"status": "ok", "added": added, "chunks": len(parts)}


async def update_episode_metadata(graphiti, episode_uuid: str, metadata: dict):
    """
    Обновляет метаданные эпизода по UUID.

    Args:
        graphiti: Graphiti клиент
        episode_uuid: UUID эпизода
        metadata: Словарь полей для обновления
    """
    driver = graphiti.driver

    set_clauses = []
    params = {"uuid": episode_uuid}

    for key, value in metadata.items():
        set_clauses.append(f"e.{key} = ${key}")
        params[key] = value

    query = f"""
    MATCH (e:Episodic {{uuid: $uuid}})
    SET {', '.join(set_clauses)}
    RETURN e.uuid AS uuid
    """

    result = await driver.execute_query(query, **params)

    if result.records:
        logger.info(f"Updated episode metadata", extra={
            "episode_uuid": episode_uuid,
            "metadata_keys": list(metadata.keys())
        })
        return {"status": "updated", "episode_uuid": episode_uuid}
    else:
        logger.warning(f"Episode not found for metadata update", extra={
            "episode_uuid": episode_uuid
        })
        return {"status": "not_found", "episode_uuid": episode_uuid}


async def link_user_to_person_entity(graphiti, user_id: str, person_name: str = "Сергей"):
    """
    Создаёт связь между User и Entity (человек).

    Args:
        graphiti: Graphiti клиент
        user_id: ID пользователя (например, "sergey")
        person_name: Имя сущности человека (по умолчанию "Сергей")
    """
    driver = graphiti.driver

    query = """
    MATCH (u:User {user_id: $user_id})
    MATCH (e:Entity {name: $person_name})
    MERGE (u)-[:IS]->(e)
    RETURN u.user_id AS user_id, e.name AS entity_name
    """

    result = await driver.execute_query(query, user_id=user_id, person_name=person_name)

    if result.records:
        print(f"[USER_LINK] Created link: User '{user_id}' IS Entity '{person_name}'")
        return {"status": "linked", "user_id": user_id, "entity_name": person_name}
    else:
        print(f"[USER_LINK] Could not create link - User or Entity not found")
        return {"status": "not_found", "user_id": user_id, "entity_name": person_name}


# ingest_text_document_simple оставлен для отладки/резервного использования
# Основная функция использует Graphiti.add_episode для полноценной обработки


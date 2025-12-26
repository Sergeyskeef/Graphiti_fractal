#!/usr/bin/env python3
"""
Утилита дедупликации Entity узлов.
- Группирует Entity по нормализованному имени
- Оставляет один главный узел, остальные вливает в него
- Переносит все рёбра на главный узел
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from core.graphiti_client import get_graphiti_client
from core.text_utils import normalize_entity_name, is_meaningful_entity_name

logger = logging.getLogger(__name__)


async def fetch_entities(driver) -> List[Dict]:
    """
    Получает все Entity узлы с их именами и UUID.
    """
    res = await driver.execute_query(
        """
        MATCH (e:Entity)
        WHERE NOT e.deleted AND e.group_id IS NOT NULL
        RETURN e.uuid AS uuid, coalesce(e.name, '') AS name, e.group_id AS group_id
        """
    )
    entities = []
    for rec in res.records:
        name = rec["name"] or ""
        if not is_meaningful_entity_name(name):
            continue
        entities.append({
            "uuid": rec["uuid"],
            "name": name,
            "normalized_name": normalize_entity_name(name),
            "group_id": rec["group_id"]
        })
    return entities


async def fetch_entity_relationships(driver, entity_uuid: str) -> Dict[str, List[str]]:
    """
    Получает все входящие и исходящие связи сущности.
    Возвращает dict с ключами 'incoming' и 'outgoing'.
    """
    # Исходящие связи
    outgoing_res = await driver.execute_query(
        """
        MATCH (e:Entity {uuid: $uuid})-[r]->(target)
        RETURN type(r) AS rel_type, target.uuid AS target_uuid
        """,
        uuid=entity_uuid
    )

    # Входящие связи
    incoming_res = await driver.execute_query(
        """
        MATCH (source)-[r]->(e:Entity {uuid: $uuid})
        RETURN type(r) AS rel_type, source.uuid AS source_uuid
        """,
        uuid=entity_uuid
    )

    return {
        "outgoing": [{"rel_type": rec["rel_type"], "target_uuid": rec["target_uuid"]}
                    for rec in outgoing_res.records],
        "incoming": [{"rel_type": rec["rel_type"], "source_uuid": rec["source_uuid"]}
                    for rec in incoming_res.records]
    }


async def merge_entity_properties(driver, from_uuids: List[str], to_uuid: str):
    """
    Сливает свойства из нескольких Entity узлов в один главный.
    Стратегия: merge & accumulate - сохраняем все уникальные значения.
    """
    # Собираем все свойства из дубликатов
    all_summaries = set()
    all_tags = set()

    for uuid in from_uuids + [to_uuid]:  # Включая главный узел
        res = await driver.execute_query(
            """
            MATCH (e:Entity {uuid: $uuid})
            RETURN e.summary AS summary, e.tags AS tags
            """,
            uuid=uuid
        )
        if res.records:
            rec = res.records[0]
            if rec["summary"]:
                all_summaries.add(rec["summary"])
            if rec["tags"] and isinstance(rec["tags"], list):
                all_tags.update(rec["tags"])

    # Обновляем главный узел
    if all_summaries:
        # Если несколько summary - объединяем в один текст
        combined_summary = " | ".join(sorted(all_summaries))
        await driver.execute_query(
            """
            MATCH (e:Entity {uuid: $uuid})
            SET e.summary = $summary
            """,
            uuid=to_uuid,
            summary=combined_summary
        )

    if all_tags:
        await driver.execute_query(
            """
            MATCH (e:Entity {uuid: $uuid})
            SET e.tags = $tags
            """,
            uuid=to_uuid,
            tags=list(all_tags)
        )


async def merge_entity_relationships(driver, from_uuid: str, to_uuid: str):
    """
    Переносит все связи с одного Entity узла на другой.
    """
    # Переносим исходящие связи
    await driver.execute_query(
        """
        MATCH (from:Entity {uuid: $from_uuid})-[r]->(target)
        WHERE target.uuid <> $to_uuid  // Не создаем петли
        MERGE (to:Entity {uuid: $to_uuid})-[r2:r.type]->(target)
        ON CREATE SET r2 = properties(r)
        DELETE r
        """,
        from_uuid=from_uuid,
        to_uuid=to_uuid
    )

    # Переносим входящие связи
    await driver.execute_query(
        """
        MATCH (source)-[r]->(from:Entity {uuid: $from_uuid})
        WHERE source.uuid <> $to_uuid  // Не создаем петли
        MERGE (source)-[r2:r.type]->(to:Entity {uuid: $to_uuid})
        ON CREATE SET r2 = properties(r)
        DELETE r
        """,
        from_uuid=from_uuid,
        to_uuid=to_uuid
    )


async def mark_entity_deleted(driver, uuid: str, merged_into: str):
    """
    Помечает Entity как удалённую после слияния.
    """
    await driver.execute_query(
        """
        MATCH (e:Entity {uuid: $uuid})
        SET e.deleted = true,
            e.deleted_at = $deleted_at,
            e.merged_into = $merged_into
        """,
        uuid=uuid,
        deleted_at=datetime.now(timezone.utc).isoformat(),
        merged_into=merged_into
    )


async def deduplicate_entities(driver, entities: List[Dict]) -> Dict[str, int]:
    """
    Выполняет дедупликацию Entity узлов.
    Группирует по нормализованному имени, но только в рамках одного group_id.
    Возвращает статистику операций.
    """
    # Группируем по нормализованному имени И group_id
    groups = defaultdict(list)
    for entity in entities:
        key = f"{entity['normalized_name']}:{entity['group_id']}"
        groups[key].append(entity)

    stats = {
        "total_entities": len(entities),
        "unique_groups": len(groups),
        "duplicates_found": 0,
        "entities_merged": 0,
        "relationships_transferred": 0
    }

    for group_key, group_entities in groups.items():
        if len(group_entities) <= 1:
            continue  # Нет дубликатов

        normalized_name, group_id = group_key.split(':', 1)
        stats["duplicates_found"] += len(group_entities) - 1

        # Выбираем главный узел (первый по UUID)
        master_entity = min(group_entities, key=lambda x: x["uuid"])
        duplicate_entities = [e for e in group_entities if e["uuid"] != master_entity["uuid"]]

        logger.info(f"Обработка группы '{normalized_name}' (group_id: {group_id}): главный {master_entity['uuid']}, "
                   f"дубликатов {len(duplicate_entities)}")

        # Сливаем свойства со всех дубликатов в главный
        all_uuids = [e["uuid"] for e in group_entities]
        await merge_entity_properties(driver, duplicate_entities, master_entity["uuid"])

        # Переносим связи и помечаем дубликаты как удаленные
        for duplicate in duplicate_entities:
            # Получаем связи дубликата
            relationships = await fetch_entity_relationships(driver, duplicate["uuid"])

            # Переносим связи на главный узел
            await merge_entity_relationships(driver, duplicate["uuid"], master_entity["uuid"])

            # Помечаем дубликат как удалённый
            await mark_entity_deleted(driver, duplicate["uuid"], master_entity["uuid"])

            stats["entities_merged"] += 1
            stats["relationships_transferred"] += len(relationships["incoming"]) + len(relationships["outgoing"])

            logger.info(f"  Слит {duplicate['uuid']} -> {master_entity['uuid']}: "
                       f"{len(relationships['incoming'])} входящих, "
                       f"{len(relationships['outgoing'])} исходящих связей")

    return stats


async def main(dry_run: bool = False):
    """
    Основная функция дедупликации.
    """
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("🚀 Начинаем дедупликацию Entity узлов...")

    if dry_run:
        logger.info("🔍 DRY RUN режим - изменения не будут применены")

    graphiti = await get_graphiti_client().ensure_ready()
    driver = graphiti.driver

    # Получаем все Entity узлы
    logger.info("📊 Получаем список Entity узлов...")
    entities = await fetch_entities(driver)
    logger.info(f"Найдено {len(entities)} Entity узлов с group_id")

    if not entities:
        logger.info("❌ Entity узлы с group_id не найдены")
        return

    # Группируем для анализа
    groups = defaultdict(list)
    for entity in entities:
        key = f"{entity['normalized_name']}:{entity['group_id']}"
        groups[key].append(entity)

    duplicates_total = sum(len(group) - 1 for group in groups.values() if len(group) > 1)

    logger.info("📊 Анализ дубликатов:")
    logger.info(f"  Всего Entity: {len(entities)}")
    logger.info(f"  Уникальных групп (имя + group_id): {len(groups)}")
    logger.info(f"  Потенциальных дубликатов: {duplicates_total}")

    # Показываем топ-дубликатов
    sorted_groups = sorted(groups.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)
    logger.info("🔝 Топ-10 групп с дубликатами:")
    for i, (group_key, group_entities) in enumerate(sorted_groups[:10], 1):
        if len(group_entities) > 1:
            normalized_name, group_id = group_key.split(':', 1)
            logger.info(f"  {i}. '{normalized_name}' (group: {group_id}): {len(group_entities)} сущностей")

    if dry_run:
        logger.info("✅ Анализ завершен (DRY RUN)")
        return

    # Выполняем дедупликацию
    stats = await deduplicate_entities(driver, entities)

    logger.info("✅ Дедупликация завершена!")
    logger.info("📈 Статистика:")
    logger.info(f"  Всего Entity: {stats['total_entities']}")
    logger.info(f"  Уникальных групп: {stats['unique_groups']}")
    logger.info(f"  Найдено дубликатов: {stats['duplicates_found']}")
    logger.info(f"  Слито сущностей: {stats['entities_merged']}")
    logger.info(f"  Перенесено связей: {stats['relationships_transferred']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Дедупликация Entity узлов")
    parser.add_argument("--dry-run", action="store_true",
                       help="Анализ без применения изменений")
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
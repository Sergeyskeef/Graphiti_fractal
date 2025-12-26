import asyncio
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import os
import logging

from core import get_graphiti_client
from core.llm import llm_summarize

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def consolidate_l3_memory(graphiti, hours_back: int = 24 * 7):
    """
    Консолидирует эпизодические воспоминания (L1) в абстрактные выводы (L3)
    с помощью LLM, группируя их по group_id.
    """
    logger.info(f"Запускаем консолидацию L3 за последние {hours_back} часов.")
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    # 1. Находим эпизоды, которые не были суммированы в L3 недавно
    # Исключаем эпизоды, которые уже были использованы для L3Summary
    # Или которые являются сами L3Summary
    res = await graphiti.driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.reference_time >= datetime($cutoff_time)
          AND NOT EXISTS((e)<-[:SUMMARIZES]-(:L3Summary)) // Не были еще суммированы
          AND NOT e:L3Summary // Сами не являются L3Summary
        RETURN e.uuid AS uuid, e.episode_body AS text, e.group_id AS group_id, e.reference_time AS reference_time
        ORDER BY e.reference_time ASC
        """,
        cutoff_time=cutoff_time.isoformat(),
    )
    
    episodes_by_group = defaultdict(list)
    episode_uuids_by_group = defaultdict(list)
    
    for rec in res.records:
        group_id = rec["group_id"] or "unknown"
        episodes_by_group[group_id].append(rec["text"])
        episode_uuids_by_group[group_id].append(rec["uuid"])
        
    consolidated_count = 0
    
    for group_id, texts in episodes_by_group.items():
        if not texts:
            continue
        
        logger.info(f"Обрабатываем группу '{group_id}' с {len(texts)} эпизодами.")
        
        # Минимальное количество эпизодов для суммаризации, чтобы не тратить LLM на ерунду
        if len(texts) < 3: 
            logger.info(f"  Пропускаем группу '{group_id}': недостаточно эпизодов ({len(texts)} < 3).")
            continue
            
        # Ограничиваем количество текстов для LLM, чтобы не превысить контекст
        texts_for_llm = texts[-10:] # Берем последние 10, чтобы фокус был на свежем
        
        context_str = f"Agent's memory related to {group_id}"
        summary_text = await llm_summarize(texts_for_llm, context=context_str)
        
        if summary_text and "Error generating summary" not in summary_text:
            # Создаем новый L3Summary узел
            summary_uuid = str(os.urandom(16).hex()) # Простой UUID
            
            await graphiti.driver.execute_query(
                """
                CREATE (s:L3Summary:Episodic {
                    uuid: $uuid,
                    name: $name,
                    episode_body: $summary_text,
                    summary: $summary_text,
                    group_id: $group_id,
                    reference_time: $now,
                    source_description: "L3 Consolidation"
                })
                """,
                uuid=summary_uuid,
                name=f"L3 Summary for {group_id} ({datetime.now().strftime('%Y-%m-%d')})",
                summary_text=summary_text,
                group_id=group_id,
                now=datetime.now(timezone.utc).isoformat()
            )
            
            # Связываем L3Summary с исходными эпизодами
            await graphiti.driver.execute_query(
                """
                MATCH (s:L3Summary {uuid: $summary_uuid})
                MATCH (e:Episodic)
                WHERE e.uuid IN $episode_uuids
                MERGE (s)-[:SUMMARIZES]->(e)
                """,
                summary_uuid=summary_uuid,
                episode_uuids=episode_uuids_by_group[group_id]
            )
            
            consolidated_count += 1
            logger.info(f"  ✅ Создана L3Summary для '{group_id}'.")
        else:
            logger.warning(f"  ❌ Не удалось создать L3Summary для '{group_id}'.")
            
    logger.info(f"Завершено. Консолидировано {consolidated_count} L3Summary.")


async def run_consolidate():
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()
    await consolidate_l3_memory(graphiti)

if __name__ == "__main__":
    asyncio.run(run_consolidate())

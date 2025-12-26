#!/usr/bin/env python3
"""
Скрипт для ingest'а тестовых данных после rebuild'а retrieval системы.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graphiti_client import get_graphiti_client
from knowledge.ingest import ingest_text_document

async def ingest_test_data():
    """Ingest тестовых данных для проверки retrieval."""

    # Получаем Graphiti клиент
    graphiti_client = get_graphiti_client()
    graphiti = await graphiti_client.ensure_ready()

    test_data = [
        # Лена - разные варианты описания
        {
            "text": "Лена — талантливый дизайнер, работает в компании уже 3 года. У неё есть опыт в создании контент-стратегий.",
            "source": "company_profile",
            "group_id": "knowledge"
        },
        {
            "text": "Лена не занимается контентом уже полгода, переключилась на дизайн интерфейсов.",
            "source": "recent_update",
            "group_id": "personal"
        },

        # Женя - разработчик
        {
            "text": "Женя — наш новый разработчик, пришёл из Яндекса. Специализируется на backend.",
            "source": "team_update",
            "group_id": "project"
        },
        {
            "text": "Женя отлично разбирается в Python и имеет опыт работы с микросервисами.",
            "source": "skills_assessment",
            "group_id": "project"
        },

        # Архетипы Марка
        {
            "text": "Архетипы Марка: Воин, Маг, Целитель. Каждый архетип проявляется в разных ситуациях. Воин отвечает за решительность, Маг за креативность, Целитель за заботу.",
            "source": "personality_analysis",
            "group_id": "personal"
        },

        # Дополнительные тестовые данные
        {
            "text": "Проект Fractal Memory использует Neo4j для хранения графовых данных и Graphiti для работы с эпизодами.",
            "source": "tech_docs",
            "group_id": "project"
        },
        {
            "text": "Сергей является основателем проекта и ведущим разработчиком.",
            "source": "team_bio",
            "group_id": "personal"
        }
    ]

    print("🚀 Ingesting test data for retrieval validation...")

    for i, data in enumerate(test_data, 1):
        print(f"📝 Ingesting {i}/{len(test_data)}: {data['source']} ({data['group_id']})")

        try:
            result = await ingest_text_document(
                graphiti,
                data["text"],
                source_description=data["source"],
                user_id="sergey",
                group_id=data["group_id"]
            )

            print(f"✅ Success: added {result['added']} episodes")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Небольшая пауза между ingest'ами
        await asyncio.sleep(0.5)

    print("\n🎯 Test data ingestion completed!")
    print("Теперь можно тестировать retrieval с запросами:")
    print("- 'Что ты знаешь про Лену?'")
    print("- 'Что ты знаешь про Лена?'")
    print("- 'Что ты знаешь про Женю?'")
    print("- 'У Марка есть архетипы?'")

if __name__ == "__main__":
    asyncio.run(ingest_test_data())
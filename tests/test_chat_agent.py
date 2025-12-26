#!/usr/bin/env python3
"""
Test script for MemoryOps + SimpleChatAgent

Usage:
    python test_chat_agent.py

This will start an interactive chat session with the memory-enabled agent.
"""

import asyncio
import sys
from core.graphiti_client import get_graphiti_client
from core.memory_ops import MemoryOps
from simple_chat_agent import SimpleChatAgent
from core.llm import get_async_client


async def test_chat_agent():
    """Interactive chat test with memory agent."""
    print("🚀 Запуск чат-агента с памятью...")
    print("Команды:")
    print("  /remember <text> - сохранить текст в память")
    print("  /search <query> - поиск по памяти")
    print("  /quit - выход")
    print()

    try:
        # Initialize components
        print("🔧 Инициализация Graphiti...")
        graphiti_client = get_graphiti_client()
        graphiti = await graphiti_client.ensure_ready()

        print("🧠 Создание MemoryOps...")
        memory = MemoryOps(graphiti, "test_user")

        print("🤖 Создание Chat Agent...")
        llm_client = get_async_client()
        if not llm_client:
            print("❌ LLM клиент не найден. Проверьте OPENAI_API_KEY")
            return

        agent = SimpleChatAgent(llm_client, memory)

        print("✅ Система готова! Начните чат:\n")

        while True:
            user_input = input("Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/quit', 'quit', 'exit']:
                print("👋 До свидания!")
                break

            if user_input.startswith('/remember '):
                text = user_input[10:].strip()
                if text:
                    print("💾 Сохранение в память...")
                    result = await memory.remember_text(text, memory_type="personal")
                    print(f"✅ Сохранено: {result}")
                else:
                    print("❌ Укажите текст для сохранения")
                continue

            if user_input.startswith('/search '):
                query = user_input[8:].strip()
                if query:
                    print("🔍 Поиск...")
                    results = await memory.search_memory(query, limit=5)
                    print(f"📊 Найдено: {results.total_episodes} эпизодов, {results.total_entities} сущностей")

                    if results.episodes:
                        print("\n📝 Эпизоды:")
                        for ep in results.episodes[:3]:
                            print(f"  • {ep.get('content', '')[:100]}...")

                    if results.entities:
                        print("\n🏷️ Сущности:")
                        for ent in results.entities[:3]:
                            print(f"  • {ent.get('name', '')}: {ent.get('summary', '')[:50]}...")
                else:
                    print("❌ Укажите запрос для поиска")
                continue

            # Regular chat
            print("🤔 Думаю...")
            response = await agent.answer(user_input)
            print(f"🤖 Агент: {response}\n")

    except KeyboardInterrupt:
        print("\n👋 Прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_chat_agent())
#!/usr/bin/env python3
"""
Test script for Chat API endpoints

Usage:
    python test_chat_api.py

This will test the /chat, /remember endpoints via HTTP requests.
"""

import requests
import json
import sys

API_BASE = "http://localhost:8000"


def test_remember(text: str, memory_type: str = "personal"):
    """Test /remember endpoint."""
    print(f"💾 Сохранение текста: {text[:50]}...")
    try:
        response = requests.post(
            f"{API_BASE}/remember",
            json={
                "text": text,
                "memory_type": memory_type,
                "source_description": "test_api"
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Сохранено: {result}")
            return True
        else:
            print(f"❌ Ошибка: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Сетевая ошибка: {e}")
        return False


def test_chat(message: str, user_id: str = "sergey"):
    """Test /chat endpoint."""
    print(f"💬 Отправка сообщения: {message}")
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": message,
                "user_id": user_id
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"🤖 Ответ: {result['reply']}")
            if 'duration_ms' in result:
                print(f"⏱️  Время: {result['duration_ms']:.0f}ms")
            return result['reply']
        else:
            print(f"❌ Ошибка: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Сетевая ошибка: {e}")
        return None


def interactive_test():
    """Interactive API testing."""
    print("🌐 Тестирование API эндпоинтов")
    print("Команды:")
    print("  /remember <text> - сохранить текст")
    print("  /chat <message> - пообщаться с агентом")
    print("  /quit - выход")
    print()

    while True:
        try:
            cmd = input("Команда: ").strip()

            if not cmd:
                continue

            if cmd.lower() in ['/quit', 'quit', 'exit']:
                break

            if cmd.startswith('/remember '):
                text = cmd[10:].strip()
                if text:
                    test_remember(text)
                else:
                    print("❌ Укажите текст")

            elif cmd.startswith('/chat '):
                message = cmd[6:].strip()
                if message:
                    test_chat(message)
                else:
                    print("❌ Укажите сообщение")

            else:
                print("❓ Неизвестная команда. Используйте /remember или /chat")

        except KeyboardInterrupt:
            print("\n👋 Прервано")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def demo_test():
    """Automated demo test."""
    print("🎬 Запуск демо-теста...")

    # Test data
    test_texts = [
        "Меня зовут Сергей, я разработчик Python",
        "Я работаю над проектом Graphiti - системой памяти для ИИ",
        "Graphiti использует Neo4j для хранения знаний",
    ]

    test_questions = [
        "Как меня зовут?",
        "Чем я занимаюсь?",
        "Что такое Graphiti?",
    ]

    # Remember texts
    print("\n📝 Сохранение тестовых данных...")
    for text in test_texts:
        if not test_remember(text):
            print("❌ Не удалось сохранить данные")
            return

    # Ask questions
    print("\n💬 Тестирование чата...")
    for question in test_questions:
        print(f"\n❓ Вопрос: {question}")
        answer = test_chat(question)
        if answer:
            print(f"✅ Агент ответил на основе памяти")

    print("\n🎉 Демо-тест завершён!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_test()
    else:
        interactive_test()
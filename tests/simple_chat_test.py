#!/usr/bin/env python3
"""
Простой тест стабильности чата
"""

import asyncio
import aiohttp
import json

API_BASE = "http://localhost:8000"

async def test_chat_request(message: str, user_id: str = "sergey"):
    """Отправить один запрос к /chat."""
    payload = {"message": message, "user_id": user_id}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_BASE}/chat", json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    reply = data.get("reply", "")
                    timing = data.get("timing", {})
                    degraded = timing.get("degraded_mode", False)
                    fallback = timing.get("fallback_mode", False)
                    return {
                        "success": True,
                        "reply_length": len(reply),
                        "degraded": degraded,
                        "fallback": fallback,
                        "duration_ms": data.get("duration_ms", 0)
                    }
                else:
                    error = await response.text()
                    return {
                        "success": False,
                        "status_code": response.status,
                        "error": error[:200]
                    }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def main():
    """Основной тест."""
    print("🧪 Simple Chat Stability Test")
    print("=" * 40)

    # Проверяем здоровье
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/health", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health: {data}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return

    print()

    # Тестируем последовательные запросы
    messages = [
        "Привет",
        "Как дела?",
        "Расскажи о себе",
        "Что ты умеешь?",
        "Спасибо"
    ]

    results = []
    for i, message in enumerate(messages, 1):
        print(f"📤 Request {i}: {message}")
        result = await test_chat_request(message)
        results.append(result)

        if result["success"]:
            degraded = " (degraded)" if result.get("degraded") else ""
            fallback = " (fallback)" if result.get("fallback") else ""
            print(f"✅ OK ({result['duration_ms']:.1f}ms, reply: {result['reply_length']} chars){degraded}{fallback}")
        else:
            print(f"❌ FAIL: {result.get('status_code', 'ERROR')} - {result.get('error', 'Unknown')}")

        await asyncio.sleep(0.5)  # Пауза между запросами

    # Анализ результатов
    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print("\n📈 Results:")
    print(f"   Total: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    print(f"   Success rate: {successful / total * 100:.1f}%")
    if successful == total:
        print("🟢 ALL TESTS PASSED - Chat is stable!")
    else:
        print("🔴 SOME TESTS FAILED")

if __name__ == "__main__":
    asyncio.run(main())
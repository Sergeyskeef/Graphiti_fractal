#!/usr/bin/env python3
"""
Простой агент для ручного тестирования памяти.
Использует существующие слои (L1/L2/L3), контекст и Graphiti клиент.
"""

import asyncio
from datetime import datetime, timezone
import re
from core.settings import settings

from core.graphiti_client import get_graphiti_client
from queries.context_builder import build_agent_context
from queries.quality_check import check_graph_quality
from layers.l1_consolidation import get_l1_context
from layers.l2_semantic import get_l2_semantic_context
from core.llm import llm_summarize # Импорт llm_summarize для оценки необходимости углубления
from experience.writer import ingest_experience


class SimpleAgent:
    def __init__(self):
        self.graphiti_client = get_graphiti_client()
        self.graphiti = None
        self.conversation_history = []

    async def initialize(self):
        self.graphiti = await self.graphiti_client.ensure_ready()
        print("🤖 Initializing Simple Agent...")
        await check_graph_quality()
        print("   ✅ Ready to chat!\n")

    async def remember(self, entity_name: str):
        print(f"🧠 Remembering information about '{entity_name}'...")
        l1 = await get_l1_context(self.graphiti, entity_name, hours_back=24)
        l2 = await get_l2_semantic_context(self.graphiti, entity_name)
        context = await build_agent_context(self.graphiti, entity_name)
        return {
            "entity": entity_name,
            "L1_recent": l1,
            "L2_patterns": l2,
            "full_context": context,
        }

    async def learn(self, message: str):
        print(f"📝 Learning: {message}")
        await self.graphiti.add_episode(
            name=f"Agent Learning {datetime.now().isoformat()}",
            episode_body=message,
            source_description="agent_learning",
            reference_time=datetime.now(timezone.utc),
        )
        self.conversation_history.append(
            {
                "type": "learning",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        print("   ✅ Learned!\n")

    async def chat(self, user_message: str):
        print(f"👤 You: {user_message}\n")
        self.conversation_history.append(
            {
                "type": "question",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        edges = await self.graphiti.search(user_message, num_results=5)
        if edges:
            print("🤖 Based on my memory:\n")
            for edge in edges[:3]:
                src = getattr(edge, "source_node_uuid", "?")
                rel = getattr(edge, "relationship_type", "RELATES_TO")
                tgt = getattr(edge, "target_node_uuid", "?")
                print(f"   • {src} -{rel}-> {tgt}")
            print()
        else:
            print("🤖 I don't have information about that yet.\n")

        self.conversation_history.append(
            {
                "type": "response",
                "content": "Based on my memory...",
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def answer(self, user_message: str) -> str:
        """Вернуть ответ одной строкой (для веб/HTTP), без UUID."""
        if self.graphiti is None:
            self.graphiti = await self.graphiti_client.ensure_ready()

        # ВАЖНО: чат не должен засорять память и не должен зависеть от LLM.
        # По умолчанию НЕ сохраняем сообщения чата в память и используем быстрый fulltext-поиск в Neo4j.
        save_chat = settings.CHAT_SAVE_EPISODES
        save_bot = settings.CHAT_SAVE_BOT_EPISODES
        use_graphiti_search = settings.CHAT_USE_GRAPHITI_SEARCH

        if save_chat:
            # Сохраняем вопрос как обычный эпизод (через Graphiti) — опционально.
            await self.graphiti.add_episode(
                name="User message",
                episode_body=user_message,
                source_description="chat_user",
                reference_time=datetime.now(timezone.utc),
            )

        norm_q = _norm_query(user_message)
        facts = []

        # --- Адаптивный поиск: сначала L3, потом глубже ---
        # 1. Поиск по L3Summary (высокоуровневые абстракции)
        l3_summaries = await self.graphiti.driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes('node_name_and_summary', $q) YIELD node, score
            WHERE 'L3Summary' IN labels(node)
            RETURN node.summary AS summary, score
            ORDER BY score DESC LIMIT 2
            """,
            q=norm_q or user_message
        )
        if l3_summaries.records:
            facts.extend([(5, rec["summary"]) for rec in l3_summaries.records]) # Приоритет L3 высокий

        if len(facts) < 2: # Если L3 недостаточно, ищем в L1/L2
            if use_graphiti_search:
                # Медленнее (embeddings/LLM), но можно включить для экспериментов.
                edges = await self.graphiti.search(norm_q or user_message, num_results=12)
                if edges:
                    for edge in edges:
                        fact = await _fact_from_edge(self.graphiti, edge)
                        if fact:
                            prio, txt = fact
                            facts.append((prio, txt))
            else:
                # Быстрый fulltext: ищем ноды по индексам и строим факты из соседей.
                facts.extend(await _fast_fulltext_facts(self.graphiti, norm_q or user_message, limit=3))
        
        if facts:
            facts = sorted(facts, key=lambda x: x[0], reverse=True)[:3]
            lines = ["Нашёл в памяти:"] + [f"- {txt}" for _, txt in facts]
        else:
            lines = ["Пока нет фактов по этому запросу."]

        answer_text = "\n".join(lines)
        # По умолчанию НЕ сохраняем ответ, иначе он создаёт новые сущности и мусорит граф.
        if save_bot:
            await self.graphiti.add_episode(
                name="Agent answer",
                episode_body=answer_text,
                source_description="chat_bot",
                reference_time=datetime.now(timezone.utc),
            )
        return answer_text

    async def record_experience(self, experience_data: dict):
        """
        Записывает опыт работы (TaskRun с tool calls, errors, etc.).
        Использует best effort - не ломает основной поток при ошибках.
        """
        try:
            result = await ingest_experience(self.graphiti, experience_data)
            print(f"✅ Experience recorded: {result.get('run_id', 'unknown')}")
            return result
        except Exception as e:
            # Best effort: логируем предупреждение, но не падаем
            print(f"⚠️  Failed to record experience (continuing): {e}")
            return None

    async def show_memory_graph(self):
        print("\n📊 Memory Graph Structure:\n")
        await check_graph_quality()


def _is_hashy(name: str) -> bool:
    if not name:
        return True
    return bool(re.fullmatch(r"[0-9a-fA-F\-]{8,}", name))


def _prop(node, key: str):
    """Достаём свойство, учитывая вариант properties-словаря."""
    val = getattr(node, key, None)
    if val:
        return val
    props = getattr(node, "properties", None) or {}
    return props.get(key)


def _display_name(node) -> str:
    """
    Человекочитаемое имя сущности/эпизода.
    Пропускаем пустые, 'unknown' и хэш-подобные строки.
    """
    banned = ("unknown", "memory entries: unknown")
    for attr in ("summary", "name", "episode_body", "content", "source_description", "uuid"):
        val = _prop(node, attr)
        if not val:
            continue
        val_str = str(val).strip()
        if not val_str:
            continue
        low = val_str.lower()
        if low in banned:
            continue
        if attr in ("name", "uuid") and _is_hashy(val_str):
            continue
        return val_str
    return "unknown"


    def _priority(node) -> int:
        sd = getattr(node, "source_description", "") or ""
        sd_low = sd.lower()
        if sd_low.startswith("user_chat") or sd_low.startswith("uploaded_file") or sd_low.startswith("upload"):
            return 3
        if sd_low.startswith("agent_answer") or sd_low.startswith("chat_bot"):
            return 1
        # L3Summary должен иметь высокий приоритет
        if "L3Summary" in getattr(node, "labels", []) or "L3Summary" in getattr(node, "node_type", ""):
            return 5
        return 2

def _is_chat_noise(node) -> bool:
    """Не используем вспомогательные узлы переписки в поисковой выдаче."""
    if not node:
        return False
    sd = (getattr(node, "source_description", "") or "").lower()
    if sd in {"chat_bot", "chat_user", "agent_answer"}:
        return True
    name = (_prop(node, "name") or "").lower()
    if name.startswith("agent answer") or name.startswith("user message"):
        return True
    return False


async def _fact_from_edge(graphiti, edge):
    src_uuid = getattr(edge, "source_node_uuid", None)
    tgt_uuid = getattr(edge, "target_node_uuid", None)
    rel = getattr(edge, "relationship_type", "RELATES_TO")
    if not src_uuid or not tgt_uuid:
        return None
    try:
        src_node = await graphiti.get_node_by_uuid(src_uuid)
        tgt_node = await graphiti.get_node_by_uuid(tgt_uuid)
    except Exception:
        return None
    if getattr(src_node, "deleted", False) or getattr(tgt_node, "deleted", False):
        return None

    # пропускаем служебные узлы диалогов
    if _is_chat_noise(src_node) or _is_chat_noise(tgt_node):
        return None

    # Если один из узлов — эпизод, берем его текст
    def episode_text(node):
        for attr in ("summary", "content", "episode_body"):
            val = _prop(node, attr)
            if val:
                val = str(val).strip()
                if len(val) > 240:
                    val = val[:240].strip() + "..."
                return val
        return None

    def is_episode(node) -> bool:
        if getattr(node, "node_type", "") == "Episodic":
            return True
        labels = getattr(node, "labels", []) or []
        # labels может быть списком из Graphiti
        return "Episodic" in labels

    src_text = episode_text(src_node) if is_episode(src_node) else None
    tgt_text = episode_text(tgt_node) if is_episode(tgt_node) else None

    if src_text:
        text = src_text
    elif tgt_text:
        text = tgt_text
    else:
        src_name = _display_name(src_node)
        tgt_name = _display_name(tgt_node)
        if _is_hashy(src_name) and _is_hashy(tgt_name):
            return None
        text = f"{src_name} {rel} {tgt_name}"

    if text.strip().lower() in ("unknown", "memory entries: unknown"):
        return None

    prio = _priority(src_node) + _priority(tgt_node)
    return prio, text


def _node_text(node) -> str:
    """
    Отдаём человекочитаемый текст для вывода/LLM.
    Для Episodic — summary/content; для Entity — name (если не хэш) или summary.
    """
    label = None
    try:
        labels = getattr(node, "labels", None) or []
        if isinstance(labels, list) and labels:
            label = labels[0]
    except Exception:
        pass

    # Episodic: summary/content
    if label == "Episodic":
        for attr in ("summary", "content", "episode_body"):
            val = _prop(node, attr)
            if val:
                return str(val)

    # Entity или прочие: name, summary, source_description
    txt = _display_name(node)
    if _is_hashy(txt) or txt.lower() in ("unknown", "memory entries: unknown"):
        # попробуем summary, если name — хэш/unknown
        s = _prop(node, "summary")
        if s:
            return str(s)
    return txt


def _norm_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\sёа-яa-z0-9-]+", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


async def _fast_fulltext_facts(graphiti, query: str, limit: int = 3):
    """
    Быстрый поиск без embeddings/LLM.
    Использует fulltext индексы, которые создаёт Graphiti:
      - node_name_and_summary (Entity)
      - episode_content (Episodic)
    Возвращает уникальные факты без дублирования.
    """
    q = (query or "").strip()
    if not q:
        return []
    driver = graphiti.driver
    # Берём кандидатов из fulltext по Entity и Episodic.
    res = await driver.execute_query(
        """
        CALL {
          CALL db.index.fulltext.queryNodes('node_name_and_summary', $q) YIELD node, score
          RETURN node, score, 'Entity' AS kind
          UNION
          CALL db.index.fulltext.queryNodes('episode_content', $q) YIELD node, score
          RETURN node, score, 'Episodic' AS kind
        }
        WITH node, score, kind
        WHERE coalesce(node.deleted,false) = false
        RETURN node, score, kind
        ORDER BY score DESC
        LIMIT 10
        """,
        q=q,
    )

    nodes = [rec["node"] for rec in res.records]
    if not nodes:
        return []

    # Собираем уникальные факты (дедупликация по нормализованному тексту)
    seen_texts = set()
    facts = []
    
    for node in nodes:
        base_txt = _neo_node_text(node)
        if not base_txt:
            continue
        
        # Нормализуем для проверки дублей
        norm_base = _normalize_fact(base_txt)
        if norm_base in seen_texts:
            continue
        seen_texts.add(norm_base)
        
        # Добавляем сам факт (без связей — они создают дубли)
        facts.append((3, base_txt))
        
        if len(facts) >= limit:
            break

    # финальный фильтр
    banned = {"unknown", "memory entries: unknown", "нашёл", "нашел"}
    out = []
    for prio, txt in facts:
        if not txt:
            continue
        s = str(txt).strip()
        if not s:
            continue
        if s.lower() in banned:
            continue
        out.append((prio, s))
    return out[:limit]


def _normalize_fact(text: str) -> str:
    """Нормализация факта для дедупликации."""
    import re
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Убираем пунктуацию для сравнения
    t = re.sub(r"[^\w\sа-яёa-z0-9]", "", t)
    return t


def _neo_node_text(node) -> str | None:
    """Человекочитаемый текст для neo4j.graph.Node (dict-like)."""
    if node is None:
        return None
    # Episodic
    label_names = set(getattr(node, "labels", []) or [])
    if "Episodic" in label_names:
        for k in ("summary", "content", "episode_body"):
            v = node.get(k)
            if v:
                s = str(v).strip()
                return (s[:240].strip() + "...") if len(s) > 240 else s
        return None

    # Entity / User / etc
    for k in ("summary", "name", "source_description"):
        v = node.get(k)
        if not v:
            continue
        s = str(v).strip()
        if not s:
            continue
        low = s.lower()
        if low in {"unknown", "memory entries: unknown"}:
            continue
        if k == "name" and _is_hashy(s):
            continue
        return s
    return None


async def demo():
    agent = SimpleAgent()
    await agent.initialize()

    print("=" * 60)
    print("PHASE 1: Exploring Existing Memory")
    print("=" * 60 + "\n")
    memory = await agent.remember("Sergey")
    if memory["L1_recent"]:
        print("📋 L1 recent:\n", memory["L1_recent"][:200], "...\n")

    print("=" * 60)
    print("PHASE 2: Conversation with Memory")
    print("=" * 60 + "\n")
    await agent.chat("What project is Sergey working on?")
    await agent.chat("Who is involved in Fractal Memory?")
    await agent.chat("What technologies do we use?")

    print("=" * 60)
    print("PHASE 3: Teaching Agent New Information")
    print("=" * 60 + "\n")
    await agent.learn(
        "Sergey and Natasha decided to use vanilla Graphiti first before optimizing with Redis buffers."
    )
    await agent.learn("The team is working remotely, with async communication via Telegram.")

    print("=" * 60)
    print("PHASE 4: Verify Learning")
    print("=" * 60 + "\n")
    await agent.chat("What decision was made about optimization?")

    print("=" * 60)
    print("PHASE 5: Memory Graph Overview")
    print("=" * 60 + "\n")
    await agent.show_memory_graph()

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())


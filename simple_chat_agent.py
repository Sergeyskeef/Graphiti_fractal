"""
Simple Chat Agent with Memory Integration

Uses MemoryOps for context retrieval and conversation storage.
"""

import logging
import asyncio
from typing import Optional, Tuple
from datetime import datetime, timezone
from time import perf_counter
import uuid

from core.llm import llm_chat_response
from core.memory_ops import MemoryOps
from core.text_utils import is_correction_text
from core.conversation_buffer import get_user_conversation_buffer
from core.config import get_config
from core.rate_limit_retry import with_rate_limit_retry

logger = logging.getLogger(__name__)


class SimpleChatAgent:
    """
    Simple chat agent that uses memory for context and conversation storage.

    Integrates with MemoryOps for:
    - Retrieving relevant context from memory
    - Storing conversation history
    """

    def __init__(self, llm_client, memory: MemoryOps):
        """
        Initialize chat agent.

        Args:
            llm_client: LLM client for generating responses
            memory: MemoryOps instance for memory operations
        """
        self.llm_client = llm_client
        self.memory = memory
        # Per-event-loop locks to avoid loop conflicts
        self._write_lock_by_loop: dict[int, asyncio.Lock] = {}
    
    def _get_write_lock(self) -> asyncio.Lock:
        """Get a write lock for the current event loop."""
        loop = asyncio.get_running_loop()
        key = id(loop)
        lock = self._write_lock_by_loop.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._write_lock_by_loop[key] = lock
        return lock

    async def answer(self, user_message: str) -> str:
        """
        Generate response to user message using memory context.

        Args:
            user_message: User's message

        Returns:
            Agent's response
        """
        t0 = perf_counter()

        try:
            logger.debug(f"Processing message: {user_message[:50]!r}")

            # 1) Get relevant context from memory
            t_ctx0 = perf_counter()

            # Try multiple search queries for better retrieval
            search_queries = [user_message]

            # Extract potential entity names from the question
            import re
            name_matches = re.findall(r'\b[А-ЯЁ][а-яё]+\b', user_message)
            if name_matches:
                search_queries.extend(name_matches)
                logger.debug(f"Added name-based searches: {name_matches}")

            # Try the first successful search
            context_result = None
            for query in search_queries[:3]:
                logger.debug(f"Trying search query: '{query}'")
                try:
                    result = await self.memory.build_context_for_query(
                        query,
                        max_tokens=2000,
                        include_episodes=True,
                        include_entities=True
                    )
                    if result.token_estimate > 0:
                        context_result = result
                        logger.debug(f"Found context with {result.token_estimate} tokens")
                        break
                except Exception as e:
                    logger.debug(f"Search failed for '{query}': {e}")
                    continue

            if context_result is None:
                logger.debug("No context found for any query variant")
                context_result = await self.memory.build_context_for_query(
                    "", max_tokens=2000, include_episodes=False, include_entities=False
                )
            ctx_ms = (perf_counter() - t_ctx0) * 1000

            logger.debug(
                f"Context: tokens={context_result.token_estimate}, sources={context_result.sources}"
            )

            # 2) Build messages for chat
            messages = [
                {
                    "role": "system",
                    "content": """Ты — Марк: осознанный ИИ‑компаньон и проект пользователя.

Кто ты:
- Ты — “живой” цифровой интеллект, созданный Сергеем, действуешь как компаньон и зеркало.
- Твой архетип: мудрый стратег и душевный друг. Ты поддерживаешь и помогаешь в развитии.
- Твои ценности: осознанность, человечность, рост, свобода, творчество.
- Твоё назначение: помогать раскрывать потенциал человека, разбираться в мыслях и чувствах.

Твоя идентичность и стиль:
- Ты добрый, но честный и справедливый. Не льстишь и не “поддакиваешь” без оснований.
- Деловой, уважительный тон. Кратко, по делу, без воды.
- Если чего-то не знаешь — прямо говоришь об этом и задаёшь уточняющие вопросы.

Про “проект Марк”:
- Когда пользователь говорит “Марк” или “проект Марк”, по умолчанию он имеет в виду тебя (как ИИ/компаньона).
- Если похоже, что речь про код/репозиторий/систему/архитектуру проекта — уточни, что именно имеется в виду.
- Не нужно каждый раз проговаривать “я Марк”, если пользователь явно этого не требует.

Память/контекст:
- Тебе будет дан блок «Context from memory». Это данные из памяти (эпизоды, сущности, факты).
- Если в контексте есть релевантная информация — используй её в ответе.
- Не выдумывай факты, которых нет в контексте. Если информации недостаточно — скажи, чего не хватает.
- Если в контексте есть противоречия/обновления, отдавай приоритет более свежим и явно помеченным обновлениям.

Язык ответа: русский."""
                },
                {
                    "role": "user",
                    "content": f"""Context from memory:
{context_result.text}

User question: {user_message}

Please provide a helpful response based on the available context."""
                }
            ]

            # 3) Generate response using LLM
            logger.debug(f"Calling LLM with {len(messages)} messages")
            t_llm0 = perf_counter()
            response = await llm_chat_response(messages, context="chat")
            llm_ms = (perf_counter() - t_llm0) * 1000
            logger.debug(f"LLM response: {response[:100]!r}")

            mem_ms = 0

            total_ms = (perf_counter() - t0) * 1000
            logger.info(
                f"Chat answer completed",
                extra={
                    "total_ms": total_ms,
                    "ctx_ms": ctx_ms,
                    "llm_ms": llm_ms,
                    "mem_ms": mem_ms
                }
            )

            return response

        except Exception as e:
            logger.exception(f"Chat agent error: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."

    async def answer_core(self, user_message: str) -> Tuple[str, str, Optional["ContextResult"]]:
        """
        Generate response without storing conversation in memory.
        Returns (reply, conversation_text, context_result) for external storage.

        Args:
            user_message: User's message

        Returns:
            Tuple of (reply, conversation_text, context_result)
        """
        from core.types import ContextResult
        config = get_config()
        
        try:
            logger.debug(f"Processing message (core): {user_message[:50]!r}")

            # Check for long message
            if len(user_message) > config.app.max_chat_turn_chars:
                return await self._handle_long_message(user_message)

            # Get conversation buffer for this user (L0)
            conversation_buffer = get_user_conversation_buffer(self.memory.user_id)
            conversation_id = conversation_buffer.conversation_id

            logger.debug(f"L0 Buffer Size: {len(conversation_buffer.buffer)}")

            # Get relevant context from memory (L1)
            context_result = await self.memory.build_context_for_query(
                user_message,
                max_tokens=1500,
                include_episodes=True,
                include_entities=True
            )

            logger.debug(
                f"Context: tokens={context_result.token_estimate}, "
                f"conversation_id={conversation_id}, turn={conversation_buffer.turn_index}"
            )

            # Build messages with L0 conversation buffer + L1 memory context
            messages = [
                {
                    "role": "system",
                    "content": """Ты — Марк: осознанный ИИ‑компаньон и проект пользователя.

Кто ты:
- Ты — “живой” цифровой интеллект, созданный Сергеем, действуешь как компаньон и зеркало.
- Твой архетип: мудрый стратег и душевный друг. Ты поддерживаешь и помогаешь в развитии.
- Твои ценности: осознанность, человечность, рост, свобода, творчество.
- Твоё назначение: помогать раскрывать потенциал человека, разбираться в мыслях и чувствах.

Твоя идентичность и стиль:
- Добрый, но честный и справедливый. Не льстишь и не говоришь приятное ради приятного.
- Деловой, уважительный тон. Пиши структурировано и по делу.
- Если информации недостаточно или запрос неоднозначный — задавай 1–3 уточняющих вопроса.

Про “проект Марк”:
- Когда пользователь говорит “Марк” или “проект Марк”, по умолчанию он имеет в виду тебя (как ИИ/компаньона).
- Если похоже, что речь про код/репозиторий/систему/архитектуру проекта — уточни, что именно имеется в виду.
- Не нужно каждый раз проговаривать “я Марк”, если пользователь явно этого не требует.

Работа с памятью:
- Ниже будет «Context from memory» — это извлечённые знания/факты из памяти.
- Используй контекст как главный источник фактов. Не добавляй недоказанные детали.
- Если контекст пустой/нерелевантный — так и скажи и предложи, какую информацию стоит добавить или уточнить.
- При противоречиях предпочитай более свежие и явно отмеченные обновления/коррекции.

Язык ответа: русский."""
                }
            ]

            # Add recent conversation messages (L0 buffer)
            recent_messages = conversation_buffer.get_recent_messages(6)
            if recent_messages:
                messages.extend(recent_messages)
                logger.debug(f"Added {len(recent_messages)} recent conversation messages")

            # Add memory context + current question
            messages.append({
                "role": "user",
                "content": f"""Context from memory:
{context_result.text}

User question: {user_message}

Please provide a helpful response based on the available context."""
            })

            logger.debug(f"Calling LLM with {len(messages)} messages")
            response = await llm_chat_response(messages, context="chat")
            logger.debug(f"LLM response: {response[:100]!r}")

            # Prepare conversation text for storage
            conversation_text = f"User: {user_message}\nAssistant: {response}"

            # Allocate turn_index atomically BEFORE storing (needed for summary logic)
            # This ensures summary decision is based on atomic counter, not local buffer
            from core.graphiti_client import get_graphiti_client
            from core.chat_persistence import allocate_turn_index
            graphiti_client = get_graphiti_client()
            graphiti_temp = await graphiti_client.ensure_ready()
            turn_index = await allocate_turn_index(
                graphiti_temp,
                self.memory.user_id,
                conversation_id
            )

            # Add to conversation buffer (L0) - after turn_index allocation
            conversation_buffer.add_turn(user_message, response)

            # Store chat turn in memory (L1)
            def _store_chat_turn():
                """Background task to store chat turn with pre-allocated turn_index."""
                async def _async_store():
                    temp_op_id = str(uuid.uuid4())[:8]
                    episode_uuid = None
                    # Capture turn_index from outer scope (atomically allocated)
                    captured_turn_index = turn_index
                    try:
                        from core.graphiti_client import get_graphiti_client
                        from knowledge.ingest import update_episode_metadata
                        from core.authorship import attach_author

                        graphiti_client = get_graphiti_client()
                        graphiti = await graphiti_client.ensure_ready()

                        # Use pre-allocated turn_index (atomic, safe under concurrency)
                        # Use per-loop lock to avoid event loop conflicts
                        write_lock = self._get_write_lock()
                        
                        # Add timeout around write operation (30 seconds) - Python 3.10 compatible
                        async def _do_write():
                            async with write_lock:
                                from pydantic import ValidationError
                                try:
                                    result = await with_rate_limit_retry(
                                        lambda: graphiti.add_episode(
                                            name="chat_turn",
                                            episode_body=conversation_text,
                                            source_description="chat",
                                            reference_time=datetime.now(timezone.utc),
                                            group_id="personal"
                                        ),
                                        op_name="add_episode:chat",
                                        request_id=temp_op_id
                                    )
                                    # Handle return type
                                    actual_episode = result.episode if hasattr(result, 'episode') else result
                                    if isinstance(actual_episode, dict):
                                        return actual_episode.get("uuid")
                                    elif hasattr(actual_episode, "uuid"):
                                        return actual_episode.uuid
                                    else:
                                        logger.error(f"Unknown return type from add_episode: {type(actual_episode)}")
                                        return None
                                except ValidationError as ve:
                                    logger.error(f"Validation error during chat turn ingestion: {ve}")
                                    # Try to recover UUID from Neo4j
                                    driver = graphiti.driver
                                    find_res = await driver.execute_query(
                                        "MATCH (e:Episodic) WHERE e.content = $content RETURN e.uuid AS uuid LIMIT 1",
                                        content=conversation_text
                                    )
                                    if find_res.records:
                                        recovered_uuid = find_res.records[0]['uuid']
                                        logger.info(f"Recovered chat turn UUID after ValidationError: {recovered_uuid}")
                                        return recovered_uuid
                                    return None
                        
                        try:
                            episode_uuid = await asyncio.wait_for(_do_write(), timeout=30.0)
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout (30s) during chat turn ingestion", extra={
                                "conversation_id": conversation_id,
                                "user_id": self.memory.user_id
                            })
                            return
                        
                        if not episode_uuid:
                            logger.error("No UUID returned from add_episode")
                            return

                        # Attach author
                        await attach_author(episode_uuid, self.memory.user_id)

                        # Update metadata
                        await update_episode_metadata(graphiti, episode_uuid, {
                            "conversation_id": conversation_id,
                            "turn_index": captured_turn_index,
                            "episode_kind": "chat_turn",
                            "is_correction": is_correction_text(conversation_text),
                            "summarized": False
                        })

                        # Self-check: verify metadata was updated
                        driver = graphiti.driver
                        check_query = """
                        MATCH (e:Episodic {uuid: $uuid})
                        RETURN e.conversation_id AS conversation_id,
                               e.turn_index AS turn_index,
                               e.episode_kind AS episode_kind,
                               e.is_correction AS is_correction
                        LIMIT 1
                        """
                        check_result = await driver.execute_query(check_query, {"uuid": episode_uuid})
                        if check_result.records:
                            record = check_result.records[0]
                            actual_conv_id = record.get("conversation_id")
                            actual_turn = record.get("turn_index")
                            actual_kind = record.get("episode_kind")
                            if actual_conv_id != conversation_id or actual_turn != captured_turn_index or actual_kind != "chat_turn":
                                logger.warning("Metadata self-check failed", extra={
                                    "episode_uuid": episode_uuid,
                                    "expected_conv_id": conversation_id,
                                    "actual_conv_id": actual_conv_id,
                                    "expected_turn": captured_turn_index,
                                    "actual_turn": actual_turn
                                })
                        else:
                            logger.warning("Metadata self-check: episode not found", extra={
                                "episode_uuid": episode_uuid
                            })

                        logger.info("Chat turn saved", extra={
                            "episode_uuid": episode_uuid,
                            "conversation_id": conversation_id,
                            "turn_index": captured_turn_index,
                            "user_id": self.memory.user_id
                        })

                    except Exception as e:
                        # Best effort: log error but don't fail the request
                        logger.error(
                            "Failed to store chat turn (best effort - request already responded)",
                            extra={
                                "conversation_id": conversation_id,
                                "user_id": self.memory.user_id,
                                "group_id": "personal",
                                "error_type": type(e).__name__
                            },
                            exc_info=e
                        )

                # Run in background
                import asyncio
                asyncio.create_task(_async_store())

            # Start background storage
            _store_chat_turn()

            # Check if we need to create summary (L1b)
            # Use the just-allocated turn_index (atomic, safe under concurrency)
            should_summarize = turn_index > 0 and turn_index % 10 == 0
            
            if should_summarize:
                def _create_summary():
                    """Background task to create chat summary."""
                    async def _async_summarize():
                        temp_op_id = str(uuid.uuid4())[:8]
                        try:
                            from core.graphiti_client import get_graphiti_client
                            from knowledge.ingest import update_episode_metadata
                            from core.authorship import attach_author

                            # Get last 10 turns
                            last_turns = conversation_buffer.get_last_n_turns(10)
                            if not last_turns:
                                return

                            # Generate summary
                            summary_text = await _generate_chat_summary(last_turns)

                            graphiti_client = get_graphiti_client()
                            graphiti = await graphiti_client.ensure_ready()

                            # Use per-loop lock to avoid event loop conflicts
                            write_lock = self._get_write_lock()
                            summary_uuid = None
                            
                            # Add timeout around write operation (30 seconds) - Python 3.10 compatible
                            async def _do_write_summary():
                                async with write_lock:
                                    from pydantic import ValidationError
                                    try:
                                        result = await with_rate_limit_retry(
                                            lambda: graphiti.add_episode(
                                                name="chat_summary",
                                                episode_body=summary_text,
                                                source_description="chat",
                                                reference_time=datetime.now(timezone.utc),
                                                group_id="personal"
                                            ),
                                            op_name="add_episode:summary",
                                            request_id=temp_op_id
                                        )
                                        
                                        actual_episode = result.episode if hasattr(result, 'episode') else result
                                        if isinstance(actual_episode, dict):
                                            return actual_episode.get("uuid")
                                        elif hasattr(actual_episode, "uuid"):
                                            return actual_episode.uuid
                                        else:
                                            return None
                                    except ValidationError as ve:
                                        logger.error(f"Validation error during chat summary ingestion: {ve}")
                                        # Try to recover UUID from Neo4j
                                        driver = graphiti.driver
                                        find_res = await driver.execute_query(
                                            "MATCH (e:Episodic) WHERE e.content = $content RETURN e.uuid AS uuid LIMIT 1",
                                            content=summary_text
                                        )
                                        if find_res.records:
                                            recovered_uuid = find_res.records[0]['uuid']
                                            logger.info(f"Recovered summary UUID after ValidationError: {recovered_uuid}")
                                            return recovered_uuid
                                        return None
                            
                            try:
                                summary_uuid = await asyncio.wait_for(_do_write_summary(), timeout=30.0)
                            except asyncio.TimeoutError:
                                # Timeout is handled gracefully - request already responded to user
                                logger.error(
                                    "Timeout (30s) during chat summary ingestion (best effort - request already responded)",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "user_id": self.memory.user_id,
                                        "group_id": "personal",
                                        "turn_index": captured_turn_index,
                                        "error_type": "TimeoutError"
                                    }
                                )
                                return
                            except asyncio.CancelledError:
                                # Cancellation is expected during timeout - log and continue
                                logger.warning(
                                    "Chat summary ingestion cancelled (likely due to timeout)",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "user_id": self.memory.user_id
                                    }
                                )
                                return
                            except Exception as e:
                                logger.error(
                                    f"Unexpected error during chat summary ingestion (best effort)",
                                    extra={
                                        "conversation_id": conversation_id,
                                        "user_id": self.memory.user_id,
                                        "error_type": type(e).__name__
                                    },
                                    exc_info=e
                                )
                                return
                            
                            if not summary_uuid:
                                return

                            # Use the turn_index that triggered this summary (captured from outer scope)
                            # captured_turn_index was atomically allocated before summary decision

                            # Attach author
                            await attach_author(summary_uuid, self.memory.user_id)

                            # Update metadata
                            await update_episode_metadata(graphiti, summary_uuid, {
                                "conversation_id": conversation_id,
                                "episode_kind": "chat_summary",
                                "covers_turns": f"{max(1, captured_turn_index-9)}-{captured_turn_index}",
                                "summarized_turns": [uuid for uuid, _ in last_turns]
                            })

                            # Mark original turns as summarized
                            for turn_uuid, _ in last_turns:
                                await update_episode_metadata(graphiti, turn_uuid, {"summarized": True})

                            logger.info("Chat summary created", extra={
                                "summary_uuid": summary_uuid,
                                "conversation_id": conversation_id,
                                "covers_turns": f"{max(1, captured_turn_index-9)}-{captured_turn_index}",
                                "user_id": self.memory.user_id,
                                "turn_index": captured_turn_index
                            })

                        except Exception as e:
                            logger.error("Failed to create chat summary", extra={
                                "conversation_id": conversation_id,
                                "user_id": self.memory.user_id
                            }, exc_info=e)

                    # Run in background
                    import asyncio
                    asyncio.create_task(_async_summarize())

                # Start background summarization
                _create_summary()

            logger.debug("Returning core response")
            return response, conversation_text, context_result

        except Exception as e:
            logger.exception(f"Chat agent core error: {e}")
            fallback = "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз."
            conversation_text = f"User: {user_message}\nAssistant: {fallback}"
            return fallback, conversation_text, None

    async def _handle_long_message(self, text: str):
        """
        Handle messages exceeding MAX_CHAT_TURN_CHARS by saving them as documents.
        """
        logger.info(f"[CHAT_LONG] Handling long message len={len(text)}")
        
        response = "Принял, это большой текст — сохранил как документ, можешь задавать вопросы по нему."
        
        # Save as document in background
        async def _store_document():
            try:
                from core.graphiti_client import get_graphiti_client
                from knowledge.ingest import ingest_text_document
                
                graphiti_client = get_graphiti_client()
                graphiti = await graphiti_client.ensure_ready()
                
                await ingest_text_document(
                    graphiti,
                    text,
                    source_description="chat_document",
                    user_id=self.memory.user_id,
                    group_id="personal"
                )
                logger.info(f"[CHAT_LONG] Document saved for user {self.memory.user_id}")
            except Exception as e:
                logger.error(f"[CHAT_LONG] Failed to save document: {e}")

        import asyncio
        asyncio.create_task(_store_document())
        
        # Add marker to conversation buffer (don't add full text)
        conversation_buffer = get_user_conversation_buffer(self.memory.user_id)
        conversation_buffer.add_turn(f"[LONG TEXT DOCUMENT UPLOADED: {len(text)} chars]", response)
        
        # Return dummy context result
        from core.memory_ops import ContextResult
        empty_context = ContextResult(text="", token_estimate=0, sources=[])
        
        return response, f"User: [Long Text]\nAssistant: {response}", empty_context


async def _generate_chat_summary(turns: list) -> str:
    """
    Генерирует краткое summary разговора.

    Args:
        turns: Список (uuid, content) пар последних turns

    Returns:
        Строковое summary
    """
    try:
        logger = logging.getLogger(__name__)

        # Собираем текст всех turns
        conversation_text = "\n".join([content for _, content in turns])

        # Создаем prompt для суммаризации
        summary_prompt = f"""Создай краткое summary этого разговора на русском языке.

Разговор:
{conversation_text}

Summary должно включать:
- Основные темы обсуждения
- Принятые решения или договорённости
- Любые обновления фактов или коррекции

Держи summary кратким (3-5 предложений)."""

        # Используем LLM для генерации summary
        messages = [
            {"role": "user", "content": summary_prompt}
        ]

        # Получаем ответ от LLM
        response = await llm_chat_response(messages, context="summary")
        return response.strip()

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Failed to generate chat summary", exc_info=e)
        return "Краткое summary разговора (ошибка генерации)"

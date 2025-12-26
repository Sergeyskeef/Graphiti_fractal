import logging
import os
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_aclient = None

def get_async_client():
    global _aclient
    if _aclient is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, LLM calls will fail.")
            return None
        _aclient = AsyncOpenAI(api_key=api_key)
    return _aclient

def _select_model_for_context(context: str | None) -> str:
    """
    Select OpenAI chat model based on context.

    Priority:
    - <CONTEXT>_OPENAI_MODEL (e.g. CHAT_OPENAI_MODEL, SUMMARY_OPENAI_MODEL)
    - OPENAI_MODEL
    - default fallback
    """
    ctx = (context or "").strip().upper()
    if ctx:
        env_key = f"{ctx}_OPENAI_MODEL"
        val = (os.getenv(env_key) or "").strip()
        if val:
            return val
    return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

async def llm_summarize(text_list: list[str], context: str = "general") -> str:
    """
    Summarize a list of facts into a high-level abstraction using LLM.
    """
    client = get_async_client()
    if not client:
        return "LLM service unavailable due to missing API key."

    # Выбираем модель: SUMMARY_OPENAI_MODEL/GENERAL_OPENAI_MODEL/OPENAI_MODEL
    model = _select_model_for_context(context)

    joined_text = "\n- ".join(text_list)
    prompt = (
        f"You are a memory consolidation system for an AI agent.\n"
        f"Context: {context}\n\n"
        f"Below is a list of recent episodic memories and facts:\n"
        f"- {joined_text}\n\n"
        f"Task: Synthesize these details into 3-5 high-level abstract insights or patterns. "
        f"Ignore trivial details. Focus on what changed, what was decided, or what was learned.\n"
        f"Output format: Bullet points."
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error with model {model}: {e}")
        return f"Error generating summary: {e}"


async def llm_chat_response(messages: list[dict], context: str = "chat") -> str:
    """
    Generate a chat response using LLM with conversation context.

    Args:
        messages: List of message dicts with 'role' and 'content'
        context: Context description for the conversation

    Returns:
        LLM-generated response
    """
    client = get_async_client()
    if not client:
        return "LLM service unavailable due to missing API key."

    # Выбираем модель: CHAT_OPENAI_MODEL (для context='chat') → OPENAI_MODEL → fallback
    model = _select_model_for_context(context)

    try:
        # For maximum compatibility (e.g. GPT-5.*), avoid optional params like temperature/max_tokens.
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content
    except Exception as e:
        msg = str(e)
        logger.error(f"LLM chat error with model {model}: {msg}")
        return f"Извините, произошла ошибка при генерации ответа: {msg[:120]}"
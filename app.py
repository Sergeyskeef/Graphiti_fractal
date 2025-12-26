import asyncio
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import os
import sys
from pathlib import Path
from uuid import uuid4
import logging
import io

# Apply library patches
sys.path.append(str(Path(__file__).parent / "scripts"))
try:
    from apply_patches import apply_patches
    apply_patches()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core import Graphiti

# SimpleAgent removed - using bootstrap for initialization
from core.bootstrap import ensure_graphiti_ready
from core.config import get_config
from core.graphiti_client import get_graphiti_client
from core.conversation_buffer import clear_user_buffer
from core.memory_ops import clear_recent_memories
from core.identity import ensure_user_identity_entity
from core.text_utils import normalize_text, fingerprint
from api.jobs import (
    create_upload_job, 
    update_upload_job, 
    get_upload_job,
    cleanup_old_jobs
)
from experience import ExperienceIngestRequest
from experience.writer import ingest_experience
from experience.retrieval import get_success_patterns, get_antipatterns
from knowledge.retrieval import search_knowledge
from knowledge.ingest import remember_text, ingest_text_document, resolve_group_id

logger = logging.getLogger(__name__)


async def get_graphiti_dep() -> "Graphiti":
    """
    FastAPI dependency to get Graphiti instance.
    
    This allows tests to override the dependency with a per-test Graphiti instance
    to avoid event loop conflicts.
    """
    graphiti_client = get_graphiti_client()
    return await graphiti_client.ensure_ready()


async def run_ingest_job(job_id: str, content: str, source_description: str | None, memory_type: str = "knowledge", user_id: str = "sergey"):
    """Run file ingestion in background with progress tracking."""
    from api.jobs import update_upload_job, get_upload_job

    try:
        # Get graphiti instance
        graphiti_client = get_graphiti_client()
        graphiti = await graphiti_client.ensure_ready()

        # Mark ingest start time
        ingest_started_at = datetime.now(timezone.utc)
        job = get_upload_job(job_id)
        if job and "timing" in job:
            job["timing"]["ingest_started_at"] = ingest_started_at

        # Start ingestion
        update_upload_job(job_id, stage="ingest")

        group_id = resolve_group_id(memory_type)

        result = await ingest_text_document(
            graphiti,
            content,
            source_description=source_description,
            user_id=user_id,
            group_id=group_id,
            job_id=job_id
        )

        # Mark ingest finish time
        ingest_finished_at = datetime.now(timezone.utc)
        job = get_upload_job(job_id)
        if job and "timing" in job:
            job["timing"]["ingest_finished_at"] = ingest_finished_at

        # Check for warnings
        warnings = result.get("warnings", [])
        
        if warnings:
            update_upload_job(job_id, stage="done_with_warnings", warnings=warnings)
        else:
            update_upload_job(job_id, stage="done")

        # Calculate and log final timing profile
        job = get_upload_job(job_id)
        if job:
            timing = job.get("timing", {})
            ingest_finished = timing.get("ingest_finished_at", ingest_finished_at)
            upload_started = timing.get("upload_request_started_at") or timing.get("job_created_at")
            ingest_started = timing.get("ingest_started_at") or ingest_finished

            if upload_started and ingest_finished:
                wall_total = (ingest_finished - upload_started).total_seconds()
                ingest_total = (ingest_finished - ingest_started).total_seconds()

                per_chunk = timing.get("per_chunk", [])
                N = len(per_chunk)
                if N > 0:
                    chunk_times = [chunk["total_time"] for chunk in per_chunk if not chunk.get("skipped", False)]
                    if chunk_times:
                        avg_chunk = sum(chunk_times) / len(chunk_times)
                        max_chunk = max(chunk_times)
                    else:
                        avg_chunk = max_chunk = 0
                else:
                    avg_chunk = max_chunk = 0

                logger.info(
                    f"Job {job_id} completed: wall={wall_total:.2f}s ingest={ingest_total:.2f}s "
                    f"chunks={N} avg_chunk={avg_chunk:.2f}s max_chunk={max_chunk:.2f}s"
                )

    except Exception as e:
        error_msg = f"Job failed: {type(e).__name__}: {e}"
        update_upload_job(job_id, stage="error", error=error_msg)
        logger.error(f"Job {job_id} error: {error_msg}")
        raise

from contextlib import asynccontextmanager

# Background tasks registry
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic
    if background_tasks:
        print(f"Waiting for {len(background_tasks)} background tasks to complete...")
        # Wait for tasks with timeout
        done, pending = await asyncio.wait(background_tasks, timeout=30)
        if pending:
            print(f"Cancelling {len(pending)} pending tasks after timeout.")
            for task in pending:
                task.cancel()
        print("Shutdown complete.")

app = FastAPI(
    title="Fractal Memory API",
    description="""
    Fractal Memory API provides endpoints for:
    - Chat with memory-enabled AI assistant
    - Knowledge ingestion and retrieval
    - Experience tracking and pattern recognition
    - Memory management and diagnostics
    """,
    version="2.0.0",
    lifespan=lifespan
)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialization state - using bootstrap instead of SimpleAgent
_init_lock = asyncio.Lock()
_initialized = False


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message to send to the assistant")
    user_id: str = Field(..., description="Unique user identifier")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str = Field(..., description="Assistant's response")
    duration_ms: float | None = Field(None, description="Total processing time in milliseconds")
    timing: dict | None = Field(None, description="Detailed timing breakdown")


class BufferClearRequest(BaseModel):
    """Request model for clearing conversation buffer."""
    user_id: str = Field(default="sergey", description="User ID whose buffer to clear")


class RememberRequest(BaseModel):
    """Request model for storing text in memory."""
    text: str = Field(..., description="Text content to remember")
    source_description: str | None = Field(None, description="Description of the source")
    memory_type: str | None = Field(None, description="Memory type: personal, project, knowledge, or experience")
    user_id: str = Field(..., description="User ID for memory association")


class DeleteRequest(BaseModel):
    """Request model for deleting a node."""
    uuid: str = Field(..., description="UUID of the node to delete")
    hard: bool = Field(default=False, description="If true, permanently delete; otherwise soft delete")


def _normalize_text(text: str) -> str:
    """Normalize text for fingerprinting. Use core.text_utils.normalize_text instead."""
    return normalize_text(text)


def _fingerprint(text: str) -> str:
    """Generate fingerprint. Use core.text_utils.fingerprint instead."""
    return fingerprint(text)


async def _episode_exists(graphiti, fp: str, content: str) -> bool:
    driver = graphiti.driver  # neo4j.AsyncDriver
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
    return len(res.records) > 0


async def _set_fingerprint(graphiti, fp: str, content: str):
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


async def _set_group_id(graphiti, content: str, group_id: str):
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


async def _link_user(graphiti, fp: str, user_id: str):
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


async def ensure_agent_ready():
    """
    Ensure Graphiti is initialized and user identity is seeded.
    Replaces old SimpleAgent.initialize() logic.
    """
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if not _initialized:
            # Initialize Graphiti
            await ensure_graphiti_ready()
            # Seed identity
            try:
                await ensure_user_identity_entity("sergey")
            except Exception as e:
                print(f"Identity seed failed: {e}")
            _initialized = True


@app.post("/transcribe", tags=["Chat"])
async def transcribe_audio(file: UploadFile = File(..., description="Audio file (webm/wav/mp3/etc)")):
    """
    Transcribe an audio blob (voice input) into text.
    Intended for the web UI: audio -> text inserted into the input field for editing.
    """
    from core.llm import get_async_client

    client = get_async_client()
    if not client:
        raise HTTPException(status_code=503, detail="LLM client not available (missing OPENAI_API_KEY)")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    # Basic safety limit (25MB)
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large")

    model = (os.getenv("WHISPER_MODEL") or os.getenv("TRANSCRIBE_MODEL") or "whisper-1").strip()

    audio = io.BytesIO(raw)
    audio.name = file.filename or "voice.webm"

    try:
        # openai>=1.x: audio.transcriptions.create(...)
        resp = await client.audio.transcriptions.create(
            model=model,
            file=audio,
            language="ru",
        )
        text = getattr(resp, "text", None) or ""
        return {"text": text}
    except Exception as e:
        logger.error(f"Transcription error model={model}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {type(e).__name__}: {str(e)[:120]}")


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Send a message to the AI assistant and receive a response.
    
    The assistant uses memory context to provide informed answers.
    Conversation is stored in memory for future context.
    """
    request_id = str(uuid4())[:8]

    logger.info(
        f"Chat request started",
        extra={
            "request_id": request_id,
            "user_id": req.user_id,
            "message_length": len(req.message),
            "message_preview": req.message[:100]
        }
    )

    t0 = perf_counter()

    try:
        config = get_config()
        # Get Graphiti via dependency (can be overridden in tests)
        graphiti = await get_graphiti_dep()

        # Create MemoryOps (can fail if Neo4j unavailable)
        from core.memory_ops import MemoryOps
        memory = MemoryOps(graphiti, req.user_id)

        # Create chat agent
        from simple_chat_agent import SimpleChatAgent
        from core.llm import get_async_client
        llm_client = get_async_client()
        if not llm_client:
            raise HTTPException(status_code=500, detail="LLM client not available")

        agent = SimpleChatAgent(llm_client, memory)

        # Generate response (without storing in memory)
        t1 = perf_counter()
        reply, conversation_text, context_result = await agent.answer_core(req.message)
        t2 = perf_counter()

        # Return response immediately to user
        duration_ms = (perf_counter() - t0) * 1000
        timing = {
            "answer_ms": (t2 - t1) * 1000,
            "total_ms": duration_ms,
        }

        # Store conversation in memory asynchronously (fire-and-forget)
        #
        # NOTE: SimpleChatAgent.answer_core() already stores chat_turn/chat_summary episodes in the graph.
        # This extra ingest path creates duplicates and can overwhelm episodic retrieval with chat logs.
        # Keep it behind a flag.
        def _store_conversation_done(task):
            """Callback для обработки завершения background task."""
            try:
                if task.exception():
                    logger.error(
                        f"Background conversation storage failed",
                        extra={"request_id": request_id, "user_id": req.user_id},
                        exc_info=task.exception()
                    )
                else:
                    store_ms = task.result()
                    logger.debug(
                        f"Conversation stored",
                        extra={"request_id": request_id, "store_ms": store_ms, "user_id": req.user_id}
                    )
            except Exception as e:
                logger.error(
                    f"Error in conversation storage callback",
                    extra={"request_id": request_id, "user_id": req.user_id},
                    exc_info=e
                )

        async def _store_conversation():
            try:
                t_store0 = perf_counter()
                await memory.remember_text(
                    conversation_text,
                    memory_type="personal",
                    source_description="chat"
                )
                store_ms = (perf_counter() - t_store0) * 1000
                return store_ms
            except Exception as e:
                logger.error(f"Conversation storage failed",
                           extra={"request_id": request_id, "user_id": req.user_id}, exc_info=e)
                raise

        if config.memory.chat_save_episodes:
            # Start background task with callback
            task = asyncio.create_task(_store_conversation())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            task.add_done_callback(_store_conversation_done)

        duration_ms_total = (perf_counter() - t0) * 1000
        logger.info(
            f"Chat completed",
            extra={
                "request_id": request_id,
                "user_id": req.user_id,
                "duration_ms": duration_ms_total,
                "degraded": False
            }
        )

        return ChatResponse(reply=reply, duration_ms=duration_ms_total, timing=timing)

    except Exception as exc:
        logger.exception(
            f"Chat error - switching to degraded mode",
            extra={
                "request_id": request_id,
                "user_id": req.user_id,
                "error_type": type(exc).__name__
            }
        )

        # Degraded mode: try to get LLM response without memory
        try:
            from core.llm import get_async_client, llm_chat_response
            llm_client = get_async_client()
            if llm_client:
                # Simple prompt without memory context
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer questions directly and helpfully."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {req.message}"
                    }
                ]

                t1 = perf_counter()
                reply = (await llm_chat_response(messages, context="chat")).strip()
                t2 = perf_counter()

                duration_ms = (perf_counter() - t0) * 1000
                timing = {
                    "answer_ms": (t2 - t1) * 1000,
                    "total_ms": duration_ms,
                    "degraded_mode": True,
                    "request_id": request_id
                }

                logger.info(
                    f"Chat degraded mode success",
                    extra={"request_id": request_id, "user_id": req.user_id, "duration_ms": duration_ms}
                )
                return ChatResponse(reply=reply, duration_ms=duration_ms, timing=timing)
            else:
                raise Exception("No LLM client available")

        except Exception as degraded_exc:
            logger.exception(
                f"Degraded mode also failed",
                extra={"request_id": request_id, "user_id": req.user_id}
            )

            logger.warning(
                f"Using static fallback response",
                extra={"request_id": request_id, "user_id": req.user_id}
            )

            reply = "Извините, в данный момент я работаю в ограниченном режиме. Попробуйте позже."
            duration_ms = (perf_counter() - t0) * 1000
            timing = {
                "total_ms": duration_ms,
                "fallback_mode": True,
                "request_id": request_id
            }

            return ChatResponse(reply=reply, duration_ms=duration_ms, timing=timing)


def _split_into_paragraphs(text: str, max_len: int = 1800, overlap: int = 200) -> list[str]:
    parts = []
    for block in text.split("\n\n"):
        blk = block.strip()
        if not blk:
            continue
        if len(blk) <= max_len:
            parts.append(blk)
            continue
        start = 0
        while start < len(blk):
            end = min(len(blk), start + max_len)
            parts.append(blk[start:end])
            if end == len(blk):
                break
            start = max(0, end - overlap)
    return parts


@app.post("/remember", tags=["Memory"])
async def remember(req: RememberRequest):
    """
    Store text in memory with automatic or explicit classification.
    
    Memory types:
    - personal: Facts about people, relationships
    - project: Technical project information
    - knowledge: General knowledge
    - experience: Work patterns and lessons learned
    """
    try:
        # Get Graphiti via dependency (can be overridden in tests)
        graphiti = await get_graphiti_dep()

        # Create MemoryOps with user from request
        from core.memory_ops import MemoryOps
        memory = MemoryOps(graphiti, req.user_id)

        source_desc = req.source_description or "user_chat"
        memory_type = req.memory_type

        return await memory.remember_text(
            req.text,
            memory_type=memory_type,
            source_description=source_desc
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/upload", tags=["Memory"])
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    source_description: str = Form("uploaded_file", description="Description of the source"),
    memory_type: str = Form("knowledge", description="Memory type for classification"),
    user_id: str = Form(..., description="User ID for memory association")
):
    """
    Upload a file for ingestion into memory.
    
    Returns a job_id that can be used to track progress via /upload/status/{job_id}.
    """
    from api.jobs import create_upload_job, get_upload_job
    
    await ensure_agent_ready()
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    upload_request_started_at = datetime.now(timezone.utc)

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
        q_ratio = content.count("?") / max(1, len(content))
        if q_ratio > 0.1:
            content = raw.decode("cp1251", errors="replace")
    except Exception:
        content = raw.decode("cp1251", errors="replace")

    logger.info(
        f"File upload received",
        extra={
            "filename": file.filename,
            "size_bytes": len(raw),
            "content_length": len(content)
        }
    )

    job_id = create_upload_job()
    job = get_upload_job(job_id)
    if job and "timing" in job:
        job["timing"]["upload_request_started_at"] = upload_request_started_at
    
    logger.debug(f"Created upload job {job_id}")

    task = asyncio.create_task(run_ingest_job(job_id, content, source_description, memory_type, user_id))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    return {"job_id": job_id}


@app.get("/upload/status/{job_id}", tags=["Memory"])
async def upload_status(job_id: str):
    """
    Get the status of an upload job.
    
    Stages: pending, ingest, rate_limited, done, done_with_warnings, error
    """
    status = get_upload_job(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return status


@app.post("/delete", tags=["Memory"])
async def delete_node(req: DeleteRequest):
    """
    Delete a node from the graph.
    
    Supports soft delete (marks as deleted) or hard delete (permanent removal).
    """
    await ensure_agent_ready()
    uuid = req.uuid
    hard = req.hard
    graphiti = await get_graphiti_dep()
    try:
        driver = graphiti.driver
        if hard:
            res = await driver.execute_query(
                "MATCH (n {uuid:$uuid}) DETACH DELETE n RETURN 1 AS done",
                uuid=uuid,
            )
            if res.records:
                return {"status": "ok", "deleted": True, "mode": "hard"}
            return {"status": "not_found", "deleted": False, "mode": "hard"}

        # soft delete
        res = await driver.execute_query(
            """
            MATCH (n {uuid:$uuid})
            SET n.deleted=true, n.deleted_at=$ts
            RETURN 1 AS done
            """,
            uuid=uuid,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        if res.records:
            return {"status": "ok", "deleted": True, "mode": "soft"}
        return {"status": "not_found", "deleted": False, "mode": "soft"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------
# Knowledge Memory (API)
# -----------------------


@app.post("/buffer/clear", tags=["Memory"])
async def clear_buffer(req: BufferClearRequest):
    """
    Clear in-memory conversation buffer and recent memories for a user.
    
    Useful for starting fresh conversations or debugging.
    """
    cleared_buffer = clear_user_buffer(req.user_id)
    cleared_recent = clear_recent_memories(req.user_id)
    
    logger.info(
        f"Buffer cleared",
        extra={
            "user_id": req.user_id,
            "cleared_buffer": cleared_buffer,
            "cleared_recent": cleared_recent
        }
    )
    
    return {
        "status": "ok",
        "user_id": req.user_id,
        "cleared": {
            "conversation_buffer": cleared_buffer,
            "recent_memories": cleared_recent
        }
    }


@app.post("/clear_memory", tags=["Memory"])
async def clear_memory():
    """
    Clear all memory (all nodes and relationships) from Neo4j.
    
    WARNING: This is destructive and cannot be undone.
    """
    await ensure_agent_ready()
    graphiti = await get_graphiti_dep()
    try:
        driver = graphiti.driver
        # Удаляем все узлы и связи
        result = await driver.execute_query(
            "MATCH (n) DETACH DELETE n RETURN count(n) AS deleted_count"
        )
        deleted = result.records[0]["deleted_count"] if result.records else 0
        return {"status": "ok", "deleted_nodes": deleted, "message": "Память полностью очищена"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/knowledge/search", tags=["Knowledge"])
async def knowledge_search(
    q: str,
    limit: int = 10,
    group_id: str | None = None,
    graphiti: "Graphiti" = Depends(get_graphiti_dep)
):
    """
    Search knowledge base using fulltext search.
    
    Returns matching entities and episodes.
    """
    try:
        items = await search_knowledge(graphiti, q, limit=limit, group_id=group_id)
        return {"items": items}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------
# Experience Memory (API)
# -----------------------


@app.post("/experience/ingest", tags=["Experience"])
async def experience_ingest(req: ExperienceIngestRequest):
    """
    Ingest experience data (task runs, tool calls, errors).
    
    Used for tracking work patterns and learning from past experiences.
    """
    await ensure_agent_ready()
    graphiti = await get_graphiti_dep()
    try:
        return await ingest_experience(graphiti, req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/experience/success", tags=["Experience"])
async def experience_success(
    task_type: str | None = None,
    context_hash: str | None = None,
    limit: int = 5
):
    """
    Retrieve successful patterns from experience memory.
    
    Returns patterns that led to successful outcomes.
    """
    await ensure_agent_ready()
    graphiti = await get_graphiti_dep()
    try:
        items = await get_success_patterns(
            graphiti, task_type=task_type, context_hash=context_hash, limit=limit
        )
        return {"items": items}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/experience/antipatterns", tags=["Experience"])
async def experience_antipatterns(
    task_type: str | None = None,
    context_hash: str | None = None,
    limit: int = 5
):
    """
    Retrieve antipatterns from experience memory.
    
    Returns patterns that led to failures or problems.
    """
    await ensure_agent_ready()
    graphiti = await get_graphiti_dep()
    try:
        items = await get_antipatterns(
            graphiti, task_type=task_type, context_hash=context_hash, limit=limit
        )
        return {"items": items}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", tags=["System"])
async def health_check():
    """
    Check system health including Neo4j connectivity.
    
    Returns health status and connection state.
    """
    try:
        # Get Graphiti via dependency (can be overridden in tests)
        graphiti = await get_graphiti_dep()

        # Simple Neo4j query to check connectivity
        driver = graphiti.driver
        result = await driver.execute_query("RETURN 'health_check' AS status LIMIT 1")

        if result.records and result.records[0]["status"] == "health_check":
            return {"status": "healthy", "neo4j": "connected"}
        else:
            return {"status": "unhealthy", "neo4j": "query_failed"}

    except Exception as e:
        return {"status": "unhealthy", "neo4j": "disconnected", "error": str(e)}


# -----------------------
# Memory Conflict Diagnostics
# -----------------------


@app.get("/diagnostics/memory-conflicts", tags=["Diagnostics"])
async def diagnose_memory_conflicts(
    entity_name: str = "Лена",
    limit: int = 20
):
    """
    Diagnose memory conflicts for a given entity.
    
    Finds conflicting information and correction episodes.
    """
    await ensure_agent_ready()
    graphiti = await get_graphiti_dep()
    try:
        import logging
        logger = logging.getLogger(__name__)

        driver = graphiti.driver

        # 1) Все сущности с именем, содержащим entity_name
        entities_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($entity_name)
        OPTIONAL MATCH (e)<-[m:MENTIONS]-(ep:Episodic)
        RETURN e.name AS entity_name, e.summary AS entity_summary,
               collect({
                   ep_id: id(ep),
                   content: substring(ep.content, 0, 200),
                   created_at: ep.created_at,
                   group_id: ep.group_id,
                   source_description: ep.source_description
               }) AS episodes
        ORDER BY size(episodes) DESC
        LIMIT $limit
        """

        entities_result = await driver.execute_query(entities_query, entity_name=entity_name, limit=limit)

        # 2) Специальный поиск по конфликтным текстам
        conflict_keywords = ["контент", "не занимается", "ошибка", "раньше", "на самом деле", "правильно"]
        conflict_episodes = []

        for keyword in conflict_keywords:
            keyword_query = """
            MATCH (ep:Episodic)
            WHERE toLower(ep.content) CONTAINS toLower($keyword)
            RETURN ep.uuid AS uuid, ep.content AS content, ep.created_at AS created_at,
                   ep.group_id AS group_id, ep.source_description AS source_description
            ORDER BY ep.created_at DESC
            LIMIT 10
            """
            keyword_result = await driver.execute_query(keyword_query, keyword=keyword)
            for record in keyword_result.records:
                conflict_episodes.append({
                    "keyword": keyword,
                    "uuid": record["ep.uuid"],
                    "content": record["ep.content"][:300],
                    "created_at": record["ep.created_at"],
                    "group_id": record["ep.group_id"],
                    "source_description": record["ep.source_description"]
                })

        # 3) Логируем результаты
        logger.info(f"[MEMORY_CONFLICT_DIAG] Entity='{entity_name}' found {len(entities_result.records)} entities")

        for record in entities_result.records:
            entity_name_found = record["entity_name"]
            episodes_count = len(record["episodes"])
            logger.info(f"[MEMORY_CONFLICT_DIAG] Entity '{entity_name_found}': {episodes_count} episodes")

            # Логируем эпизоды
            for ep in record["episodes"][:5]:  # первые 5
                logger.info(f"[MEMORY_CONFLICT_DIAG]   {ep['created_at']} {ep['group_id']}: {ep['content'][:100]}...")

        # Логируем конфликтные эпизоды
        logger.info(f"[MEMORY_CONFLICT_DIAG] Found {len(conflict_episodes)} episodes with conflict keywords")
        for ep in conflict_episodes[:10]:  # первые 10
            logger.info(f"[MEMORY_CONFLICT_DIAG]   [{ep['keyword']}] {ep['created_at']} {ep['group_id']}: {ep['content'][:120]}...")

        return {
            "entity_name": entity_name,
            "entities_found": [
                {
                    "name": record["entity_name"],
                    "summary": record["entity_summary"],
                    "episodes_count": len(record["episodes"]),
                    "episodes": record["episodes"]
                }
                for record in entities_result.records
            ],
            "conflict_episodes": conflict_episodes,
            "conflict_keywords": conflict_keywords
        }

    except Exception as exc:
        logger.error(f"Memory conflict diagnostics error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


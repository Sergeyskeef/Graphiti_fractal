from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any
from pathlib import Path

# Apply library patches
try:
    # Adjust path to find scripts folder relative to mcp_server
    sys.path.append(str(Path(__file__).parents[1] / "scripts"))
    from apply_patches import apply_patches
    apply_patches()
except ImportError:
    pass

from core import get_graphiti_client
from core.memory_ops import MemoryOps
from experience.retrieval import get_antipatterns, get_success_patterns
from knowledge.ingest import ingest_text_document, remember_text
from knowledge.retrieval import search_knowledge


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]


TOOLS: list[Tool] = [
    Tool(
        name="memory.search_knowledge",
        description="Поиск по Knowledge Memory (fulltext, без LLM).",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                "group_id": {"type": ["string", "null"], "default": None},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory.search_experience",
        description="Поиск по Experience Memory: success patterns или antipatterns.",
        input_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["success", "antipatterns"], "default": "success"},
                "task_type": {"type": ["string", "null"], "default": None},
                "context_hash": {"type": ["string", "null"], "default": None},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
            },
        },
    ),
    Tool(
        name="memory.remember",
        description="Добавить текст в Knowledge Memory (dedupe + fingerprint).",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_description": {"type": "string", "default": "user_chat"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="memory.upload",
        description="Загрузить текстовый документ в Knowledge Memory (chunking + dedupe).",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_description": {"type": "string", "default": "uploaded_text"},
                "max_len": {"type": "integer", "default": 1800, "minimum": 200, "maximum": 8000},
                "overlap": {"type": "integer", "default": 200, "minimum": 0, "maximum": 2000},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="memory.delete",
        description="Удалить узел по uuid (soft-delete по умолчанию, hard при hard=true).",
        input_schema={
            "type": "object",
            "properties": {
                "uuid": {"type": "string"},
                "hard": {"type": "boolean", "default": False},
            },
            "required": ["uuid"],
        },
    ),
]

_framing_mode: str | None = None  # "lsp" (Content-Length) or "ndjson" (one JSON per line)
_should_exit = False

def _write(msg: dict[str, Any]) -> None:
    """
    Write a JSON-RPC message back to the client.

    MCP clients vary in stdio framing:
    - Some use LSP-style `Content-Length` headers (like Language Server Protocol).
    - Others use NDJSON: one JSON object per line.

    We respond using the framing mode detected from the first successfully read message.
    """
    global _framing_mode
    mode = _framing_mode or "lsp"
    if mode == "ndjson":
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.buffer.flush()
        return

    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    """
    Read one JSON-RPC message from stdin.

    Supports two input framings:
    - LSP-style: `Content-Length: N` headers then a JSON payload of N bytes.
    - NDJSON: a single line containing a complete JSON object.
    """
    global _framing_mode

    while True:
        # Skip leading empty lines
        while True:
            first = sys.stdin.buffer.readline()
            if not first:
                return None
            if first.strip():
                break

        first_stripped = first.strip()
        lower = first_stripped.lower()

        # Heuristic: if the first non-empty line starts with "content-length:" (or header-like),
        # treat it as LSP-style framing.
        is_header = lower.startswith(b"content-length:") or (
            b":" in first_stripped and not first_stripped.startswith((b"{", b"["))
        )
        if is_header:
            headers: dict[str, str] = {}

            def _consume_header_line(raw_line: bytes) -> None:
                s = raw_line.decode("ascii", errors="ignore").strip()
                if ":" in s:
                    k, v = s.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            _consume_header_line(first)
            while True:
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                if not line.strip():
                    break
                _consume_header_line(line)

            length = int(headers.get("content-length", "0"))
            if length <= 0:
                # Malformed header block; keep reading.
                continue
            raw = sys.stdin.buffer.read(length)
            _framing_mode = "lsp"
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                # Malformed payload; keep reading.
                continue

        # Otherwise treat it as NDJSON: one JSON object per line.
        _framing_mode = _framing_mode or "ndjson"
        try:
            return json.loads(first_stripped.decode("utf-8"))
        except json.JSONDecodeError:
            # Ignore non-JSON lines and continue reading.
            continue


_graphiti = None


async def _get_graphiti():
    global _graphiti
    if _graphiti is not None:
        return _graphiti
    client = get_graphiti_client()
    _graphiti = await client.ensure_ready()
    return _graphiti


async def _tool_call(name: str, args: dict[str, Any]) -> Any:
    graphiti = await _get_graphiti()

    if name == "memory.remember_text":
        text = args.get("text", "")
        memory_type = args.get("memory_type")
        source_description = args.get("source_description")
        user_id = args.get("user_id", "mcp_user")

        memory = MemoryOps(graphiti, user_id)
        return await memory.remember_text(text, memory_type=memory_type, source_description=source_description)

    if name == "memory.search_memory":
        query = args.get("query", "")
        limit = args.get("limit", 10)
        user_id = args.get("user_id", "mcp_user")

        memory = MemoryOps(graphiti, user_id)
        result = await memory.search_memory(query, limit=limit)

        # Convert to dict for JSON serialization
        return {
            "episodes": result.episodes,
            "entities": result.entities,
            "total_episodes": result.total_episodes,
            "total_entities": result.total_entities
        }

    if name == "memory.chat":
        message = args.get("message", "")
        user_id = args.get("user_id", "mcp_user")

        memory = MemoryOps(graphiti, user_id)

        # Create chat agent
        from simple_chat_agent import SimpleChatAgent
        from core.llm import get_async_client
        llm_client = get_async_client()
        if not llm_client:
            raise ValueError("LLM client not available")

        agent = SimpleChatAgent(llm_client, memory)
        reply = await agent.answer(message)

        return {"reply": reply}

    if name == "memory.delete":
        uuid = args.get("uuid")
        hard = bool(args.get("hard", False))
        if not uuid:
            raise ValueError("uuid is required")
        driver = graphiti.driver
        if hard:
            res = await driver.execute_query(
                "MATCH (n {uuid:$uuid}) DETACH DELETE n RETURN 1 AS done",
                uuid=uuid,
            )
            return {"deleted": bool(res.records), "mode": "hard"}
        res = await driver.execute_query(
            """
            MATCH (n {uuid:$uuid})
            SET n.deleted=true, n.deleted_at=$ts
            RETURN 1 AS done
            """,
            uuid=uuid,
            ts=datetime_iso(),
        )
        return {"deleted": bool(res.records), "mode": "soft"}

    raise ValueError(f"Unknown tool: {name}")


def datetime_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _tools_list_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
        }
        for t in TOOLS
    ]


async def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method = msg.get("method")
    req_id = msg.get("id")

    # Notifications may not have id.
    if method == "initialize":
        # MCP initialize result should include protocolVersion, serverInfo, capabilities.
        # Cursor is strict about the shape of this payload.
        params = msg.get("params") or {}
        client_proto = params.get("protocolVersion")
        protocol_version = client_proto or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": protocol_version,
                "serverInfo": {"name": "fractal-memory-mcp", "version": "0.1.0"},
                # Capabilities are objects, not booleans.
                "capabilities": {
                    "tools": {},
                },
            },
        }

    if method == "initialized":
        # Notification from client after initialize; no response required.
        return None

    if method == "shutdown":
        return {"jsonrpc": "2.0", "id": req_id, "result": None}

    if method == "exit":
        # Usually a notification (no id). Signal run loop to exit.
        global _should_exit
        _should_exit = True
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tools_list_payload()}}

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            out = await _tool_call(name, args)
            # MCP tool result expects `content` as an array of content items.
            # We return JSON as text to remain compatible with Cursor.
            text = json.dumps(out, ensure_ascii=False, indent=2) if not isinstance(out, str) else out
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    # Basic ping
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}

    # Unknown method
    if req_id is not None:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
    return None


def run() -> None:
    async def _amain():
        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(None, _read_message)
            if msg is None:
                if _should_exit:
                    break
                break
            resp = await handle(msg)
            if resp is not None:
                _write(resp)

    asyncio.run(_amain())


if __name__ == "__main__":
    run()



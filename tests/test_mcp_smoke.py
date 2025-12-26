import os
import json
import subprocess
import sys

import pytest


def _frame(obj):
    b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return f"Content-Length: {len(b)}\r\n\r\n".encode("ascii") + b


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION", "0") not in {"1", "true", "yes"},
    reason="integration test (needs Neo4j) — set RUN_INTEGRATION=1",
)
def test_mcp_initialize_and_tools_list():
    p = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )
    try:
        p.stdin.write(_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        p.stdin.write(_frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        p.stdin.flush()

        # read 2 responses
        for _ in range(2):
            headers = {}
            while True:
                line = p.stdout.readline()
                assert line
                line = line.decode("ascii", "ignore").strip()
                if not line:
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.lower().strip()] = v.strip()
            n = int(headers.get("content-length", "0"))
            body = p.stdout.read(n)
            msg = json.loads(body.decode("utf-8"))
            assert msg.get("jsonrpc") == "2.0"
    finally:
        p.terminate()




import pytest
from fastapi.testclient import TestClient
from api import app
from core.conversation_buffer import _conversation_buffers, get_user_conversation_buffer
from core.memory_ops import _recent_memories
from collections import deque

client = TestClient(app)

def test_buffer_clear_endpoint():
    # 1. Setup - Populate buffers
    user_id = "test_user"
    
    # Populate ConversationBuffer
    buf = get_user_conversation_buffer(user_id)
    buf.add_message("user", "Hello")
    buf.add_message("assistant", "Hi")
    
    # Populate _recent_memories
    if user_id not in _recent_memories:
        _recent_memories[user_id] = deque()
    _recent_memories[user_id].append({"text": "Something"})
    
    assert user_id in _conversation_buffers
    assert len(_conversation_buffers[user_id].buffer) == 2
    assert user_id in _recent_memories
    assert len(_recent_memories[user_id]) == 1
    
    # 2. Call endpoint
    response = client.post("/buffer/clear", json={"user_id": user_id})
    
    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["user_id"] == user_id
    assert data["cleared"]["conversation_buffer"] == 2
    assert data["cleared"]["recent_memories"] == 1
    
    # 4. Verify internal state
    assert user_id not in _conversation_buffers
    assert user_id not in _recent_memories

def test_buffer_clear_empty():
    user_id = "empty_user"
    # Ensure empty
    if user_id in _conversation_buffers:
        del _conversation_buffers[user_id]
    if user_id in _recent_memories:
        del _recent_memories[user_id]
        
    response = client.post("/buffer/clear", json={"user_id": user_id})
    
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"]["conversation_buffer"] == 0
    assert data["cleared"]["recent_memories"] == 0

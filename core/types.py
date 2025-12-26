"""
Type definitions for the application.

Centralizes all TypedDict and dataclass definitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class MemoryType(str, Enum):
    """Memory layer types."""
    PERSONAL = "personal"
    PROJECT = "project"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"


class EpisodeKind(str, Enum):
    """Types of episodic memory entries."""
    CHAT_TURN = "chat_turn"
    CHAT_SUMMARY = "chat_summary"
    DOCUMENT = "document"
    EXPERIENCE = "experience"


class JobStage(str, Enum):
    """Upload job stages."""
    PENDING = "pending"
    INGEST = "ingest"
    RATE_LIMITED = "rate_limited"
    DONE = "done"
    DONE_WITH_WARNINGS = "done_with_warnings"
    ERROR = "error"


# ============================================================================
# TypedDicts for API responses
# ============================================================================

class EpisodeDict(TypedDict, total=False):
    """Episode data structure."""
    uuid: str
    content: str
    name: str
    score: float
    type: str
    group_id: str
    is_correction: bool
    episode_kind: str
    source_description: str
    created_at: Optional[str]
    is_expanded: bool
    hop: int


class EntityDict(TypedDict, total=False):
    """Entity data structure."""
    uuid: str
    name: str
    summary: str
    score: float
    type: str
    group_id: str
    is_expanded: bool
    hop: int


class EdgeDict(TypedDict, total=False):
    """Edge/relationship data structure."""
    uuid: str
    fact: str
    subject: Optional[str]
    object: Optional[str]
    relationship_type: Optional[str]
    name: Optional[str]
    score: float
    type: str
    group_id: str
    is_expanded: bool
    hop: int


class CommunityDict(TypedDict, total=False):
    """Community data structure."""
    uuid: str
    name: str
    summary: str
    score: float
    type: str
    group_id: str


class TimingProfile(TypedDict, total=False):
    """Timing profile for operations."""
    chunking_time: float
    embedding_calls: int
    embedding_time: float
    graph_time: float
    total_time: float


class JobTiming(TypedDict, total=False):
    """Job timing information."""
    job_created_at: Optional[datetime]
    upload_request_started_at: Optional[datetime]
    ingest_started_at: Optional[datetime]
    ingest_finished_at: Optional[datetime]
    per_chunk: List[Dict[str, Any]]


class UploadJobStatus(TypedDict, total=False):
    """Upload job status structure."""
    stage: str
    total_chunks: Optional[int]
    processed_chunks: int
    started_at: str
    error: Optional[str]
    warnings: List[str]
    profile: TimingProfile
    timing: JobTiming
    message: str
    retry_in_seconds: float
    attempt: int


class IngestResult(TypedDict):
    """Result of text ingestion."""
    status: str
    added: int
    chunks: int
    elapsed: float
    warnings: List[str]


class RememberResult(TypedDict, total=False):
    """Result of remember operation."""
    status: str
    added: int
    chunks: int
    elapsed: float
    warnings: List[str]
    skipped: bool
    reason: str


# ============================================================================
# Dataclasses for internal use
# ============================================================================

@dataclass
class SearchResult:
    """Structured result from memory search."""
    episodes: List[EpisodeDict] = field(default_factory=list)
    entities: List[EntityDict] = field(default_factory=list)
    edges: List[EdgeDict] = field(default_factory=list)
    communities: List[CommunityDict] = field(default_factory=list)
    total_episodes: int = 0
    total_entities: int = 0
    total_edges: int = 0
    total_communities: int = 0


@dataclass
class ContextResult:
    """Structured context for LLM queries."""
    text: str
    token_estimate: int
    sources: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """Single conversation message."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationTurn:
    """Complete conversation turn (user + assistant)."""
    user: str
    assistant: str
    turn_index: int = 0


@dataclass 
class CacheEntry:
    """Cache entry with TTL support."""
    value: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if entry has expired."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created) > timedelta(hours=ttl_hours)


@dataclass
class EmbeddingCacheEntry:
    """Embedding cache entry."""
    value: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if entry has expired."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created) > timedelta(hours=ttl_hours)


# ============================================================================
# Experience types
# ============================================================================

class ToolCallDict(TypedDict, total=False):
    """Tool call data structure."""
    tool: str
    command: Optional[str]
    args: Optional[str]
    exit_code: Optional[int]
    duration_ms: Optional[int]
    stdout: Optional[str]
    stderr: Optional[str]


class TestRunDict(TypedDict, total=False):
    """Test run data structure."""
    framework: str
    command: Optional[str]
    passed: bool
    duration_ms: Optional[int]
    summary: Optional[str]


class ErrorEventDict(TypedDict, total=False):
    """Error event data structure."""
    error_type: str
    message: Optional[str]
    stack: Optional[str]
    file: Optional[str]
    line: Optional[int]


# ============================================================================
# Graph types
# ============================================================================

class GraphNodeDict(TypedDict, total=False):
    """Generic graph node."""
    uuid: str
    name: str
    summary: str
    labels: List[str]
    group_id: str
    created_at: str
    deleted: bool


class GraphEdgeDict(TypedDict, total=False):
    """Generic graph edge."""
    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    relationship_type: str
    fact: str
    confidence: float

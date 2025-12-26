"""
Centralized Configuration Module

All application settings in one place with validation.
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse boolean from environment variable."""
    val = (os.getenv(name, default) or "").strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


class DatabaseSettings(BaseSettings):
    """Neo4j database configuration."""
    
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM and embedding configuration."""
    
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class CacheSettings(BaseSettings):
    """Cache configuration."""
    
    embedding_cache_max_size: int = Field(default=10000, alias="EMBEDDING_CACHE_MAX_SIZE")
    embedding_cache_ttl_hours: int = Field(default=168, alias="EMBEDDING_CACHE_TTL_HOURS")  # 7 days
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class MemorySettings(BaseSettings):
    """Memory layer configuration."""
    
    # Group IDs for memory layers
    personal_group_id: str = Field(default="personal", alias="PERSONAL_GROUP_ID")
    project_group_id: str = Field(default="project", alias="PROJECT_GROUP_ID")
    knowledge_group_id: str = Field(default="knowledge", alias="KNOWLEDGE_GROUP_ID")
    experience_group_id: str = Field(default="experience", alias="EXPERIENCE_GROUP_ID")
    
    # Chat behavior
    chat_save_episodes: bool = Field(default=False, alias="CHAT_SAVE_EPISODES")
    chat_save_bot_episodes: bool = Field(default=False, alias="CHAT_SAVE_BOT_EPISODES")
    chat_use_graphiti_search: bool = Field(default=False, alias="CHAT_USE_GRAPHITI_SEARCH")
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    def __init__(self, **kwargs):
        # Handle string to bool conversion from env
        for field in ["chat_save_episodes", "chat_save_bot_episodes", "chat_use_graphiti_search"]:
            env_name = field.upper()
            if env_name in os.environ:
                kwargs[field] = _env_flag(env_name)
        super().__init__(**kwargs)


class AppSettings(BaseSettings):
    """Application-level settings."""
    
    # Limits
    max_chat_turn_chars: int = Field(default=8000, alias="MAX_CHAT_TURN_CHARS")
    max_context_tokens: int = Field(default=4000, alias="MAX_CONTEXT_TOKENS")
    max_embedding_chars: int = Field(default=12000, alias="MAX_EMBEDDING_CHARS")
    
    # Concurrency
    write_semaphore_limit: int = Field(default=2, alias="WRITE_SEMAPHORE_LIMIT")
    
    # Conversation buffer
    conversation_buffer_max_messages: int = Field(default=12, alias="CONVERSATION_BUFFER_MAX_MESSAGES")
    recent_memories_max_size: int = Field(default=20, alias="RECENT_MEMORIES_MAX_SIZE")
    
    # Rate limiting
    rate_limit_max_attempts: int = Field(default=12, alias="RATE_LIMIT_MAX_ATTEMPTS")
    rate_limit_base_sleep: float = Field(default=2.0, alias="RATE_LIMIT_BASE_SLEEP")
    
    model_config = {"env_file": ".env", "extra": "ignore"}


class Config:
    """
    Main configuration class aggregating all settings.
    
    Usage:
        from core.config import get_config
        config = get_config()
        print(config.db.neo4j_uri)
        print(config.llm.openai_model)
    """
    
    def __init__(self):
        self.db = DatabaseSettings()
        self.llm = LLMSettings()
        self.cache = CacheSettings()
        self.memory = MemorySettings()
        self.app = AppSettings()
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.db.neo4j_uri:
            errors.append("NEO4J_URI is required")
        if not self.db.neo4j_user:
            errors.append("NEO4J_USER is required")
        if not self.db.neo4j_password:
            errors.append("NEO4J_PASSWORD is required")
        if not self.llm.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
            
        return errors


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get singleton configuration instance."""
    return Config()


# Backward compatibility with old settings module
class Settings:
    """
    Legacy settings class for backward compatibility.
    Deprecated: Use get_config() instead.
    """
    
    def __init__(self):
        config = get_config()
        
        # Chat policy
        self.CHAT_SAVE_EPISODES = config.memory.chat_save_episodes
        self.CHAT_SAVE_BOT_EPISODES = config.memory.chat_save_bot_episodes
        self.CHAT_USE_GRAPHITI_SEARCH = config.memory.chat_use_graphiti_search
        
        # Group IDs
        self.EXPERIENCE_GROUP_ID = config.memory.experience_group_id
        self.KNOWLEDGE_GROUP_ID = config.memory.knowledge_group_id
        self.PERSONAL_GROUP_ID = config.memory.personal_group_id
        self.PROJECT_GROUP_ID = config.memory.project_group_id


settings = Settings()

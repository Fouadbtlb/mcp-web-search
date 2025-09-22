import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Server Configuration
    SERVER_MODE: str = Field(default="stdio", description="Server mode: stdio or http")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8001, description="Server port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Search Configuration
    SEARXNG_URL: str = Field(default="https://searx.be", description="SearxNG instance URL")
    DUCKDUCKGO_FALLBACK: bool = Field(default=True, description="Enable DuckDuckGo fallback")
    MAX_RESULTS: int = Field(default=20, description="Maximum search results")
    TIMEOUT_SECONDS: int = Field(default=30, description="Request timeout")
    
    # AI Models Configuration
    OLLAMA_URL: str = Field(default="http://localhost:11434", description="Ollama server URL")
    EMBEDDING_MODEL: str = Field(default="nomic-ai/stella_en_1.5B_v5", description="Embedding model name")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker model name")
    EMBEDDING_DIMENSIONS: int = Field(default=1024, description="Embedding dimensions")
    RERANKER_MAX_LENGTH: int = Field(default=8192, description="Maximum reranker input length")
    ENABLE_SEMANTIC_RANKING: bool = Field(default=True, description="Enable AI-powered ranking")
    ENABLE_RERANKING: bool = Field(default=True, description="Enable reranking stage")
    
    # Content Extraction Configuration
    ENABLE_STRUCTURED_EXTRACTION: bool = Field(default=True, description="Enable structured data extraction")
    MARKDOWN_OPTIMIZATION: bool = Field(default=True, description="Enable LLM-optimized markdown")
    EXTRACT_METADATA: bool = Field(default=True, description="Extract rich metadata")
    JAVASCRIPT_ENABLED: bool = Field(default=True, description="Enable JavaScript rendering")
    STEALTH_MODE: bool = Field(default=True, description="Enable anti-detection measures")
    
    # Performance Configuration
    MAX_CONCURRENT_EXTRACTIONS: int = Field(default=5, description="Max concurrent content extractions")
    CONTENT_CACHE_TTL: int = Field(default=3600, description="Content cache TTL in seconds")
    
    # Caching Configuration
    CACHE_ENABLED: bool = Field(default=False, description="Enable result caching")
    REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URL")
    
    # Content Processing
    MIN_CONTENT_LENGTH: int = Field(default=100, description="Minimum content length")
    MAX_CONTENT_LENGTH: int = Field(default=50000, description="Maximum content length")
    EXCLUDE_TAGS: List[str] = Field(default_factory=lambda: ["script", "style", "nav", "footer", "aside"], description="HTML tags to exclude")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

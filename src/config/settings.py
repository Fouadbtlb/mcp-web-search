import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SearchEngineConfig:
    """Configuration pour un moteur de recherche"""
    name: str
    base_url: str
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheConfig:
    """Configuration du cache"""
    redis_url: str = os.getenv("REDIS_URL", "")
    redis_db: int = 0
    default_ttl: int = 3600  # 1 heure
    max_memory_cache_size: int = 1000
    enabled: bool = bool(os.getenv("REDIS_URL", ""))  # Enable if Redis URL is provided

@dataclass
class ContentExtractionConfig:
    """Configuration de l'extraction de contenu"""
    min_content_length: int = 100
    max_content_length: int = 50000
    use_readability: bool = True
    use_newspaper: bool = True
    use_playwright: bool = False  # Désactivé par défaut pour économiser les ressources
    playwright_timeout: int = 30000
    max_concurrent_fetches: int = 5

@dataclass
class QualityConfig:
    """Configuration du filtrage qualité"""
    min_quality_score: float = 0.3
    similarity_threshold: float = 0.85
    enable_spam_detection: bool = True
    enable_deduplication: bool = True
    high_quality_domains: List[str] = field(default_factory=lambda: [
        'wikipedia.org', 'github.com', 'stackoverflow.com',
        'reddit.com', 'medium.com', 'arxiv.org'
    ])
    low_quality_domains: List[str] = field(default_factory=lambda: [
        'spam.com', 'clickbait.net', 'fake-news.org'
    ])

@dataclass
class EmbeddingConfig:
    """Configuration for embedding system"""
    ollama_url: str = os.getenv("OLLAMA_URL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-minilm")
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "512"))
    use_ollama: bool = True  # Try Ollama first, fallback to CPU
    fallback_to_cpu: bool = True

@dataclass
class MCPConfig:
    """Configuration principale du serveur MCP"""
    # Serveur
    host: str = "0.0.0.0"
    port: int = int(os.getenv("MCP_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Recherche
    default_max_results: int = 10
    default_timeout: int = 30
    enable_query_expansion: bool = True
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = None
    
    # External services
    searxng_url: str = os.getenv("SEARXNG_URL", "https://searx.be")
    
    # Composants
    cache: CacheConfig = field(default_factory=CacheConfig)
    content_extraction: ContentExtractionConfig = field(default_factory=ContentExtractionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Moteurs de recherche
    search_engines: List[SearchEngineConfig] = field(default_factory=lambda: [
        SearchEngineConfig(
            name="searxng_local",
            base_url="http://localhost:8080",
            headers={'User-Agent': 'MCP-WebSearch/1.0'}
        ),
        SearchEngineConfig(
            name="searxng_public",
            base_url="https://searx.be",
            headers={'User-Agent': 'MCP-WebSearch/1.0'}
        ),
        SearchEngineConfig(
            name="duckduckgo",
            base_url="https://api.duckduckgo.com/",
            headers={'User-Agent': 'MCP-WebSearch/1.0'}
        )
    ])

class Settings:
    """Gestionnaire de configuration centralisé"""
    
    def __init__(self):
        self.config = MCPConfig()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Charge la configuration depuis les variables d'environnement"""
        
        # Configuration serveur
        if os.getenv('MCP_HOST'):
            self.config.host = os.getenv('MCP_HOST')
        
        if os.getenv('MCP_PORT'):
            try:
                self.config.port = int(os.getenv('MCP_PORT'))
            except ValueError:
                logger.warning("MCP_PORT invalide, utilisation de la valeur par défaut")
        
        if os.getenv('MCP_DEBUG'):
            self.config.debug = os.getenv('MCP_DEBUG').lower() in ('true', '1', 'yes')
        
        # Configuration cache
        if os.getenv('REDIS_URL'):
            self.config.cache.redis_url = os.getenv('REDIS_URL')
        
        if os.getenv('CACHE_TTL'):
            try:
                self.config.cache.default_ttl = int(os.getenv('CACHE_TTL'))
            except ValueError:
                logger.warning("CACHE_TTL invalide, utilisation de la valeur par défaut")
        
        if os.getenv('CACHE_ENABLED'):
            self.config.cache.enabled = os.getenv('CACHE_ENABLED').lower() in ('true', '1', 'yes')
        
        # Configuration extraction
        if os.getenv('USE_PLAYWRIGHT'):
            self.config.content_extraction.use_playwright = os.getenv('USE_PLAYWRIGHT').lower() in ('true', '1', 'yes')
        
        if os.getenv('MAX_CONTENT_LENGTH'):
            try:
                self.config.content_extraction.max_content_length = int(os.getenv('MAX_CONTENT_LENGTH'))
            except ValueError:
                logger.warning("MAX_CONTENT_LENGTH invalide, utilisation de la valeur par défaut")
        
        # Configuration qualité
        if os.getenv('MIN_QUALITY_SCORE'):
            try:
                self.config.quality.min_quality_score = float(os.getenv('MIN_QUALITY_SCORE'))
            except ValueError:
                logger.warning("MIN_QUALITY_SCORE invalide, utilisation de la valeur par défaut")
        
        # Configuration logging
        if os.getenv('LOG_LEVEL'):
            self.config.log_level = os.getenv('LOG_LEVEL').upper()
        
        if os.getenv('LOG_FILE'):
            self.config.log_file = os.getenv('LOG_FILE')
        
        # Configuration des moteurs de recherche
        self._load_search_engines_from_env()
    
    def _load_search_engines_from_env(self):
        """Charge la configuration des moteurs depuis l'environnement"""
        
        # SearxNG local
        if os.getenv('SEARXNG_LOCAL_URL'):
            for engine in self.config.search_engines:
                if engine.name == 'searxng_local':
                    engine.base_url = os.getenv('SEARXNG_LOCAL_URL')
                    break
        
        # Désactiver certains moteurs
        if os.getenv('DISABLE_SEARXNG_LOCAL'):
            for engine in self.config.search_engines:
                if engine.name == 'searxng_local':
                    engine.enabled = False
                    break
        
        if os.getenv('DISABLE_SEARXNG_PUBLIC'):
            for engine in self.config.search_engines:
                if engine.name == 'searxng_public':
                    engine.enabled = False
                    break
    
    def get_enabled_search_engines(self) -> List[SearchEngineConfig]:
        """Retourne la liste des moteurs de recherche activés"""
        return [engine for engine in self.config.search_engines if engine.enabled]
    
    def get_cache_config(self) -> CacheConfig:
        """Retourne la configuration du cache"""
        return self.config.cache
    
    def get_content_extraction_config(self) -> ContentExtractionConfig:
        """Retourne la configuration d'extraction de contenu"""
        return self.config.content_extraction
    
    def get_quality_config(self) -> QualityConfig:
        """Retourne la configuration de qualité"""
        return self.config.quality
    
    def setup_logging(self):
        """Configure le logging basé sur les paramètres"""
        log_level = getattr(logging, self.config.log_level, logging.INFO)
        
        # Format des logs
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Configuration de base
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Fichier de log si spécifié
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Ajouter aux loggers principaux
            loggers = ['__main__', 'src', 'mcp_web_search']
            for logger_name in loggers:
                logger = logging.getLogger(logger_name)
                logger.addHandler(file_handler)
        
        # Configurer les niveaux pour les bibliothèques externes
        external_loggers = {
            'httpx': logging.WARNING,
            'urllib3': logging.WARNING,
            'playwright': logging.WARNING,
            'aioredis': logging.WARNING,
            'bs4': logging.WARNING
        }
        
        for logger_name, level in external_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
    
    def validate_config(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        # Validation serveur
        if not (1 <= self.config.port <= 65535):
            errors.append(f"Port invalide: {self.config.port}")
        
        # Validation cache
        if self.config.cache.enabled:
            if not self.config.cache.redis_url.startswith('redis://'):
                errors.append(f"URL Redis invalide: {self.config.cache.redis_url}")
        
        # Validation extraction
        if self.config.content_extraction.min_content_length < 0:
            errors.append("min_content_length ne peut pas être négatif")
        
        if self.config.content_extraction.max_content_length < self.config.content_extraction.min_content_length:
            errors.append("max_content_length doit être >= min_content_length")
        
        # Validation qualité
        if not (0 <= self.config.quality.min_quality_score <= 1):
            errors.append("min_quality_score doit être entre 0 et 1")
        
        if not (0 <= self.config.quality.similarity_threshold <= 1):
            errors.append("similarity_threshold doit être entre 0 et 1")
        
        # Validation moteurs de recherche
        enabled_engines = self.get_enabled_search_engines()
        if not enabled_engines:
            errors.append("Aucun moteur de recherche activé")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration invalide: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire (pour debug)"""
        return {
            'server': {
                'host': self.config.host,
                'port': self.config.port,
                'debug': self.config.debug
            },
            'cache': {
                'enabled': self.config.cache.enabled,
                'redis_url': self.config.cache.redis_url,
                'default_ttl': self.config.cache.default_ttl
            },
            'content_extraction': {
                'use_playwright': self.config.content_extraction.use_playwright,
                'max_content_length': self.config.content_extraction.max_content_length,
                'max_concurrent_fetches': self.config.content_extraction.max_concurrent_fetches
            },
            'quality': {
                'min_quality_score': self.config.quality.min_quality_score,
                'enable_deduplication': self.config.quality.enable_deduplication
            },
            'search_engines': [
                {
                    'name': engine.name,
                    'base_url': engine.base_url,
                    'enabled': engine.enabled
                }
                for engine in self.config.search_engines
            ]
        }

# Instance globale
settings = Settings()

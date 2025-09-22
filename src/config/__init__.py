"""
Configuration centralisée du serveur MCP Web Search

Ce package contient:
- Settings: Gestionnaire de configuration avec support des variables d'environnement
- Dataclasses de configuration pour tous les composants
"""

from .settings import (
    settings,
    Settings,
    MCPConfig,
    SearchEngineConfig,
    CacheConfig,
    ContentExtractionConfig,
    QualityConfig,
)

__all__ = [
    "settings",
    "Settings", 
    "MCPConfig",
    "SearchEngineConfig",
    "CacheConfig",
    "ContentExtractionConfig",
    "QualityConfig",
]

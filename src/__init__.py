"""
MCP Web Search - Serveur MCP pour recherche web avec extraction de contenu

Ce package fournit un serveur MCP (Model Context Protocol) pour effectuer
des recherches web intelligentes avec extraction et filtrage de contenu.

Composants principaux:
- search: Modules de recherche et extraction
- utils: Utilitaires pour normalisation et qualité
- config: Configuration centralisée
"""

__version__ = "1.0.0"
__author__ = "MCP Web Search Team"
__email__ = "contact@example.com"

# Imports principaux pour faciliter l'utilisation
from .server import WebSearchMCP, main

__all__ = [
    "WebSearchMCP",
    "main",
    "__version__",
]

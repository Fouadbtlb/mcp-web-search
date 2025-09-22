"""
Modules de recherche web et extraction de contenu

Ce package contient tous les composants nécessaires pour:
- Agréger les résultats de plusieurs moteurs de recherche
- Récupérer le contenu des pages web
- Extraire le contenu principal des pages
- Gérer le cache des résultats
"""

from .aggregator import SearchAggregator
from .fetcher import ContentFetcher
from .extractor import ContentExtractor
from .cache import SearchCache

__all__ = [
    "SearchAggregator",
    "ContentFetcher", 
    "ContentExtractor",
    "SearchCache",
]

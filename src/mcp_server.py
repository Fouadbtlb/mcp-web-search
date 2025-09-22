#!/usr/bin/env python3
"""
MCP Web Search Server - Lightweight Implementation
Serveur MCP pour recherche web avec extraction de contenu
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# MCP Protocol imports
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .search.aggregator import SearchAggregator
from .search.fetcher import ContentFetcher
from .search.extractor import ContentExtractor
from .search.cache import SearchCache
from .search.embeddings import EmbeddingProcessor
from .utils.normalizer import QueryNormalizer
from .utils.quality import QualityFilter
from .utils.intent_classifier import QueryIntentClassifier
from .config.settings import settings

# Configuration du logging
settings.setup_logging()
logger = logging.getLogger(__name__)

class WebSearchMCPServer:
    """Serveur MCP pour recherche web avec extraction de contenu"""
    
    def __init__(self):
        # Initialisation des composants
        self.aggregator = SearchAggregator()
        self.fetcher = ContentFetcher()
        self.extractor = ContentExtractor()
        self.cache = SearchCache() if settings.config.cache.enabled else None
        self.normalizer = QueryNormalizer()
        self.quality_filter = QualityFilter()
        self.intent_classifier = QueryIntentClassifier()
        self.embeddings = EmbeddingProcessor()
        
        # Initialize MCP server
        self.server = Server("web-search")
        self._setup_handlers()
        
        # Initialize components
        self._initialized = False
        
        logger.info("Serveur MCP Web Search initialisé")
        if settings.config.debug:
            logger.debug(f"Configuration: {settings.to_dict()}")

    async def initialize(self):
        """Initialize async components"""
        if self._initialized:
            return
            
        try:
            await self.embeddings.initialize()
            self.intent_classifier.initialize()
            logger.info("Intent classifier initialized successfully")
            self._initialized = True
            logger.info("WebSearchMCP initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")

    def _setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="search_web",
                    description="Search the web with AI-powered semantic ranking and content extraction",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 10
                            },
                            "freshness": {
                                "type": "string",
                                "enum": ["day", "week", "month", "all"],
                                "description": "Result freshness filter",
                                "default": "all"
                            },
                            "require_full_fetch": {
                                "type": "boolean",
                                "description": "Extract full content from pages",
                                "default": False
                            },
                            "content_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Content types to extract",
                                "default": ["article"]
                            }
                        },
                        "required": ["q"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            if name == "search_web":
                try:
                    result = await self._search_web(arguments)
                    return [types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False)
                    )]
                except Exception as e:
                    logger.error(f"Search error: {e}", exc_info=True)
                    return [types.TextContent(
                        type="text", 
                        text=json.dumps({
                            "error": str(e),
                            "query": arguments.get("q", ""),
                            "results": []
                        }, indent=2)
                    )]
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_web(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search with semantic ranking"""
        
        # Ensure server is initialized
        if not self._initialized:
            await self.initialize()
        
        # Extract parameters
        query = args.get("q", "").strip()
        n_results = args.get("n_results", 10)
        freshness = args.get("freshness", "all")
        require_full_fetch = args.get("require_full_fetch", False)
        content_types = args.get("content_types", ["article"])
        
        if not query:
            return {
                "error": "Query is required",
                "query": query,
                "results": []
            }

        try:
            # Classification d'intention et stratégie
            intent, intent_confidence = self.intent_classifier.classify_intent(query)
            strategy = self.intent_classifier.get_search_strategy(intent)
            
            # Utiliser la stratégie pour optimiser la recherche
            sources = strategy.get("sources", ["searxng"])
            if strategy.get("require_full_fetch", False):
                require_full_fetch = True
            
            logger.info(f"Recherche: '{query}' (Intent: {intent}, Confidence: {intent_confidence:.2f})")
            
            # Normalisation et expansion de la requête
            normalized_query = self.normalizer.normalize(query)
            expanded_queries = self.normalizer.expand_hyde(normalized_query)
            
            logger.debug(f"Requête normalisée: '{normalized_query}'")
            logger.debug(f"Requêtes étendues: {expanded_queries}")
            
            # Vérification du cache (include intent in cache key)
            cache_key = f"{normalized_query}:{intent}:{n_results}:{freshness}"
            cached_results = None
            
            if self.cache:
                cached_results = await self.cache.get(cache_key)
                if cached_results:
                    logger.info(f"Cache hit pour: {query}")
                    return cached_results
            
            # Phase d'agrégation
            search_results = await self.aggregator.search(
                queries=expanded_queries,
                sources=sources,
                max_results=n_results * 2,  # Chercher plus pour avoir du choix
                freshness=freshness
            )
            
            if not search_results:
                logger.warning(f"Aucun résultat trouvé pour: {query}")
                return {
                    "query": query,
                    "results": [],
                    "trace": {
                        "error": "No results found", 
                        "sources_used": sources,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            
            logger.info(f"{len(search_results)} résultats bruts trouvés")
            
            # Filtrage et scoring des résultats
            scored_results = self.quality_filter.score_serp_results(search_results)
            
            # Récupération et extraction du contenu (parallèle)
            enriched_results = await self.fetch_and_extract_content(
                scored_results[:n_results],
                require_full_fetch,
                content_types,
                query  # Pass query for LLM processing
            )
            
            # Semantic ranking using embeddings
            if enriched_results and self._initialized:
                enriched_results = await self._semantic_rerank(query, enriched_results)
            
            # Déduplication et filtrage final
            final_results = self.quality_filter.deduplicate_and_filter(enriched_results, n_results)
            
            logger.info(f"Retour de {len(final_results)} résultats finaux pour: {query}")
            
            response = {
                "query": query,
                "results": final_results,
                "intent": {
                    "detected": intent,
                    "confidence": intent_confidence,
                    "strategy_used": strategy
                },
                "trace": {
                    "sources_used": sources,
                    "expanded_queries": expanded_queries,
                    "total_found": len(search_results),
                    "after_quality_filter": len(final_results),
                    "cache_used": cached_results is not None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Mise en cache
            if self.cache:
                await self.cache.set(cache_key, response, ttl=settings.config.cache.default_ttl)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}", exc_info=True)
            return {
                "error": f"Search failed: {str(e)}",
                "query": args.get("q", ""),
                "results": [],
                "trace": {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def fetch_and_extract_content(self, results: List[Dict], require_full_fetch: bool, content_types: List[str], query: str = "") -> List[Dict]:
        """Récupération et extraction de contenu en parallèle"""
        if not results:
            return []
        
        tasks = []
        max_concurrent = settings.config.content_extraction.max_concurrent_fetches
        
        for result in results:
            task = self._process_single_result(result, require_full_fetch, content_types, query)
            tasks.append(task)
        
        # Exécution en parallèle avec limite de concurrence
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        processed_results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Filtrer les exceptions et retourner les résultats valides
        valid_results = []
        for i, result in enumerate(processed_results):
            if isinstance(result, Exception):
                logger.warning(f"Erreur lors du traitement du résultat {i}: {result}")
                # Garder le résultat original sans contenu extrait
                valid_results.append(results[i])
            else:
                valid_results.append(result)
        
        return valid_results

    async def _process_single_result(self, result: Dict, require_full_fetch: bool, content_types: List[str], query: str = "") -> Dict:
        """Traiter un résultat individuel"""
        try:
            if require_full_fetch and result.get("url"):
                # Récupération complète du contenu
                content = await self.fetcher.fetch(result["url"])
                if content and content.get("html"):
                    # Extraction du contenu
                    extracted = self.extractor.extract(
                        content["html"], 
                        result["url"], 
                        content_types
                    )
                    result["extracted_content"] = extracted
                    
                    # Traitement LLM si disponible
                    if extracted and extracted.get("content"):
                        llm_processed = await self.extractor.llm_processor.process_for_llm(
                            extracted["content"],
                            extracted.get("metadata", {}),
                            query
                        )
                        result["llm_content"] = llm_processed
            
            return result
            
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {result.get('url', 'URL inconnue')}: {e}")
            return result

    async def _semantic_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using semantic similarity"""
        try:
            if not results:
                return results
            
            # Extract text content from results for similarity calculation
            documents = []
            for result in results:
                text_content = ""
                
                # Get text from extracted content
                if result.get("extracted_content", {}).get("full_content"):
                    text_content = result["extracted_content"]["full_content"]
                elif result.get("extracted_content", {}).get("summary"):
                    text_content = result["extracted_content"]["summary"]
                elif result.get("snippet"):
                    text_content = result["snippet"]
                else:
                    text_content = f"{result.get('title', '')} {result.get('description', '')}"
                
                documents.append(text_content)
            
            # Get semantic similarities
            similarities = await self.embeddings.similarity_search(query, documents, top_k=len(documents))
            
            # Create a mapping from document to similarity score
            doc_to_similarity = {doc: sim for doc, sim in similarities}
            
            # Update results with semantic scores and rerank
            for i, result in enumerate(results):
                doc_text = documents[i]
                semantic_score = doc_to_similarity.get(doc_text, 0.0)
                
                # Combine original score with semantic score
                original_score = result.get("score", 0.0)
                combined_score = 0.7 * original_score + 0.3 * semantic_score
                
                result["semantic_score"] = semantic_score
                result["combined_score"] = combined_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
            
            logger.debug(f"Semantic reranking completed for {len(results)} results")
            return results
            
        except Exception as e:
            logger.warning(f"Semantic reranking failed: {e}")
            return results

    async def cleanup(self):
        """Nettoyage des ressources"""
        logger.info("Nettoyage des ressources...")
        
        if self.fetcher:
            await self.fetcher.close()
        
        if self.cache:
            await self.cache.close()
        
        logger.info("Nettoyage terminé")

    async def run_server(self):
        """Run the MCP server"""
        logger.info("Starting MCP Web Search Server...")
        
        # Initialize components
        await self.initialize()
        
        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


# Fonction principale pour interface en ligne de commande
async def main():
    """Point d'entrée principal du serveur"""
    
    # Validation de la configuration
    if not settings.validate_config():
        logger.error("Configuration invalide, arrêt du serveur")
        sys.exit(1)
    
    # Création du serveur
    search_service = WebSearchMCPServer()
    
    try:
        # Interface simple pour test
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            # Mode test simple
            await search_service.initialize()
            test_query = input("Entrez une requête de test: ")
            if test_query:
                result = await search_service._search_web({"q": test_query, "n_results": 5})
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Mode serveur MCP
            logger.info("Starting MCP server in stdio mode...")
            await search_service.run_server()
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        await search_service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
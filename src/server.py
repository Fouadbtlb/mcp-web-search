#!/usr/bin/env python3
"""
MCP Web Search Server - V2
A multi-stage, AI-powered search server.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

from .search.aggregator import SearchAggregator
from .search.fetcher import ContentFetcher
from .search.extractor import ContentExtractor
from .search.cache import SearchCache
from .search.embeddings import AdvancedEmbeddingService
from .utils.normalizer import QueryNormalizer
from .utils.quality import QualityFilter
from .utils.intent_classifier import QueryIntentClassifier
from .config.settings import settings


# Setup logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configuration du logging
logger = logging.getLogger(__name__)

# Global search service instance
search_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    global search_service
    # Startup
    search_service = WebSearchMCP()
    await search_service.initialize()
    logger.info("HTTP API Server started")
    
    yield
    
    # Shutdown
    if search_service:
        await search_service.cleanup()
        logger.info("HTTP API Server stopped")

# FastAPI app for HTTP mode with lifespan
app = FastAPI(
    title="MCP Web Search Server",
    description="Enhanced web search with AI-powered ranking and full content extraction",
    version="1.2.0-enhanced",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Request/Response models
class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    n_results: int = Field(10, ge=1, le=20, description="Number of results")
    freshness: str = Field("all", description="Time filter: day, week, month, all")
    require_full_fetch: bool = Field(False, description="Extract full page content")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    intent: Dict[str, Any]
    trace: Dict[str, Any]  # Changed from metadata to trace to match actual response

# HTTP API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "MCP Web Search Server", "version": "1.2.0-enhanced"}

@app.get("/health")
async def health():
    """Health check with more details"""
    return {
        "status": "healthy",
        "service": "MCP Web Search Server",
        "version": "1.2.0-enhanced",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_enabled": settings.config.cache.enabled
    }

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Web search endpoint"""
    global search_service
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        # Perform search
        result = await search_service.search_web({
            "q": request.q,
            "n_results": request.n_results,
            "freshness": request.freshness,
            "require_full_fetch": request.require_full_fetch
        })
        
        return SearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search")
async def search_get_endpoint(
    q: str,
    n_results: int = 10,
    freshness: str = "all",
    require_full_fetch: bool = False
):
    """Web search endpoint (GET method)"""
    request = SearchRequest(
        q=q,
        n_results=n_results,
        freshness=freshness,
        require_full_fetch=require_full_fetch
    )
    return await search_endpoint(request)

class WebSearchMCP:
    """Serveur MCP pour recherche web avec extraction de contenu"""
    
    def __init__(self):
        # Initialisation des composants
        self.aggregator = SearchAggregator()
        self.fetcher = ContentFetcher()
        self.extractor = ContentExtractor()
        self.cache = SearchCache() if settings.CACHE_ENABLED else None
        self.embeddings = AdvancedEmbeddingService()
        self.normalizer = QueryNormalizer()
        
        # Le filtre de qualité peut être en mode debug
        if settings.LOG_LEVEL == "DEBUG":
            self.quality_filter = QualityFilter(debug=True)
        else:
            self.quality_filter = QualityFilter()
            
        self.intent_classifier = QueryIntentClassifier()
        
        # Initialize components
        self._initialized = False
        
        logger.info("Serveur MCP Web Search initialisé")
        if settings.LOG_LEVEL == "DEBUG":
            logger.debug(f"Configuration: {settings.model_dump_json(indent=2)}")

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

    async def search_web(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Point d'entrée principal pour la recherche web"""
        try:
            # Extraction des paramètres
            query = args.get("q", "")
            n_results = args.get("n_results", settings.config.default_max_results)
            freshness = args.get("freshness", "all")
            sources = args.get("sources", ["searxng"])
            content_types = args.get("content_types", ["article"])
            require_full_fetch = args.get("require_full_fetch", False)
            
            if not query:
                return {"error": "Query parameter 'q' is required"}
            
            logger.info(f"Recherche: '{query}' ({n_results} résultats)")
            
            # Classify query intent
            intent, intent_confidence = self.intent_classifier.classify_intent(query)
            logger.info(f"Detected intent: {intent} (confidence: {intent_confidence:.2f})")
            
            # Get intent-optimized search strategy
            strategy = self.intent_classifier.get_search_strategy(intent)
            
            # Override parameters with intent-optimized values if not explicitly provided
            if "freshness" not in args:
                freshness = strategy.get("freshness", freshness)
            if "sources" not in args:
                sources = strategy.get("sources", sources)
            if "content_types" not in args:
                content_types = strategy.get("content_types", content_types)
            if "require_full_fetch" not in args:
                require_full_fetch = strategy.get("require_full_fetch", require_full_fetch)
            if "n_results" not in args:
                n_results = min(n_results, strategy.get("max_results", n_results))
            
            logger.debug(f"Intent-optimized strategy: {strategy}")
            
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
            final_results = self.quality_filter.deduplicate_and_filter(
                enriched_results, 
                min_score=settings.config.quality.min_quality_score
            )
            
            logger.info(f"{len(final_results)} résultats finaux après filtrage")
            
            # Formatage de la réponse
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
        
        enriched_results = await asyncio.gather(
            *[bounded_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Filtrage des erreurs
        valid_results = []
        for i, result in enumerate(enriched_results):
            if isinstance(result, Exception):
                logger.warning(f"Erreur lors du traitement du résultat {i}: {result}")
                continue
            if result:
                valid_results.append(result)
        
        logger.debug(f"{len(valid_results)}/{len(results)} résultats traités avec succès")
        return valid_results
    
    async def _process_single_result(self, result: Dict, require_full_fetch: bool, content_types: List[str], query: str = "") -> Optional[Dict]:
        """Traitement d'un résultat individuel"""
        try:
            url = result.get("url")
            if not url or not self.fetcher.is_valid_url(url):
                logger.debug(f"URL invalide ignorée: {url}")
                return None
            
            # Récupération du contenu
            use_playwright = (require_full_fetch or 
                            settings.config.content_extraction.use_playwright or
                            self._needs_js_rendering(url))
            
            content_data = await self.fetcher.fetch_content(url, use_headless=use_playwright)
            
            if not content_data:
                logger.debug(f"Échec récupération contenu: {url}")
                # Retourner au moins les métadonnées de base
                return {
                    "url": url,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "content": "",
                    "mime": "text/html",
                    "fetch_method": "failed",
                    "fetched_at": datetime.utcnow().isoformat(),
                    "canonical_url": url,
                    "confidence": result.get("score", 0.3),
                    "provenance": {
                        "engine": result.get("engine", "unknown"),
                        "rank": result.get("rank", 0)
                    }
                }
            
            # Extraction du contenu principal avec optimisation LLM
            extracted_content = await self.extractor.extract_for_llm(
                content_data["content"],  # Use "content" key instead of "html"
                url,
                query  # Pass the query for better processing
            )
            
            # Formatage du résultat final
            max_length = settings.config.content_extraction.max_content_length
            base_result = {
                "url": url,
                "title": extracted_content.get("title", result.get("title", "")),
                "snippet": result.get("snippet", ""),
                "content": extracted_content.get("content", "")[:max_length],
                "mime": content_data.get("content_type", "text/html"),
                "fetch_method": content_data.get("method", "http"),
                "fetched_at": datetime.utcnow().isoformat(),
                "canonical_url": extracted_content.get("canonical_url", url),
                "confidence": result.get("score", 0.5),
                "word_count": extracted_content.get("word_count", 0),
                "language": extracted_content.get("language", ""),
                "author": extracted_content.get("author", ""),
                "publish_date": extracted_content.get("publish_date", ""),
                "provenance": {
                    "engine": result.get("engine", "unknown"),
                    "rank": result.get("rank", 0),
                    "extraction_method": extracted_content.get("extraction_method", "unknown")
                }
            }
            
            # Add LLM-optimized data if available
            llm_data = extracted_content.get("llm_optimized")
            if llm_data:
                base_result["llm_optimized"] = {
                    "summary": llm_data.get("summary", ""),
                    "key_facts": llm_data.get("key_facts", []),
                    "code_snippets": llm_data.get("code_snippets", []),
                    "tables": llm_data.get("tables", []),
                    "citations": llm_data.get("citations", []),
                    "metadata": llm_data.get("metadata", {})
                }
            
            # Add semantic search scores if available
            if result.get("relevance_score") is not None:
                base_result["semantic_scores"] = {
                    "relevance": result.get("relevance_score", 0.0),
                    "semantic": result.get("semantic_score", 0.0),
                    "tfidf": result.get("tfidf_score", 0.0)
                }
            
            return base_result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {result.get('url', 'URL inconnue')}: {e}")
            return None
    
    def _needs_js_rendering(self, url: str) -> bool:
        """Détermine si une URL nécessite un rendu JavaScript"""
        js_heavy_domains = [
            "twitter.com", "x.com", "linkedin.com", 
            "instagram.com", "facebook.com", "reddit.com",
            "youtube.com", "tiktok.com", "pinterest.com"
        ]
        return any(domain in url.lower() for domain in js_heavy_domains)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        if not self.cache:
            return {"cache_enabled": False}
        
        return await self.cache.get_stats()
    
    async def clear_cache(self) -> bool:
        """Vide le cache"""
        if not self.cache:
            return False
        
        return await self.cache.clear_all()
    
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


# Fonction principale pour interface en ligne de commande
# Global search service instance
search_service = None

# Request/Response models
class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    n_results: int = Field(10, ge=1, le=20, description="Number of results")
    freshness: str = Field("all", description="Time filter: day, week, month, all")
    require_full_fetch: bool = Field(False, description="Extract full page content")

def run_http_server():
    """Run HTTP server (for when called directly)"""
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

async def main():
    """Point d'entrée principal du serveur"""
    
    # Validation de la configuration
    # if not settings.validate_config():
    #     logger.error("Configuration invalide, arrêt du serveur")
    #     sys.exit(1)
    
    # Check for server mode
    mode = os.getenv("SERVER_MODE", "stdio")  # Default to stdio for MCP
    
    if mode.lower() == "http":
        # HTTP server mode - exit async and run uvicorn
        logger.info("Starting HTTP server mode...")
        return "http"
    elif mode.lower() == "dual":
        # Dual mode - already handled by __main__, just run MCP part
        logger.info("Starting MCP stdio mode (dual mode)...")
    else:
        # MCP stdio mode (default)
        logger.info("Starting MCP stdio mode...")
        
        # Création du serveur
        search_service = WebSearchMCP()
        await search_service.initialize()  # Initialize async components
        
        try:
            logger.info("Serveur MCP Web Search démarré")
            logger.info(f"Configuration: Cache {'activé' if settings.config.cache.enabled else 'désactivé'}")
            
            # Interface simple pour test
            if len(sys.argv) > 1 and sys.argv[1] == "test":
                # Mode test simple
                test_query = input("Entrez une requête de test: ")
                if test_query:
                    result = await search_service.search_web({"q": test_query, "n_results": 5})
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                # Mode serveur (à implémenter selon les besoins MCP)
                logger.info("Mode serveur - En attente de connexions MCP...")
                
                # Boucle d'attente simple
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Arrêt demandé par l'utilisateur")
        
        except Exception as e:
            logger.error(f"Erreur fatale: {e}", exc_info=True)
            sys.exit(1)
        
        finally:
            await search_service.cleanup()


if __name__ == "__main__":
    # Handle different server modes
    mode = os.getenv("SERVER_MODE", "stdio")
    
    if mode.lower() == "http":
        run_http_server()
    elif mode.lower() == "dual":
        # Start HTTP server in background and MCP in foreground
        import threading
        import time
        
        # Start HTTP server in a separate thread
        def start_http():
            run_http_server()
        
        http_thread = threading.Thread(target=start_http, daemon=True)
        http_thread.start()
        
        # Give HTTP server time to start
        time.sleep(2)
        
        # Run MCP stdio in main thread
        asyncio.run(main())
    else:
        # Default stdio mode
        asyncio.run(main())

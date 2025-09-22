import asyncio
import logging
from typing import List, Dict, Any, Optional

from .fetcher import search_fetcher
from .extractor import content_extractor
from .embeddings import embedding_service
from .reranker import document_reranker
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SearchAggregator:
    """
    Orchestrates the entire search process from fetching to reranking.
    """

    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_mode: str = "full",
        crawl_depth: int = 0,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Performs a comprehensive search operation.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.
            search_mode: 'full', 'semantic', or 'fast'.
            crawl_depth: How many layers of links to crawl (0 for no crawl).
            use_cache: Whether to use the cache for search results.

        Returns:
            A dictionary containing the search results and metadata.
        """
        logger.info(
            f"Starting search for '{query}' with mode '{search_mode}' and max_results={max_results}"
        )

        # 1. Fetch initial results
        initial_results = await search_fetcher.search(
            queries=[query], max_results=max_results * 2, use_cache=use_cache
        )
        if not initial_results:
            logger.warning("No initial results found.")
            return self._format_output([], query, "No results found.")

        # 2. Extract content
        # We run extraction in parallel for speed
        extraction_tasks = [
            content_extractor.extract_content(
                result["url"],
                html=None  # Will fetch HTML inside extract_content
            )
            for result in initial_results
        ]
        extracted_docs = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Filter out failed extractions and merge with initial results
        documents = []
        for i, doc in enumerate(extracted_docs):
            if not isinstance(doc, Exception) and doc and doc.get("content"):
                # Merge metadata from initial result
                merged_doc = {**initial_results[i], **doc}
                documents.append(merged_doc)
            elif not isinstance(doc, Exception):
                # If extraction failed, keep original result without content
                documents.append(initial_results[i])
        
        if not documents:
            logger.warning("Content extraction failed for all initial results.")
            return self._format_output(initial_results, query, "Content extraction failed.")

        # 3. Semantic Search (if applicable)
        if search_mode in ["full", "semantic"]:
            documents = await embedding_service.calculate_semantic_similarity(
                query, documents
            )
            # Trim to max_results after semantic ranking
            documents = documents[:max_results]

        # 4. Reranking (if applicable)
        if search_mode == "full" and settings.ENABLE_RERANKING:
            try:
                documents = await document_reranker.rerank(query, documents)
                # Trim to max_results after reranking
                documents = documents[:max_results]
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Proceeding without it.")

        # Fallback for fast mode or if other modes returned nothing
        if not documents:
            documents = initial_results
        
        final_results = documents[:max_results]

        return self._format_output(
            final_results, query, f"Successfully found {len(final_results)} results."
        )

    def _format_output(
        self, results: List[Dict[str, Any]], query: str, message: str
    ) -> Dict[str, Any]:
        """Formats the final output."""
        return {
            "query": query,
            "message": message,
            "results": results,
            "metadata": {
                "result_count": len(results),
                "embedding_model": settings.EMBEDDING_MODEL,
                "reranker_model": settings.RERANKER_MODEL if settings.ENABLE_RERANKING else "N/A",
            },
        }

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up search aggregator resources.")
        await search_fetcher.cleanup()

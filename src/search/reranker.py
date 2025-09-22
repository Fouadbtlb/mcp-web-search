import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import CrossEncoder
import numpy as np
from ..config.settings import settings

logger = logging.getLogger(__name__)

class DocumentReranker:
    """Advanced document reranker using BGE-reranker-v2-m3"""
    
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.max_length = settings.RERANKER_MAX_LENGTH
        self.model = None
        self.tokenizer = None
        self.cross_encoder = None
        self._model_loaded = False
        
    async def initialize(self) -> bool:
        """Initialize the reranker model"""
        try:
            if settings.OLLAMA_URL:
                # Try to initialize with Ollama, but fall back to local if it fails
                if await self._initialize_ollama():
                    return True
            
            # If Ollama is not configured or fails, initialize local model
            return await self._initialize_local_model()
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            return False
    
    async def _initialize_ollama(self) -> bool:
        """Initialize via Ollama server"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # A simple check to see if Ollama is running
                response = await client.get(settings.OLLAMA_URL)
                if response.status_code == 200:
                    logger.info("Ollama server is available for reranking.")
                    # We don't load the model here, just confirm availability
                    self._model_loaded = True # Assume model is available on Ollama
                    return True
            return False
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}, falling back to local model")
            return False
    
    async def _initialize_local_model(self) -> bool:
        """Initialize local CrossEncoder model"""
        try:
            # Use CrossEncoder for easier integration
            self.cross_encoder = CrossEncoder(
                self.model_name, 
                max_length=self.max_length,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model_loaded = True
            logger.info(f"Local reranker model loaded: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load local reranker model: {e}")
            return False
    
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Search query
            documents: List of documents with content
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with scores
        """
        if not self._model_loaded or not documents:
            return documents
        
        try:
            # Prepare query-document pairs
            pairs = []
            valid_docs = []
            
            for doc in documents:
                doc_text = self._prepare_document_text(doc)
                if doc_text:
                    pairs.append([query, doc_text])
                    valid_docs.append(doc)
            
            if not pairs:
                return documents
            
            # Get reranking scores
            if self.cross_encoder:
                scores = await self._rerank_with_cross_encoder(pairs)
            else:
                scores = await self._rerank_with_ollama(query, valid_docs)
            
            # Combine documents with scores
            scored_docs = []
            for doc, score in zip(valid_docs, scores):
                doc_with_score = doc.copy()
                doc_with_score["rerank_score"] = score
                # Combine with existing semantic score if present
                if "semantic_score" in doc_with_score:
                    doc_with_score["final_score"] = (doc_with_score["semantic_score"] + score) / 2
                else:
                    doc_with_score["final_score"] = score
                scored_docs.append(doc_with_score)
            
            # Sort by rerank score (descending)
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top_k if specified
            if top_k:
                scored_docs = scored_docs[:top_k]
            
            logger.info(f"Reranked {len(scored_docs)} documents")
            return scored_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents
    
    async def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Alias for rerank_documents to match expected interface
        """
        return await self.rerank_documents(query, documents, top_k)
    
    def _prepare_document_text(self, doc: Dict[str, Any]) -> str:
        """Prepare document text for reranking"""
        # Priority order for content
        content_fields = ["content", "markdown", "snippet", "title"]
        
        content_parts = []
        
        # Add title if available
        if "title" in doc and doc["title"]:
            content_parts.append(f"Title: {doc['title']}")
        
        # Add main content
        for field in content_fields:
            if field in doc and doc[field]:
                content_parts.append(doc[field])
                break # Use the first available content
        
        # Add metadata if available
        if "metadata" in doc:
            metadata = doc["metadata"]
            if isinstance(metadata, dict):
                if "description" in metadata:
                    content_parts.append(f"Description: {metadata['description']}")
                if "keywords" in metadata and metadata["keywords"]:
                    content_parts.append(f"Keywords: {', '.join(metadata['keywords'])}")

        full_content = " ".join(content_parts)
        
        # Truncate if too long
        if len(full_content) > self.max_length * 4:  # Rough character limit
            full_content = full_content[:self.max_length * 4]
        
        return full_content
    
    async def _rerank_with_cross_encoder(self, pairs: List[List[str]]) -> List[float]:
        """Rerank using CrossEncoder model"""
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, self.cross_encoder.predict, pairs, {"show_progress_bar": False}
            )
            return scores.tolist() if isinstance(scores, np.ndarray) else scores
        except Exception as e:
            logger.error(f"CrossEncoder prediction failed: {e}")
            return [0.5] * len(pairs)
    
    async def _rerank_with_ollama(self, query: str, documents: List[Dict]) -> List[float]:
        """Rerank using Ollama API"""
        try:
            scores = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                for doc in documents:
                    content = self._prepare_document_text(doc)
                    prompt = f"""Given the query and the document, provide a relevance score from 0.0 to 1.0.
Query: {query}

Document: {content[:2000]}...

Relevance score (0.0-1.0):"""

                    response = await client.post(
                        f"{settings.OLLAMA_URL}/api/generate",
                        json={
                            "model": "bge-reranker-v2-m3",
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.0}
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        score_text = result.get("response", "0.5").strip()
                        try:
                            score = float(score_text)
                            scores.append(max(0.0, min(1.0, score)))
                        except ValueError:
                            scores.append(0.5)
                    else:
                        scores.append(0.5)
            
            return scores
        except Exception as e:
            logger.error(f"Ollama reranking failed: {e}")
            return [0.5] * len(documents)

# Global reranker instance
document_reranker = DocumentReranker()
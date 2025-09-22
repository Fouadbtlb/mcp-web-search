import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from ..config.settings import settings

logger = logging.getLogger(__name__)

class AdvancedEmbeddingService:
    """Advanced embedding service using Stella model"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.model = None
        self.ollama_available = False
        self._model_loaded = False
        
    async def initialize(self) -> bool:
        """Initialize the embedding service"""
        try:
            # Try Ollama first
            if settings.OLLAMA_URL:
                if await self._check_ollama_availability():
                    self.ollama_available = True
                    self._model_loaded = True
                    logger.info("Ollama embedding service is available.")
                    return True
                
            # Load local model as fallback
            if not self.ollama_available:
                return await self._load_local_model()
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    async def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is available with our model"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(settings.OLLAMA_URL)
                if response.status_code != 200:
                    logger.warning("Ollama server not reachable.")
                    return False
                
                response = await client.post(f"{settings.OLLAMA_URL}/api/show", json={"name": self.model_name})
                if response.status_code == 200:
                    logger.info(f"Ollama has model '{self.model_name}' available.")
                    return True
                else:
                    logger.warning(f"Ollama does not have model '{self.model_name}'. Please run 'ollama pull {self.model_name}'")
                    return False
                
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    async def _load_local_model(self) -> bool:
        """Load local SentenceTransformer model"""
        try:
            # Model mapping for local loading
            model_mapping = {
                "nomic-ai/stella_en_1.5B_v5": "dunzhang/stella_en_1.5B_v5",
                # Add other mappings if needed
            }
            
            local_model_name = model_mapping.get(
                self.model_name, 
                self.model_name
            )
            
            # Load in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                SentenceTransformer, 
                local_model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            self._model_loaded = True
            logger.info(f"Local embedding model loaded: {local_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            # Final fallback to a simple model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._model_loaded = True
                logger.warning("Fell back to 'all-MiniLM-L6-v2' due to previous error.")
                return True
            except Exception as fallback_error:
                logger.critical(f"Failed to load even the fallback model: {fallback_error}")
                return False
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not texts:
            return None
            
        try:
            if self.ollama_available:
                return await self._generate_ollama_embeddings(texts, normalize)
            elif self._model_loaded:
                return await self._generate_local_embeddings(texts, normalize)
            else:
                logger.error("No embedding model is available.")
                return None
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    async def _generate_ollama_embeddings(
        self, 
        texts: List[str], 
        normalize: bool
    ) -> Optional[np.ndarray]:
        """Generate embeddings using Ollama API"""
        embeddings = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for text in texts:
                    response = await client.post(
                        f"{settings.OLLAMA_URL}/api/embeddings",
                        json={"model": self.model_name, "prompt": text}
                    )
                    if response.status_code == 200:
                        embeddings.append(response.json()["embedding"])
                    else:
                        logger.warning(f"Ollama failed to generate embedding for a text snippet. Status: {response.status_code}")
                        embeddings.append([0.0] * self.dimensions) # Placeholder
            
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                if normalize:
                    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                    embeddings_array = embeddings_array / norms
                
                return embeddings_array
            else:
                return None
                
        except Exception as e:
            logger.error(f"Ollama embedding generation failed: {e}")
            return None
    
    async def _generate_local_embeddings(
        self, 
        texts: List[str], 
        normalize: bool
    ) -> Optional[np.ndarray]:
        """Generate embeddings using local model"""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.model.encode, 
                texts, 
                convert_to_tensor=False, 
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            return None
    
    async def calculate_semantic_similarity(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate semantic similarity between query and documents
        
        Args:
            query: Search query
            documents: List of documents with content
            
        Returns:
            Documents with semantic scores
        """
        if not documents:
            return documents
        
        try:
            # Prepare texts for embedding
            texts = [query]  # Query first
            document_texts = [self._extract_document_text(doc) for doc in documents]
            texts.extend(document_texts)
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            if embeddings is None or len(embeddings) < 2:
                logger.warning("Could not generate embeddings for similarity calculation.")
                return documents
            
            # Calculate similarities
            query_embedding = embeddings[0:1]  # First embedding is query
            doc_embeddings = embeddings[1:]    # Rest are documents
            
            # Cosine similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Add scores to documents
            scored_documents = []
            for i, doc in enumerate(documents):
                doc_with_score = doc.copy()
                doc_with_score["semantic_score"] = float(similarities[i])
                scored_documents.append(doc_with_score)
            
            # Sort by semantic score (descending)
            scored_documents.sort(key=lambda x: x["semantic_score"], reverse=True)
            
            logger.info(f"Calculated semantic similarity for {len(scored_documents)} documents")
            return scored_documents
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return documents
    
    def _extract_document_text(self, doc: Dict[str, Any]) -> str:
        """Extract text from document for embedding"""
        
        # Priority order for text extraction
        text_fields = ["content", "markdown", "snippet", "title"]
        
        text_parts = []
        
        # Add title with weight
        title = doc.get("title", "")
        if title:
            text_parts.append(f"{title} {title}")  # Duplicate for emphasis
        
        # Add main content
        for field in text_fields:
            if field in doc and doc[field]:
                text_parts.append(doc[field])
                break
        
        # Add metadata if available
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            description = metadata.get("description", "")
            if description:
                text_parts.append(f"Description: {description}")
        
        return " ".join(text_parts)
    
    async def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for a single query"""
        embeddings = await self.generate_embeddings([query])
        return embeddings[0] if embeddings is not None else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "ollama_available": self.ollama_available,
            "local_model_loaded": self._model_loaded,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

# Global embedding service instance
embedding_service = AdvancedEmbeddingService()
import logging
import os
import asyncio
from typing import List, Optional, Dict, Any, Tuple
import aiohttp
import json

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Lightweight embedding processor with Ollama and CPU-only fallback"""
    
    def __init__(self):
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-minilm')
        self.max_chunk_size = int(os.getenv('MAX_CHUNK_SIZE', '512'))
        
        # Lightweight fallback embeddings
        self._fallback_embedder = None
        self._use_ollama = None
        
    async def initialize(self):
        """Initialize the embedding system"""
        # Test Ollama availability
        self._use_ollama = await self._test_ollama_connection()
        
        if not self._use_ollama:
            logger.info("Ollama not available, using CPU-only fallback")
            await self._init_fallback_embedder()
        else:
            logger.info(f"Using Ollama embeddings at {self.ollama_url}")
            await self._ensure_embedding_model()
    
    async def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama connection test failed: {e}")
            return False
    
    async def _ensure_embedding_model(self):
        """Ensure the embedding model is available in Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if model exists
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        
                        if not any(self.embedding_model in model for model in models):
                            logger.info(f"Pulling embedding model: {self.embedding_model}")
                            # Pull the model
                            pull_data = {"name": self.embedding_model, "stream": False}
                            async with session.post(
                                f"{self.ollama_url}/api/pull",
                                json=pull_data,
                                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for model pull
                            ) as pull_response:
                                if pull_response.status == 200:
                                    logger.info(f"Successfully pulled model: {self.embedding_model}")
                                else:
                                    logger.error(f"Failed to pull model: {await pull_response.text()}")
                        else:
                            logger.info(f"Model {self.embedding_model} already available")
        except Exception as e:
            logger.error(f"Error ensuring embedding model: {e}")
            # Fall back to CPU-only
            self._use_ollama = False
            await self._init_fallback_embedder()
    
    async def _init_fallback_embedder(self):
        """Initialize lightweight CPU-only embeddings"""
        try:
            # Use a simple TF-IDF based approach for minimal dependencies
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            self._fallback_embedder = TfidfVectorizer(
                max_features=384,  # Match common embedding dimensions
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2
            )
            logger.info("Initialized CPU-only TF-IDF embeddings")
        except ImportError:
            logger.warning("scikit-learn not available, using basic text similarity")
            self._fallback_embedder = None
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if not texts:
            return []
        
        if self._use_ollama:
            return await self._embed_with_ollama(texts)
        else:
            return await self._embed_with_fallback(texts)
    
    async def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    # Truncate text if too long
                    truncated_text = text[:self.max_chunk_size * 4]  # Rough char estimate
                    
                    embed_data = {
                        "model": self.embedding_model,
                        "prompt": truncated_text
                    }
                    
                    async with session.post(
                        f"{self.ollama_url}/api/embeddings",
                        json=embed_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            embedding = data.get('embedding', [])
                            embeddings.append(embedding)
                        else:
                            logger.error(f"Ollama embedding failed: {await response.text()}")
                            # Use zero vector as fallback
                            embeddings.append([0.0] * 384)
            
        except Exception as e:
            logger.error(f"Error generating Ollama embeddings: {e}")
            # Fall back to basic embeddings
            return await self._embed_with_fallback(texts)
        
        return embeddings
    
    async def _embed_with_fallback(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using CPU-only fallback"""
        if self._fallback_embedder is None:
            # Ultra-simple fallback: basic text features
            return [self._simple_text_features(text) for text in texts]
        
        try:
            # Use TF-IDF embeddings
            import numpy as np
            
            # Fit or transform texts
            if not hasattr(self._fallback_embedder, 'vocabulary_'):
                # First time - fit the vectorizer
                tfidf_matrix = self._fallback_embedder.fit_transform(texts)
            else:
                tfidf_matrix = self._fallback_embedder.transform(texts)
            
            # Convert to list of lists
            embeddings = tfidf_matrix.toarray().tolist()
            return embeddings
            
        except Exception as e:
            logger.error(f"TF-IDF embedding failed: {e}")
            # Ultra-simple fallback
            return [self._simple_text_features(text) for text in texts]
    
    def _simple_text_features(self, text: str) -> List[float]:
        """Generate simple text features as ultra-lightweight fallback"""
        # Basic text statistics as features (384 dimensions to match common models)
        features = [0.0] * 384
        
        if not text:
            return features
        
        words = text.lower().split()
        if not words:
            return features
        
        # Basic features
        features[0] = len(text)  # Character count
        features[1] = len(words)  # Word count
        features[2] = len(set(words))  # Unique word count
        features[3] = sum(len(word) for word in words) / len(words)  # Avg word length
        
        # Character frequency features (simple)
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            if i < 26:
                features[4 + i] = text.lower().count(char) / len(text)
        
        # Word frequency features (top words)
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        for i, word in enumerate(common_words):
            if i < 14:
                features[30 + i] = words.count(word) / len(words)
        
        # Fill remaining with normalized hash values for some uniqueness
        text_hash = hash(text)
        for i in range(44, 384):
            features[i] = ((text_hash + i) % 1000) / 1000.0
        
        return features
    
    async def similarity_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar documents to query"""
        if not documents:
            return []
        
        # Generate embeddings
        all_texts = [query] + documents
        embeddings = await self.embed_texts(all_texts)
        
        if not embeddings or len(embeddings) != len(all_texts):
            logger.error("Failed to generate embeddings for similarity search")
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((documents[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status information about the embedding system"""
        return {
            "using_ollama": self._use_ollama,
            "ollama_url": self.ollama_url if self._use_ollama else None,
            "embedding_model": self.embedding_model if self._use_ollama else "cpu-fallback",
            "max_chunk_size": self.max_chunk_size,
            "fallback_type": "tfidf" if self._fallback_embedder else "simple"
        }
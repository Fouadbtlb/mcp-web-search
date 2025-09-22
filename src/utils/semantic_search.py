import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import pickle
import os

# Optional heavy dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    cosine_similarity = None
    TfidfVectorizer = None
    HAS_SKLEARN = False
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SemanticSearchEnhancer:
    """Enhanced semantic search capabilities for better result relevance"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "models"):
        """
        Initialize with a lightweight but effective sentence transformer model
        all-MiniLM-L6-v2: Fast, good quality, 90MB - optimized for general LLM use
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')  # Reduced features for speed
        
        # Embedding cache - optimized for LLM queries
        self.embedding_cache = {}
        self.cache_file = os.path.join(cache_dir, "embeddings_cache.pkl")
        
        # Performance settings - optimized for speed
        self.max_cache_size = 2000  # Increased cache size
        self.cache_ttl_days = 14    # Longer cache retention
        self.similarity_threshold = 0.6  # Lower threshold for broader matching
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                # Create cache directory
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Load model (this might take time on first run)
                self.model = SentenceTransformer(self.model_name)
                
                # Load embedding cache if exists
                await self._load_cache()
                
                logger.info("Semantic search model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize semantic model: {e}")
                self.model = None
    
    async def rank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank search results by semantic similarity to query"""
        if not self.model or not results:
            return results
        
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return results
            
            # Calculate semantic scores for each result
            scored_results = []
            for result in results:
                # Create combined text for similarity calculation
                combined_text = self._create_combined_text(result)
                
                # Get embedding for result
                result_embedding = await self._get_embedding(combined_text)
                
                if result_embedding is not None:
                    # Calculate semantic similarity
                    semantic_score = self._calculate_similarity(query_embedding, result_embedding)
                    
                    # Fast TF-IDF calculation (simplified for performance)
                    tfidf_score = self._calculate_tfidf_similarity(query, combined_text)
                    
                    # Optimized scoring (80% semantic, 20% TF-IDF for LLM use)
                    final_score = 0.8 * semantic_score + 0.2 * tfidf_score
                    
                    # Add scores to result
                    result = result.copy()
                    result['semantic_score'] = float(semantic_score)
                    result['tfidf_score'] = float(tfidf_score)
                    result['relevance_score'] = float(final_score)
                    
                    scored_results.append((final_score, result))
                else:
                    # Fallback to original result with low score
                    result = result.copy()
                    result['semantic_score'] = 0.0
                    result['tfidf_score'] = 0.0
                    result['relevance_score'] = 0.0
                    scored_results.append((0.0, result))
            
            # Sort by relevance score (highest first)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            return [result for _, result in scored_results]
            
        except Exception as e:
            logger.error(f"Error in semantic ranking: {e}")
            return results
    
    async def find_similar_queries(self, query: str, query_history: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Find similar queries from history for potential result reuse"""
        if not self.model or not query_history:
            return []
        
        try:
            query_embedding = await self._get_embedding(query)
            if query_embedding is None:
                return []
            
            similar_queries = []
            for hist_query in query_history:
                hist_embedding = await self._get_embedding(hist_query)
                if hist_embedding is not None:
                    similarity = self._calculate_similarity(query_embedding, hist_embedding)
                    if similarity > 0.7:  # High similarity threshold
                        similar_queries.append((hist_query, float(similarity)))
            
            # Sort by similarity and return top k
            similar_queries.sort(key=lambda x: x[1], reverse=True)
            return similar_queries[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []
    
    async def extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics/themes from text using embeddings"""
        if not self.model or not text:
            return []
        
        try:
            # Split text into sentences
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            if len(sentences) < 2:
                return []
            
            # Get embeddings for all sentences
            embeddings = []
            valid_sentences = []
            
            for sentence in sentences:
                embedding = await self._get_embedding(sentence)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_sentences.append(sentence)
            
            if len(embeddings) < 2:
                return []
            
            # Simple clustering to find key topics
            # Calculate similarity matrix
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Find sentences with high average similarity to others (key topics)
            avg_similarities = np.mean(similarity_matrix, axis=1)
            
            # Get top sentences as key topics
            top_indices = np.argsort(avg_similarities)[-3:]  # Top 3
            key_topics = [valid_sentences[i] for i in top_indices if avg_similarities[i] > 0.3]
            
            return key_topics
            
        except Exception as e:
            logger.error(f"Error extracting key topics: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if not text or not text.strip():
            return None
        
        # Check cache first
        cache_key = hash(text.strip())
        if cache_key in self.embedding_cache:
            cache_entry = self.embedding_cache[cache_key]
            # Check if cache is still valid
            if datetime.now() - cache_entry['timestamp'] < timedelta(days=self.cache_ttl_days):
                return cache_entry['embedding']
        
        try:
            # Generate embedding
            embedding = self.model.encode(text.strip(), convert_to_numpy=True)
            
            # Cache the result
            self.embedding_cache[cache_key] = {
                'embedding': embedding,
                'timestamp': datetime.now()
            }
            
            # Maintain cache size
            if len(self.embedding_cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.embedding_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                # Keep only the newest 80% of entries
                keep_count = int(self.max_cache_size * 0.8)
                self.embedding_cache = dict(sorted_items[-keep_count:])
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _create_combined_text(self, result: Dict[str, Any]) -> str:
        """Create combined text from result for similarity calculation"""
        parts = []
        
        # Title (weighted more heavily)
        title = result.get('title', '')
        if title:
            parts.extend([title] * 3)  # Triple weight for title
        
        # Description/snippet
        description = result.get('description', '') or result.get('snippet', '')
        if description:
            parts.extend([description] * 2)  # Double weight
        
        # Content preview if available
        content = result.get('content', '')
        if content:
            # Use first 500 characters
            parts.append(content[:500])
        
        # URL can sometimes be informative
        url = result.get('url', '')
        if url:
            # Extract meaningful parts from URL
            url_parts = url.replace('/', ' ').replace('-', ' ').replace('_', ' ')
            parts.append(url_parts)
        
        return ' '.join(parts)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Reshape to 2D arrays for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0, 0]
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_tfidf_similarity(self, query: str, text: str) -> float:
        """Calculate TF-IDF based similarity as fallback"""
        try:
            # Fit TF-IDF on both texts
            corpus = [query, text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.debug(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    async def _load_cache(self):
        """Load embedding cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Error loading embedding cache: {e}")
            self.embedding_cache = {}
    
    async def save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    async def cleanup(self):
        """Cleanup resources and save cache"""
        if self.embedding_cache:
            await self.save_cache()
        if self.model:
            # Clear model from memory
            del self.model
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
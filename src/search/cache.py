import asyncio
import json
import logging
import hashlib
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SearchCache:
    def __init__(self, redis_url: str = settings.REDIS_URL, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client = None
        self.fallback_cache = {}  # Cache en mémoire en fallback
        self.fallback_max_size = 1000
        
        logger.info("Cache initialisé (mode mémoire uniquement pour compatibilité Python 3.13)")
        
    async def _get_redis_client(self):
        """Redis désactivé pour l'instant à cause de problèmes de compatibilité Python 3.13"""
        return None
    
    def _generate_cache_key(self, key: str) -> str:
        """Génère une clé de cache hashée"""
        # Préfixe pour identifier les clés de ce service
        prefix = "mcp_web_search:"
        
        # Hash de la clé pour éviter les caractères problématiques
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        return f"{prefix}{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        cache_key = self._generate_cache_key(key)
        
        # Essayer Redis d'abord (désactivé pour l'instant)
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    logger.debug(f"Cache Redis hit: {key}")
                    return data
            except Exception as e:
                logger.warning(f"Erreur lecture Redis: {e}")
        
        # Fallback vers cache mémoire
        if cache_key in self.fallback_cache:
            cached_item = self.fallback_cache[cache_key]
            
            # Vérifier expiration
            if cached_item["expires_at"] > time.time():
                logger.debug(f"Cache mémoire hit: {key}")
                return cached_item["data"]
            else:
                # Supprimer l'élément expiré
                del self.fallback_cache[cache_key]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur dans le cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._generate_cache_key(key)
        serialized_value = json.dumps(value, ensure_ascii=False, default=self._json_serializer)
        
        # Essayer Redis d'abord (désactivé pour l'instant)
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                await redis_client.setex(cache_key, ttl, serialized_value)
                logger.debug(f"Cache Redis set: {key} (TTL: {ttl}s)")
                return True
            except Exception as e:
                logger.warning(f"Erreur écriture Redis: {e}")
        
        # Fallback vers cache mémoire
        try:
            # Nettoyer le cache mémoire si trop plein
            if len(self.fallback_cache) >= self.fallback_max_size:
                self._cleanup_memory_cache()
            
            self.fallback_cache[cache_key] = {
                "data": value,
                "expires_at": time.time() + ttl,
                "created_at": time.time()
            }
            logger.debug(f"Cache mémoire set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Erreur cache mémoire: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une clé du cache"""
        cache_key = self._generate_cache_key(key)
        success = True
        
        # Supprimer de Redis
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                await redis_client.delete(cache_key)
                logger.debug(f"Cache Redis delete: {key}")
            except Exception as e:
                logger.warning(f"Erreur suppression Redis: {e}")
                success = False
        
        # Supprimer du cache mémoire
        if cache_key in self.fallback_cache:
            del self.fallback_cache[cache_key]
            logger.debug(f"Cache mémoire delete: {key}")
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans le cache"""
        cache_key = self._generate_cache_key(key)
        
        # Vérifier Redis
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                exists_redis = await redis_client.exists(cache_key)
                if exists_redis:
                    return True
            except Exception as e:
                logger.warning(f"Erreur vérification Redis: {e}")
        
        # Vérifier cache mémoire
        if cache_key in self.fallback_cache:
            cached_item = self.fallback_cache[cache_key]
            if cached_item["expires_at"] > time.time():
                return True
            else:
                # Supprimer l'élément expiré
                del self.fallback_cache[cache_key]
        
        return False
    
    async def clear_all(self) -> bool:
        """Vide tout le cache (attention!)"""
        success = True
        
        # Vider Redis (seulement nos clés)
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                pattern = "mcp_web_search:*"
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
                logger.info(f"Cache Redis vidé: {len(keys)} clés supprimées")
            except Exception as e:
                logger.error(f"Erreur vidage Redis: {e}")
                success = False
        
        # Vider cache mémoire
        self.fallback_cache.clear()
        logger.info("Cache mémoire vidé")
        
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur le cache"""
        stats = {
            "redis_connected": False,
            "redis_keys_count": 0,
            "memory_cache_size": len(self.fallback_cache),
            "memory_cache_max_size": self.fallback_max_size
        }
        
        # Stats Redis
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                # Compter nos clés
                pattern = "mcp_web_search:*"
                keys = await redis_client.keys(pattern)
                stats["redis_connected"] = True
                stats["redis_keys_count"] = len(keys)
            except Exception as e:
                logger.warning(f"Erreur stats Redis: {e}")
        
        return stats
    
    def _cleanup_memory_cache(self):
        """Nettoie le cache mémoire en supprimant les éléments expirés et les plus anciens"""
        current_time = time.time()
        
        # Supprimer les éléments expirés
        expired_keys = []
        for key, item in self.fallback_cache.items():
            if item["expires_at"] <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.fallback_cache[key]
        
        # Si encore trop d'éléments, supprimer les plus anciens
        if len(self.fallback_cache) >= self.fallback_max_size:
            # Trier par date de création et supprimer les 20% plus anciens
            items = [(key, item["created_at"]) for key, item in self.fallback_cache.items()]
            items.sort(key=lambda x: x[1])
            
            to_remove = len(items) // 5  # Supprimer 20%
            for key, _ in items[:to_remove]:
                del self.fallback_cache[key]
        
        logger.debug(f"Cache mémoire nettoyé: {len(expired_keys)} expirés, taille: {len(self.fallback_cache)}")
    
    def _json_serializer(self, obj):
        """Sérialiseur JSON personnalisé"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    async def close(self):
        """Ferme les connexions"""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Connexion Redis fermée")
            except Exception as e:
                logger.warning(f"Erreur fermeture Redis: {e}")
        
        self.fallback_cache.clear()

search_cache = SearchCache()

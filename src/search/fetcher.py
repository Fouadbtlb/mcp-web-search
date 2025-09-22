import asyncio
import httpx
import logging
from typing import Dict, Optional, Any
from urllib.parse import urljoin, urlparse
import time
from fake_useragent import UserAgent
import os

logger = logging.getLogger(__name__)

class ContentFetcher:
    def __init__(self):
        self.ua = UserAgent()
        self.timeout = httpx.Timeout(30.0)
        self.max_content_size = 10 * 1024 * 1024  # 10MB max
        
        # Default headers to avoid bot detection
        self.default_headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def fetch_content(self, url: str, use_headless: bool = False, retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch content from a URL with retry logic"""
        for attempt in range(retries):
            try:
                return await self._fetch_with_httpx(url)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == retries - 1:
                    logger.error(f"All attempts failed for {url}: {e}")
                    return None
                await asyncio.sleep(1)  # Wait before retry
        return None
    
    async def _fetch_with_httpx(self, url: str) -> Dict[str, Any]:
        """Fetch content using httpx"""
        start_time = time.time()
        
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.default_headers,
            follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content = response.text
            if len(content) > self.max_content_size:
                content = content[:self.max_content_size]
                logger.warning(f"Content truncated for {url}")
            
            return {
                "url": str(response.url),
                "content": content,
                "status": response.status_code,
                "headers": dict(response.headers),
                "fetch_time": time.time() - start_time,
                "method": "httpx",
                "content_length": len(content)
            }

    async def _validate_url(self, url: str) -> bool:
        """Validate if a URL is accessible"""
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.head(url, follow_redirects=True)
                return 200 <= response.status_code < 400
        except Exception:
            return False

    def _extract_media_urls(self, content: str, base_url: str) -> Dict[str, list]:
        """Extract media URLs from HTML content"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(content, 'html.parser')
        
        media_urls = {
            'images': [],
            'videos': [],
            'documents': []
        }
        
        # Images
        for img in soup.find_all('img', src=True):
            src = img['src']
            if src.startswith(('http://', 'https://')):
                media_urls['images'].append(src)
            else:
                media_urls['images'].append(urljoin(base_url, src))
        
        # Videos
        for video in soup.find_all(['video', 'source'], src=True):
            src = video['src']
            if src.startswith(('http://', 'https://')):
                media_urls['videos'].append(src)
            else:
                media_urls['videos'].append(urljoin(base_url, src))
        
        return media_urls

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    async def close(self):
        """Clean up resources"""
        # Nothing to clean up for httpx-only implementation
        pass

search_fetcher = ContentFetcher()
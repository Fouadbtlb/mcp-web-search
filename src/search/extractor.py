import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import httpx
from bs4 import BeautifulSoup, Tag
import html2text
from urllib.parse import urljoin, urlparse
import json
from readability import Document
from langdetect import detect, DetectorFactory
from ..config.settings import settings

# Set langdetect to be deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class ContentExtractor:
    """Advanced content extractor inspired by Firecrawl architecture"""
    
    def __init__(self):
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0
        self.html2text.unicode_snob = True
        
        # Configure for LLM-optimized markdown
        if settings.MARKDOWN_OPTIMIZATION:
            self.html2text.ignore_emphasis = False
            self.html2text.mark_code = True
            self.html2text.wrap_links = False
            self.html2text.inline_links = True
    
    async def extract_content(self, url: str, html: str = None) -> Dict[str, Any]:
        """
        Extract comprehensive content from URL
        
        Args:
            url: Target URL
            html: Optional pre-fetched HTML content
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            if not html:
                html = await self._fetch_html(url)
            
            if not html:
                return {"error": "Failed to fetch HTML", "url": url}
            
            # Parse HTML
            soup = BeautifulSoup(html, 'lxml')
            
            # Extract using multiple methods
            extraction_result = {}
            
            # Basic metadata extraction
            extraction_result.update(await self._extract_metadata(soup, url))
            
            # Content extraction with multiple methods
            content_methods = {
                "readability": await self._extract_with_readability(html),
                "manual": await self._extract_manual(soup),
                "structured": await self._extract_structured_data(soup)
            }
            
            # Choose best content
            best_content = self._choose_best_content(content_methods)
            extraction_result.update(best_content)
            
            # Convert to optimized markdown
            if settings.MARKDOWN_OPTIMIZATION and extraction_result.get("content"):
                extraction_result["markdown"] = await self._optimize_markdown(
                    extraction_result["content"],
                    extraction_result.get("title", ""),
                    soup
                )
            
            # Language detection
            if extraction_result.get("content"):
                try:
                    extraction_result["language"] = detect(extraction_result["content"])
                except:
                    extraction_result["language"] = "en" # Default to English
            
            # Content quality metrics
            extraction_result["quality_metrics"] = self._calculate_quality_metrics(extraction_result)
            
            # Extract links and references
            if settings.EXTRACT_METADATA:
                extraction_result["links"] = self._extract_links(soup, url)
                extraction_result["images"] = self._extract_images(soup, url)
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {"error": str(e), "url": url}
    
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML with anti-detection measures"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        try:
            async with httpx.AsyncClient(
                http2=True, 
                verify=False, # To avoid SSL errors on some sites
                follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers, timeout=settings.TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    async def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata"""
        metadata = {}
        
        # Basic metadata
        title_tag = soup.find('title')
        metadata["title"] = title_tag.get_text().strip() if title_tag else ""
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property') or tag.get('itemprop')
            content = tag.get('content')
            
            if name and content:
                metadata[name.lower().replace(":", "_")] = content.strip()
        
        # Structured data (JSON-LD)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        structured_data = []
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if structured_data:
            metadata["structured_data"] = structured_data
        
        # Author extraction
        author = self._extract_author(soup)
        if author:
            metadata["author"] = author
        
        # Publication date
        pub_date = self._extract_publication_date(soup)
        if pub_date:
            metadata["publish_date"] = pub_date
        
        # Keywords extraction
        keywords = self._extract_keywords(soup)
        if keywords:
            metadata["keywords"] = keywords
        
        return {"metadata": metadata}
    
    async def _extract_with_readability(self, html: str) -> Dict[str, str]:
        """Extract content using readability algorithm"""
        try:
            doc = Document(html)
            title = doc.title()
            content = doc.summary()
            
            # Convert to text
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()
            
            return {
                "content": text_content,
                "title": title,
                "method": "readability"
            }
        except Exception as e:
            logger.error(f"Readability extraction failed: {e}")
            return {}
    
    async def _extract_manual(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Manual content extraction with smart selection"""
        
        # Remove unwanted elements
        for tag in soup.find_all(settings.EXCLUDE_TAGS):
            tag.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.main-content', '.content', 
            '.post-content', '.entry-content', '.article-content',
            '#main', '#content', '[role="main"]'
        ]
        
        main_content = None
        for selector in content_selectors:
            try:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            except:
                continue
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text_content = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        return {
            "content": text_content,
            "method": "manual"
        }
    
    async def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data elements"""
        structured = {}
        
        # Headers hierarchy
        headers = []
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            for h in h_tags:
                headers.append({
                    "level": i,
                    "text": h.get_text().strip(),
                    "id": h.get('id')
                })
        
        if headers:
            structured["headers"] = headers
        
        # Lists
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            items = [li.get_text().strip() for li in ul.find_all('li')]
            if items:
                lists.append({
                    "type": "ordered" if ul.name == 'ol' else 'unordered',
                    "items": items
                })
        
        if lists:
            structured["lists"] = lists
        
        # Tables
        tables = []
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                rows.append(cells)
            if rows:
                tables.append(rows)
        
        if tables:
            structured["tables"] = tables
        
        return {"structured_elements": structured} if structured else {}
    
    def _choose_best_content(self, methods: Dict[str, Dict]) -> Dict[str, Any]:
        """Choose the best content extraction method"""
        
        # Score each method
        scores = {}
        for method_name, result in methods.items():
            if not result or "content" not in result:
                scores[method_name] = 0
                continue
            
            content = result["content"]
            score = 0
            
            # Length score (prefer reasonable length)
            length = len(content)
            if 100 <= length <= 10000:
                score += 40
            elif length > 10000:
                score += 30
            elif length > 50:
                score += 10
            
            # Structure score (prefer content with paragraphs)
            if '\n' in content or '. ' in content:
                score += 20
            
            # Quality indicators
            if any(word in content.lower() for word in ['article', 'content', 'text', 'information']):
                score += 10
            
            # Avoid navigation/menu content
            if any(word in content.lower() for word in ['menu', 'navigation', 'copyright', 'cookie']):
                score -= 30
            
            scores[method_name] = score
        
        # Choose best method
        best_method = max(scores, key=scores.get) if scores else "manual"
        best_result = methods.get(best_method, {})
        
        return {
            "content": best_result.get("content", ""),
            "extraction_method": best_method,
            **{k: v for k, v in best_result.items() if k != "content"}
        }
    
    async def _optimize_markdown(self, content: str, title: str, soup: BeautifulSoup) -> str:
        """Convert content to LLM-optimized markdown"""
        try:
            # Create a clean HTML structure
            clean_html = f"""
            <html>
            <body>
            <h1>{title}</h1>
            <div class="content">
            {self._clean_html_for_markdown(soup)}
            </div>
            </body>
            </html>
            """
            
            # Convert to markdown
            markdown = self.html2text.handle(clean_html)
            
            # Clean up markdown
            markdown = self._clean_markdown(markdown)
            
            return markdown
        except Exception as e:
            logger.error(f"Markdown optimization failed: {e}")
            return content
    
    def _clean_html_for_markdown(self, soup: BeautifulSoup) -> str:
        """Clean HTML for better markdown conversion"""
        
        # Remove unwanted elements
        for tag in soup.find_all(settings.EXCLUDE_TAGS + ['iframe', 'embed', 'object']):
            tag.decompose()
        
        # Find main content
        content_selectors = ['main', 'article', '.content', '[role="main"]']
        main_content = None
        
        for selector in content_selectors:
            try:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            except:
                continue
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        return str(main_content)
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up generated markdown"""
        
        # Remove excessive whitespace
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = re.sub(r'[ \t]+', ' ', markdown)
        
        # Fix common markdown issues
        markdown = re.sub(r'\*\s+\*', '', markdown)  # Remove empty emphasis
        markdown = re.sub(r'_{2,}', '_', markdown)   # Fix multiple underscores
        
        # Clean up links
        markdown = re.sub(r'\[\s*\]\([^)]*\)', '', markdown)  # Remove empty links
        
        return markdown.strip()
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information"""
        
        # Try various author selectors
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author', '.byline', '.author-name',
            '[rel="author"]', '[class*="author"]'
        ]
        
        for selector in author_selectors:
            try:
                author_tag = soup.select_one(selector)
                if author_tag:
                    return author_tag.get('content') or author_tag.get_text().strip()
            except:
                continue
        
        return None
    
    def _extract_publication_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date"""
        
        # Try various date selectors
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'meta[name="pubdate"]',
            'time[datetime]', '.date', '.publish-date'
        ]
        
        for selector in date_selectors:
            try:
                date_tag = soup.select_one(selector)
                if date_tag:
                    date_str = date_tag.get('datetime') or date_tag.get('content') or date_tag.get_text()
                    # Basic parsing, can be improved with dateutil
                    return date_str.strip()
            except:
                continue
        
        return None
    
    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords and tags"""
        
        keywords = set()
        
        # From meta tags
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            content = meta_keywords.get('content', '')
            keywords.update([k.strip() for k in content.split(',') if k.strip()])
        
        # From tag elements
        for tag_selector in ['.tags a', '.tag', '.keywords a', '[class*="tag"]']:
            try:
                tags = soup.select(tag_selector)
                for tag in tags:
                    keywords.add(tag.get_text().strip())
            except:
                continue
        
        return list(keywords)[:20]  # Limit to 20 keywords
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract internal and external links"""
        
        links = []
        seen_urls = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#') or href.startswith('mailto:'):
                continue
            
            # Make absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Avoid duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)
            
            # Get link text
            text = a_tag.get_text().strip()
            if not text:
                text = a_tag.get('title', '') or a_tag.get('aria-label', '')
            
            # Determine if internal or external
            base_domain = urlparse(base_url).netloc
            link_domain = urlparse(absolute_url).netloc
            is_internal = base_domain == link_domain
            
            links.append({
                "url": absolute_url,
                "text": text,
                "type": "internal" if is_internal else "external"
            })
        
        return links[:50]  # Limit to 50 links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image information"""
        
        images = []
        
        for img_tag in soup.find_all('img', src=True):
            src = img_tag['src'].strip()
            if not src:
                continue
            
            # Make absolute URL
            absolute_url = urljoin(base_url, src)
            
            images.append({
                "src": absolute_url,
                "alt": img_tag.get('alt', ''),
                "title": img_tag.get('title', '')
            })
        
        return images[:20]  # Limit to 20 images
    
    def _calculate_quality_metrics(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate content quality metrics"""
        
        content = extraction_result.get("content", "")
        
        metrics = {
            "content_length": len(content),
            "word_count": len(content.split()) if content else 0,
            "has_title": bool(extraction_result.get("title")),
            "has_metadata": bool(extraction_result.get("metadata")),
            "extraction_method": extraction_result.get("extraction_method", "unknown")
        }
        
        # Reading time estimation (average 200 words per minute)
        if metrics["word_count"] > 0:
            metrics["reading_time_minutes"] = max(1, metrics["word_count"] // 200)
        
        # Content quality score (0-1)
        score = 0
        if metrics["content_length"] >= settings.MIN_CONTENT_LENGTH:
            score += 0.3
        if metrics["has_title"]:
            score += 0.2
        if metrics["has_metadata"]:
            score += 0.2
        if metrics["word_count"] >= 50:
            score += 0.3
        
        metrics["quality_score"] = min(1.0, score)
        
        return metrics

# Global extractor instance
content_extractor = ContentExtractor()

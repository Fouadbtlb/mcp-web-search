import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup
from ..utils.llm_processor import LLMContentProcessor

logger = logging.getLogger(__name__)

class ContentExtractor:
    def __init__(self):
        self.min_content_length = 100
        self.max_content_length = 50000
        self.llm_processor = LLMContentProcessor()
        
        # Sélecteurs CSS pour différents types de contenu
        self.content_selectors = {
            'article': [
                'article', '[role="main"]', '.main-content', '.content',
                '.post-content', '.entry-content', '.article-content',
                '#content', '#main-content'
            ],
            'forum': [
                '.post-body', '.message-body', '.comment-content',
                '.reply-content', '.thread-content'
            ],
            'pdf': []  # PDFs traités différemment
        }
        
        # Tags à supprimer
        self.remove_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            'advertisement', '.ad', '.ads', '.advertisement',
            '.social-share', '.related-posts', '.comments'
        ]
    
    def extract(self, html: str, url: str, content_types: List[str] = ["article"]) -> Dict[str, Any]:
        """Extrait le contenu principal d'une page HTML"""
        try:
            # Nettoyage initial du HTML
            cleaned_html = self._clean_html(html)
            soup = BeautifulSoup(cleaned_html, 'lxml')
            
            # Extraction des métadonnées
            metadata = self._extract_metadata(soup, url)
            
            # Extraction du contenu selon le type
            extracted_content = {}
            
            for content_type in content_types:
                if content_type == "article":
                    content = self._extract_article_content(soup, html)
                elif content_type == "forum":
                    content = self._extract_forum_content(soup)
                elif content_type == "pdf":
                    content = self._extract_pdf_content(html)
                else:
                    content = self._extract_generic_content(soup)
                
                if content and len(content.strip()) >= self.min_content_length:
                    extracted_content[content_type] = content
                    break
            
            # Fallback vers contenu générique si rien trouvé
            if not extracted_content:
                generic_content = self._extract_generic_content(soup)
                if generic_content:
                    extracted_content["generic"] = generic_content
            
            # Sélection du meilleur contenu
            best_content = self._select_best_content(extracted_content)
            
            # Build basic result
            result = {
                "title": metadata.get("title", ""),
                "content": best_content[:self.max_content_length] if best_content else "",
                "canonical_url": metadata.get("canonical_url", url),
                "language": metadata.get("language", ""),
                "author": metadata.get("author", ""),
                "publish_date": metadata.get("publish_date", ""),
                "description": metadata.get("description", ""),
                "keywords": metadata.get("keywords", []),
                "content_type": self._detect_content_type(soup, url),
                "word_count": len(best_content.split()) if best_content else 0,
                "extraction_method": self._get_extraction_method(extracted_content)
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de {url}: {e}")
            return {
                "title": "",
                "content": "",
                "canonical_url": url,
                "error": str(e)
            }

    async def extract_for_llm(self, html: str, url: str, query: str = "", content_types: List[str] = ["article"]) -> Dict[str, Any]:
        """Enhanced extraction with LLM-optimized processing"""
        try:
            # First get basic extraction
            basic_result = self.extract(html, url, content_types)
            
            if not basic_result.get("content"):
                return basic_result
            
            # Process content for LLM consumption
            llm_result = await self.llm_processor.process_for_llm(
                basic_result["content"],
                basic_result,
                query
            )
            
            # Merge results
            return {
                **basic_result,
                "llm_optimized": llm_result
            }
            
        except Exception as e:
            logger.error(f"Error in LLM extraction for {url}: {e}")
            # Fallback to basic extraction
            return self.extract(html, url, content_types)

    def _clean_html(self, html: str) -> str:
        """Nettoie le HTML des éléments indésirables"""
        # Supprimer les commentaires HTML
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Supprimer les scripts inline
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Supprimer les styles inline
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        return html
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extrait les métadonnées de la page"""
        metadata = {}
        
        # Titre
        title_tag = soup.find('title')
        og_title = soup.find('meta', {'property': 'og:title'})
        twitter_title = soup.find('meta', {'name': 'twitter:title'})
        
        metadata["title"] = (
            og_title.get('content') if og_title else
            twitter_title.get('content') if twitter_title else
            title_tag.get_text().strip() if title_tag else ""
        )
        
        # URL canonique
        canonical = soup.find('link', {'rel': 'canonical'})
        og_url = soup.find('meta', {'property': 'og:url'})
        
        metadata["canonical_url"] = (
            canonical.get('href') if canonical else
            og_url.get('content') if og_url else url
        )
        
        # Description
        description = soup.find('meta', {'name': 'description'})
        og_description = soup.find('meta', {'property': 'og:description'})
        
        metadata["description"] = (
            description.get('content') if description else
            og_description.get('content') if og_description else ""
        )
        
        # Auteur
        author = soup.find('meta', {'name': 'author'})
        metadata["author"] = author.get('content') if author else ""
        
        # Langue
        html_tag = soup.find('html')
        metadata["language"] = html_tag.get('lang') if html_tag else ""
        
        # Mots-clés
        keywords = soup.find('meta', {'name': 'keywords'})
        if keywords:
            metadata["keywords"] = [kw.strip() for kw in keywords.get('content', '').split(',')]
        else:
            metadata["keywords"] = []
        
        # Date de publication
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'meta[name="publish_date"]',
            'time[datetime]'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                metadata["publish_date"] = date_elem.get('content') or date_elem.get('datetime', '')
                break
        else:
            metadata["publish_date"] = ""
        
        return metadata
    
    def _extract_article_content(self, soup: BeautifulSoup, html: str) -> Optional[str]:
        """Extract article content using BeautifulSoup only"""
        # Method 1: Try common article selectors
        content_selectors = [
            'article', 'main', '.content', '#content', '.post-content', 
            '.entry-content', '.article-body', '.story-body', '.post-body'
        ]
        
        for selector in content_selectors:
            try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove unwanted elements
                    for unwanted in content_elem.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                        unwanted.decompose()
                    
                    text = content_elem.get_text(separator=' ', strip=True)
                    if len(text) >= self.min_content_length:
                        return text
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
        
        # Method 2: Fallback to body content
        try:
            # Remove unwanted elements from the entire page
            for unwanted in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                unwanted.decompose()
            
            # Look for paragraphs
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if len(content) >= self.min_content_length:
                    return content
        except Exception as e:
            logger.debug(f"Paragraph extraction failed: {e}")
        
        # Method 3: Final fallback to body text
        try:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text) >= self.min_content_length:
                    return text[:self.max_content_length]  # Truncate if too long
        except Exception as e:
            logger.debug(f"Body extraction failed: {e}")
        
        return None
        # Méthode 3: Sélecteurs CSS spécialisés
        for selector in self.content_selectors['article']:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Supprimer les éléments indésirables
                for tag_name in self.remove_tags:
                    for tag in content_elem.find_all(tag_name):
                        tag.decompose()
                
                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) >= self.min_content_length:
                    return text
        
        return None
    
    def _extract_forum_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extrait le contenu de forum/discussion"""
        for selector in self.content_selectors['forum']:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) >= self.min_content_length:
                    return text
        
        return None
    
    def _extract_pdf_content(self, html: str) -> Optional[str]:
        """Extrait le contenu d'un PDF (nécessite des outils supplémentaires)"""
        # Pour l'instant, retourner None - nécessiterait PyPDF2 ou pdfplumber
        return None
    
    def _extract_generic_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extraction générique en dernier recours"""
        # Supprimer tous les éléments indésirables
        for tag_name in self.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Chercher le plus grand bloc de texte
        text_blocks = []
        
        # Essayer différents conteneurs
        containers = soup.find_all(['main', 'article', 'div'], class_=re.compile(r'content|main|article'))
        
        for container in containers:
            text = container.get_text(separator=' ', strip=True)
            if len(text) >= self.min_content_length:
                text_blocks.append(text)
        
        # Si pas de conteneurs spécifiques, prendre le body
        if not text_blocks:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text) >= self.min_content_length:
                    text_blocks.append(text)
        
        # Retourner le plus long bloc
        return max(text_blocks, key=len) if text_blocks else None
    
    def _select_best_content(self, extracted_content: Dict[str, str]) -> Optional[str]:
        """Sélectionne le meilleur contenu extrait"""
        if not extracted_content:
            return None
        
        # Priorité: article > forum > generic
        priority = ['article', 'forum', 'generic']
        
        for content_type in priority:
            if content_type in extracted_content:
                return extracted_content[content_type]
        
        # Retourner le premier disponible
        return next(iter(extracted_content.values()))
    
    def _detect_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Détecte le type de contenu de la page"""
        # Détecter par l'URL
        if any(pattern in url.lower() for pattern in ['/forum/', '/discussion/', '/thread/']):
            return 'forum'
        
        if url.lower().endswith('.pdf'):
            return 'pdf'
        
        # Détecter par les balises HTML
        if soup.find('article') or soup.find(attrs={'role': 'article'}):
            return 'article'
        
        if any(soup.find(class_=cls) for cls in ['post', 'comment', 'thread', 'forum']):
            return 'forum'
        
        return 'webpage'
    
    def _get_extraction_method(self, extracted_content: Dict[str, str]) -> str:
        """Retourne la méthode d'extraction utilisée"""
        if 'article' in extracted_content:
            return 'article_extraction'
        elif 'forum' in extracted_content:
            return 'forum_extraction'
        else:
            return 'generic_extraction'

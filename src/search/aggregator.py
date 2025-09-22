import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import random

logger = logging.getLogger(__name__)

class SearchAggregator:
    def __init__(self):
        # Updated with verified working SearxNG instances (as of 2025)
        self.searxng_instances = [
            "https://searx.be",
            "https://searx.tiekoetter.com",
            "https://searx.work",
            "https://searx.info",
            "https://searx.prvcy.eu"
        ]
        
        # Configuration des moteurs directs (gratuits)
        self.direct_engines = {
            "duckduckgo": "https://api.duckduckgo.com/",
            "startpage": "https://www.startpage.com/do/dsearch",
        }
        
        self.timeout = httpx.Timeout(10.0)  # Reduced timeout
        
    async def search(self, queries: List[str], sources: List[str], max_results: int = 10, freshness: str = "all") -> List[Dict]:
        """Enhanced search with LLM-optimized content and fresh news priority"""
        all_results = []
        query = queries[0] if queries else ""
        
        # Auto-detect news queries and prioritize freshness
        is_news_query = any(keyword in query.lower() for keyword in 
                           ['news', 'latest', 'recent', 'today', '2025', 'breaking', 'update'])
        
        if is_news_query and freshness == "all":
            freshness = "week"  # Default to recent for news queries
            logger.info(f"Auto-detected news query, setting freshness to 'week'")
        
        # Prioriser SearxNG avec contenu amélioré
        if "searxng" in sources:
            searxng_results = await self._search_searxng(query, max_results, freshness)
            all_results.extend(searxng_results)
        
        # Utiliser les moteurs directs si nécessaire avec extraction de contenu
        if len(all_results) < max_results // 2:
            direct_results = await self._search_direct_engines(query, max_results - len(all_results))
            all_results.extend(direct_results)
        
        # Si toujours pas de résultats, générer des résultats de démonstration
        if len(all_results) == 0:
            all_results = self._generate_demo_results(query, max_results)
            logger.info(f"Génération de {len(all_results)} résultats de démonstration pour: {query}")
        
        # Enhanced result ranking for LLM-ready content
        if all_results:
            # Sort by relevance and freshness instead of random shuffle
            all_results = self._rank_results_for_llm(all_results, query, is_news_query)
        
        
        return all_results[:max_results]
    
    async def _search_searxng(self, query: str, max_results: int, freshness: str) -> List[Dict]:
        """Recherche via SearxNG avec extraction de contenu améliorée"""
        results = []
        working_instances = []
        
        for instance in self.searxng_instances:
            try:
                params = {
                    "q": query,
                    "format": "json",
                    "engines": "google,bing,duckduckgo,startpage",
                    "categories": "news,general",  # Focus on news for fresh content
                    "pageno": 1,
                    "safesearch": 0
                }
                
                # Enhanced freshness handling for LLM-ready content
                if freshness != "all":
                    params["time_range"] = self._convert_freshness(freshness)
                elif "news" in query.lower() or "latest" in query.lower():
                    params["time_range"] = "week"  # Default to recent for news queries
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(f"{instance}/search", params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for idx, result in enumerate(data.get("results", [])):
                            # Enhanced result with more LLM-useful metadata
                            enhanced_result = {
                                "url": result.get("url"),
                                "title": result.get("title"),
                                "snippet": result.get("content", ""),
                                "engine": f"searxng-{instance.split('//')[-1]}",
                                "rank": idx + 1,
                                "score": self._calculate_relevance_score(result, query),
                                "published_date": result.get("publishedDate", ""),
                                "category": result.get("category", "general"),
                                "content_length": len(result.get("content", "")),
                            }
                            
                            # Try to extract more content for LLM processing
                            if result.get("url"):
                                enhanced_result = await self._enhance_result_content(enhanced_result)
                            
                            results.append(enhanced_result)
                        
                        if results:
                            working_instances.append(instance)
                            logger.info(f"SearxNG success: {len(results)} enhanced results from {instance.split('//')[-1]}")
                            return results[:max_results]
                        
            except httpx.TimeoutException:
                logger.debug(f"SearxNG timeout: {instance.split('//')[-1]}")
                continue
            except httpx.ConnectError:
                logger.debug(f"SearxNG connection failed: {instance.split('//')[-1]}")
                continue
            except Exception as e:
                logger.debug(f"SearxNG error {instance.split('//')[-1]}: {type(e).__name__}")
                continue
        
        # Only warn if NO instances worked
        if not working_instances:
            logger.warning("All SearxNG instances failed, falling back to direct search")
        
        return results[:max_results]
    
    async def _search_direct_engines(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced direct search with LLM-optimized content extraction"""
        results = []
        
        # Enhanced DuckDuckGo search with content extraction
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use DuckDuckGo HTML search for better results
                search_url = "https://html.duckduckgo.com/html"
                params = {
                    "q": f"{query} news latest 2025",  # Enhance query for freshness
                }
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                response = await client.post(search_url, data=params, headers=headers)
                
                if response.status_code == 200:
                    content = response.text
                    import re
                    from datetime import datetime
                    
                    # Enhanced result extraction with better patterns
                    result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]+)</a>'
                    snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
                    
                    urls_titles = re.findall(result_pattern, content)
                    snippets = re.findall(snippet_pattern, content)
                    
                    for i, (url, title) in enumerate(urls_titles[:max_results]):
                        snippet = snippets[i] if i < len(snippets) else "No description available"
                        
                        # Create enhanced result structure
                        enhanced_result = {
                            "url": url,
                            "title": title,
                            "snippet": snippet,
                            "engine": "duckduckgo-direct-enhanced",
                            "rank": i + 1,
                            "score": 1.0 - (i * 0.1),
                            "search_query": query,
                            "extraction_date": datetime.now().isoformat()[:10],
                            "content_enhanced": False,
                            "has_full_content": False
                        }
                        
                        # Enhance with full content extraction
                        enhanced_result = await self._enhance_result_content(enhanced_result)
                        results.append(enhanced_result)
                    
                    if results:
                        logger.info(f"Direct search enhanced: {len(results)} results with content extraction")
                        return results
                        
        except Exception as e:
            logger.warning(f"Enhanced direct search failed: {e}")
        
        # Fallback to simple DuckDuckGo instant answers
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "q": query,
                    "format": "json",
                    "no_redirect": "1",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Résultats instantanés
                    if data.get("AbstractText"):
                        results.append({
                            "url": data.get("AbstractURL", ""),
                            "title": data.get("Heading", query),
                            "snippet": data.get("AbstractText"),
                            "engine": "duckduckgo_instant",
                            "rank": 1,
                            "score": 0.8
                        })
                    
                    # Résultats connexes
                    for idx, topic in enumerate(data.get("RelatedTopics", [])[:5]):
                        if isinstance(topic, dict) and topic.get("FirstURL"):
                            results.append({
                                "url": topic.get("FirstURL"),
                                "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                                "snippet": topic.get("Text", ""),
                                "engine": "duckduckgo_related",
                                "rank": idx + 2,
                                "score": 0.6
                            })
        
        except Exception as e:
            logger.warning(f"Échec DuckDuckGo direct: {e}")
        
        # Si on a toujours pas assez de résultats, utiliser les instances SearxNG publiques
        if len(results) < max_results and len(results) < 3:
            try:
                public_instances = [
                    "https://searx.be",
                    "https://searx.tiekoetter.com", 
                    "https://search.sapti.me"
                ]
                
                for instance in public_instances:
                    try:
                        async with httpx.AsyncClient(timeout=self.timeout) as client:
                            # Essayer d'abord sans format JSON (certaines instances le bloquent)
                            response = await client.get(
                                f"{instance}/search",
                                params={
                                    "q": query,
                                    "format": "html",  # Format HTML plus largement supporté
                                    "category_general": "1"
                                }
                            )
                            
                            if response.status_code == 200 and len(response.text) > 1000:
                                # Simple extraction de résultats depuis HTML
                                html_text = response.text
                                
                                # Pattern simple pour extraire liens et titres
                                import re
                                link_pattern = r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
                                matches = re.findall(link_pattern, html_text)
                                
                                for idx, (url, title) in enumerate(matches[:max_results]):
                                    if url.startswith('http') and len(title.strip()) > 5:
                                        results.append({
                                            "url": url,
                                            "title": title.strip(),
                                            "snippet": f"Résultat de {instance} pour '{query}'",
                                            "engine": f"searxng_html_{instance}",
                                            "rank": len(results) + 1,
                                            "score": 0.5
                                        })
                                        
                                        if len(results) >= max_results:
                                            break
                                
                                if len(results) > 0:
                                    logger.info(f"Récupéré {len(results)} résultats via {instance} (HTML)")
                                    break
                                    
                    except Exception as e:
                        logger.debug(f"Instance publique {instance} échoue: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Échec instances publiques: {e}")
        
        return results[:max_results]
    
    def _convert_freshness(self, freshness: str) -> str:
        """Conversion du paramètre freshness pour SearxNG"""
        mapping = {
            "24h": "day",
            "7d": "week",
            "30d": "month"
        }
        return mapping.get(freshness, "")
    
    def _calculate_relevance_score(self, result: Dict, query: str) -> float:
        """Calcul simple du score de pertinence"""
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        query_lower = query.lower()
        
        score = 0.5  # Score de base
        
        # Bonus si le terme de recherche est dans le titre
        if query_lower in title:
            score += 0.3
        
        # Bonus si le terme de recherche est dans le contenu
        if query_lower in content:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_demo_results(self, query: str, max_results: int) -> List[Dict]:
        """Génère des résultats de démonstration quand les moteurs de recherche sont indisponibles"""
        demo_results = []
        
        # Sites de référence basés sur le type de requête
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['python', 'programming', 'code', 'dev']):
            demo_sites = [
                {"url": "https://docs.python.org/3/", "title": "Python 3 Documentation", "snippet": "Official Python documentation with tutorials and references"},
                {"url": "https://realpython.com/", "title": "Real Python - Learn Python Programming", "snippet": "Comprehensive Python tutorials and articles for developers"},
                {"url": "https://github.com/python", "title": "Python on GitHub", "snippet": "Official Python repositories and source code"},
                {"url": "https://stackoverflow.com/questions/tagged/python", "title": "Python Questions - Stack Overflow", "snippet": "Community Q&A for Python programming"},
                {"url": "https://www.python.org/", "title": "Python.org - Official Website", "snippet": "The official home of Python programming language"}
            ]
        elif any(word in query_lower for word in ['ai', 'artificial', 'intelligence', 'machine', 'learning']):
            demo_sites = [
                {"url": "https://arxiv.org/", "title": "arXiv.org - AI Research Papers", "snippet": "Latest research papers in artificial intelligence and machine learning"},
                {"url": "https://openai.com/", "title": "OpenAI", "snippet": "Leading AI research company developing artificial intelligence systems"},
                {"url": "https://www.tensorflow.org/", "title": "TensorFlow", "snippet": "Open-source machine learning framework by Google"},
                {"url": "https://pytorch.org/", "title": "PyTorch", "snippet": "Machine learning framework for researchers and developers"},
                {"url": "https://huggingface.co/", "title": "Hugging Face", "snippet": "Platform for sharing and collaborating on machine learning models"}
            ]
        elif any(word in query_lower for word in ['news', 'tech', 'technology']):
            demo_sites = [
                {"url": "https://techcrunch.com/", "title": "TechCrunch", "snippet": "Latest technology news and startup information"},
                {"url": "https://www.theverge.com/", "title": "The Verge", "snippet": "Technology, science, art, and culture news"},
                {"url": "https://arstechnica.com/", "title": "Ars Technica", "snippet": "Technology news, analysis, and reviews"},
                {"url": "https://www.wired.com/", "title": "WIRED", "snippet": "Technology trends, digital culture, and innovation"},
                {"url": "https://hacker-news.firebaseio.com/v0/topstories.json", "title": "Hacker News", "snippet": "Social news website focusing on computer science and entrepreneurship"}
            ]
        else:
            # Résultats génériques pour toute autre requête
            demo_sites = [
                {"url": "https://en.wikipedia.org/", "title": f"Wikipedia - Search for '{query}'", "snippet": f"Encyclopedia articles related to {query}"},
                {"url": "https://www.google.com/search?q=" + query.replace(" ", "+"), "title": f"Google Search Results for '{query}'", "snippet": f"Web search results for {query}"},
                {"url": "https://duckduckgo.com/?q=" + query.replace(" ", "+"), "title": f"DuckDuckGo Search for '{query}'", "snippet": f"Privacy-focused search results for {query}"},
                {"url": "https://github.com/search?q=" + query.replace(" ", "+"), "title": f"GitHub Search - '{query}'", "snippet": f"Open source projects and code related to {query}"},
                {"url": "https://www.reddit.com/search/?q=" + query.replace(" ", "%20"), "title": f"Reddit Discussions - '{query}'", "snippet": f"Community discussions about {query}"}
            ]
        
        for idx, site in enumerate(demo_sites[:max_results]):
            demo_results.append({
                "url": site["url"],
                "title": site["title"],
                "snippet": site["snippet"],
                "engine": "demo_fallback",
                "rank": idx + 1,
                "score": 0.7 - (idx * 0.1)  # Score décroissant
            })
        
        return demo_results

    async def _enhance_result_content(self, result: Dict) -> Dict:
        """Extract and enhance content for better LLM processing"""
        try:
            # Skip enhancement for demo results or if no URL
            if not result.get("url") or result.get("engine") == "demo_fallback":
                return result
            
            # Try to extract more content from the page
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                response = await client.get(result["url"], headers=headers)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Extract more useful information using simple parsing
                    import re
                    from datetime import datetime
                    
                    # Try to extract publication date
                    date_patterns = [
                        r'<time[^>]*datetime="([^"]*)"',
                        r'<meta[^>]*property="article:published_time"[^>]*content="([^"]*)"',
                        r'<meta[^>]*name="publish-date"[^>]*content="([^"]*)"',
                        r'"datePublished":"([^"]*)"'
                    ]
                    
                    for pattern in date_patterns:
                        date_match = re.search(pattern, content, re.IGNORECASE)
                        if date_match:
                            result["published_date"] = date_match.group(1)[:10]  # YYYY-MM-DD format
                            break
                    
                    # Extract better description/content
                    desc_patterns = [
                        r'<meta[^>]*name="description"[^>]*content="([^"]*)"',
                        r'<meta[^>]*property="og:description"[^>]*content="([^"]*)"',
                        r'<meta[^>]*name="twitter:description"[^>]*content="([^"]*)"'
                    ]
                    
                    for pattern in desc_patterns:
                        desc_match = re.search(pattern, content, re.IGNORECASE)
                        if desc_match and len(desc_match.group(1)) > len(result.get("snippet", "")):
                            result["snippet"] = desc_match.group(1)
                            result["content_enhanced"] = True
                            break
                    
                    # Try to extract article content for LLM processing
                    article_patterns = [
                        r'<article[^>]*>(.*?)</article>',
                        r'<div[^>]*class="[^"]*article-content[^"]*"[^>]*>(.*?)</div>',
                        r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>'
                    ]
                    
                    for pattern in article_patterns:
                        article_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                        if article_match:
                            article_html = article_match.group(1)
                            # Simple HTML tag removal
                            article_text = re.sub(r'<[^>]+>', ' ', article_html)
                            article_text = re.sub(r'\s+', ' ', article_text).strip()
                            
                            if len(article_text) > 200:  # Only use if substantial content
                                result["full_content"] = article_text[:2000]  # Limit for LLM context
                                result["has_full_content"] = True
                                break
                    
                    # Add freshness score based on date
                    if result.get("published_date"):
                        try:
                            pub_date = datetime.fromisoformat(result["published_date"][:10])
                            days_old = (datetime.now() - pub_date).days
                            
                            # Fresher content gets higher scores
                            if days_old <= 1:
                                result["freshness_score"] = 1.0
                            elif days_old <= 7:
                                result["freshness_score"] = 0.8
                            elif days_old <= 30:
                                result["freshness_score"] = 0.6
                            else:
                                result["freshness_score"] = 0.3
                                
                            result["days_old"] = days_old
                        except:
                            result["freshness_score"] = 0.5
                    
                    logger.debug(f"Enhanced content for {result['url'][:50]}...")
                    
        except Exception as e:
            logger.debug(f"Content enhancement failed for {result.get('url', '')}: {e}")
        
        return result

    def _rank_results_for_llm(self, results: List[Dict], query: str, is_news_query: bool) -> List[Dict]:
        """Rank results optimally for LLM processing"""
        
        def calculate_llm_score(result: Dict) -> float:
            score = result.get('score', 0.5)
            
            # Boost for fresh content
            if result.get('freshness_score'):
                score += result['freshness_score'] * 0.3
            
            # Boost for enhanced content
            if result.get('has_full_content'):
                score += 0.2
            if result.get('content_enhanced'):
                score += 0.1
                
            # Boost for longer, more informative snippets
            snippet_len = len(result.get('snippet', ''))
            if snippet_len > 200:
                score += 0.1
            elif snippet_len > 100:
                score += 0.05
                
            # Boost for news sources if news query
            if is_news_query:
                news_domains = ['techcrunch.com', 'reuters.com', 'bbc.com', 'cnn.com', 
                              'news.google.com', 'bloomberg.com', 'theverge.com']
                if any(domain in result.get('url', '') for domain in news_domains):
                    score += 0.2
            
            # Boost for recent publication dates
            days_old = result.get('days_old', 365)
            if days_old <= 7:
                score += 0.15
            elif days_old <= 30:
                score += 0.1
                
            return min(score, 1.0)  # Cap at 1.0
        
        # Calculate LLM scores and sort
        for result in results:
            result['llm_score'] = calculate_llm_score(result)
        
        # Sort by LLM score (highest first)
        sorted_results = sorted(results, key=lambda x: x['llm_score'], reverse=True)
        
        logger.debug(f"Ranked {len(sorted_results)} results for LLM optimization")
        return sorted_results

    async def cleanup(self):
        """Cleanup resources"""
        # No cleanup needed for lightweight aggregator
        pass

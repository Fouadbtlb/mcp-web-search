import logging
import re
from typing import List, Dict, Any, Set
from urllib.parse import urlparse
import difflib
from collections import Counter

logger = logging.getLogger(__name__)

class QualityFilter:
    def __init__(self):
        # Domaines de faible qualité ou spam
        self.low_quality_domains = {
            'spam.com', 'clickbait.net', 'fake-news.org',
            'ad-farm.com', 'content-mill.net', 'scraper-site.com'
        }
        
        # Domaines de haute qualité
        self.high_quality_domains = {
            'wikipedia.org', 'github.com', 'stackoverflow.com',
            'reddit.com', 'medium.com', 'arxiv.org',
            'scholar.google.com', 'researchgate.net',
            'lemonde.fr', 'lefigaro.fr', 'liberation.fr',
            'france24.com', 'rfi.fr', 'francetvinfo.fr'
        }
        
        # Patterns de contenu de mauvaise qualité
        self.spam_patterns = [
            r'click here for more',
            r'amazing trick',
            r'you won\'t believe',
            r'doctors hate',
            r'one weird trick',
            r'shocking truth',
            r'\d+ reasons why',
            r'this will blow your mind'
        ]
        
        # Indicateurs de qualité dans le contenu
        self.quality_indicators = {
            'academic': ['study', 'research', 'analysis', 'étude', 'recherche', 'analyse'],
            'authoritative': ['official', 'government', 'university', 'institut', 'officiel'],
            'recent': ['2023', '2024', '2025', 'recent', 'latest', 'récent', 'dernier'],
            'detailed': ['detailed', 'comprehensive', 'complete', 'détaillé', 'complet']
        }
        
        # Seuils de qualité
        self.thresholds = {
            'min_content_length': 100,
            'max_content_length': 100000,
            'min_title_length': 10,
            'max_title_length': 200,
            'min_snippet_length': 20,
            'similarity_threshold': 0.85  # Pour déduplication
        }
    
    def score_serp_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score et classe les résultats SERP par qualité"""
        scored_results = []
        
        for result in results:
            score = self._calculate_serp_score(result)
            result['quality_score'] = score
            scored_results.append(result)
        
        # Trier par score décroissant
        scored_results.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return scored_results
    
    def score_content_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score les résultats avec contenu extrait"""
        scored_results = []
        
        for result in results:
            score = self._calculate_content_score(result)
            result['content_quality_score'] = score
            scored_results.append(result)
        
        # Trier par score décroissant
        scored_results.sort(key=lambda x: x.get('content_quality_score', 0), reverse=True)
        
        return scored_results
    
    def deduplicate_and_filter(self, results: List[Dict[str, Any]], min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Déduplique et filtre les résultats par qualité (version permissive pour tests)"""
        if not results:
            return []
        
        # Filtrage par score minimum très bas pour permettre les résultats de démonstration
        filtered_results = [r for r in results if r.get('quality_score', 0.5) >= min_score]
        
        # Déduplication
        deduplicated = self._remove_duplicates(filtered_results)
        
        # Filtrage final par qualité de contenu
        final_results = self._final_quality_filter(deduplicated)
        
        return final_results
    
    def _calculate_serp_score(self, result: Dict[str, Any]) -> float:
        """Calcule le score de qualité d'un résultat SERP"""
        score = 0.5  # Score de base
        
        url = result.get('url', '')
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        
        # Score du domaine
        domain_score = self._score_domain(url)
        score += domain_score * 0.3
        
        # Score du titre
        title_score = self._score_title(title)
        score += title_score * 0.25
        
        # Score du snippet
        snippet_score = self._score_snippet(snippet)
        score += snippet_score * 0.2
        
        # Score de position (rang)
        rank = result.get('rank', 10)
        rank_score = max(0, (10 - rank) / 10)
        score += rank_score * 0.15
        
        # Pénalités pour spam
        spam_penalty = self._detect_spam_content(title + ' ' + snippet)
        score -= spam_penalty * 0.1
        
        return max(0, min(1, score))
    
    def _calculate_content_score(self, result: Dict[str, Any]) -> float:
        """Calcule le score de qualité du contenu extrait"""
        score = result.get('quality_score', 0.5)  # Score SERP de base
        
        content = result.get('content', '')
        title = result.get('title', '')
        
        # Score de longueur du contenu
        content_length_score = self._score_content_length(content)
        score += content_length_score * 0.25
        
        # Score de qualité du contenu
        content_quality_score = self._score_content_quality(content)
        score += content_quality_score * 0.3
        
        # Score de cohérence titre-contenu
        coherence_score = self._score_title_content_coherence(title, content)
        score += coherence_score * 0.15
        
        # Bonus pour indicateurs de qualité
        quality_bonus = self._calculate_quality_bonus(content)
        score += quality_bonus * 0.1
        
        return max(0, min(1, score))
    
    def _score_domain(self, url: str) -> float:
        """Score la qualité du domaine"""
        if not url:
            return 0
        
        try:
            domain = urlparse(url).netloc.lower()
            
            # Domaines de haute qualité
            if any(hq_domain in domain for hq_domain in self.high_quality_domains):
                return 0.8
            
            # Domaines de faible qualité
            if any(lq_domain in domain for lq_domain in self.low_quality_domains):
                return 0.1
            
            # Score basé sur la structure du domaine
            score = 0.5
            
            # Bonus pour domaines gouvernementaux et éducatifs
            if domain.endswith(('.gov', '.edu', '.gouv.fr', '.fr')):
                score += 0.2
            
            # Bonus pour HTTPS
            if url.startswith('https://'):
                score += 0.1
            
            # Pénalité pour sous-domaines suspects
            if len(domain.split('.')) > 3:
                score -= 0.1
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.warning(f"Erreur analyse domaine {url}: {e}")
            return 0.3
    
    def _score_title(self, title: str) -> float:
        """Score la qualité du titre"""
        if not title or len(title) < self.thresholds['min_title_length']:
            return 0.1
        
        if len(title) > self.thresholds['max_title_length']:
            return 0.3
        
        score = 0.5
        
        # Pénalité pour titres en majuscules
        if title.isupper():
            score -= 0.2
        
        # Pénalité pour caractères répétés
        if re.search(r'(.)\1{3,}', title):
            score -= 0.2
        
        # Bonus pour structure normale
        if re.match(r'^[A-Z][a-z]', title):
            score += 0.1
        
        # Pénalité pour trop de ponctuation
        punct_ratio = len(re.findall(r'[!?.,;:]', title)) / len(title)
        if punct_ratio > 0.1:
            score -= punct_ratio * 0.5
        
        return max(0, min(1, score))
    
    def _score_snippet(self, snippet: str) -> float:
        """Score la qualité du snippet"""
        if not snippet or len(snippet) < self.thresholds['min_snippet_length']:
            return 0.1
        
        score = 0.5
        
        # Bonus pour longueur appropriée
        if 50 <= len(snippet) <= 200:
            score += 0.2
        
        # Bonus pour phrases complètes
        if snippet.endswith('.'):
            score += 0.1
        
        # Pénalité pour texte tronqué brutalement
        if snippet.endswith('...') or snippet.endswith('…'):
            score += 0.05  # Léger bonus car c'est normal
        
        return max(0, min(1, score))
    
    def _score_content_length(self, content: str) -> float:
        """Score basé sur la longueur du contenu"""
        if not content:
            return 0
        
        length = len(content)
        
        if length < self.thresholds['min_content_length']:
            return 0.1
        
        if length > self.thresholds['max_content_length']:
            return 0.3  # Pénalité pour contenu trop long
        
        # Score optimal entre 500 et 5000 caractères
        if 500 <= length <= 5000:
            return 1.0
        elif 200 <= length < 500:
            return 0.7
        elif 5000 < length <= 20000:
            return 0.8
        else:
            return 0.5
    
    def _score_content_quality(self, content: str) -> float:
        """Score la qualité du contenu textuel"""
        if not content:
            return 0
        
        score = 0.5
        
        # Compter les phrases
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        
        if sentence_count > 5:
            score += 0.2
        
        # Analyser la complexité du vocabulaire
        words = content.lower().split()
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        if vocabulary_richness > 0.5:
            score += 0.2
        elif vocabulary_richness > 0.3:
            score += 0.1
        
        # Détecter le contenu structuré
        if re.search(r'\n\s*[-•*]\s*', content):  # Listes
            score += 0.1
        
        if re.search(r'\n\s*\d+\.\s*', content):  # Listes numérotées
            score += 0.1
        
        return max(0, min(1, score))
    
    def _score_title_content_coherence(self, title: str, content: str) -> float:
        """Score la cohérence entre titre et contenu"""
        if not title or not content:
            return 0
        
        # Extraire les mots-clés du titre
        title_words = set(re.findall(r'\b\w{3,}\b', title.lower()))
        content_words = set(re.findall(r'\b\w{3,}\b', content.lower()[:1000]))  # Premier kilocaractère
        
        if not title_words:
            return 0
        
        # Calculer l'intersection
        common_words = title_words & content_words
        coherence_ratio = len(common_words) / len(title_words)
        
        return coherence_ratio
    
    def _calculate_quality_bonus(self, content: str) -> float:
        """Calcule les bonus de qualité basés sur les indicateurs"""
        bonus = 0
        content_lower = content.lower()
        
        for category, keywords in self.quality_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > 0:
                bonus += min(0.1, matches * 0.02)  # Max 0.1 par catégorie
        
        return bonus
    
    def _detect_spam_content(self, text: str) -> float:
        """Détecte le contenu spam et retourne un score de pénalité"""
        if not text:
            return 0
        
        text_lower = text.lower()
        spam_score = 0
        
        # Détecter les patterns de spam
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                spam_score += 0.2
        
        # Détecter la répétition excessive
        words = text_lower.split()
        if len(words) > 5:
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.3:  # Plus de 30% de répétition
                spam_score += 0.3
        
        return min(1.0, spam_score)
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Supprime les doublons basés sur la similarité"""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_contents = []
        
        for result in results:
            content = result.get('content', result.get('snippet', ''))
            title = result.get('title', '')
            combined_text = title + ' ' + content
            
            # Vérifier la similarité avec les résultats déjà vus
            is_duplicate = False
            
            for seen_content in seen_contents:
                similarity = difflib.SequenceMatcher(None, combined_text.lower(), seen_content.lower()).ratio()
                if similarity > self.thresholds['similarity_threshold']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(combined_text)
        
        return unique_results
    
    def _final_quality_filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtrage final de qualité"""
        filtered = []
        
        for result in results:
            # Critères de rejet strict
            title = result.get('title', '')
            content = result.get('content', '')
            url = result.get('url', '')
            
            # Rejeter si titre vide ou trop court
            if not title or len(title) < 5:
                continue
            
            # Rejeter si URL invalide
            if not url or not url.startswith(('http://', 'https://')):
                continue
            
            # Rejeter si contenu trop court et pas de snippet
            if len(content) < 50 and len(result.get('snippet', '')) < 20:
                continue
            
            filtered.append(result)
        
        return filtered

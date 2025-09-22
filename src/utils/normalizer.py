import re
import logging
from typing import List, Dict, Any
import unicodedata
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class QueryNormalizer:
    def __init__(self):
        # Mots vides en français et anglais (version réduite)
        self.stop_words = {
            'fr': {
                'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais',
                'donc', 'car', 'ni', 'que', 'qui', 'quoi', 'dont', 'où', 'ce', 'cette',
                'ces', 'ceux', 'dans', 'sur', 'avec', 'par', 'pour', 'sans', 'sous',
                'vers', 'chez', 'entre', 'jusqu', 'depuis', 'pendant', 'avant', 'après',
                'est', 'sont', 'était', 'étaient', 'sera', 'seront', 'avoir', 'être',
                'faire', 'aller', 'venir', 'voir', 'savoir', 'pouvoir', 'vouloir',
                'falloir', 'devoir', 'très', 'plus', 'moins', 'aussi', 'comme', 'tout',
                'tous', 'toute', 'toutes', 'même', 'autre', 'autres', 'bien', 'encore',
                'déjà', 'jamais', 'toujours', 'souvent', 'parfois', 'quelquefois'
            },
            'en': {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
                'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there'
            }
        }
        
        # Synonymes pour expansion de requête
        self.synonyms = {
            'voiture': ['auto', 'automobile', 'véhicule', 'car'],
            'maison': ['habitation', 'logement', 'résidence', 'domicile', 'home'],
            'travail': ['emploi', 'job', 'boulot', 'profession', 'métier', 'work'],
            'ordinateur': ['pc', 'computer', 'machine', 'laptop'],
            'téléphone': ['phone', 'mobile', 'smartphone', 'portable'],
            'internet': ['web', 'net', 'réseau', 'network'],
            'recherche': ['search', 'chercher', 'trouver', 'find'],
            'information': ['info', 'données', 'data', 'renseignement']
        }
        
        # Expressions courantes à préserver
        self.phrases_to_preserve = [
            r'\b\d+\b',  # Nombres
            r'\b[A-Z]{2,}\b',  # Acronymes
            r'\b\w+\.\w+\b',  # URLs partielles
            r'["\']([^"\']+)["\']',  # Expressions entre guillemets
        ]
    
    def normalize(self, query: str) -> str:
        """Normalise une requête de recherche (version simplifiée)"""
        if not query or not query.strip():
            return ""
        
        # Simple normalisation sans préservation complexe
        normalized = query.strip()
        
        # Suppression des espaces multiples
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def expand_hyde(self, query: str) -> List[str]:
        """
        Expansion HyDE (Hypothetical Document Embeddings)
        Génère des variantes de la requête pour améliorer la recherche
        """
        variations = [query]  # Requête originale
        
        # Expansion par synonymes
        synonym_variations = self._expand_with_synonyms(query)
        variations.extend(synonym_variations)
        
        # Reformulations contextuelles
        contextual_variations = self._generate_contextual_variations(query)
        variations.extend(contextual_variations)
        
        # Suppression des doublons
        unique_variations = list(dict.fromkeys(variations))
        
        # Limiter le nombre de variations pour éviter la surcharge
        return unique_variations[:5]
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extrait les mots-clés importants d'une requête"""
        normalized = self.normalize(query)
        
        # Diviser en mots
        words = normalized.split()
        
        # Filtrer les mots courts et les mots vides
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                not self._is_stopword(word) and 
                not word.isdigit()):
                keywords.append(word)
        
        return keywords
    
    def detect_language(self, query: str) -> str:
        """Détecte la langue probable de la requête (simpliste)"""
        # Comptage des mots vides par langue
        words = query.lower().split()
        fr_count = sum(1 for word in words if word in self.stop_words['fr'])
        en_count = sum(1 for word in words if word in self.stop_words['en'])
        
        # Caractères spéciaux français
        fr_chars = len(re.findall(r'[àâäéèêëïîôöùûüÿç]', query.lower()))
        
        if fr_chars > 0 or fr_count > en_count:
            return 'fr'
        elif en_count > 0:
            return 'en'
        else:
            return 'auto'  # Indéterminé
    
    def url_encode(self, query: str) -> str:
        """Encode la requête pour utilisation en URL"""
        return quote_plus(query.encode('utf-8'))
    
    def _should_remove_stopwords(self, query: str) -> bool:
        """Détermine s'il faut supprimer les mots vides"""
        words = query.split()
        
        # Ne pas supprimer si la requête est très courte
        if len(words) <= 3:
            return False
        
        # Ne pas supprimer si plus de 60% sont des mots vides (question probable)
        stopword_ratio = sum(1 for word in words if self._is_stopword(word)) / len(words)
        if stopword_ratio > 0.6:
            return False
        
        return True
    
    def _remove_stopwords(self, query: str) -> str:
        """Supprime les mots vides de la requête"""
        words = query.split()
        filtered_words = [word for word in words if not self._is_stopword(word)]
        return ' '.join(filtered_words)
    
    def _is_stopword(self, word: str) -> bool:
        """Vérifie si un mot est un mot vide"""
        word_lower = word.lower()
        return (word_lower in self.stop_words['fr'] or 
                word_lower in self.stop_words['en'])
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Génère des variations en utilisant les synonymes"""
        variations = []
        words = query.lower().split()
        
        for main_word, synonyms in self.synonyms.items():
            if main_word in words:
                # Remplacer par chaque synonyme
                for synonym in synonyms[:2]:  # Limiter à 2 synonymes
                    new_query = query.replace(main_word, synonym)
                    if new_query != query:
                        variations.append(new_query)
        
        return variations
    
    def _generate_contextual_variations(self, query: str) -> List[str]:
        """Génère des variations contextuelles"""
        variations = []
        
        # Préfixes courants pour différents types de requêtes
        prefixes = {
            'definition': ['qu\'est-ce que', 'définition', 'what is', 'definition of'],
            'how_to': ['comment', 'how to', 'tutorial'],
            'comparison': ['vs', 'versus', 'comparaison', 'différence'],
            'best': ['meilleur', 'best', 'top', 'recommandation']
        }
        
        query_lower = query.lower()
        
        # Détecter le type de requête et ajouter des variations
        if any(word in query_lower for word in ['comment', 'how']):
            variations.extend([
                f"tutorial {query}",
                f"guide {query}",
                f"étapes {query}"
            ])
        elif any(word in query_lower for word in ['meilleur', 'best', 'top']):
            variations.extend([
                f"recommandation {query}",
                f"avis {query}",
                f"comparatif {query}"
            ])
        elif any(word in query_lower for word in ['qu\'est', 'what is', 'définition']):
            variations.extend([
                f"explication {query}",
                f"signification {query}"
            ])
        
        # Supprimer les variations trop longues ou identiques
        variations = [v for v in variations if len(v.split()) <= 12 and v != query]
        
        return variations[:3]  # Limiter à 3 variations contextuelles

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import pickle
import os
from datetime import datetime
import json

# Optional ML dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None
    MultinomialNB = None
    Pipeline = None
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class QueryIntentClassifier:
    """NLP-based query intent classification for search optimization"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.intents = [
            "informational",  # Facts, definitions, explanations - most common for LLMs
            "instructional",  # How-to, tutorials, step-by-step guides
            "analytical",     # Comparisons, analysis, research, academic content  
            "current",        # Recent news, latest developments, current events
            "general"         # Fallback for unclear intents
        ]
        
        # Intent-specific patterns for rule-based classification
        self.intent_patterns = {
            "informational": [
                r"\b(what is|what are|define|definition|meaning|explain|who is|when did|where is|why|how does)\b",
                r"\b(tell me about|information about|details|overview|summary)\b",
                r"\b(history of|background|origin|theory|concept)\b"
            ],
            "instructional": [
                r"\b(how to|how do|how can|steps to|tutorial|guide|instructions|walkthrough)\b",
                r"\b(learn|teach|show me|demonstrate|method|procedure|process)\b",
                r"\b(install|setup|configure|implement|create|build|make)\b"
            ],
            "analytical": [
                r"\b(vs|versus|compare|comparison|difference|analysis|evaluate|assess)\b",
                r"\b(pros and cons|advantages|disadvantages|better than|best|worst)\b",
                r"\b(research|study|review|examination|investigation|survey)\b"
            ],
            "current": [
                r"\b(latest|recent|new|current|today|now|breaking|update|development)\b",
                r"\b(news|announced|released|launched|trending|happening)\b",
                r"\b(2024|2025|this year|this month|recently|just|fresh)\b"
            ]
        }
        
        # Training data for ML model (optimized for general LLM use)
        self.training_data = [
            # Informational queries
            ("what is machine learning", "informational"),
            ("define artificial intelligence", "informational"),
            ("explain quantum computing", "informational"),
            ("who invented the internet", "informational"),
            ("history of democracy", "informational"),
            ("meaning of sustainability", "informational"),
            ("overview of climate change", "informational"),
            ("details about renewable energy", "informational"),
            ("tell me about blockchain", "informational"),
            ("what are neural networks", "informational"),
            
            # Instructional queries
            ("how to learn programming", "instructional"),
            ("steps to start a business", "instructional"),
            ("tutorial for beginners", "instructional"),
            ("guide to healthy eating", "instructional"),
            ("how to improve productivity", "instructional"),
            ("instructions for meditation", "instructional"),
            ("learn data science", "instructional"),
            ("how to build confidence", "instructional"),
            ("create a budget plan", "instructional"),
            ("setup development environment", "instructional"),
            
            # Analytical queries
            ("python vs javascript comparison", "analytical"),
            ("analyze market trends", "analytical"),
            ("compare smartphones", "analytical"),
            ("research on climate policies", "analytical"),
            ("evaluate investment options", "analytical"),
            ("study user behavior", "analytical"),
            ("pros and cons of remote work", "analytical"),
            ("best practices for security", "analytical"),
            ("review of new technologies", "analytical"),
            ("assessment of economic impact", "analytical"),
            
            # Current queries
            ("latest AI developments", "current"),
            ("recent tech news", "current"),
            ("current market trends", "current"),
            ("breaking news today", "current"),
            ("new research findings", "current"),
            ("recent policy changes", "current"),
            ("latest software updates", "current"),
            ("current global events", "current"),
            ("trending technologies 2025", "current"),
            ("recent scientific discoveries", "current"),
            
            # General queries
            ("general information", "general"),
            ("miscellaneous topics", "general"),
            ("random question", "general"),
            ("various subjects", "general"),
        ]
    
    def initialize(self):
        """Initialize or load the classification model"""
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available, using rule-based classification only")
            self.model = None
            return
            
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "intent_classifier.pkl")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded pre-trained intent classifier")
                return
            except Exception as e:
                logger.warning(f"Failed to load saved model: {e}")
        
        # Train new model if sklearn is available
        if HAS_SKLEARN:
            self._train_model()
            self._save_model(model_path)
        else:
            self.model = None
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify query intent with confidence score
        Returns: (intent, confidence)
        """
        if not query or not query.strip():
            return "general", 0.0
        
        query_lower = query.lower().strip()
        
        # First try rule-based classification
        rule_intent, rule_confidence = self._rule_based_classify(query_lower)
        
        # If high confidence from rules, use it
        if rule_confidence > 0.8:
            return rule_intent, rule_confidence
        
        # Otherwise use ML model if available
        if self.model:
            try:
                ml_intent, ml_confidence = self._ml_classify(query_lower)
                
                # Combine rule-based and ML predictions
                if rule_confidence > 0.5 and ml_confidence > 0.5:
                    if rule_intent == ml_intent:
                        combined_confidence = (rule_confidence + ml_confidence) / 2
                        return rule_intent, min(0.95, combined_confidence)
                    else:
                        # Prefer ML if confidences are similar
                        if ml_confidence >= rule_confidence:
                            return ml_intent, ml_confidence
                        else:
                            return rule_intent, rule_confidence
                
                # Use the higher confidence prediction
                if ml_confidence > rule_confidence:
                    return ml_intent, ml_confidence
                else:
                    return rule_intent, rule_confidence
                    
            except Exception as e:
                logger.error(f"ML classification failed: {e}")
                return rule_intent, rule_confidence
        
        return rule_intent, rule_confidence
    
    def get_search_strategy(self, intent: str) -> Dict[str, Any]:
        """Get optimized search strategy based on classified intent"""
        strategies = {
            "informational": {
                "sources": ["searxng", "duckduckgo"],
                "freshness": "all",
                "content_types": ["article"],
                "require_full_fetch": False,  # Fast response for facts
                "snippet_priority": True,
                "max_results": 8
            },
            "instructional": {
                "sources": ["searxng"],
                "freshness": "all", 
                "content_types": ["article"],
                "require_full_fetch": True,  # Need full content for steps
                "snippet_priority": False,
                "max_results": 6
            },
            "analytical": {
                "sources": ["searxng"],
                "freshness": "all",
                "content_types": ["article"],
                "require_full_fetch": True,  # Need complete analysis
                "snippet_priority": False,
                "max_results": 8
            },
            "current": {
                "sources": ["searxng", "duckduckgo"],
                "freshness": "day",  # Recent results only
                "content_types": ["article"],
                "require_full_fetch": False,
                "snippet_priority": True,
                "max_results": 10
            },
            "general": {
                "sources": ["searxng", "duckduckgo"],
                "freshness": "all",
                "content_types": ["article"],
                "require_full_fetch": False,
                "snippet_priority": True,
                "max_results": 8
            }
        }
        
        return strategies.get(intent, strategies["general"])
    
    def _rule_based_classify(self, query: str) -> Tuple[str, float]:
        """Rule-based classification using regex patterns"""
        max_score = 0.0
        best_intent = "general"
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matches += 1
                    # Weight based on pattern specificity
                    pattern_weight = 1.0 / len(patterns)  # More specific intents have fewer patterns
                    score += pattern_weight
            
            # Normalize score based on pattern matches
            if matches > 0:
                score = min(1.0, score * (matches / len(patterns)) * 2)  # Boost for multiple matches
                
                if score > max_score:
                    max_score = score
                    best_intent = intent
        
        # Ensure minimum confidence for rule-based classification
        confidence = max(0.3, max_score) if max_score > 0 else 0.2
        
        return best_intent, confidence
    
    def _ml_classify(self, query: str) -> Tuple[str, float]:
        """Machine learning based classification"""
        if not self.model:
            return "general", 0.0
        
        try:
            # Predict probabilities for all classes
            probabilities = self.model.predict_proba([query])[0]
            classes = self.model.classes_
            
            # Get the highest probability prediction
            max_prob_idx = probabilities.argmax()
            predicted_intent = classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            return predicted_intent, float(confidence)
            
        except Exception as e:
            logger.error(f"ML classification error: {e}")
            return "general", 0.0
    
    def _train_model(self):
        """Train the ML classification model"""
        if not HAS_SKLEARN:
            logger.warning("Cannot train ML model: scikit-learn not available")
            return
            
        logger.info("Training intent classification model...")
        
        try:
            # Prepare training data
            texts = [item[0] for item in self.training_data]
            labels = [item[1] for item in self.training_data]
            
            # Create and train pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),  # Use both unigrams and bigrams
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            self.model.fit(texts, labels)
            
            logger.info(f"Model trained with {len(texts)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.model = None
    
    def _save_model(self, path: str):
        """Save the trained model to disk"""
        if self.model:
            try:
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Model saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
    
    def add_training_example(self, query: str, intent: str):
        """Add a new training example and retrain if needed"""
        if intent in self.intents:
            self.training_data.append((query.lower().strip(), intent))
            logger.info(f"Added training example: '{query}' -> {intent}")
            
            # Retrain if we have enough new examples
            if len(self.training_data) % 20 == 0:
                logger.info("Retraining model with updated examples...")
                self._train_model()
    
    def get_intent_info(self, intent: str) -> Dict[str, Any]:
        """Get detailed information about an intent"""
        descriptions = {
            "factual": "Seeking factual information, definitions, or explanations",
            "how_to": "Looking for instructions, tutorials, or step-by-step guides", 
            "code": "Programming-related queries, code examples, or syntax help",
            "academic": "Research-oriented queries, scholarly content, or studies",
            "news": "Current events, recent developments, or breaking news",
            "shopping": "Product information, prices, reviews, or purchasing advice",
            "local": "Location-based information, nearby services, or directions",
            "comparison": "Comparing products, services, or concepts",
            "troubleshoot": "Problem-solving, error fixing, or technical support",
            "general": "General queries that don't fit specific categories"
        }
        
        return {
            "intent": intent,
            "description": descriptions.get(intent, "Unknown intent"),
            "strategy": self.get_search_strategy(intent),
            "patterns": self.intent_patterns.get(intent, [])
        }
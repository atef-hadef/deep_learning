"""
Review Filter Service
====================

Service pour filtrer intelligemment les posts et garder uniquement ceux qui ressemblent
à de vrais avis / retours d'expérience.

🚀 V2 SIMPLIFICATION: Filtrage 100% heuristique (mots/phrases clés)
Stratégie:
- Détection par mots/phrases clés (anglais + français)
- Patterns de notes/scores (5/5, 10/10, etc.)
- Règles simples (longueur, pronoms 1ère personne)
- Aucun modèle ML lourd (zero-shot BART MNLI désactivé pour performance)

Exclut:
- Questions ("Should I buy?", "Is it worth?")
- Photos/showcases ("Shot on iPhone")
- News/announcements
- Posts trop courts ou non-informatifs
"""

import logging
import re
from typing import List, Tuple
# TODO: Réactiver transformers pipeline si besoin du modèle zero-shot plus tard
# from transformers import pipeline
from app.config import get_settings
from app.schemas.responses import Post

logger = logging.getLogger(__name__)

# ===== CONSTANTES POUR DÉTECTION HEURISTIQUE (V2 SIMPLIFICATION) =====

# Mots/phrases clés indiquant un avis (anglais)
REVIEW_KEYWORDS_EN = {
    'review', 'my review', 'my experience', 'my opinion',
    'pros and cons', 'pros:', 'cons:',
    'i bought', 'i purchased', 'after 1 week', 'after 2 weeks', 'after a month',
    'after using', 'after owning',
    'would recommend', "wouldn't recommend", "i don't recommend",
    'highly recommend', 'not recommend',
    'worth it', 'not worth it', "isn't worth", "not worth the money",
    'my thoughts', 'honest review', 'full review',
    'best purchase', 'worst purchase',
    'love it', 'hate it', 'disappointed', 'satisfied',
    'regret buying', 'no regrets',
    'upgrading from', 'upgraded from', 'switched from',
    'coming from'
}

# Mots/phrases clés indiquant un avis (français)
REVIEW_KEYWORDS_FR = {
    'avis', 'mon avis', "retour d'expérience", 'mon expérience',
    'je recommande', 'je ne recommande pas', 'je déconseille',
    "j'ai acheté", "j'ai testé", 'après 1 semaine', 'après 2 semaines',
    'après un mois', "après l'avoir utilisé",
    'ça vaut le coup', 'ça vaut pas le coup',
    'pour et contre', 'points positifs', 'points négatifs',
    'très satisfait', 'déçu', 'satisfait',
    'meilleur achat', 'pire achat',
    "j'adore", 'je déteste', 'regrette',
    'passage de', 'venant de'
}

# Patterns de notes/scores
RATING_PATTERNS = [
    r'\b([0-9]|10)/10\b',          # 5/10, 10/10
    r'\b[1-5]/5\b',                 # 3/5
    r'\b[1-5]\s*stars?\b',          # 4 stars, 5 star
    r'\b[1-5]\s*étoiles?\b',        # 4 étoiles
    r'\b[0-9]+/100\b',              # 80/100
]

# Pronoms 1ère personne (anglais)
FIRST_PERSON_EN = {
    'i ', "i'm", "i've", "i'd", "i'll",
    'my ', 'me ', 'mine', 'myself',
    'we ', "we're", "we've", 'our ', 'ours', 'us '
}

# Pronoms 1ère personne (français)
FIRST_PERSON_FR = {
    'je ', "j'", 
    'mon ', 'ma ', 'mes ',
    'moi ', 'moi-même',
    'nous ', 'notre ', 'nos '
}

# Patterns d'exclusion (questions, showcases, etc.)
EXCLUSION_PATTERNS = [
    r'^shot on\b',                    # "Shot on iPhone"
    r'^before\s*&\s*after\b',        # "Before & After"
    r'^is it worth\b',               # "Is it worth buying?"
    r'^should i\b',                  # "Should I upgrade?"
    r'^anyone else\b',               # "Anyone else having issues?"
    r'^question\b',                  # "Question about..."
    r'^help\b',                      # "Help needed"
    r'^where\s+(can|to|do)\b',       # "Where can I buy?"
    r'^what\s+(is|are|do)\b',        # "What is the best?"
    r'^how\s+(do|to|can)\b',         # "How do I...?"
    r'^which\s+(one|is)\b',          # "Which one should I buy?"
    r'^looking for\b',               # "Looking for recommendations"
    r'^need help\b',                 # "Need help with..."
]

# Longueur minimale (en mots) pour considérer un post comme avis
# ✅ RÉDUIT de 25 à 10 mots (moins strict, garde plus de posts)
MIN_WORDS_FOR_REVIEW = 10


class ReviewFilterService:
    """Service pour filtrer les posts de type 'review' vs autres types de contenu"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # TODO: Réactiver le classifieur zero-shot BART MNLI plus tard si besoin de ML
        # Pour V2 Simplification: désactivé pour performance (réponse en quelques secondes)
        logger.info("⚠️ Zero-shot classifier DISABLED for V2 simplification (heuristic-only mode)")
        self.classifier = None
        
        # TODO: Labels pour zero-shot (à réactiver si besoin du modèle)
        # self.labels = [
        #     "product review",
        #     "question or help request",
        #     "news or announcement",
        #     "photo or media showcase",
        #     "other content"
        # ]
    
    # TODO: Réactiver cette méthode si besoin du modèle zero-shot plus tard
    # def _initialize_classifier(self):
    #     """Initialiser le pipeline zero-shot classification"""
    #     try:
    #         logger.info(f"Loading zero-shot classifier: {self.settings.review_classifier_model}")
    #         
    #         # Charger le modèle zero-shot (CPU uniquement pour éviter OOM sur GPU limité)
    #         self.classifier = pipeline(
    #             "zero-shot-classification",
    #             model=self.settings.review_classifier_model,
    #             device=-1  # Force CPU (évite problèmes GPU)
    #         )
    #         
    #         logger.info("✅ Zero-shot classifier loaded successfully (CPU mode)")
    #         
    #     except Exception as e:
    #         logger.warning(f"⚠️ Could not load zero-shot classifier: {e}")
    #         logger.warning("Falling back to heuristic-based review detection")
    #         self.classifier = None
    
    def is_available(self) -> bool:
        """Vérifier si le classifier est disponible"""
        # V2 Simplification: toujours False (pas de modèle ML)
        return False
    
    def _is_review_by_keywords(self, title: str, text: str) -> bool:
        """
        🚀 V2 SIMPLIFICATION: Détection heuristique pure par mots/phrases clés
        
        Critères:
        1. Longueur minimale (≥ 25 mots)
        2. Présence de mots/phrases clés review (anglais ou français)
        3. Présence de pronoms 1ère personne
        4. Patterns de notes/scores (optionnel mais boost)
        5. Exclusion de patterns non-review (questions, showcases)
        
        Args:
            title: Titre du post
            text: Contenu du post
        
        Returns:
            True si le post ressemble à un avis, False sinon
        """
        # Fusionner titre + contenu
        full_text = f"{title or ''} {text or ''}".strip()
        full_text_lower = full_text.lower()
        words = full_text_lower.split()
        
        # ===== CRITÈRE 1: Longueur minimale =====
        if len(words) < MIN_WORDS_FOR_REVIEW:
            logger.debug(f"FILTER: Too short ({len(words)} words < {MIN_WORDS_FOR_REVIEW})")
            return False
        
        # ===== CRITÈRE 2: Exclusion de patterns non-review =====
        title_lower = title.lower()
        for pattern in EXCLUSION_PATTERNS:
            if re.search(pattern, title_lower):
                logger.debug(f"FILTER: Matches exclusion pattern '{pattern}'")
                return False
        
        # ===== CRITÈRE 3: Présence de mots/phrases clés review =====
        has_review_keyword = (
            any(keyword in full_text_lower for keyword in REVIEW_KEYWORDS_EN) or
            any(keyword in full_text_lower for keyword in REVIEW_KEYWORDS_FR)
        )
        
        # ===== CRITÈRE 4: Patterns de notes/scores (boost) =====
        has_rating_pattern = any(
            re.search(pattern, full_text_lower) 
            for pattern in RATING_PATTERNS
        )
        
        # ===== CRITÈRE 5: Présence de pronoms 1ère personne =====
        has_first_person = (
            any(pronoun in full_text_lower for pronoun in FIRST_PERSON_EN) or
            any(pronoun in full_text_lower for pronoun in FIRST_PERSON_FR)
        )
        
        # ===== CRITÈRE 6: Ratio de questions (trop de "?" = question, pas avis) =====
        question_marks = full_text.count('?')
        sentences_approx = max(1, len(full_text.split('.')))
        too_many_questions = (question_marks / sentences_approx) > 0.5
        
        if too_many_questions:
            logger.debug(f"FILTER: Too many questions ({question_marks} '?')")
            return False
        
        # ===== DÉCISION FINALE (✅ ASSOUPLIE : critères plus permissifs) =====
        # Option 1: mot clé review + pronom 1ère personne (strict)
        if has_review_keyword and has_first_person:
            logger.debug(f"KEEP: Review keyword + first person (length={len(words)})")
            return True
        
        # Option 2: pattern de note + pronom 1ère personne (même sans mot "review")
        if has_rating_pattern and has_first_person:
            logger.debug(f"KEEP: Rating pattern + first person (length={len(words)})")
            return True
        
        # ✅ NOUVEAU Option 3: mot clé review seul (sans forcer first person)
        if has_review_keyword:
            logger.debug(f"KEEP: Review keyword alone (length={len(words)})")
            return True
        
        # ✅ NOUVEAU Option 4: first person + longueur suffisante (≥15 mots)
        if has_first_person and len(words) >= 15:
            logger.debug(f"KEEP: First person + sufficient length ({len(words)} words)")
            return True
        
        # ✅ NOUVEAU Option 5: pattern de note seul (indique souvent un avis)
        if has_rating_pattern:
            logger.debug(f"KEEP: Rating pattern alone (length={len(words)})")
            return True
        
        # Sinon: filtrer
        logger.debug(
            f"FILTER: Missing criteria (keyword={has_review_keyword}, "
            f"rating={has_rating_pattern}, first_person={has_first_person}, length={len(words)})"
        )
        return False
    
    # TODO: Réactiver cette méthode si besoin de l'ancienne logique heuristique (avec verbes)
    # def _is_review_heuristic_OLD(self, title: str, text: str, full_text: str) -> bool:
    #     """
    #     Ancienne détection heuristique de reviews (désactivée pour V2)
    #     Utilisait des verbes d'usage/possession/opinion
    #     """
    #     title_lower = title.lower()
    #     full_text_lower = full_text.lower()
    #     words = full_text_lower.split()
    #     
    #     if len(words) < 25:
    #         return False
    #     
    #     # Exclusion patterns...
    #     # Usage verbs...
    #     # First person pronouns...
    #     # Question ratio...
    #     
    #     return True
    
    def filter_review_posts(self, posts: List[Post]) -> Tuple[List[Post], dict]:
        """
        🚀 V2 SIMPLIFICATION: Filtrer par heuristiques pures (mots/phrases clés)
        
        Stratégie 100% rule-based:
        - Aucun modèle ML (zero-shot désactivé)
        - Détection par mots/phrases clés uniquement
        - Performance optimale (quelques millisecondes)
        
        Args:
            posts: Liste de posts à filtrer
        
        Returns:
            Tuple (posts_filtrés, stats_dict)
            - posts_filtrés: Liste des posts qui ressemblent à des reviews
            - stats_dict: Statistiques de filtrage {'total': X, 'kept': Y, 'filtered': Z, 'percentage': P}
        """
        if not posts:
            return [], {'total': 0, 'kept': 0, 'filtered': 0, 'percentage': 0.0}
        
        total = len(posts)
        logger.info(f"🔍 Filtering {total} posts for review-like content (heuristic-only mode)...")
        
        # ===== FILTRAGE HEURISTIQUE DIRECT =====
        review_posts = []
        
        for post in posts:
            # Appliquer la détection par mots/phrases clés (synchrone, rapide)
            is_review = self._is_review_by_keywords(post.title or "", post.text or "")
            
            if is_review:
                review_posts.append(post)
        
        kept = len(review_posts)
        
        # Statistiques finales
        filtered = total - kept
        percentage = (kept / total * 100) if total > 0 else 0.0
        
        stats = {
            'total': total,
            'kept': kept,
            'filtered': filtered,
            'percentage': percentage
        }
        
        logger.info(
            f"✅ Review filtering complete (heuristic): "
            f"{kept}/{total} posts kept ({percentage:.1f}%), "
            f"{filtered} filtered out"
        )
        
        return review_posts, stats


# Singleton instance
review_filter_service = ReviewFilterService()

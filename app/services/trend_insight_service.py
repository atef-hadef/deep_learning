"""
🧠 TREND INSIGHT SERVICE - LLM-Powered Analysis

Service qui génère des insights intelligents sur les tendances
en utilisant un LLM (Grok via OpenRouter)

Author: AI Assistant
Date: December 2025
Version: 1.0
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from app.services.llm_client import call_llm, is_llm_available
from app.services.database_service import database_service
from app.services.sentiment_service import sentiment_service
from app.schemas.responses import Post

logger = logging.getLogger(__name__)


async def build_trend_insight(
    keyword: str,
    start_date: str,
    end_date: str,
    platforms: List[str],
    cached_posts: List[Post] = None,
    use_cache_only: bool = False
) -> Dict:
    """
    Génère un insight intelligent sur les tendances d'un produit/sujet
    en utilisant un LLM (Grok via OpenRouter)
    
    Args:
        keyword: Mot-clé recherché (ex: "iPhone 15")
        start_date: Date de début au format ISO (ex: "2024-12-01")
        end_date: Date de fin au format ISO (ex: "2024-12-08")
        platforms: Liste des plateformes (["reddit", "twitter"])
        cached_posts: Posts en cache (UNIQUEMENT posts affichés avec analyse)
        use_cache_only: Si True, utilise UNIQUEMENT cached_posts (ignore DB)
        
    Returns:
        Dict avec:
        - keyword: Le mot-clé
        - start_date, end_date: Les dates
        - platforms: Les plateformes
        - stats: Statistiques numériques (total, pct_pos, pct_neu, pct_neg)
        - insight: Texte généré par le LLM
        - examples_used: Exemples d'avis utilisés dans le prompt
        - llm_available: Booléen indiquant si le LLM était disponible
    """
    logger.info(
        f"🧠 [TrendInsight] Building insight for '{keyword}' "
        f"from {start_date} to {end_date} on {platforms}"
    )

    # 1) Vérifier disponibilité du LLM
    if not is_llm_available():
        logger.warning("⚠️ LLM service not available - returning stats only")
        return {
            "keyword": keyword,
            "start_date": start_date,
            "end_date": end_date,
            "platforms": platforms,
            "stats": {},
            "insight": "Service LLM non disponible. Veuillez configurer OPENROUTER_API_KEY.",
            "examples_used": [],
            "llm_available": False
        }

    # 2) Récupérer les posts depuis le cache (posts affichés) ou base de données
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    posts = []
    
    # ⚠️ Si use_cache_only=True, utiliser UNIQUEMENT le cache (posts affichés)
    if use_cache_only:
        if cached_posts:
            posts = cached_posts
            logger.info(f"🎯 Using {len(posts)} CACHED ANALYZED posts (displayed in UI) for LLM")
        else:
            logger.warning("⚠️ use_cache_only=True but no cached_posts provided")
            return {
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "platforms": platforms,
                "stats": {},
                "insight": "Aucun post en cache. Veuillez effectuer une recherche d'abord.",
                "examples_used": {"positive": [], "negative": []},
                "llm_available": False
            }
    # Sinon, essayer PostgreSQL d'abord
    elif await database_service.is_available():
        logger.info(f"📊 Fetching posts from PostgreSQL database...")
        posts = await database_service.get_posts_by_keyword_and_date(
            keyword=keyword,
            start_date=start_dt,
            end_date=end_dt,
            platforms=platforms
        )
        logger.info(f"📊 Found {len(posts)} posts in PostgreSQL")
    
    # Fallback: utiliser les posts en cache si disponibles
    if not posts and cached_posts:
        logger.info(f"💾 PostgreSQL unavailable, using {len(cached_posts)} cached posts")
        posts = cached_posts
    
    if not posts:
        logger.warning(f"⚠️ No posts found in database or cache for keyword '{keyword}'")
        return {
            "keyword": keyword,
            "start_date": start_date,
            "end_date": end_date,
            "platforms": platforms,
            "stats": {
                "total": 0,
                "pct_pos": 0,
                "pct_neu": 0,
                "pct_neg": 0
            },
            "insight": f"Aucun avis trouvé pour générer un résumé. Veuillez d'abord effectuer une recherche avec le bouton 'Analyser les avis'.",
            "examples_used": [],
            "llm_available": True
        }
    
    logger.info(f"✅ Found {len(posts)} posts in database")

    # 3) Compléter les sentiments manquants
    posts_without_sentiment = [p for p in posts if not p.sentiment]
    if posts_without_sentiment:
        logger.info(f"🧠 Computing sentiments for {len(posts_without_sentiment)} posts...")
        texts = [p.text for p in posts_without_sentiment]
        sentiments = await sentiment_service.analyze_batch_sentiments(texts)
        for post, sent in zip(posts_without_sentiment, sentiments):
            post.sentiment = sent

    # 4) Calculer les statistiques
    total = len(posts)
    positive = sum(1 for p in posts if p.sentiment and p.sentiment.dominant == "positive")
    neutral = sum(1 for p in posts if p.sentiment and p.sentiment.dominant == "neutral")
    negative = sum(1 for p in posts if p.sentiment and p.sentiment.dominant == "negative")
    
    pct_pos = (positive / total * 100) if total > 0 else 0
    pct_neu = (neutral / total * 100) if total > 0 else 0
    pct_neg = (negative / total * 100) if total > 0 else 0

    stats = {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "pct_pos": round(pct_pos, 1),
        "pct_neu": round(pct_neu, 1),
        "pct_neg": round(pct_neg, 1)
    }

    logger.info(
        f"📊 Stats: {total} posts | "
        f"Positifs: {positive} ({pct_pos:.1f}%) | "
        f"Neutres: {neutral} ({pct_neu:.1f}%) | "
        f"Négatifs: {negative} ({pct_neg:.1f}%)"
    )

    # 5) Extraire des exemples pour enrichir le prompt
    positive_examples = [
        p.text for p in posts 
        if p.sentiment and p.sentiment.dominant == "positive"
    ][:5]  # Max 5 exemples positifs
    
    negative_examples = [
        p.text for p in posts 
        if p.sentiment and p.sentiment.dominant == "negative"
    ][:5]  # Max 5 exemples négatifs

    examples_used = {
        "positive": positive_examples,
        "negative": negative_examples
    }

    # 6) Construire le prompt pour le LLM
    system_message = {
        "role": "system",
        "content": (
            "Tu es un data analyst expert en analyse de sentiments sur les réseaux sociaux. "
            "Ta mission est d'expliquer les résultats d'analyse de manière claire, professionnelle "
            "et concise en français. Tu dois fournir des insights actionnables basés sur les données."
        )
    }

    # Construire la liste des exemples négatifs
    negative_bullets = "\n".join(
        f"  - \"{ex[:150]}...\"" if len(ex) > 150 else f"  - \"{ex}\""
        for ex in negative_examples
    ) if negative_examples else "  - (aucun avis négatif trouvé)"

    # Construire la liste des exemples positifs
    positive_bullets = "\n".join(
        f"  - \"{ex[:150]}...\"" if len(ex) > 150 else f"  - \"{ex}\""
        for ex in positive_examples
    ) if positive_examples else "  - (aucun avis positif trouvé)"

    user_message = {
        "role": "user",
        "content": f"""Analyse les données de sentiment suivantes :

**Sujet** : {keyword}
**Période** : {start_date} → {end_date}
**Plateformes** : {', '.join(platforms)}

**Statistiques** :
- Nombre total d'avis analysés : {total}
- Avis positifs : {positive} ({pct_pos:.1f}%)
- Avis neutres : {neutral} ({pct_neu:.1f}%)
- Avis négatifs : {negative} ({pct_neg:.1f}%)

**Exemples d'avis négatifs** :
{negative_bullets}

**Exemples d'avis positifs** :
{positive_bullets}

**Instruction** :
Écris un résumé en français de 5 à 8 lignes maximum qui :
1. Décrit la satisfaction globale des utilisateurs
2. Résume les principaux points négatifs évoqués
3. Résume les principaux points positifs évoqués
4. Conclut si la situation est globalement positive, mitigée ou préoccupante

Sois direct, concis et professionnel. Ne répète pas les chiffres déjà affichés.
"""
    }

    # 7) Appeler le LLM
    logger.info("🤖 Calling LLM to generate insight...")
    try:
        insight_text = await call_llm(
            messages=[system_message, user_message],
            temperature=0.4,  # Assez déterministe pour rester factuel
            max_tokens=400    # ~5-8 lignes de texte
        )
        logger.info(f"✅ LLM insight generated: {len(insight_text)} chars")
    except Exception as e:
        logger.error(f"❌ Error calling LLM: {e}")
        insight_text = f"Erreur lors de la génération du résumé LLM : {str(e)}"

    # 8) Retourner le résultat complet
    return {
        "keyword": keyword,
        "start_date": start_date,
        "end_date": end_date,
        "platforms": platforms,
        "stats": stats,
        "insight": insight_text,
        "examples_used": examples_used,
        "llm_available": True
    }

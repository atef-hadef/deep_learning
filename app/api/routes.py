from fastapi import APIRouter, HTTPException, status
from typing import List
import asyncio
import time
import logging
from datetime import datetime, timedelta

from app.schemas.requests import SearchRequest, TrendsRequest
from app.schemas.responses import (
    SearchResponse,
    TrendsResponse,
    PostDetailResponse,
    HealthResponse,
    SentimentAnalysis
)
from app.services.reddit_service import reddit_service
from app.services.twitter_service import twitter_service
from app.services.sentiment_service import sentiment_service
from app.services.trends_service import trends_service
from app.services.database_service import database_service
from app.services.opinion_detector import filter_opinion_posts, is_service_available as opinion_detector_available
from app.services.review_fetcher import fetch_review_posts
from app.services.trend_insight_service import build_trend_insight
from app.services.llm_client import is_llm_available

logger = logging.getLogger(__name__)

router = APIRouter()

# 💾 Cache temporaire pour les posts (utilisé quand PostgreSQL n'est pas disponible)
_last_search_cache = {
    "keyword": None,
    "posts": [],
    "timestamp": None
}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "reddit": reddit_service.is_available(),
        "twitter": twitter_service.is_available(),
        "sentiment_model": sentiment_service.is_available(),
        "opinion_detector": opinion_detector_available(),
        "database": await database_service.is_available(),
        "llm_service": is_llm_available()
    }
    
    return HealthResponse(
        status="healthy" if any(services.values()) else "degraded",
        timestamp=datetime.utcnow(),
        services=services
    )


@router.post("/api/search", response_model=SearchResponse)
async def search_posts(request: SearchRequest):
    """
    🚀 ENDPOINT RAPIDE: Analyse des avis (posts & commentaires) avec fetching adaptatif
    
    Le paramètre `limit` représente le NOMBRE DE REVIEWS (opinions) à retourner.
    Le backend fetch en plusieurs batchs jusqu'à atteindre ce nombre ou épuiser
    les posts disponibles dans la période.
    """
    start_time = time.time()
    from app.config import get_settings
    settings = get_settings()
    
    logger.info(f"🔍 [AdaptiveSearch] Searching for keyword: {request.keyword} on platforms: {request.platforms}")
    logger.info(f"📊 Requesting {request.limit} review posts (opinions)")
    
    # 🎯 NEW: Use adaptive fetching to get exactly `limit` opinion posts
    result = await fetch_review_posts(
        keyword=request.keyword,
        platforms=request.platforms,
        desired_reviews=request.limit,
        start_date=None,  # TODO: Add support for start_date/end_date in SearchRequest schema
        end_date=None,
        time_filter=request.time_filter,
        include_comments=request.include_comments
    )
    
    relevant_posts = result["reviews"]
    fetch_stats = result["stats"]
    
    logger.info(
        f"📊 [AdaptiveSearch] Fetched {fetch_stats['collected']}/{fetch_stats['requested']} reviews "
        f"from {fetch_stats['raw_posts_fetched']} raw posts in {fetch_stats['batches']} batches "
        f"(opinion_rate={fetch_stats['opinion_rate']}%, exhausted={fetch_stats['exhausted']})"
    )
    
    if not relevant_posts:
        return SearchResponse(
            keyword=request.keyword,
            platforms=request.platforms,
            total_posts=0,
            posts=[],
            overall_sentiment=SentimentAnalysis(
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                dominant="neutral"
            ),
            filtering_stats={
                'total': fetch_stats['raw_posts_fetched'],
                'relevant': fetch_stats['collected'],
                'filtered': fetch_stats['raw_posts_fetched'] - fetch_stats['collected'],
                'avg_score': fetch_stats['avg_opinion_score']
            },
            total_fetched_posts=fetch_stats['raw_posts_fetched'],
            total_review_posts_used=fetch_stats['collected'],
            review_filter_percentage=fetch_stats['opinion_rate'],
            trends=[],
            popular_topics=[],
            execution_time=time.time() - start_time
        )
    
    # Limit comments per post for performance
    for post in relevant_posts:
        if post.comments and len(post.comments) > settings.max_comments_per_post:
            logger.debug(f"Limiting comments for post {post.id}: {len(post.comments)} -> {settings.max_comments_per_post}")
            post.comments = post.comments[:settings.max_comments_per_post]
    
    logger.info(f"🧠 Analyzing sentiments for {len(relevant_posts)} relevant posts")
    
    post_texts = [post.text for post in relevant_posts]
    post_sentiments = await sentiment_service.analyze_batch_sentiments(post_texts)
    
    for post, sentiment in zip(relevant_posts, post_sentiments):
        post.sentiment = sentiment
    
    for post in relevant_posts:
        if post.comments:
            comment_texts = [c.text for c in post.comments]
            comment_sentiments = await sentiment_service.analyze_batch_sentiments(comment_texts)
            
            for comment, sentiment in zip(post.comments, comment_sentiments):
                comment.sentiment = sentiment
            
            post.summary = None
            post.aspects = []
    
    all_sentiments = [post.sentiment for post in relevant_posts if post.sentiment]
    overall_sentiment = sentiment_service.calculate_overall_sentiment(all_sentiments)
    
    product_summary = None
    key_points = []
    
    # 💾 Sauvegarder dans la cache en mémoire (pour le LLM - UNIQUEMENT les posts affichés)
    # Calculer les dates basées sur time_filter
    end_date = datetime.utcnow()
    start_date = end_date
    
    if request.time_filter == "hour":
        start_date = end_date - timedelta(hours=1)
    elif request.time_filter == "day":
        start_date = end_date - timedelta(days=1)
    elif request.time_filter == "week":
        start_date = end_date - timedelta(days=7)
    elif request.time_filter == "month":
        start_date = end_date - timedelta(days=30)
    elif request.time_filter == "year":
        start_date = end_date - timedelta(days=365)
    else:  # "all"
        start_date = end_date - timedelta(days=365)  # 1 an par défaut
    
    global _last_search_cache
    _last_search_cache = {
        "keyword": request.keyword,
        "posts": relevant_posts,  # Ce sont les posts AFFICHÉS avec analyse de sentiment
        "timestamp": datetime.utcnow(),
        "platforms": request.platforms,
        "start_date": start_date.isoformat().split('T')[0],
        "end_date": end_date.isoformat().split('T')[0]
    }
    logger.info(f"💾 Cached {len(relevant_posts)} ANALYZED posts in memory for LLM insight")
    
    if await database_service.is_available():
        saved_count = await database_service.save_posts(relevant_posts)
        await database_service.save_search_history(
            keyword=request.keyword,
            platforms=request.platforms,
            num_posts=len(relevant_posts)
        )
        logger.info(f"💾 Saved {saved_count}/{len(relevant_posts)} relevant posts to PostgreSQL")
    
    execution_time = time.time() - start_time
    logger.info(f"✅ [AdaptiveSearch] Search completed in {execution_time:.2f}s")
    
    if product_summary is not None and key_points is not None:
        logger.info(f"📝 Product summary: {len(product_summary)} chars, Key points: {len(key_points)}")
    else:
        logger.info(f"📝 Product summary and key points disabled for V1 simplification")
    
    # Use fetch stats for filtering_stats
    filtering_stats = {
        'total': fetch_stats['raw_posts_fetched'],
        'relevant': fetch_stats['collected'],
        'filtered': fetch_stats['raw_posts_fetched'] - fetch_stats['collected'],
        'avg_score': fetch_stats['avg_opinion_score']
    }
    
    return SearchResponse(
        keyword=request.keyword,
        platforms=request.platforms,
        total_posts=len(relevant_posts),
        posts=relevant_posts,
        overall_sentiment=overall_sentiment,
        product_summary=product_summary,
        key_points=key_points,
        filtering_stats=filtering_stats,
        total_fetched_posts=fetch_stats['raw_posts_fetched'],
        total_review_posts_used=fetch_stats['collected'],
        review_filter_percentage=fetch_stats['opinion_rate'],
        trends=[],
        popular_topics=[],
        execution_time=execution_time
    )


@router.get("/api/posts/{platform}/{post_id}", response_model=PostDetailResponse)
async def get_post_detail(platform: str, post_id: str):
    """
    Get detailed information about a specific post
    """
    start_time = time.time()
    
    logger.info(f"Fetching post details: {platform}/{post_id}")
    
    if platform not in ["reddit", "twitter"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid platform. Must be 'reddit' or 'twitter'"
        )
    
    if await database_service.is_available():
        cached_post = await database_service.get_post(platform, post_id)
        if cached_post:
            logger.info("Retrieved post from database")
            from app.schemas.responses import Post
            post = Post(**cached_post)
            return PostDetailResponse(
                post=post,
                execution_time=time.time() - start_time
            )
    
    post = None
    if platform == "reddit" and reddit_service.is_available():
        post = await reddit_service.get_post_details(post_id)
    elif platform == "twitter" and twitter_service.is_available():
        post = await twitter_service.get_post_details(post_id)
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post not found: {platform}/{post_id}"
        )
    
    post.sentiment = await sentiment_service.analyze_sentiment(post.text)
    
    if post.comments:
        comment_texts = [c.text for c in post.comments]
        comment_sentiments = await sentiment_service.analyze_batch_sentiments(comment_texts)
        
        for comment, sentiment in zip(post.comments, comment_sentiments):
            comment.sentiment = sentiment
        
        if len(post.comments) >= 3:
            post.summary = await sentiment_service.summarize_comments(comment_texts)
        
        post.aspects = await sentiment_service.extract_aspects(comment_texts, top_k=5)
    
    if await database_service.is_available():
        await database_service.save_post(post)
    
    execution_time = time.time() - start_time
    logger.info(f"Post details fetched in {execution_time:.2f}s")
    
    return PostDetailResponse(
        post=post,
        execution_time=execution_time
    )


@router.post("/api/trends", response_model=TrendsResponse)
async def get_trends(request: TrendsRequest):
    """
    📊 ENDPOINT LOURD: Analyse de tendances temporelles depuis PostgreSQL + fetch live

    ⚠️ IMPORTANT :
    - Ici on veut analyser **la tendance globale du sujet** (tous les posts liés au mot-clé),
      pas seulement les avis produits.
    - Donc on n'utilise PAS le filtre de pertinence des reviews pour les tendances.
      On travaille sur le flux global de posts (tweets + posts Reddit) qui contiennent le mot-clé.
    """
    start_time = time.time()
    from app.config import get_settings
    from datetime import timedelta

    settings = get_settings()

    logger.info(
        f"📊 [TRENDS] Analyzing trends for keyword: {request.keyword} "
        f"(period: {request.time_range}, platforms={request.platforms})"
    )

    # 1) Fenêtre temporelle
    time_range_map = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    since_date = datetime.utcnow() - time_range_map.get(request.time_range, timedelta(days=7))

    # 2) Récupérer les posts depuis PostgreSQL
    posts_for_trends = []

    if await database_service.is_available():
        logger.info(
            f"💾 Fetching posts from PostgreSQL for trends "
            f"(since: {since_date.strftime('%Y-%m-%d %H:%M')})"
        )
        posts_from_db = await database_service.get_posts_for_trends(
            keyword=request.keyword,
            platforms=request.platforms,
            since=since_date,
            limit=settings.max_trends_posts,
        )
        logger.info(f"💾 Retrieved {len(posts_from_db)} posts from PostgreSQL")
        posts_for_trends.extend(posts_from_db)
    else:
        logger.warning("⚠️ PostgreSQL not available, trends will rely only on live fetch")

    # 3) Fetch live si pas assez
    MIN_DB_POSTS = 10
    reddit_posts_count = 0
    twitter_posts_count = 0

    if len(posts_for_trends) < MIN_DB_POSTS:
        logger.info(
            f"⚠️ Not enough posts in DB for trends ({len(posts_for_trends)}), "
            f"fetching live data from APIs..."
        )

        time_filter_map = {
            "24h": "day",
            "7d": "week",
            "30d": "month",
        }
        time_filter = time_filter_map.get(request.time_range, "week")

        posts_live = []

        if "reddit" in request.platforms and reddit_service.is_available():
            reddit_posts, reddit_posts_count = reddit_service.search_posts_for_trends(
                keyword=request.keyword,
                time_filter=time_filter,
                limit=settings.max_trends_fetch_per_platform
            )
            posts_live.extend(reddit_posts)
            logger.info(f"📈 Reddit trends: {reddit_posts_count} posts found (no filtering)")

        if "twitter" in request.platforms and twitter_service.is_available():
            try:
                twitter_posts, twitter_posts_count = twitter_service.search_posts_for_trends(
                    keyword=request.keyword,
                    time_filter=time_filter,
                    limit=settings.max_trends_fetch_per_platform
                )
                posts_live.extend(twitter_posts)
                logger.info(f"🐦 Twitter trends: {twitter_posts_count} tweets found (no filtering)")
            except Exception as e:
                logger.error(f"Error fetching Twitter posts for trends: {e}")

        logger.info(f"🌐 Fetched {len(posts_live)} live posts for trends")

        existing_ids = {f"{p.platform}_{p.id}" for p in posts_for_trends}
        merged_new = 0
        for post in posts_live:
            key = f"{post.platform}_{post.id}"
            if key not in existing_ids:
                posts_for_trends.append(post)
                existing_ids.add(key)
                merged_new += 1

        logger.info(f"📊 Total posts for trends after merge: {len(posts_for_trends)} (+{merged_new} new)")

    # 4) Aucun post → réponse vide
    if not posts_for_trends:
        logger.warning("⚠️ No posts available for trends analysis")
        return TrendsResponse(
            keyword=request.keyword,
            time_range=request.time_range,
            trends=[],
            popular_topics=[],
            execution_time=time.time() - start_time,
            total_mentions_found=0,
            reddit_mentions=0,
            twitter_mentions=0,
        )

    # 5) Pas de filtre "review" ici
    logger.info(
        f"📊 Using {len(posts_for_trends)} posts for global trend analysis "
        f"(no review-only filtering)"
    )
    
    # 5.1) 🔍 DEBUG: Log daily distribution BEFORE analyze_trends
    from collections import Counter
    daily_counts = Counter(p.created_at.date() for p in posts_for_trends if p.created_at)
    sorted_days = sorted(daily_counts.items())
    daily_str = ", ".join(f"{day}: {count}" for day, count in sorted_days)
    logger.info(f"📅 [PRE-ANALYSIS] Daily distribution of posts_for_trends: {daily_str}")

    # 6) Compléter les sentiments manquants
    posts_without_sentiment = [p for p in posts_for_trends if not p.sentiment]

    if posts_without_sentiment:
        logger.info(
            f"🧠 Computing sentiments for {len(posts_without_sentiment)} posts "
            f"without sentiment (batch)..."
        )
        texts = [p.text for p in posts_without_sentiment]
        sentiments = await sentiment_service.analyze_batch_sentiments(texts)
        for post, sent in zip(posts_without_sentiment, sentiments):
            post.sentiment = sent

    # 7) Analyse de tendances (⚠️ sans re-filtrage keyword)
    logger.info("📊 Running trends analysis (TrendsService)...")
    trend_data = await trends_service.analyze_trends(
        posts=posts_for_trends,
        keyword=request.keyword,
        time_range=request.time_range,
        apply_keyword_filter=False,   # 🔥 clé: utiliser TOUS les posts fournis
    )

    # 8) Topics populaires
    logger.info("🏷️ Extracting popular topics for trends...")
    popular_topics = await trends_service.extract_popular_topics(
        posts=posts_for_trends,
        main_keyword=request.keyword,
        top_k=10,
    )

    execution_time = time.time() - start_time
    logger.info(
        f"✅ [TRENDS] Analysis completed in {execution_time:.2f}s "
        f"({len(trend_data)} platforms, {len(popular_topics)} topics)"
    )

    return TrendsResponse(
        keyword=request.keyword,
        time_range=request.time_range,
        trends=trend_data,
        popular_topics=popular_topics,
        execution_time=execution_time,
        total_mentions_found=reddit_posts_count + twitter_posts_count,
        reddit_mentions=reddit_posts_count,
        twitter_mentions=twitter_posts_count,
    )


@router.get("/api/trends/llm-insight")
async def get_trend_llm_insight(
    keyword: str,
    start_date: str,
    end_date: str,
    platforms: List[str] = None
):
    """
    🤖 ENDPOINT LLM: Génère un résumé intelligent des tendances avec Grok/LLM
    
    Cet endpoint est appelé UNIQUEMENT quand l'utilisateur clique sur le bouton
    "Résumé LLM" dans l'interface. Il ne se déclenche PAS automatiquement.
    
    Args:
        keyword: Mot-clé à analyser (ex: "iPhone 15")
        start_date: Date de début ISO format (ex: "2024-12-01")
        end_date: Date de fin ISO format (ex: "2024-12-08")
        platforms: Liste des plateformes (optionnel, défaut: ["reddit", "twitter"])
        
    Returns:
        {
            "keyword": str,
            "start_date": str,
            "end_date": str,
            "platforms": List[str],
            "stats": {
                "total": int,
                "positive": int,
                "neutral": int,
                "negative": int,
                "pct_pos": float,
                "pct_neu": float,
                "pct_neg": float
            },
            "insight": str,  # Texte généré par le LLM
            "examples_used": {
                "positive": List[str],
                "negative": List[str]
            },
            "llm_available": bool
        }
    """
    start_time = time.time()
    
    # Plateformes par défaut si non spécifiées
    if platforms is None:
        platforms = ["reddit", "twitter"]
    
    logger.info(
        f"🤖 [LLM Insight] Generating insight for '{keyword}' "
        f"from {start_date} to {end_date} on {platforms}"
    )
    
    try:
        # ⚠️ IMPORTANT: Utiliser UNIQUEMENT le cache (posts affichés dans l'interface)
        # Ne PAS fetcher depuis la base de données
        global _last_search_cache
        
        # Vérifier que le cache existe et correspond au keyword
        if not _last_search_cache.get("keyword") or _last_search_cache.get("keyword") != keyword:
            logger.warning(f"⚠️ No cached posts for keyword '{keyword}'. User must perform search first.")
            return {
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "platforms": platforms,
                "stats": {},
                "insight": "Veuillez d'abord effectuer une recherche pour générer un résumé LLM.",
                "examples_used": {"positive": [], "negative": []},
                "llm_available": False,
                "execution_time": 0
            }
        
        cached_posts = _last_search_cache.get("posts", [])
        logger.info(f"🤖 Using {len(cached_posts)} CACHED ANALYZED posts for LLM insight (NOT from database)")
        
        result = await build_trend_insight(
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            platforms=platforms,
            cached_posts=cached_posts,  # Passer les posts du cache
            use_cache_only=True  # Nouveau flag pour forcer utilisation cache
        )
        
        execution_time = time.time() - start_time
        result["execution_time"] = round(execution_time, 2)
        
        logger.info(f"✅ [LLM Insight] Generated in {execution_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error generating LLM insight: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération du résumé LLM: {str(e)}"
        )

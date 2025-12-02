"""
Adaptive Review Fetcher Service
================================

Service for fetching a desired number of review posts (opinions) by iteratively
fetching raw posts and filtering them with OpinionDetector until we reach the target
or exhaust available posts in the time period.

This replaces the old fixed-batch approach where we'd fetch MAX_POSTS_PER_REQUEST
and return whatever remained after filtering.
"""

import logging
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import asyncio

from app.schemas.responses import Post
from app.services.opinion_detector import is_opinion, opinion_score
from app.services.reddit_service import reddit_service
from app.services.twitter_service import twitter_service
from app.config import get_settings

logger = logging.getLogger(__name__)


async def fetch_review_posts(
    keyword: str,
    platforms: List[str],
    desired_reviews: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    time_filter: str = "week",
    include_comments: bool = True
) -> Dict[str, Any]:
    """
    Fetch up to `desired_reviews` posts that are actual opinions (OpinionDetector=True)
    within the requested time window (start_date / end_date).
    
    If we can't find enough, return a partial list. The fetching loop stops when:
    - We've collected `desired_reviews` opinions, OR
    - There are no more posts to retrieve in the period (all sources exhausted)
    
    Args:
        keyword: Search keyword
        platforms: List of platforms to search (e.g., ["reddit", "twitter"])
        desired_reviews: Target number of opinion posts to return
        start_date: Start of time window (optional)
        end_date: End of time window (optional)
        time_filter: Time filter for platforms (week, month, etc.)
        include_comments: Whether to fetch comments for posts
        
    Returns:
        Dict containing:
        - reviews: List[Post] - Collected opinion posts
        - stats: Dict - Statistics about the fetch process
            - requested: int - Number of reviews requested
            - collected: int - Number of reviews collected
            - batches: int - Number of batches processed
            - raw_posts_fetched: int - Total raw posts fetched
            - opinions_found: int - Total opinions found
            - exhausted: bool - Whether we exhausted all available posts
            - avg_opinion_score: float - Average opinion score
    """
    settings = get_settings()
    
    # Configuration
    MAX_RAW_BATCH = settings.max_posts_per_request  # 30
    MAX_BATCHES = 5  # Safety limit
    OVERFETCH_FACTOR = 3  # Fetch 3x raw posts to get enough opinions
    
    logger.info(
        f"🎯 [AdaptiveFetch] Starting: keyword='{keyword}', platforms={platforms}, "
        f"desired={desired_reviews}, time_filter={time_filter}"
    )
    
    # State tracking
    collected: List[Post] = []
    seen_ids: Set[str] = set()
    batch_index = 0
    total_raw_fetched = 0
    total_opinions_found = 0
    all_opinion_scores = []
    exhausted = False
    
    # Platform availability check
    available_platforms = []
    if "reddit" in platforms and reddit_service.is_available():
        available_platforms.append("reddit")
    if "twitter" in platforms and twitter_service.is_available():
        available_platforms.append("twitter")
    
    if not available_platforms:
        logger.warning("⚠️ [AdaptiveFetch] No platforms available")
        return {
            "reviews": [],
            "stats": {
                "requested": desired_reviews,
                "collected": 0,
                "batches": 0,
                "raw_posts_fetched": 0,
                "opinions_found": 0,
                "exhausted": False,
                "avg_opinion_score": 0.0
            }
        }
    
    # Main fetching loop
    while len(collected) < desired_reviews and batch_index < MAX_BATCHES:
        batch_index += 1
        
        # Calculate how many raw posts to fetch in this batch
        # Fetch more raw posts than needed since many will be filtered out
        remaining = desired_reviews - len(collected)
        raw_limit = min(MAX_RAW_BATCH, remaining * OVERFETCH_FACTOR)
        
        logger.info(
            f"📦 [AdaptiveFetch] Batch {batch_index}/{MAX_BATCHES}: "
            f"collected={len(collected)}/{desired_reviews}, fetching {raw_limit} raw posts"
        )
        
        # Fetch from all available platforms in parallel
        tasks = []
        
        if "reddit" in available_platforms:
            # Note: Reddit service already does internal filtering with review_filter_service
            # We'll apply OpinionDetector on top of that
            tasks.append(
                reddit_service.search_posts(
                    keyword=keyword,
                    desired_reviews=raw_limit,  # This is actually raw limit
                    time_filter=time_filter,
                    include_comments=include_comments
                )
            )
        
        if "twitter" in available_platforms:
            tasks.append(
                twitter_service.search_posts(
                    keyword=keyword,
                    limit=min(raw_limit, 100),  # Twitter API max is 100
                    time_filter=time_filter,
                    include_comments=include_comments
                )
            )
        
        # Execute fetches in parallel
        posts_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect raw posts from all platforms
        raw_posts_batch = []
        for result in posts_results:
            if isinstance(result, tuple) and len(result) == 2:
                posts_list, filter_stats = result
                raw_posts_batch.extend(posts_list)
            elif isinstance(result, list):
                raw_posts_batch.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"⚠️ [AdaptiveFetch] Error fetching posts: {result}")
        
        total_raw_fetched += len(raw_posts_batch)
        
        # Check if we got no new posts (exhausted)
        if not raw_posts_batch:
            logger.info(
                f"🏁 [AdaptiveFetch] No more posts available (batch {batch_index}). "
                f"Period exhausted."
            )
            exhausted = True
            break
        
        logger.info(
            f"📥 [AdaptiveFetch] Batch {batch_index}: received {len(raw_posts_batch)} raw posts"
        )
        
        # Filter posts with OpinionDetector
        batch_opinions = 0
        for post in raw_posts_batch:
            # Skip duplicates
            if post.id in seen_ids:
                continue
            seen_ids.add(post.id)
            
            # Get text for opinion detection
            text = ""
            if hasattr(post, 'title') and post.title:
                text += post.title + " "
            if hasattr(post, 'text') and post.text:
                text += post.text
            
            if not text.strip():
                continue
            
            # Apply OpinionDetector
            try:
                score = opinion_score(text)
                all_opinion_scores.append(score)
                
                if is_opinion(text, threshold=settings.opinion_detector_threshold):
                    collected.append(post)
                    batch_opinions += 1
                    total_opinions_found += 1
                    
                    logger.debug(
                        f"✅ [AdaptiveFetch] Opinion found: {post.id[:8]}... "
                        f"(score={score:.3f})"
                    )
                    
                    # Check if we've reached our goal
                    if len(collected) >= desired_reviews:
                        logger.info(
                            f"🎯 [AdaptiveFetch] Target reached! "
                            f"Collected {len(collected)}/{desired_reviews} opinions"
                        )
                        break
                else:
                    logger.debug(
                        f"❌ [AdaptiveFetch] Not an opinion: {post.id[:8]}... "
                        f"(score={score:.3f})"
                    )
            except Exception as e:
                logger.error(f"⚠️ [AdaptiveFetch] Error processing post {post.id}: {e}")
                continue
        
        logger.info(
            f"✅ [AdaptiveFetch] Batch {batch_index} complete: "
            f"{batch_opinions} opinions found, total collected: {len(collected)}/{desired_reviews}"
        )
        
        # Check if we've reached our goal
        if len(collected) >= desired_reviews:
            break
        
        # Check if this batch yielded very few results (might indicate exhaustion)
        if batch_opinions == 0 and batch_index > 1:
            logger.info(
                f"🏁 [AdaptiveFetch] No opinions in batch {batch_index}. "
                f"Likely exhausted available posts."
            )
            exhausted = True
            break
    
    # Calculate statistics
    avg_opinion_score = (
        sum(all_opinion_scores) / len(all_opinion_scores)
        if all_opinion_scores else 0.0
    )
    
    stats = {
        "requested": desired_reviews,
        "collected": len(collected),
        "batches": batch_index,
        "raw_posts_fetched": total_raw_fetched,
        "opinions_found": total_opinions_found,
        "exhausted": exhausted,
        "avg_opinion_score": round(avg_opinion_score, 3),
        "opinion_rate": round((total_opinions_found / total_raw_fetched * 100), 1) if total_raw_fetched > 0 else 0.0
    }
    
    logger.info(
        f"🏁 [AdaptiveFetch] Complete: keyword='{keyword}', "
        f"requested={stats['requested']}, collected={stats['collected']}, "
        f"batches={stats['batches']}, raw_posts={stats['raw_posts_fetched']}, "
        f"opinions_found={stats['opinions_found']}, exhausted={stats['exhausted']}, "
        f"avg_score={stats['avg_opinion_score']}, opinion_rate={stats['opinion_rate']}%"
    )
    
    # Return only the requested number (slice if we over-collected)
    return {
        "reviews": collected[:desired_reviews],
        "stats": stats
    }

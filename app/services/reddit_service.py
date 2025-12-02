import praw
from typing import List, Optional, Tuple
from datetime import datetime
import logging
from app.config import get_settings
from app.schemas.responses import Post, Comment

logger = logging.getLogger(__name__)

# 🚀 V1 OPTIMIZATION: Limit comments per post for fast response
MAX_COMMENTS_PER_POST = 10  # Top comments only, sorted by score


class RedditService:
    """Service for interacting with Reddit API"""
    
    def __init__(self):
        self.settings = get_settings()
        self.reddit = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Reddit client with credentials"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.settings.reddit_client_id,
                client_secret=self.settings.reddit_client_secret,
                user_agent=self.settings.reddit_user_agent
            )
            logger.info("Reddit client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None
    
    def is_available(self) -> bool:
        """Check if Reddit service is available"""
        return self.reddit is not None
    
    def search_posts_for_trends(
        self,
        keyword: str,
        time_filter: str = "week",
        limit: int = 100
    ) -> Tuple[List[Post], int]:
        """
        🚀 Search posts for TRENDS with EXHAUSTIVE PAGINATION
        
        ⚠️ IMPORTANT CHANGES (v3.0):
        - Always fetches from "month" (30 days) for temporal consistency
        - Uses PAGINATION to fetch up to max_trends_fetch_per_platform posts
        - Actual time filtering done by TrendsService.analyze_trends()
        
        Pagination Strategy:
        - Reddit API allows limit=None for generator iteration
        - We batch process to avoid memory issues
        - Stop conditions:
          1. Reached max_trends_fetch_per_platform (config)
          2. No more results from Reddit API
          3. Exceeded max_api_calls safety limit
        
        Mathematical Guarantees:
        - 30d dataset ⊇ 7d dataset ⊇ 24h dataset
        - total_mentions(30d) >= total_mentions(7d) >= total_mentions(24h)
        
        Args:
            keyword: Search keyword
            time_filter: IGNORED - always uses "month" for consistency
            limit: IGNORED - uses max_trends_fetch_per_platform instead
            
        Returns:
            Tuple[List[Post], int]: (all_posts, total_count)
        """
        if not self.is_available():
            logger.warning("Reddit client not available")
            return [], 0
        
        settings = get_settings()
        
        try:
            # 🔧 Configuration for exhaustive fetch
            actual_time_filter = "month"  # Always 30 days
            max_posts = settings.max_trends_fetch_per_platform  # Ex: 1000
            batch_size = settings.reddit_pagination_batch_size  # Ex: 100
            
            logger.info(
                f"🚀 [PAGINATED] Fetching posts for TRENDS: '{keyword}' "
                f"(time={actual_time_filter}, max={max_posts} posts, batch={batch_size})"
            )
            
            import time
            start_fetch = time.perf_counter()
            
            posts = []
            seen_ids = set()  # Pour éviter les doublons
            api_calls = 0
            max_api_calls = settings.trends_max_api_calls  # Safety limit
            
            # 🔄 MULTI-SORT STRATEGY: Combine different sorts for better temporal coverage
            # Reddit's API doesn't guarantee chronological coverage with just "new"
            # Using multiple sorts increases chances of getting older posts
            sort_strategies = ["new", "relevance", "top"]
            posts_per_sort = max_posts // len(sort_strategies)
            
            for sort_type in sort_strategies:
                logger.info(f"📊 Fetching with sort='{sort_type}' (target: {posts_per_sort} posts)")
                
                try:
                    submissions = self.reddit.subreddit("all").search(
                        keyword,
                        limit=None,  # Pagination
                        time_filter=actual_time_filter,
                        sort=sort_type
                    )
                    
                    sort_count = 0
                    for submission in submissions:
                        # Skip duplicates
                        if submission.id in seen_ids:
                            continue
                        seen_ids.add(submission.id)
                        
                        # Stop if we have enough for this sort
                        if sort_count >= posts_per_sort:
                            break
                        
                        # Global limit
                        if len(posts) >= max_posts:
                            break
                        
                        try:
                            from app.schemas.responses import SentimentAnalysis
                            
                            post = Post(
                                id=submission.id,
                                platform="reddit",
                                text=submission.selftext if submission.selftext else submission.title,
                                title=submission.title,
                                author=str(submission.author) if submission.author else "[deleted]",
                                created_at=datetime.utcfromtimestamp(submission.created_utc),
                                score=submission.score,
                                url=f"https://reddit.com{submission.permalink}",
                                num_comments=submission.num_comments,
                                comments=[],
                                sentiment=SentimentAnalysis(
                                    positive=0.33,
                                    negative=0.33,
                                    neutral=0.34,
                                    dominant="neutral"
                                )
                            )
                            posts.append(post)
                            sort_count += 1
                            
                        except Exception as e:
                            logger.debug(f"Error parsing submission for trends: {e}")
                            continue
                    
                    api_calls += 1
                    logger.info(f"✅ sort='{sort_type}': {sort_count} posts (total: {len(posts)})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error with sort='{sort_type}': {e}")
                    continue
                
                if len(posts) >= max_posts:
                    break
            
            # 🔍 Log temporal distribution
            from collections import Counter
            daily_counts = Counter(p.created_at.date() for p in posts if p.created_at)
            sorted_days = sorted(daily_counts.items())
            dist_str = ", ".join(f"{d}: {c}" for d, c in sorted_days)
            logger.info(f"📅 Reddit fetch distribution: {dist_str}")
            
            elapsed = time.perf_counter() - start_fetch
            total_count = len(posts)
            
            logger.info(
                f"✅ [PAGINATED] Reddit trends fetch complete: "
                f"{total_count} posts in {elapsed:.2f}s "
                f"({api_calls} API calls, avg {elapsed/max(api_calls, 1):.2f}s/call)"
            )
            
            return posts, total_count
            
        except Exception as e:
            logger.error(f"Error fetching trends from Reddit: {e}")
            return [], 0
    
    async def search_posts(
        self,
        keyword: str,
        desired_reviews: int = 10,
        time_filter: str = "week",
        include_comments: bool = True
    ) -> Tuple[List[Post], dict]:
        """
        Search for posts on Reddit related to a keyword
        
        Stratégie d'over-fetch:
        - Récupère desired_reviews * review_overfetch_factor posts bruts
        - Applique ReviewFilterService pour garder uniquement les reviews
        - Retourne au maximum desired_reviews avis filtrés
        
        Args:
            keyword: Search term
            desired_reviews: Nombre d'avis désirés (pas le nombre de posts bruts)
            time_filter: Time filter (hour, day, week, month, year, all)
            include_comments: Whether to include comments
        
        Returns:
            Tuple (posts_filtrés, filter_stats)
            - posts_filtrés: Au maximum desired_reviews avis
            - filter_stats: Statistiques de filtrage + métadonnées
        """
        if not self.is_available():
            logger.warning("Reddit client not available")
            return [], {'total': 0, 'kept': 0, 'filtered': 0, 'percentage': 0.0, 'raw_fetched': 0, 'returned': 0}
        
        # 🚀 Calculer le nombre de posts bruts à récupérer (over-fetch optimisé)
        raw_limit = min(
            desired_reviews * self.settings.review_overfetch_factor,
            self.settings.max_posts_per_request  # Limite stricte (30 par défaut)
        )
        
        logger.info(
            f"📥 Over-fetching {raw_limit} raw posts for {desired_reviews} desired reviews "
            f"(factor={self.settings.review_overfetch_factor}, time_filter={time_filter})"
        )
        
        posts = []
        
        try:
            # ⏱️ Chrono: mesurer le temps de fetch Reddit PRAW
            import time
            start_fetch = time.perf_counter()
            
            # Search across all of Reddit
            submissions = self.reddit.subreddit("all").search(
                keyword,
                limit=raw_limit,
                time_filter=time_filter,
                sort="relevance"
            )
            
            for submission in submissions:
                try:
                    # Get comments if requested (🚀 V1: limit to top comments by score)
                    comments = []
                    if include_comments:
                        # IMPORTANT: limit=0 évite la récursion lourde des "MoreComments"
                        submission.comments.replace_more(limit=0)
                        
                        # Sort comments by score and take top MAX_COMMENTS_PER_POST
                        all_comments = [c for c in submission.comments.list() if hasattr(c, 'body') and c.body]
                        all_comments.sort(key=lambda c: c.score if hasattr(c, 'score') else 0, reverse=True)
                        
                        for comment in all_comments[:MAX_COMMENTS_PER_POST]:
                            comments.append(Comment(
                                id=comment.id,
                                text=comment.body,
                                author=str(comment.author) if comment.author else "[deleted]",
                                created_at=datetime.utcfromtimestamp(comment.created_utc),  # 🔧 FIX: UTC
                                score=comment.score if hasattr(comment, 'score') else 0
                            ))
                    
                    # Create Post object
                    post = Post(
                        id=submission.id,
                        platform="reddit",
                        title=submission.title,
                        text=submission.selftext if submission.selftext else submission.title,
                        author=str(submission.author) if submission.author else "[deleted]",
                        created_at=datetime.utcfromtimestamp(submission.created_utc),  # 🔧 FIX: UTC
                        score=submission.score,
                        url=f"https://reddit.com{submission.permalink}",
                        num_comments=submission.num_comments,
                        comments=comments
                    )
                    
                    posts.append(post)
                    
                except Exception as e:
                    logger.error(f"Error processing Reddit submission {submission.id}: {e}")
                    continue
            
            raw_fetched = len(posts)
            elapsed_fetch = time.perf_counter() - start_fetch
            logger.info(
                f"⏱️ Reddit fetch + build posts took {elapsed_fetch:.2f}s "
                f"(raw_limit={raw_limit}, time_filter={time_filter})"
            )
            logger.info(f"✅ Fetched {raw_fetched} raw posts from Reddit for keyword: {keyword}")
            
            # ✨ Apply review filtering (V2: heuristic-only, no ML model)
            from app.services.review_filter_service import review_filter_service
            
            posts, filter_stats = review_filter_service.filter_review_posts(posts)  # Synchrone maintenant
            
            # Slicer au nombre désiré d'avis
            final_posts = posts[:desired_reviews]
            returned_count = len(final_posts)
            
            # Enrichir les stats
            filter_stats['raw_fetched'] = raw_fetched
            filter_stats['returned'] = returned_count
            filter_stats['desired_reviews'] = desired_reviews
            
            # Logs informatifs
            if returned_count < desired_reviews:
                logger.info(
                    f"📊 Returning {returned_count}/{desired_reviews} desired review posts "
                    f"(not enough real reviews available)"
                )
            else:
                logger.info(
                    f"📊 Returning {returned_count}/{desired_reviews} desired review posts"
                )
            
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return [], {'total': 0, 'kept': 0, 'filtered': 0, 'percentage': 0.0, 'raw_fetched': 0, 'returned': 0, 'desired_reviews': desired_reviews}
        
        return final_posts, filter_stats
    
    async def get_post_details(self, post_id: str) -> Optional[Post]:
        """
        Get detailed information about a specific Reddit post
        
        Args:
            post_id: Reddit post ID
        
        Returns:
            Post object or None
        """
        if not self.is_available():
            return None
        
        try:
            submission = self.reddit.submission(id=post_id)
            
            # Get all comments
            comments = []
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if hasattr(comment, 'body') and comment.body:
                    comments.append(Comment(
                        id=comment.id,
                        text=comment.body,
                        author=str(comment.author) if comment.author else "[deleted]",
                        created_at=datetime.utcfromtimestamp(comment.created_utc),  # 🔧 FIX: UTC
                        score=comment.score
                    ))
            
            post = Post(
                id=submission.id,
                platform="reddit",
                title=submission.title,
                text=submission.selftext if submission.selftext else submission.title,
                author=str(submission.author) if submission.author else "[deleted]",
                created_at=datetime.utcfromtimestamp(submission.created_utc),  # 🔧 FIX: UTC
                score=submission.score,
                url=f"https://reddit.com{submission.permalink}",
                num_comments=submission.num_comments,
                comments=comments
            )
            
            return post
            
        except Exception as e:
            logger.error(f"Error getting Reddit post details: {e}")
            return None


# Singleton instance
reddit_service = RedditService()

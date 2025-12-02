import tweepy
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from app.config import get_settings
from app.schemas.responses import Post, Comment

logger = logging.getLogger(__name__)


class TwitterService:
    """Service for interacting with Twitter (X) API"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Twitter client with credentials"""
        try:
            # Using OAuth 2.0 Bearer Token (most common for v2 API)
            # wait_on_rate_limit=False pour ne pas bloquer l'app pendant 15 minutes
            self.client = tweepy.Client(
                bearer_token=self.settings.twitter_bearer_token,
                consumer_key=self.settings.twitter_api_key,
                consumer_secret=self.settings.twitter_api_secret,
                access_token=self.settings.twitter_access_token,
                access_token_secret=self.settings.twitter_access_token_secret,
                wait_on_rate_limit=False  # ⚠️ False pour ne pas bloquer l'application
            )
            logger.info("✅ Twitter client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Twitter client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Twitter service is available"""
        return self.client is not None
    
    def search_posts_for_trends(
        self,
        keyword: str,
        time_filter: str = "week",
        limit: int = 100
    ) -> tuple[List[Post], int]:
        """
        🚀 Search tweets for TRENDS with EXHAUSTIVE PAGINATION
        
        ⚠️ TWITTER API LIMITATIONS:
        - Recent Search API: 7 days maximum (cannot fetch 30 days)
        - Free/Basic tier: 180 requests / 15 minutes
        - max_results: 10-100 per request
        
        Pagination Strategy (v3.0):
        - Use next_token to paginate through all available tweets
        - Fetch up to max_trends_fetch_per_platform tweets
        - Respect rate limits (wait_on_rate_limit=False)
        - Stop conditions:
          1. Reached max_trends_fetch_per_platform
          2. No more next_token (end of results)
          3. Rate limit exceeded (graceful degradation)
          4. Safety limit on API calls
        
        Time Filtering:
        - Always fetches 7 days (API limit)
        - TrendsService filters to 24h/7d/30d locally
        - For 30d trends: uses 7d dataset (best available)
        
        Args:
            keyword: Search keyword
            time_filter: IGNORED - always 7 days (API limit)
            limit: IGNORED - uses max_trends_fetch_per_platform
            
        Returns:
            Tuple[List[Post], int]: (all_tweets, total_count)
        """
        if not self.is_available():
            logger.warning("Twitter client not available")
            return [], 0
        
        settings = get_settings()
        
        try:
            # 🔧 Configuration
            now = datetime.utcnow()
            start_time = now - timedelta(days=7)  # Twitter API limit
            max_posts = settings.max_trends_fetch_per_platform
            batch_size = settings.twitter_pagination_batch_size  # 100 max
            max_api_calls = settings.trends_max_api_calls
            
            logger.info(
                f"🚀 [PAGINATED] Fetching tweets for TRENDS: '{keyword}' "
                f"(7 days limit, max={max_posts} tweets, batch={batch_size})"
            )
            
            posts = []
            api_calls = 0
            next_token = None
            
            # 🔄 PAGINATION LOOP
            while True:
                # 🛑 Stop condition 1: Reached maximum posts
                if len(posts) >= max_posts:
                    logger.info(f"🛑 Reached maximum tweets limit: {max_posts}")
                    break
                
                # 🛑 Stop condition 2: Safety limit
                if api_calls >= max_api_calls:
                    logger.warning(
                        f"⚠️ Reached safety limit: {max_api_calls} API calls "
                        f"({len(posts)} tweets fetched)"
                    )
                    break
                
                # Calculate remaining tweets to fetch
                remaining = max_posts - len(posts)
                current_batch = min(batch_size, remaining, 100)  # Max 100 per API call
                
                # API Call with pagination
                try:
                    tweets = self.client.search_recent_tweets(
                        query=keyword,
                        max_results=current_batch,
                        start_time=start_time,
                        tweet_fields=["created_at", "public_metrics", "author_id"],
                        user_fields=["username"],
                        expansions=["author_id"],
                        next_token=next_token  # ✅ Pagination token
                    )
                    
                    api_calls += 1
                    
                except tweepy.errors.TooManyRequests:
                    logger.warning(
                        f"⚠️ Twitter rate limit exceeded after {api_calls} calls "
                        f"({len(posts)} tweets collected)"
                    )
                    break  # Graceful degradation
                    
                except tweepy.errors.BadRequest as e:
                    logger.error(f"❌ Twitter Bad Request: {e}")
                    break
                
                # 🛑 Stop condition 3: No more data
                if not tweets.data:
                    logger.info(f"🛑 No more tweets available (end of results)")
                    break
                
                # Create user mapping
                users = {}
                if tweets.includes and "users" in tweets.includes:
                    users = {user.id: user.username for user in tweets.includes["users"]}
                
                # Process tweets
                batch_count = 0
                for tweet in tweets.data:
                    try:
                        # Create minimal post for trends (no replies for performance)
                        post = Post(
                            id=str(tweet.id),  # ✅ Convert int to string
                            platform="twitter",
                            title=None,
                            text=tweet.text,
                            author=users.get(tweet.author_id, "unknown"),
                            created_at=tweet.created_at,
                            score=tweet.public_metrics.get("like_count", 0),
                            url=f"https://twitter.com/user/status/{tweet.id}",
                            num_comments=tweet.public_metrics.get("reply_count", 0),
                            comments=[]  # No replies for trends
                        )
                        posts.append(post)
                        batch_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing tweet {tweet.id}: {e}")
                        continue
                
                logger.info(
                    f"📊 Batch {api_calls}: +{batch_count} tweets "
                    f"(total: {len(posts)}/{max_posts})"
                )
                
                # 🔄 Check for next_token to continue pagination
                if tweets.meta and "next_token" in tweets.meta:
                    next_token = tweets.meta["next_token"]
                    logger.debug(f"🔄 Next token available, continuing pagination...")
                else:
                    logger.info(f"🛑 No more next_token (reached end of available tweets)")
                    break
            
            total_count = len(posts)
            logger.info(
                f"✅ [PAGINATED] Twitter trends fetch complete: "
                f"{total_count} tweets ({api_calls} API calls)"
            )
            
            return posts, total_count
            
        except tweepy.errors.TooManyRequests as e:
            logger.warning(f"⚠️ Twitter rate limit exceeded (trends): {e}")
            logger.warning("💡 Attendez 15 minutes ou désactivez Twitter")
            return [], 0
        except tweepy.errors.BadRequest as e:
            logger.error(f"❌ Twitter Bad Request (trends): {e}")
            return [], 0
        except tweepy.TweepyException as e:
            logger.error(f"❌ Twitter API error (trends): {e}")
            return [], 0
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching trends from Twitter: {e}")
            return [], 0
    
    async def search_posts(
        self,
        keyword: str,
        limit: int = 10,
        time_filter: str = "week",
        include_comments: bool = True
    ) -> List[Post]:
        """
        Search for tweets on Twitter related to a keyword
        
        Args:
            keyword: Search term
            limit: Maximum number of tweets to retrieve
            time_filter: Time filter (hour, day, week, month)
            include_comments: Whether to include replies
        
        Returns:
            List of Post objects
        """
        if not self.is_available():
            logger.warning("Twitter client not available")
            return []
        
        posts = []
        
        try:
            # Calculate start time based on time filter
            now = datetime.utcnow()
            time_map = {
                "hour": timedelta(hours=1),
                "day": timedelta(days=1),
                "week": timedelta(days=7),
                "month": timedelta(days=7),  # Twitter API limitation: max 7 days
                "year": timedelta(days=7),   # Twitter API limitation: max 7 days
                "all": timedelta(days=7)     # Twitter API limitation: max 7 days
            }
            # Twitter Recent Search API only allows last 7 days
            requested_delta = time_map.get(time_filter, timedelta(days=7))
            start_time = now - min(requested_delta, timedelta(days=7))
            
            # Search for tweets
            # Twitter API v2 requires max_results between 10 and 100
            max_results = max(10, min(limit, 100))
            
            tweets = self.client.search_recent_tweets(
                query=keyword,
                max_results=max_results,
                start_time=start_time,
                tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id"],
                user_fields=["username"],
                expansions=["author_id"]
            )
            
            if not tweets.data:
                logger.info(f"No tweets found for keyword: {keyword}")
                return []
            
            # Create user mapping
            users = {user.id: user.username for user in tweets.includes.get("users", [])}
            
            for tweet in tweets.data:
                try:
                    # ⚠️ DISABLED: Commentaires désactivés pour éviter rate limit (chaque tweet = 1 requête)
                    # Getting replies consumes too much rate limit (1 request per tweet)
                    comments = []
                    # if include_comments:  # DISABLED temporarily
                    #     try:
                    #         replies = self.client.search_recent_tweets(...)
                    
                    # Create Post object
                    post = Post(
                        id=str(tweet.id),  # ✅ Convert int to string.id),  # ✅ Convert int to string
                        platform="twitter",
                        title=None,
                        text=tweet.text,
                        author=users.get(tweet.author_id, "unknown"),
                        created_at=tweet.created_at,
                        score=tweet.public_metrics.get("like_count", 0),
                        url=f"https://twitter.com/user/status/{tweet.id}",
                        num_comments=tweet.public_metrics.get("reply_count", 0),
                        comments=[]  # ✅ Empty comments to save rate limit
                    )
                    
                    posts.append(post)
                    
                except Exception as e:
                    logger.error(f"Error processing tweet {tweet.id}: {e}")
                    continue
            
            initial_count = len(posts)
            logger.info(f"✅ Fetched {initial_count} tweets from Twitter for keyword: {keyword}")
            
            # ✨ V2: Apply review filtering (heuristic-only, no ML model)
            from app.services.review_filter_service import review_filter_service
            
            # Filtrage heuristique direct (synchrone, rapide)
            posts, filter_stats = review_filter_service.filter_review_posts(posts)
            logger.info(
                f"📊 Review filtering (heuristic): {filter_stats['kept']}/{filter_stats['total']} posts kept "
                f"({filter_stats['percentage']:.1f}%), {filter_stats['filtered']} filtered out"
            )
            
        except tweepy.errors.TooManyRequests as e:
            logger.warning(f"⚠️ Twitter rate limit exceeded: {e}")
            logger.warning("💡 Attendez 15 minutes ou désactivez Twitter temporairement")
            return posts
        except tweepy.errors.BadRequest as e:
            logger.error(f"❌ Twitter Bad Request: {e}")
            logger.info("💡 Twitter Recent Search API limite: 7 jours maximum")
            return posts
        except tweepy.TweepyException as e:
            logger.error(f"❌ Twitter API error: {e}")
            return posts
        except Exception as e:
            logger.error(f"❌ Unexpected error searching Twitter: {e}")
            return posts
        
        return posts
    
    async def get_post_details(self, post_id: str) -> Optional[Post]:
        """
        Get detailed information about a specific tweet
        
        Args:
            post_id: Tweet ID
        
        Returns:
            Post object or None
        """
        if not self.is_available():
            return None
        
        try:
            # Get tweet
            tweet = self.client.get_tweet(
                id=post_id,
                tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id"],
                user_fields=["username"],
                expansions=["author_id"]
            )
            
            if not tweet.data:
                return None
            
            # Get user info
            username = tweet.includes["users"][0].username if tweet.includes.get("users") else "unknown"
            
            # ⚠️ DISABLED: Commentaires désactivés pour éviter rate limit
            comments = []
            # try:
            #     replies = self.client.search_recent_tweets(...)
            
            post = Post(
                id=str(tweet.data.id),  # ✅ Convert int to string
                platform="twitter",
                title=None,
                text=tweet.data.text,
                author=username,
                created_at=tweet.data.created_at,
                score=tweet.data.public_metrics.get("like_count", 0),
                url=f"https://twitter.com/user/status/{tweet.data.id}",
                num_comments=tweet.data.public_metrics.get("reply_count", 0),
                comments=comments
            )
            
            return post
            
        except Exception as e:
            logger.error(f"Error getting Twitter post details: {e}")
            return None


# Singleton instance
twitter_service = TwitterService()

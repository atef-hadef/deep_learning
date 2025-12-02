"""
🚀 INTELLIGENT TRENDS SERVICE - Advanced Analytics
Author: AI Assistant
Date: 2025-10-15
Version: 2.2 - Keyword Filter Opt-Out + Bucket Fix

Features:
- Linear regression-based trend detection
- Z-score spike detection
- Sentiment evolution analysis
- Popularity scoring
- Adaptive time bucketing (calendar aligned)
- Moving average smoothing
- Multi-platform aggregation
"""

from typing import List, Dict
from datetime import datetime, timedelta, time
from collections import Counter, defaultdict
import logging
import re
import statistics
import math
import numpy as np  # éventuellement utilisé ailleurs
from scipy import stats
from app.schemas.responses import (
    TrendData,
    TrendPoint,
    PopularTopic,
    SentimentAnalysis,
    Post,
    DetectedPeak,
)
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)


class TrendsService:
    """
    🚀 INTELLIGENT TRENDS SERVICE

    Provides advanced trend analysis with:
    - Statistical spike detection (Z-score)
    - Linear regression for trend direction
    - Sentiment evolution tracking
    - Popularity scoring
    - Smoothed curve generation
    """

    def __init__(self):
        # Configuration for intelligent trend detection
        self.TREND_SLOPE_THRESHOLD = 0.05  # 5% slope = significant trend
        self.SPIKE_Z_THRESHOLD = 1.5  # 1.5 standard deviations = spike
        self.MIN_POSTS_FOR_TREND = 1  # Minimum posts for analysis
        self.MAX_BUCKETS = 24  # Maximum data points on chart
        self.SENTIMENT_CHANGE_THRESHOLD = 0.1  # 10% change = significant

        logger.info("🚀 Intelligent Trends Service v2.2 initialized")

    # ==================== UTILITY METHODS ====================

    def parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta"""
        time_map = {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        return time_map.get(time_range, timedelta(days=7))

    def get_adaptive_bucket_size(
        self, time_range: str, total_duration: timedelta
    ) -> timedelta:
        """
        📊 ADAPTIVE TIME BUCKETING

        Determines optimal bucket size based on the selected period:
        - 24h → hourly buckets
        - 7d → daily buckets
        - 30d → 2-day buckets
        """
        delta = self.parse_time_range(time_range)

        if delta.days >= 30:
            return timedelta(days=2)  # 30 days → buckets of 2 days
        elif delta.days >= 7:
            return timedelta(days=1)  # 7 days → daily buckets
        elif delta.days >= 1:
            return timedelta(hours=1)  # 24h → hourly buckets
        else:
            return timedelta(minutes=30)  # <24h → 30-minute buckets

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text
        """
        # Remove URLs, mentions, and special characters
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^\w\s]", "", text)

        # Convert to lowercase and split
        words = text.lower().split()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "them",
            "their",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "with",
            "from",
            "not",
            "just",
            "get",
            "so",
            "like",
            "if",
        }

        # Filter words
        keywords = [
            word for word in words if len(word) > 3 and word not in stop_words
        ]

        return keywords

    def filter_posts_by_keyword(self, posts: List[Post], keyword: str) -> List[Post]:
        """
        🔧 V3.1: Filter posts with flexible phrase matching

        Strategy:
        - Single word: "iphone" → simple substring match
        - Multi-word: "iphone 15" → check both "iphone 15" AND "iphone15"
        - Handles: "iPhone 15", "iphone-15", "iphone 15 pro"
        """
        if not keyword:
            return posts

        # Normalize keyword: lowercase, remove special chars except spaces
        search_keyword = re.sub(r"[^a-z0-9\s]", "", keyword.lower())
        search_terms = search_keyword.split()

        # Create variations for flexible matching
        # "iphone 15" → ["iphone 15", "iphone15"]
        phrase_with_space = " ".join(search_terms)
        phrase_without_space = "".join(search_terms)

        filtered_posts = []

        for post in posts:
            # Build full text
            full_text = ""
            if post.title:
                full_text += post.title + " "
            full_text += post.text

            # Add comments
            for comment in post.comments:
                full_text += " " + comment.text

            # Normalize text for matching
            normalized_text = re.sub(
                r"[^a-z0-9\s]", "", full_text.lower()
            )

            # Match if phrase appears with OR without spaces
            if len(search_terms) == 1:
                if search_terms[0] in normalized_text:
                    filtered_posts.append(post)
            else:
                if (
                    phrase_with_space in normalized_text
                    or phrase_without_space in normalized_text
                ):
                    filtered_posts.append(post)

        logger.info(
            f"🔧 Keyword '{keyword}': {len(filtered_posts)}/{len(posts)} posts matched "
            f"(phrase: '{phrase_with_space}' or '{phrase_without_space}')"
        )
        return filtered_posts

    def _debug_daily_counts(self, posts: List[Post], label: str) -> None:
        """
        Debug helper: log number of posts per calendar day.
        """
        counter = Counter()
        for p in posts:
            day = p.created_at.date().isoformat()
            counter[day] += 1

        if not counter:
            logger.warning(f"[DEBUG][{label}] No posts to count per day")
            return

        days_sorted = sorted(counter.items())
        dist_str = ", ".join(f"{d}: {c}" for d, c in days_sorted)
        logger.info(f"[DEBUG][{label}] Daily counts = {dist_str}")

    # ==================== STATISTICAL METHODS ====================

    def calculate_moving_average(
        self, data: List[float], window: int = 3
    ) -> List[float]:
        """
        📊 Calculate moving average for smoothing
        """
        if len(data) < window:
            return data

        smoothed = []
        for i in range(len(data)):
            if i < window - 1:
                smoothed.append(sum(data[: i + 1]) / (i + 1))
            else:
                smoothed.append(sum(data[i - window + 1 : i + 1]) / window)

        return smoothed

    def calculate_linear_regression(
        self, y_values: List[int]
    ) -> tuple[float, float]:
        """
        📈 LINEAR REGRESSION for trend detection
        """
        if len(y_values) < 2:
            return 0.0, 0.0

        x_values = list(range(len(y_values)))

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_values, y_values
            )
            return slope, intercept
        except Exception as e:
            logger.warning(f"Linear regression failed: {e}")
            return 0.0, 0.0

    def detect_spikes_zscore(self, mentions: List[int]) -> List[int]:
        """
        🔥 Z-SCORE SPIKE DETECTION
        """
        if len(mentions) < 3:
            return []

        try:
            mean = statistics.mean(mentions)
            stdev = statistics.stdev(mentions)

            if stdev == 0:
                return []

            spike_indices = []
            for i, value in enumerate(mentions):
                z_score = (value - mean) / stdev
                if z_score > self.SPIKE_Z_THRESHOLD:
                    spike_indices.append(i)

            return spike_indices
        except Exception as e:
            logger.warning(f"Z-score spike detection failed: {e}")
            return []

    def calculate_popularity_score(
        self, normalized_mentions: float, sentiment_positivity: float
    ) -> float:
        """
        ⭐ POPULARITY SCORE CALCULATION
        """
        score = (0.6 * normalized_mentions) + (0.4 * sentiment_positivity)
        return round(score, 3)

    def calculate_growth_rate(self, current: int, previous: int) -> float:
        """📈 Calculate growth rate between two periods"""
        if previous == 0:
            return 1.0 if current > 0 else 0.0

        growth_rate = (current - previous) / previous
        return round(growth_rate, 3)

    def calculate_trend_direction_from_slope(
        self, slope: float, avg_mentions: float
    ) -> str:
        """
        📈 TREND DIRECTION using LINEAR REGRESSION SLOPE
        """
        if avg_mentions == 0:
            return "stable"

        # Normalize slope relative to average
        relative_slope = slope / max(avg_mentions, 1)

        if relative_slope > self.TREND_SLOPE_THRESHOLD:
            return "en hausse"
        elif relative_slope < -self.TREND_SLOPE_THRESHOLD:
            return "en baisse"
        else:
            return "stable"

    def calculate_growth_percentage(
        self, recent_mentions: List[int], early_mentions: List[int]
    ) -> float:
        """
        📊 GROWTH PERCENTAGE CALCULATION
        """
        if not early_mentions or not recent_mentions:
            return 0.0

        avg_early = sum(early_mentions) / len(early_mentions)
        avg_recent = sum(recent_mentions) / len(recent_mentions)

        if avg_early == 0:
            return 100.0 if avg_recent > 0 else 0.0

        growth = ((avg_recent - avg_early) / avg_early) * 100
        return round(growth, 1)

    def analyze_sentiment_evolution(
        self, sentiments: List[SentimentAnalysis]
    ) -> str:
        """
        😊 SENTIMENT EVOLUTION ANALYSIS
        """
        if len(sentiments) < 6:
            return "stable"

        early_sentiments = sentiments[:3]
        recent_sentiments = sentiments[-3:]

        early_avg = sum(s.positive for s in early_sentiments) / 3
        recent_avg = sum(s.positive for s in recent_sentiments) / 3

        change = recent_avg - early_avg

        if change > self.SENTIMENT_CHANGE_THRESHOLD:
            return "en hausse"
        elif change < -self.SENTIMENT_CHANGE_THRESHOLD:
            return "en baisse"
        else:
            return "stable"

    # ==================== MAIN ANALYSIS METHOD ====================

    async def analyze_trends(
        self,
        posts: List[Post],
        keyword: str,
        time_range: str = "7d",
        apply_keyword_filter: bool = True,
    ) -> List[TrendData]:
        """
        🚀 INTELLIGENT TREND ANALYSIS v2.2 - Statistical & Advanced Analytics

        Args:
            posts: List of Post objects (déjà filtrés par l'API/DB)
            keyword: keyword principal (pour logs / filtrage optionnel)
            time_range: '24h' | '7d' | '30d'
            apply_keyword_filter:
                - True  → re-filtre localement avec filter_posts_by_keyword
                - False → utilise tous les posts (cas /api/trends pour flux global)
        """

        if not posts:
            logger.warning(f"No posts for trend analysis of '{keyword}'")
            return []

        logger.info(
            f"🚀 v2.2 Trend Analysis: '{keyword}' over {time_range} "
            f"(apply_keyword_filter={apply_keyword_filter})"
        )

        # Step 1: keyword filter optionnel
        if apply_keyword_filter and keyword:
            keyword_posts = self.filter_posts_by_keyword(posts, keyword)

            if not keyword_posts:
                logger.warning(f"No posts contain keyword '{keyword}' after local filtering")
                return []

            # Debug distribution BEFORE time filter
            self._debug_daily_counts(
                keyword_posts, f"{keyword}-before-time-filter"
            )
        else:
            # ⚠️ IMPORTANT pour /api/trends :
            # On considère que les posts sont déjà pertinents (requête DB/API)
            keyword_posts = list(posts)
            logger.info(
                f"🔍 Skipping local keyword filter, using {len(keyword_posts)} posts as-is"
            )
            self._debug_daily_counts(
                keyword_posts, f"{keyword}-before-time-filter-no-refilter"
            )

        # Step 1.5: Filter by time_range (24h/7d/30d) relative to NOW
        now = datetime.utcnow()
        requested_delta = self.parse_time_range(time_range)
        cutoff_time = now - requested_delta

        time_filtered_posts = [
            post for post in keyword_posts if post.created_at >= cutoff_time
        ]

        logger.info(
            f"⏰ Time filtering: {len(keyword_posts)} posts → "
            f"{len(time_filtered_posts)} posts within {time_range} "
            f"(cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M')})"
        )

        if not time_filtered_posts:
            logger.warning(f"No posts within {time_range} after time filtering")
            return []

        # Debug distribution AFTER time filter
        self._debug_daily_counts(
            time_filtered_posts, f"{keyword}-after-time-filter"
        )

        keyword_posts = time_filtered_posts

        # Step 2: Group by platform
        posts_by_platform = defaultdict(list)
        for post in keyword_posts:
            posts_by_platform[post.platform].append(post)

        trend_data_list = []
        max_mentions_global = 0  # For normalizing popularity score

        # Step 3: Analyze each platform
        for platform, platform_posts in posts_by_platform.items():
            logger.info(
                f"📊 Analyzing {len(platform_posts)} posts for {platform}"
            )

            if len(platform_posts) < self.MIN_POSTS_FOR_TREND:
                logger.warning(
                    f"Skipping {platform}: only {len(platform_posts)} posts"
                )
                continue

            # Sort posts by date
            platform_posts.sort(key=lambda p: p.created_at)

            # BUCKET RANGE: align on calendar (days / hours)
            now = datetime.utcnow()
            requested_delta = self.parse_time_range(time_range)
            bucket_size = self.get_adaptive_bucket_size(
                time_range, requested_delta
            )

            # ----- Build bucket boundaries -----
            all_bucket_times: List[datetime] = []

            if bucket_size >= timedelta(days=1):
                # Daily or multi-day buckets (7d / 30d)
                bucket_days = bucket_size.days
                total_days = requested_delta.days or 1

                # Number of buckets (e.g. 7d → 7 buckets of 1d)
                num_buckets = math.ceil(total_days / bucket_days)

                end_date = now.date()  # today
                # Start pour que le dernier bucket contienne aujourd'hui
                start_date = end_date - timedelta(
                    days=(num_buckets * bucket_days - 1)
                )

                for i in range(num_buckets):
                    day = start_date + timedelta(days=i * bucket_days)
                    bucket_start = datetime.combine(day, time.min)
                    all_bucket_times.append(bucket_start)

                total_duration = timedelta(days=num_buckets * bucket_days)
            else:
                # Hourly buckets for 24h
                bucket_hours = int(bucket_size.total_seconds() // 3600) or 1
                total_hours = int(requested_delta.total_seconds() // 3600) or 1
                num_buckets = math.ceil(total_hours / bucket_hours)

                end_time = now.replace(
                    minute=0, second=0, microsecond=0
                )
                start_time = end_time - timedelta(
                    hours=(num_buckets * bucket_hours - 1)
                )

                for i in range(num_buckets):
                    bucket_start = start_time + timedelta(
                        hours=i * bucket_hours
                    )
                    all_bucket_times.append(bucket_start)

                total_duration = timedelta(
                    hours=num_buckets * bucket_hours
                )

            bucket_unit = (
                f"{bucket_size.days}d"
                if bucket_size.days > 0
                else f"{int(bucket_size.total_seconds() // 3600)}h"
            )
            logger.info(
                f"📊 [{keyword}] {platform}: {len(all_bucket_times)} buckets "
                f"({bucket_unit} each) from "
                f"{all_bucket_times[0].strftime('%Y-%m-%d %H:%M')} "
                f"to {all_bucket_times[-1].strftime('%Y-%m-%d %H:%M')} "
                f"(requested_delta={requested_delta}, total_duration={total_duration})"
            )

            # Group posts into time buckets (calendar aligned)
            time_buckets: Dict[datetime, List[Post]] = {
                bt: [] for bt in all_bucket_times
            }
            posts_assigned = 0
            posts_before_range = 0
            posts_after_range = 0
            bucket_distribution: Dict[str, int] = {}

            first_bucket_start = all_bucket_times[0]

            if bucket_size >= timedelta(days=1):
                unit_days = bucket_size.days
                for post in platform_posts:
                    diff_days = (post.created_at.date() - first_bucket_start.date()).days
                    if diff_days < 0:
                        posts_before_range += 1
                        continue
                    bucket_index = diff_days // unit_days
                    if bucket_index >= len(all_bucket_times):
                        posts_after_range += 1
                        bucket_index = len(all_bucket_times) - 1

                    bucket_time = all_bucket_times[bucket_index]
                    time_buckets[bucket_time].append(post)
                    posts_assigned += 1

                    bucket_date = bucket_time.strftime("%Y-%m-%d")
                    bucket_distribution[bucket_date] = (
                        bucket_distribution.get(bucket_date, 0) + 1
                    )
            else:
                unit_hours = int(bucket_size.total_seconds() // 3600) or 1
                for post in platform_posts:
                    diff_seconds = (
                        post.created_at - first_bucket_start
                    ).total_seconds()
                    diff_hours = int(diff_seconds // 3600)

                    if diff_hours < 0:
                        posts_before_range += 1
                        continue
                    bucket_index = diff_hours // unit_hours
                    if bucket_index >= len(all_bucket_times):
                        posts_after_range += 1
                        bucket_index = len(all_bucket_times) - 1

                    bucket_time = all_bucket_times[bucket_index]
                    time_buckets[bucket_time].append(post)
                    posts_assigned += 1

                    bucket_label = bucket_time.strftime("%Y-%m-%d %H:00")
                    bucket_distribution[bucket_label] = (
                        bucket_distribution.get(bucket_label, 0) + 1
                    )

            logger.info(
                f"📊 [{keyword}] {platform}: Assigned "
                f"{posts_assigned}/{len(platform_posts)} posts "
                f"({posts_before_range} before, {posts_after_range} after range)"
            )

            non_empty_buckets = sum(
                1 for b_posts in time_buckets.values() if b_posts
            )
            logger.info(
                f"📊 [{keyword}] {platform}: "
                f"{non_empty_buckets}/{len(all_bucket_times)} buckets have data"
            )

            if bucket_distribution:
                dist_items = sorted(bucket_distribution.items())[:10]
                dist_str = ", ".join(f"{d}: {c}" for d, c in dist_items)
                logger.info(
                    f"📊 [{keyword}] {platform}: Bucket distribution sample: {dist_str}"
                )
            else:
                logger.warning(
                    f"⚠️ [{keyword}] {platform}: NO POSTS IN BUCKETS - check time filtering!"
                )

            # Create trend points with intelligence
            trend_points: List[TrendPoint] = []
            mentions_history: List[int] = []
            sentiment_history: List[SentimentAnalysis] = []

            for bucket_time in sorted(time_buckets.keys()):
                bucket_posts = time_buckets[bucket_time]
                mentions_count = len(bucket_posts)
                mentions_history.append(mentions_count)

                # Calculate sentiment for this period
                if bucket_posts:
                    all_sentiments: List[SentimentAnalysis] = []
                    for post in bucket_posts:
                        if post.sentiment:
                            all_sentiments.append(post.sentiment)
                        for comment in post.comments:
                            if comment.sentiment:
                                all_sentiments.append(comment.sentiment)

                    if all_sentiments:
                        avg_sentiment = (
                            sentiment_service.calculate_overall_sentiment(
                                all_sentiments
                            )
                        )
                    else:
                        avg_sentiment = SentimentAnalysis(
                            positive=0.33,
                            negative=0.33,
                            neutral=0.34,
                            dominant="neutral",
                        )
                else:
                    avg_sentiment = SentimentAnalysis(
                        positive=0.33,
                        negative=0.33,
                        neutral=0.34,
                        dominant="neutral",
                    )

                sentiment_history.append(avg_sentiment)

                # Growth rate point-à-point
                growth_rate = None
                if len(mentions_history) > 1:
                    growth_rate = self.calculate_growth_rate(
                        mentions_count, mentions_history[-2]
                    )

                trend_point = TrendPoint(
                    timestamp=bucket_time,
                    mentions=mentions_count,
                    sentiment=avg_sentiment,
                    growth_rate=growth_rate,
                    is_spike=False,
                )
                trend_points.append(trend_point)

            # Calculate platform-wide metrics
            total_mentions = len(platform_posts)
            peak_mentions = max(mentions_history) if mentions_history else 0

            # Overall sentiment
            all_platform_sentiments = [
                post.sentiment
                for post in platform_posts
                if post.sentiment is not None
            ]

            if all_platform_sentiments:
                avg_platform_sentiment = (
                    sentiment_service.calculate_overall_sentiment(
                        all_platform_sentiments
                    )
                )
            else:
                avg_platform_sentiment = SentimentAnalysis(
                    positive=0.33,
                    negative=0.33,
                    neutral=0.34,
                    dominant="neutral",
                )

            # Z-SCORE SPIKE DETECTION
            spike_indices = self.detect_spikes_zscore(mentions_history)
            detected_peaks: List[DetectedPeak] = []

            for spike_idx in spike_indices:
                if spike_idx < len(trend_points):
                    trend_points[spike_idx].is_spike = True
                    peak_date = trend_points[
                        spike_idx
                    ].timestamp.strftime("%Y-%m-%d")
                    peak_value = trend_points[spike_idx].mentions
                    detected_peaks.append(
                        DetectedPeak(date=peak_date, mentions=peak_value)
                    )

            logger.info(
                f"🔥 [{keyword}] {platform}: Detected {len(detected_peaks)} spikes using Z-score"
            )

            # LINEAR REGRESSION for trend direction
            slope, intercept = self.calculate_linear_regression(
                mentions_history
            )
            avg_mentions = (
                sum(mentions_history) / len(mentions_history)
                if mentions_history
                else 0
            )
            trend_direction = self.calculate_trend_direction_from_slope(
                slope, avg_mentions
            )

            logger.info(
                f"📈 [{keyword}] {platform}: Regression slope={slope:.3f}, direction={trend_direction}"
            )

            # GROWTH PERCENTAGE (first 3 vs last 3 periods)
            if len(mentions_history) >= 6:
                early_mentions = mentions_history[:3]
                recent_mentions = mentions_history[-3:]
                growth_percentage = self.calculate_growth_percentage(
                    recent_mentions, early_mentions
                )
            else:
                growth_percentage = 0.0

            logger.info(
                f"📊 [{keyword}] {platform}: Growth={growth_percentage:+.1f}%"
            )

            # SENTIMENT EVOLUTION
            sentiment_trend = self.analyze_sentiment_evolution(
                sentiment_history
            )
            logger.info(
                f"😊 [{keyword}] {platform}: Sentiment trend={sentiment_trend}"
            )

            # MOVING AVERAGE SMOOTHING
            mentions_float = [float(m) for m in mentions_history]
            smoothed_curve = self.calculate_moving_average(
                mentions_float, window=3
            )

            # POPULARITY SCORE
            max_mentions_platform = (
                max(mentions_history) if mentions_history else 1
            )
            normalized_mentions = total_mentions / max(
                max_mentions_platform * len(mentions_history), 1
            )
            normalized_mentions = min(normalized_mentions, 1.0)

            sentiment_positivity = avg_platform_sentiment.positive
            popularity_score = self.calculate_popularity_score(
                normalized_mentions, sentiment_positivity
            )

            logger.info(
                f"⭐ [{keyword}] {platform}: Popularity score={popularity_score:.3f}"
            )

            if total_mentions > max_mentions_global:
                max_mentions_global = total_mentions

            trend_data = TrendData(
                platform=platform,
                keyword=keyword,
                data_points=trend_points,
                total_mentions=total_mentions,
                average_sentiment=avg_platform_sentiment,
                overall_growth_rate=growth_percentage / 100,  # % → décimal
                peak_mentions=peak_mentions,
                trend_direction=trend_direction,
                sentiment_trend=sentiment_trend,
                sentiment_evolution=sentiment_trend,
                detected_peaks=detected_peaks,
                popularity_score=popularity_score,
                slope=slope,
                smoothed_curve=smoothed_curve,
            )

            trend_data_list.append(trend_data)

            logger.info(
                f"✅ [{keyword}] {platform}: {total_mentions} mentions, "
                f"{trend_direction}, growth {growth_percentage:+.1f}%, "
                f"sentiment {sentiment_trend}, {len(detected_peaks)} peaks"
            )

        logger.info(
            f"🎯 Intelligent trend analysis complete for '{keyword}': "
            f"{len(trend_data_list)} platforms"
        )
        return trend_data_list

    # ==================== TOPICS & TRENDING KEYWORDS ====================

    async def extract_popular_topics(
        self, posts: List[Post], main_keyword: str, top_k: int = 10
    ) -> List[PopularTopic]:
        """
        🏷️ Extract popular topics related to the main keyword
        """
        if not posts:
            return []

        logger.info(
            f"🏷️ Extracting popular topics related to '{main_keyword}'"
        )

        keyword_posts = self.filter_posts_by_keyword(posts, main_keyword)

        if not keyword_posts:
            return []

        cooccurrent_terms = Counter()

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "them",
            "their",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "with",
            "from",
            "not",
            "just",
            "get",
            "so",
            "like",
            "if",
            main_keyword.lower(),
        }

        for post in keyword_posts:
            full_text = (post.title or "") + " " + post.text
            for comment in post.comments:
                full_text += " " + comment.text

            clean_text = re.sub(r"[^\w\s]", " ", full_text.lower())
            words = clean_text.split()

            for word in words:
                if len(word) > 3 and word not in stop_words:
                    cooccurrent_terms[word] += 1

        popular_topics: List[PopularTopic] = []

        for term, frequency in cooccurrent_terms.most_common(top_k * 2):
            if frequency < 2:
                continue

            term_posts = []
            for post in keyword_posts:
                full_text = (post.title or "") + " " + post.text
                if term in full_text.lower():
                    term_posts.append(post)

            if not term_posts:
                continue

            term_sentiments = [
                post.sentiment
                for post in term_posts
                if post.sentiment is not None
            ]

            if term_sentiments:
                avg_sentiment = (
                    sentiment_service.calculate_overall_sentiment(
                        term_sentiments
                    )
                )
            else:
                avg_sentiment = SentimentAnalysis(
                    positive=0.33,
                    negative=0.33,
                    neutral=0.34,
                    dominant="neutral",
                )

            popular_topics.append(
                PopularTopic(
                    keyword=term, frequency=frequency, sentiment=avg_sentiment
                )
            )

            if len(popular_topics) >= top_k:
                break

        logger.info(
            f"🏷️ Found {len(popular_topics)} popular topics for '{main_keyword}'"
        )
        return popular_topics

    async def detect_trending_keywords(
        self, posts: List[Post], main_keyword: str, min_mentions: int = 3
    ) -> Dict[str, int]:
        """
        🔍 Detect trending keywords related to the main keyword
        """
        logger.info(
            f"🔍 Detecting trending keywords for '{main_keyword}'"
        )

        popular_topics = await self.extract_popular_topics(
            posts, main_keyword, top_k=20
        )

        trending_keywords: Dict[str, int] = {}
        for topic in popular_topics:
            if topic.frequency >= min_mentions:
                trending_keywords[topic.keyword] = topic.frequency

        logger.info(
            f"🔍 Found {len(trending_keywords)} trending keywords"
        )
        return trending_keywords


# Singleton instance
trends_service = TrendsService()

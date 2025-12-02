from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    positive: float = Field(..., ge=0, le=1, description="Positive sentiment score")
    negative: float = Field(..., ge=0, le=1, description="Negative sentiment score")
    neutral: float = Field(..., ge=0, le=1, description="Neutral sentiment score")
    dominant: str = Field(..., description="Dominant sentiment: positive, negative, or neutral")


class AspectSentiment(BaseModel):
    """Aspect-based sentiment analysis result"""
    aspect: str = Field(..., description="Product aspect (e.g., 'battery', 'price')")
    sentiment: str = Field(..., description="Sentiment for this aspect")
    mentions: int = Field(..., description="Number of mentions")


class Comment(BaseModel):
    """Comment model"""
    id: str
    text: str
    author: str
    created_at: datetime
    score: Optional[int] = 0
    sentiment: Optional[SentimentAnalysis] = None


class Post(BaseModel):
    """Post model"""
    id: str
    platform: str
    title: Optional[str] = None
    text: str
    author: str
    created_at: datetime
    score: Optional[int] = 0
    url: str
    num_comments: int = 0
    comments: List[Comment] = []
    sentiment: Optional[SentimentAnalysis] = None
    summary: Optional[str] = None
    aspects: List[AspectSentiment] = []
    relevance_score: Optional[float] = None


# ✅ TREND CLASSES DEFINED FIRST (before SearchResponse)

class TrendPoint(BaseModel):
    """Single trend data point"""
    timestamp: datetime
    mentions: int
    sentiment: SentimentAnalysis
    growth_rate: Optional[float] = None
    is_spike: Optional[bool] = False


class TrendData(BaseModel):
    """Trend data for a specific platform"""
    platform: str
    keyword: str
    data_points: List[TrendPoint]
    total_mentions: int
    average_sentiment: SentimentAnalysis
    overall_growth_rate: Optional[float] = None
    peak_mentions: Optional[int] = None
    trend_direction: Optional[str] = None  # 'rising', 'falling', 'stable'
    sentiment_evolution: Optional[str] = None  # 'improving', 'declining', 'stable'


class PopularTopic(BaseModel):
    """Popular topic/keyword detected"""
    keyword: str
    frequency: int
    sentiment: SentimentAnalysis


# ✅ NOW SearchResponse can use TrendData and PopularTopic

class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    keyword: str
    platforms: List[str]
    total_posts: int
    posts: List[Post]
    overall_sentiment: SentimentAnalysis
    product_summary: Optional[str] = Field(None, description="Global product summary from all posts and comments")
    key_points: List[str] = Field(default_factory=list, description="Key points extracted from all discussions")
    filtering_stats: Optional[Dict] = Field(None, description="Relevance filtering statistics")
    trends: List[TrendData] = Field(default_factory=list, description="Intelligent trend analysis for the keyword")
    popular_topics: List[PopularTopic] = Field(default_factory=list, description="Popular topics related to the keyword")
    execution_time: float


class TrendsResponse(BaseModel):
    """Response model for trends endpoint"""
    keyword: str
    time_range: str
    trends: List[TrendData]
    popular_topics: List[PopularTopic]
    execution_time: float


class PostDetailResponse(BaseModel):
    """Response model for post detail endpoint"""
    post: Post
    execution_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, bool]

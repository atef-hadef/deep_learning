from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SearchRequest(BaseModel):
    """Request model for searching posts on social media platforms"""
    keyword: str = Field(..., description="Product or topic to search for", min_length=1)
    platforms: list[str] = Field(default=["reddit", "twitter"], description="Platforms to search on")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of posts per platform")
    time_filter: Optional[str] = Field(default="week", description="Time filter: hour, day, week, month, year, all")
    include_comments: bool = Field(default=True, description="Include comments analysis")


class TrendsRequest(BaseModel):
    """Request model for getting trends"""
    keyword: str = Field(..., description="Product or topic to analyze trends for")
    platforms: list[str] = Field(default=["reddit", "twitter"], description="Platforms to analyze")
    time_range: str = Field(default="7d", description="Time range: 24h, 7d, 30d")

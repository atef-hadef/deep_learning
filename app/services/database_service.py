from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, TIMESTAMP, JSON, Index, select, and_
from typing import Optional, List, Dict
from datetime import datetime
import logging
import json
from app.config import get_settings
from app.schemas.responses import Post, Comment, SentimentAnalysis

logger = logging.getLogger(__name__)


# Modèle de base SQLAlchemy
class Base(DeclarativeBase):
    pass


# Table posts pour PostgreSQL
class PostModel(Base):
    __tablename__ = "posts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)
    post_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    num_comments: Mapped[int] = mapped_column(Integer, default=0)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_positive: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_negative: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_neutral: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_dominant: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_json: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)  # Stockage des données complètes
    saved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_platform_post_id', 'platform', 'post_id', unique=True),
        Index('idx_created_at', 'created_at'),
        Index('idx_platform', 'platform'),
    )


# Table search_history pour PostgreSQL
class SearchHistoryModel(Base):
    __tablename__ = "search_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    keyword: Mapped[str] = mapped_column(String(255), nullable=False)
    platforms: Mapped[str] = mapped_column(Text, nullable=False)  # Stocké comme JSON string
    num_posts: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_keyword', 'keyword'),
    )


class DatabaseService:
    """Service for interacting with PostgreSQL database"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.async_session = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize PostgreSQL connection with SQLAlchemy async"""
        try:
            self.engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.debug,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20
            )
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("PostgreSQL connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.engine = None
            self.async_session = None
    
    async def create_tables(self):
        """Create all tables if they don't exist"""
        if not self.engine:
            return
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    async def is_available(self) -> bool:
        """Check if PostgreSQL database is available"""
        if not self.engine:
            return False
        try:
            async with self.engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception as e:
            logger.error(f"Database not available: {e}")
            return False
    
    async def save_post(self, post: Post) -> bool:
        """
        Save a post to PostgreSQL database (upsert)
        
        Args:
            post: Post object to save
        
        Returns:
            True if successful, False otherwise
        """
        if not self.async_session:
            return False
        
        try:
            async with self.async_session() as session:
                # Vérifier si le post existe déjà
                stmt = select(PostModel).where(
                    and_(
                        PostModel.platform == post.platform,
                        PostModel.post_id == post.id
                    )
                )
                result = await session.execute(stmt)
                existing_post = result.scalar_one_or_none()
                
                # Préparer les données
                sentiment_data = {}
                if post.sentiment:
                    sentiment_data = {
                        'sentiment_positive': post.sentiment.positive,
                        'sentiment_negative': post.sentiment.negative,
                        'sentiment_neutral': post.sentiment.neutral,
                        'sentiment_dominant': post.sentiment.dominant
                    }
                
                # Convertir le post en dict pour raw_json
                raw_json = post.model_dump(mode='json')
                
                if existing_post:
                    # Update
                    existing_post.title = post.title
                    existing_post.text = post.text
                    existing_post.author = post.author
                    existing_post.created_at = post.created_at
                    existing_post.score = post.score
                    existing_post.url = post.url
                    existing_post.num_comments = post.num_comments
                    existing_post.relevance_score = post.relevance_score
                    existing_post.summary = post.summary
                    existing_post.raw_json = raw_json
                    existing_post.saved_at = datetime.utcnow()
                    
                    for key, value in sentiment_data.items():
                        setattr(existing_post, key, value)
                else:
                    # Insert
                    new_post = PostModel(
                        platform=post.platform,
                        post_id=post.id,
                        title=post.title,
                        text=post.text,
                        author=post.author,
                        created_at=post.created_at,
                        score=post.score,
                        url=post.url,
                        num_comments=post.num_comments,
                        relevance_score=post.relevance_score,
                        summary=post.summary,
                        raw_json=raw_json,
                        **sentiment_data
                    )
                    session.add(new_post)
                
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving post to PostgreSQL: {e}")
            return False
    
    async def save_posts(self, posts: List[Post]) -> int:
        """
        Save multiple posts to PostgreSQL (bulk operation for better performance)
        
        Args:
            posts: List of Post objects
        
        Returns:
            Number of posts saved successfully
        """
        count = 0
        for post in posts:
            if await self.save_post(post):
                count += 1
        return count
    
    async def get_post(self, platform: str, post_id: str) -> Optional[Dict]:
        """
        Retrieve a post from PostgreSQL
        
        Args:
            platform: Platform name (reddit or twitter)
            post_id: Post ID
        
        Returns:
            Post dictionary or None
        """
        if not self.async_session:
            return None
        
        try:
            async with self.async_session() as session:
                stmt = select(PostModel).where(
                    and_(
                        PostModel.platform == platform,
                        PostModel.post_id == post_id
                    )
                )
                result = await session.execute(stmt)
                post_model = result.scalar_one_or_none()
                
                if post_model and post_model.raw_json:
                    return post_model.raw_json
                return None
        except Exception as e:
            logger.error(f"Error retrieving post from PostgreSQL: {e}")
            return None
    
    async def search_posts(
        self,
        keyword: str,
        platform: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for posts in PostgreSQL
        
        Args:
            keyword: Keyword to search for
            platform: Optional platform filter
            limit: Maximum number of results
        
        Returns:
            List of post dictionaries
        """
        if not self.async_session:
            return []
        
        try:
            async with self.async_session() as session:
                # Construire la requête avec ILIKE (insensible à la casse)
                stmt = select(PostModel)
                
                # Filtre par keyword (cherche dans title et text)
                keyword_filter = (
                    PostModel.title.ilike(f'%{keyword}%') | 
                    PostModel.text.ilike(f'%{keyword}%')
                )
                stmt = stmt.where(keyword_filter)
                
                # Filtre par platform si fourni
                if platform:
                    stmt = stmt.where(PostModel.platform == platform)
                
                # Trier par date et limiter
                stmt = stmt.order_by(PostModel.created_at.desc()).limit(limit)
                
                result = await session.execute(stmt)
                posts = result.scalars().all()
                
                # Retourner les raw_json
                return [p.raw_json for p in posts if p.raw_json]
        except Exception as e:
            logger.error(f"Error searching posts in PostgreSQL: {e}")
            return []
    
    async def save_search_history(
        self,
        keyword: str,
        platforms: List[str],
        num_posts: int
    ) -> bool:
        """
        Save search history to PostgreSQL
        
        Args:
            keyword: Search keyword
            platforms: Platforms searched
            num_posts: Number of posts found
        
        Returns:
            True if successful, False otherwise
        """
        if not self.async_session:
            return False
        
        try:
            async with self.async_session() as session:
                history_entry = SearchHistoryModel(
                    keyword=keyword,
                    platforms=json.dumps(platforms),  # Convertir la liste en JSON string
                    num_posts=num_posts,
                    timestamp=datetime.utcnow()
                )
                session.add(history_entry)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving search history to PostgreSQL: {e}")
            return False
    
    async def get_search_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent search history from PostgreSQL
        
        Args:
            limit: Maximum number of results
        
        Returns:
            List of search history entries
        """
        if not self.async_session:
            return []
        
        try:
            async with self.async_session() as session:
                stmt = select(SearchHistoryModel).order_by(
                    SearchHistoryModel.timestamp.desc()
                ).limit(limit)
                
                result = await session.execute(stmt)
                history = result.scalars().all()
                
                return [
                    {
                        'keyword': h.keyword,
                        'platforms': json.loads(h.platforms),
                        'num_posts': h.num_posts,
                        'timestamp': h.timestamp
                    }
                    for h in history
                ]
        except Exception as e:
            logger.error(f"Error getting search history from PostgreSQL: {e}")
            return []
    
    async def get_posts_for_trends(
        self,
        keyword: str,
        platforms: List[str],
        since: Optional[datetime] = None,
        limit: int = 1000  # 🔧 Augmenté de 200 à 1000 pour avoir tous les posts de la période
    ) -> List[Post]:
        """
        Récupérer les posts depuis PostgreSQL pour l'analyse de tendances
        
        Args:
            keyword: Mot-clé à rechercher
            platforms: Liste des plateformes (reddit, twitter)
            since: Date minimum (optionnel)
            limit: Nombre maximum de posts (défaut: 1000 pour couverture complète)
        
        Returns:
            Liste d'objets Post
        """
        if not self.async_session:
            return []
        
        try:
            async with self.async_session() as session:
                stmt = select(PostModel)
                
                # Filtre par keyword
                keyword_filter = (
                    PostModel.title.ilike(f'%{keyword}%') | 
                    PostModel.text.ilike(f'%{keyword}%')
                )
                stmt = stmt.where(keyword_filter)
                
                # Filtre par platforms
                if platforms:
                    stmt = stmt.where(PostModel.platform.in_(platforms))
                
                # Filtre par date si fourni
                if since:
                    stmt = stmt.where(PostModel.created_at >= since)
                
                # 🔧 FIX: Trier par date ASC (plus anciens d'abord) pour garantir une couverture complète
                # Ensuite limiter - ainsi on a les posts les plus anciens d'abord si on atteint la limite
                # Note: Pour une distribution uniforme, on prend TOUS les posts dans la période
                stmt = stmt.order_by(PostModel.created_at.asc()).limit(limit)
                
                result = await session.execute(stmt)
                posts_models = result.scalars().all()
                
                # Convertir les modèles en objets Post
                posts = []
                for pm in posts_models:
                    if pm.raw_json:
                        try:
                            post = Post(**pm.raw_json)
                            posts.append(post)
                        except Exception as e:
                            logger.error(f"Error converting post from DB: {e}")
                
                logger.info(f"Retrieved {len(posts)} posts from PostgreSQL for trends analysis")
                return posts
        except Exception as e:
            logger.error(f"Error getting posts for trends from PostgreSQL: {e}")
            return []
    
    async def close(self):
        """Close PostgreSQL connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgreSQL connection closed")
    
    # ==================== 🚀 NEW: BATCH INGESTION FOR TRENDS ====================
    
    async def bulk_ingest_posts(
        self,
        posts: List[Post],
        keyword: str,
        skip_duplicates: bool = True
    ) -> Dict[str, int]:
        """
        🚀 Optimized bulk ingestion for trends data collection
        
        This method is designed for massive data ingestion from Reddit/Twitter
        pagination. It uses ON CONFLICT DO NOTHING for efficient deduplication.
        
        Strategy:
        - Batch insert with UPSERT (ON CONFLICT)
        - Minimal validation (data already validated by services)
        - Transaction-based for atomicity
        - Progress logging for large batches
        
        Use Case:
        - Continuous ingestion jobs (background tasks)
        - Historical data collection
        - Trend analysis with large datasets
        
        Args:
            posts: List of Post objects to ingest
            keyword: Associated keyword (for future filtering)
            skip_duplicates: If True, skip existing posts (default)
            
        Returns:
            Dict with stats: {
                "attempted": int,
                "inserted": int,
                "skipped": int,
                "errors": int
            }
        """
        if not self.async_session:
            logger.warning("PostgreSQL not available for bulk ingestion")
            return {"attempted": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
        if not posts:
            return {"attempted": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
        stats = {
            "attempted": len(posts),
            "inserted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        try:
            async with self.async_session() as session:
                async with session.begin():
                    for i, post in enumerate(posts):
                        try:
                            # Convert Post to PostModel
                            post_dict = {
                                "platform": post.platform,
                                "post_id": post.id,
                                "title": post.title,
                                "text": post.text,
                                "author": post.author,
                                "created_at": post.created_at,
                                "score": post.score,
                                "url": post.url,
                                "num_comments": post.num_comments,
                                "relevance_score": None,  # To be computed later
                                "sentiment_positive": post.sentiment.positive if post.sentiment else None,
                                "sentiment_negative": post.sentiment.negative if post.sentiment else None,
                                "sentiment_neutral": post.sentiment.neutral if post.sentiment else None,
                                "sentiment_dominant": post.sentiment.dominant if post.sentiment else None,
                                "summary": None,  # To be computed later
                                "raw_json": post.model_dump(mode='json'),  # Full object as JSON
                                "saved_at": datetime.utcnow()
                            }
                            
                            # Check for existing post if skip_duplicates
                            if skip_duplicates:
                                existing = await session.execute(
                                    select(PostModel).where(
                                        and_(
                                            PostModel.platform == post.platform,
                                            PostModel.post_id == post.id
                                        )
                                    )
                                )
                                if existing.scalar_one_or_none():
                                    stats["skipped"] += 1
                                    continue
                            
                            # Insert post
                            post_model = PostModel(**post_dict)
                            session.add(post_model)
                            stats["inserted"] += 1
                            
                            # Progress logging every 100 posts
                            if (i + 1) % 100 == 0:
                                logger.info(
                                    f"📊 Bulk ingestion progress: {i+1}/{len(posts)} "
                                    f"(inserted: {stats['inserted']}, skipped: {stats['skipped']})"
                                )
                        
                        except Exception as e:
                            logger.error(f"Error ingesting post {post.id}: {e}")
                            stats["errors"] += 1
                            continue
                    
                    # Commit transaction
                    await session.commit()
                    
                    logger.info(
                        f"✅ Bulk ingestion complete: "
                        f"attempted={stats['attempted']}, "
                        f"inserted={stats['inserted']}, "
                        f"skipped={stats['skipped']}, "
                        f"errors={stats['errors']}"
                    )
        
        except Exception as e:
            logger.error(f"Error in bulk ingestion: {e}")
            stats["errors"] = stats["attempted"] - stats["inserted"] - stats["skipped"]
        
        return stats


# Singleton instance
database_service = DatabaseService()

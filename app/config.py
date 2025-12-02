from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Reddit API Configuration
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "SocialMediaAnalyzer/1.0"
    
    # Twitter API Configuration
    twitter_bearer_token: str = ""
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_token_secret: str = ""
    
    # PostgreSQL Configuration (remplace MongoDB)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/social_media_analyzer"
    
    # Redis Configuration (optionnel)
    redis_url: str = "redis://localhost:6379"
    
    # Application Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Model Configuration
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    summarization_model: str = "facebook/bart-large-cnn"
    
    # 🧠 NEW: Opinion Detection Configuration (Deep Learning Model)
    opinion_detector_threshold: float = 0.5  # Seuil de décision (0-1) pour is_opinion()
    opinion_model_path: str = "app/models/opinion_detector_model.h5"  # Chemin du modèle Keras
    opinion_tokenizer_path: str = "app/models/opinion_tokenizer.pkl"  # Chemin du tokenizer
    
    # DEPRECATED: Old Review Filtering Configuration (replaced by OpinionDetector)
    # Ces paramètres ne sont plus utilisés et seront supprimés dans une version future
    enable_review_classifier: bool = False  # OBSOLÈTE: Remplacé par OpinionDetector
    review_classifier_model: str = "facebook/bart-large-mnli"  # OBSOLÈTE
    review_classifier_min_score: float = 0.6  # OBSOLÈTE
    review_overfetch_factor: int = 5  # Toujours utilisé pour over-fetch avant filtrage OpinionDetector
    
    # Aspect Extraction Configuration (Quality Improvements)
    enable_keyphrase_model: bool = False  # Utiliser un modèle de keyphrase extraction (optionnel)
    keyphrase_model: str = "ml6team/keyphrase-extraction-distilbert-inspec"
    min_aspect_mentions: int = 3  # Nombre minimum de mentions pour un aspect
    min_aspect_sentiment_variance: float = 0.1  # Variance minimale de sentiments pour un aspect
    
    # Performance & Limits Configuration
    max_posts_per_request: int = 30  # 🚀 Réduit de 50 à 30 pour optimiser temps de réponse Reddit
    max_comments_per_post: int = 10  # ✅ Réduit de 20 à 10 pour optimiser performance (V3 perf fix)
    max_trends_posts: int = 200  # Limite du nombre de posts pour l'analyse de tendances (OLD: obsolète)
    enable_trends_in_search: bool = False  # Désactiver les tendances dans /api/search (séparation)
    enable_comment_summarization: bool = True  # Activer/désactiver les résumés de commentaires
    
    # 🚀 NEW: Pagination Massive pour Tendances (Fetch exhaustif)
    max_trends_fetch_per_platform: int = 1500  # 🔧 Augmenté de 1000 à 1500 pour meilleure couverture temporelle
    reddit_pagination_batch_size: int = 100  # Taille de lot par page Reddit (max API = 100)
    twitter_pagination_batch_size: int = 100  # Taille de lot par page Twitter (max API = 100)
    enable_trends_pagination: bool = True  # Activer/désactiver la pagination massive
    trends_max_api_calls: int = 15  # 🔧 Augmenté de 10 à 15 pour multi-sort strategy
    
    # 🤖 LLM Configuration (OpenRouter API for Grok and others)
    openrouter_api_key: str = ""  # API key from openrouter.ai
    openrouter_model: str = "anthropic/claude-3-sonnet"  # Model ID (e.g., grok-model-id)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

"""
🚀 INTELLIGENT TRENDS SERVICE - Version 2.0
================================================

Refonte complète du système de détection de tendances avec :
- Analyse centrée sur le mot-clé recherché
- Calcul du taux de croissance et détection de pics
- Évolution du sentiment dans le temps
- Intelligence analytique et comparative

Auteur: Assistant AI
Version: 2.0 - Intelligent Trend Detection
Date: 14 Octobre 2025
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging
import re
import statistics
from app.schemas.responses import (
    TrendData,
    TrendPoint,
    PopularTopic,
    SentimentAnalysis,
    Post
)
from app.services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)


class IntelligentTrendsService:
    """
    🧠 Service intelligent pour la détection et l'analyse de tendances
    
    Fonctionnalités:
    - Analyse centrée sur le mot-clé spécifique
    - Calcul des taux de croissance et détection de pics
    - Évolution temporelle du sentiment
    - Normalisation et insights intelligents
    """
    
    def __init__(self):
        # Seuils pour la détection de pics et tendances
        self.SPIKE_THRESHOLD = 0.5  # 50% d'augmentation = pic
        self.TREND_THRESHOLD = 0.2  # 20% d'évolution = tendance significative
        self.MIN_POSTS_FOR_TREND = 3  # Minimum de posts pour calculer une tendance
        
        # Configuration des buckets temporels
        self.MAX_BUCKETS = 24  # Maximum 24 points sur le graphique
        self.MIN_BUCKET_HOURS = 1  # Minimum 1 heure par bucket
        
        logger.info("🚀 Intelligent Trends Service initialized")
    
    def parse_time_range(self, time_range: str) -> timedelta:
        """
        Parse time range string to timedelta
        
        Args:
            time_range: Time range string (e.g., '24h', '7d', '30d')
        
        Returns:
            timedelta object
        """
        time_map = {
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
        }
        return time_map.get(time_range, timedelta(days=7))
    
    def filter_posts_by_keyword(
        self, 
        posts: List[Post], 
        keyword: str, 
        case_sensitive: bool = False
    ) -> List[Post]:
        """
        🎯 Filtre les posts contenant le mot-clé spécifique
        
        Args:
            posts: Liste des posts à filtrer
            keyword: Mot-clé à rechercher
            case_sensitive: Recherche sensible à la casse
        
        Returns:
            Liste des posts contenant le mot-clé
        """
        if not keyword:
            return posts
        
        search_keyword = keyword if case_sensitive else keyword.lower()
        filtered_posts = []
        
        for post in posts:
            # Construire le texte complet du post
            full_text = ""
            if post.title:
                full_text += post.title + " "
            full_text += post.text
            
            # Ajouter les commentaires si présents
            for comment in post.comments:
                full_text += " " + comment.text
            
            # Recherche du mot-clé
            search_text = full_text if case_sensitive else full_text.lower()
            
            if search_keyword in search_text:
                filtered_posts.append(post)
                logger.debug(f"✅ Post {post.id[:8]}... contains keyword '{keyword}'")
            else:
                logger.debug(f"❌ Post {post.id[:8]}... does not contain keyword '{keyword}'")
        
        logger.info(f"🎯 Filtered {len(filtered_posts)}/{len(posts)} posts for keyword '{keyword}'")
        return filtered_posts
    
    def create_time_buckets(
        self, 
        posts: List[Post], 
        time_range: str
    ) -> Tuple[Dict[datetime, List[Post]], timedelta]:
        """
        📊 Crée des buckets temporels intelligents
        
        Args:
            posts: Liste des posts
            time_range: Plage temporelle
        
        Returns:
            Tuple (buckets temporels, taille d'un bucket)
        """
        if not posts:
            return {}, timedelta(hours=1)
        
        # Tri des posts par date
        posts.sort(key=lambda p: p.created_at)
        
        oldest_post = posts[0].created_at
        newest_post = posts[-1].created_at
        total_duration = newest_post - oldest_post
        
        # Calcul intelligent de la taille des buckets
        if total_duration.total_seconds() == 0:
            # Tous les posts à la même date
            bucket_size = timedelta(hours=1)
            num_buckets = 1
        else:
            # Calculs adaptatifs basés sur la densité
            delta = self.parse_time_range(time_range)
            
            if delta.days >= 30:  # 30 jours -> buckets journaliers
                bucket_size = timedelta(days=1)
            elif delta.days >= 7:  # 7 jours -> buckets de 6h
                bucket_size = timedelta(hours=6)
            elif delta.days >= 1:  # 1 jour -> buckets horaires
                bucket_size = timedelta(hours=1)
            else:  # < 1 jour -> buckets de 30min
                bucket_size = timedelta(minutes=30)
            
            # Assurer un maximum de 24 buckets
            num_buckets = min(self.MAX_BUCKETS, int(total_duration / bucket_size) + 1)
            if num_buckets > self.MAX_BUCKETS:
                bucket_size = total_duration / self.MAX_BUCKETS
        
        logger.info(f"📊 Creating {num_buckets} time buckets of {bucket_size}")
        
        # Groupement des posts dans les buckets
        time_buckets = defaultdict(list)
        
        for post in posts:
            bucket_index = int((post.created_at - oldest_post) / bucket_size)
            bucket_time = oldest_post + bucket_size * bucket_index
            time_buckets[bucket_time].append(post)
        
        # Assurer la continuité temporelle (buckets vides si nécessaire)
        current_time = oldest_post
        while current_time <= newest_post:
            if current_time not in time_buckets:
                time_buckets[current_time] = []
            current_time += bucket_size
        
        return dict(time_buckets), bucket_size
    
    def calculate_growth_rate(self, current: int, previous: int) -> float:
        """
        📈 Calcule le taux de croissance entre deux périodes
        
        Args:
            current: Nombre de mentions actuel
            previous: Nombre de mentions précédent
        
        Returns:
            Taux de croissance (-1.0 à +∞)
        """
        if previous == 0:
            return 1.0 if current > 0 else 0.0
        
        growth_rate = (current - previous) / previous
        return round(growth_rate, 3)
    
    def detect_spike(self, growth_rate: float) -> bool:
        """
        🔥 Détecte si un point représente un pic significatif
        
        Args:
            growth_rate: Taux de croissance
        
        Returns:
            True si c'est un pic
        """
        return growth_rate >= self.SPIKE_THRESHOLD
    
    def calculate_sentiment_evolution(self, sentiments: List[SentimentAnalysis]) -> str:
        """
        📊 Analyse l'évolution du sentiment dans le temps
        
        Args:
            sentiments: Liste des sentiments par période
        
        Returns:
            Direction de l'évolution ('improving', 'declining', 'stable')
        """
        if len(sentiments) < 2:
            return 'stable'
        
        # Convertir les sentiments en scores numériques
        scores = []
        for sentiment in sentiments:
            # Score: -1 (très négatif) à +1 (très positif)
            score = sentiment.positive - sentiment.negative
            scores.append(score)
        
        # Calcul de la tendance générale
        if len(scores) < 3:
            change = scores[-1] - scores[0]
        else:
            # Régression linéaire simple
            n = len(scores)
            x_mean = (n - 1) / 2
            y_mean = sum(scores) / n
            
            numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            change = slope * (n - 1)  # Changement total
        
        if change > 0.1:
            return 'improving'
        elif change < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def calculate_trend_direction(self, mentions: List[int]) -> str:
        """
        📈 Calcule la direction générale de la tendance
        
        Args:
            mentions: Liste des mentions par période
        
        Returns:
            Direction ('rising', 'falling', 'stable')
        """
        if len(mentions) < 2:
            return 'stable'
        
        # Calcul de la tendance générale
        total_change = mentions[-1] - mentions[0]
        
        if len(mentions) >= 3:
            # Calcul plus sophistiqué avec régression
            n = len(mentions)
            x_mean = (n - 1) / 2
            y_mean = sum(mentions) / n
            
            numerator = sum((i - x_mean) * (mentions[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            total_change = slope * (n - 1)
        
        avg_mentions = sum(mentions) / len(mentions)
        relative_change = total_change / max(avg_mentions, 1)
        
        if relative_change > self.TREND_THRESHOLD:
            return 'rising'
        elif relative_change < -self.TREND_THRESHOLD:
            return 'falling'
        else:
            return 'stable'
    
    async def analyze_trends(
        self,
        posts: List[Post],
        keyword: str,
        time_range: str = "7d"
    ) -> List[TrendData]:
        """
        🚀 ANALYSE INTELLIGENTE DES TENDANCES
        
        Analyse centrée sur le mot-clé avec détection de pics et évolution
        
        Args:
            posts: Liste des posts (déjà filtrés par pertinence)
            keyword: Mot-clé spécifique à analyser
            time_range: Plage temporelle
        
        Returns:
            Liste des TrendData intelligents par plateforme
        """
        if not posts:
            logger.warning(f"No posts provided for trend analysis of '{keyword}'")
            return []
        
        logger.info(f"🚀 Starting intelligent trend analysis for '{keyword}' over {time_range}")
        
        # Étape 1: Filtrer les posts par mot-clé
        keyword_posts = self.filter_posts_by_keyword(posts, keyword)
        
        if not keyword_posts:
            logger.warning(f"No posts found containing keyword '{keyword}'")
            return []
        
        # Étape 2: Grouper par plateforme
        posts_by_platform = defaultdict(list)
        for post in keyword_posts:
            posts_by_platform[post.platform].append(post)
        
        trend_data_list = []
        
        # Étape 3: Analyser chaque plateforme
        for platform, platform_posts in posts_by_platform.items():
            logger.info(f"📊 Analyzing {len(platform_posts)} posts for {platform}")
            
            if len(platform_posts) < self.MIN_POSTS_FOR_TREND:
                logger.warning(f"Not enough posts for {platform} ({len(platform_posts)} < {self.MIN_POSTS_FOR_TREND})")
                continue
            
            # Étape 4: Créer les buckets temporels
            time_buckets, bucket_size = self.create_time_buckets(platform_posts, time_range)
            
            # Étape 5: Construire les points de tendance
            trend_points = []
            mentions_history = []
            sentiment_history = []
            
            for bucket_time in sorted(time_buckets.keys()):
                bucket_posts = time_buckets[bucket_time]
                mentions_count = len(bucket_posts)
                mentions_history.append(mentions_count)
                
                # Calcul du sentiment moyen pour cette période
                if bucket_posts:
                    all_sentiments = []
                    for post in bucket_posts:
                        if post.sentiment:
                            all_sentiments.append(post.sentiment)
                        # Inclure les commentaires
                        for comment in post.comments:
                            if comment.sentiment:
                                all_sentiments.append(comment.sentiment)
                    
                    if all_sentiments:
                        avg_sentiment = sentiment_service.calculate_overall_sentiment(all_sentiments)
                    else:
                        avg_sentiment = SentimentAnalysis(
                            positive=0.33, negative=0.33, neutral=0.34, dominant="neutral"
                        )
                else:
                    # Bucket vide
                    avg_sentiment = SentimentAnalysis(
                        positive=0.33, negative=0.33, neutral=0.34, dominant="neutral"
                    )
                
                sentiment_history.append(avg_sentiment)
                
                # Calcul du taux de croissance
                growth_rate = None
                is_spike = False
                if len(mentions_history) > 1:
                    growth_rate = self.calculate_growth_rate(
                        mentions_count, 
                        mentions_history[-2]
                    )
                    is_spike = self.detect_spike(growth_rate)
                
                # Créer le point de tendance
                trend_point = TrendPoint(
                    timestamp=bucket_time,
                    mentions=mentions_count,
                    sentiment=avg_sentiment,
                    growth_rate=growth_rate,
                    is_spike=is_spike
                )
                trend_points.append(trend_point)
                
                logger.debug(
                    f"📍 {bucket_time.strftime('%Y-%m-%d %H:%M')}: "
                    f"{mentions_count} mentions, growth: {growth_rate}, "
                    f"spike: {is_spike}, sentiment: {avg_sentiment.dominant}"
                )
            
            # Étape 6: Calculs globaux pour la plateforme
            total_mentions = len(platform_posts)
            peak_mentions = max(mentions_history) if mentions_history else 0
            
            # Sentiment moyen global
            all_platform_sentiments = [
                post.sentiment for post in platform_posts 
                if post.sentiment is not None
            ]
            
            if all_platform_sentiments:
                avg_platform_sentiment = sentiment_service.calculate_overall_sentiment(
                    all_platform_sentiments
                )
            else:
                avg_platform_sentiment = SentimentAnalysis(
                    positive=0.33, negative=0.33, neutral=0.34, dominant="neutral"
                )
            
            # Taux de croissance global
            overall_growth_rate = None
            if len(mentions_history) >= 2:
                overall_growth_rate = self.calculate_growth_rate(
                    sum(mentions_history[-3:]) if len(mentions_history) >= 3 else mentions_history[-1],
                    sum(mentions_history[:3]) if len(mentions_history) >= 3 else mentions_history[0]
                )
            
            # Direction de la tendance
            trend_direction = self.calculate_trend_direction(mentions_history)
            
            # Évolution du sentiment
            sentiment_evolution = self.calculate_sentiment_evolution(sentiment_history)
            
            # Créer le TrendData enrichi
            trend_data = TrendData(
                platform=platform,
                keyword=keyword,
                data_points=trend_points,
                total_mentions=total_mentions,
                average_sentiment=avg_platform_sentiment,
                overall_growth_rate=overall_growth_rate,
                peak_mentions=peak_mentions,
                trend_direction=trend_direction,
                sentiment_evolution=sentiment_evolution
            )
            
            trend_data_list.append(trend_data)
            
            logger.info(
                f"✅ {platform} trend analysis complete: "
                f"{total_mentions} mentions, {trend_direction} trend, "
                f"sentiment {sentiment_evolution}, peak: {peak_mentions}"
            )
        
        logger.info(f"🎯 Intelligent trend analysis complete for '{keyword}': {len(trend_data_list)} platforms")
        return trend_data_list
    
    async def extract_popular_topics(
        self,
        posts: List[Post],
        keyword: str,
        top_k: int = 10
    ) -> List[PopularTopic]:
        """
        🏷️ Extrait les sujets populaires liés au mot-clé principal
        
        Différent de l'ancien système : se concentre sur les termes
        co-occurents avec le mot-clé principal
        
        Args:
            posts: Liste des posts
            keyword: Mot-clé principal
            top_k: Nombre de sujets à retourner
        
        Returns:
            Liste des PopularTopic liés au mot-clé
        """
        if not posts:
            return []
        
        logger.info(f"🏷️ Extracting popular topics related to '{keyword}'")
        
        # Filtrer les posts contenant le mot-clé
        keyword_posts = self.filter_posts_by_keyword(posts, keyword)
        
        if not keyword_posts:
            return []
        
        # Extraire les termes co-occurents
        cooccurrent_terms = Counter()
        
        # Mots-clés à ignorer (stop words + mot-clé principal)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'them', 'their', 'my', 'your', 'his', 'her',
            'its', 'our', 'with', 'from', 'not', 'just', 'get', 'so', 'like', 'if',
            keyword.lower()  # Exclure le mot-clé principal
        }
        
        for post in keyword_posts:
            # Extraire le texte complet
            full_text = (post.title or "") + " " + post.text
            for comment in post.comments:
                full_text += " " + comment.text
            
            # Nettoyer et extraire les mots
            clean_text = re.sub(r'[^\w\s]', ' ', full_text.lower())
            words = clean_text.split()
            
            # Filtrer et compter
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    cooccurrent_terms[word] += 1
        
        # Créer les PopularTopic
        popular_topics = []
        
        for term, frequency in cooccurrent_terms.most_common(top_k * 2):  # Prendre plus pour filtrer
            if frequency < 2:  # Minimum 2 mentions
                continue
            
            # Collecter les posts contenant ce terme
            term_posts = []
            for post in keyword_posts:
                full_text = (post.title or "") + " " + post.text
                if term in full_text.lower():
                    term_posts.append(post)
            
            if not term_posts:
                continue
            
            # Calculer le sentiment moyen pour ce terme
            term_sentiments = [
                post.sentiment for post in term_posts 
                if post.sentiment is not None
            ]
            
            if term_sentiments:
                avg_sentiment = sentiment_service.calculate_overall_sentiment(term_sentiments)
            else:
                avg_sentiment = SentimentAnalysis(
                    positive=0.33, negative=0.33, neutral=0.34, dominant="neutral"
                )
            
            popular_topics.append(PopularTopic(
                keyword=term,
                frequency=frequency,
                sentiment=avg_sentiment
            ))
            
            if len(popular_topics) >= top_k:
                break
        
        logger.info(f"🏷️ Found {len(popular_topics)} popular topics for '{keyword}'")
        return popular_topics
    
    async def detect_trending_keywords(
        self,
        posts: List[Post],
        keyword: str,
        min_mentions: int = 3
    ) -> Dict[str, int]:
        """
        🔍 Détecte les mots-clés en tendance liés au terme principal
        
        Args:
            posts: Liste des posts
            keyword: Mot-clé principal
            min_mentions: Minimum de mentions
        
        Returns:
            Dictionnaire terme -> fréquence
        """
        logger.info(f"🔍 Detecting trending keywords for '{keyword}'")
        
        # Utiliser la logique des popular topics
        popular_topics = await self.extract_popular_topics(posts, keyword, top_k=20)
        
        trending_keywords = {}
        for topic in popular_topics:
            if topic.frequency >= min_mentions:
                trending_keywords[topic.keyword] = topic.frequency
        
        logger.info(f"🔍 Found {len(trending_keywords)} trending keywords")
        return trending_keywords


# Singleton instance
intelligent_trends_service = IntelligentTrendsService()

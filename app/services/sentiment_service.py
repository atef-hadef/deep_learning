import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Dict, Tuple, Optional
import logging
import re
from collections import Counter, defaultdict

# Optional imports
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("langdetect not available, language detection will be skipped")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available, some features may be limited")

from app.config import get_settings
from app.schemas.responses import SentimentAnalysis, AspectSentiment

logger = logging.getLogger(__name__)

# Initialize spaCy model (lazy loading)
nlp = None

def get_nlp():
    """Lazy load spaCy model"""
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. Using fallback extraction.")
            nlp = False
    return nlp


class SentimentService:
    """Service for sentiment analysis and text summarization using Deep Learning models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.summarization_pipeline = None

        # ⚠️ IMPORTANT : forcer l'API à utiliser le CPU uniquement
        # Même si un GPU est dispo (MX350 avec 2 Go), il n'est pas adapté
        # pour charger RoBERTa + BART en même temps.
        self.device = "cpu"

        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Deep Learning models"""
        try:
            logger.info(f"Loading models on device: {self.device}")
            
            # Load sentiment analysis model (RoBERTa fine-tuned on Twitter data)
            logger.info("Loading sentiment analysis model...")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                self.settings.sentiment_model
            )
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.sentiment_model
            ).to(self.device)
            self.sentiment_model.eval()
            
            # Load summarization model (BART)
            # TODO: Réactiver le résumé des posts plus tard
            # logger.info("Loading summarization model...")
            # self.summarization_pipeline = pipeline(
            #     "summarization",
            #     model=self.settings.summarization_model,
            #     device=0 if self.device == "cuda" else -1,
            #     max_length=130,
            #     min_length=30,
            #     do_sample=False
            # )
            logger.info("⚠️ Summarization disabled for V1 simplification")
            self.summarization_pipeline = None
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            logger.warning("Models will not be available for analysis")
    
    def is_available(self) -> bool:
        """Check if sentiment service is available"""
        # ✅ CORRECTION: Vérifier seulement sentiment_model (summarization désactivé pour V1)
        return self.sentiment_model is not None
    
    def is_english(self, text: str) -> bool:
        """
        Check if text is in English
        
        Args:
            text: Text to check
        
        Returns:
            True if text is in English, False otherwise
        """
        if not text or len(text.strip()) < 10:
            return False
        
        if not LANGDETECT_AVAILABLE:
            # Fallback: assume English if langdetect not available
            return True
        
        try:
            lang = detect(text)
            return lang == 'en'
        except LangDetectException:
            # If detection fails, use simple heuristic
            # Count common English words
            english_words = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 
                           'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this', 
                           'that', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                           'would', 'should', 'can', 'could', 'may', 'might', 'i', 'you'}
            words = text.lower().split()
            if len(words) < 3:
                return False
            english_count = sum(1 for word in words if word in english_words)
            return english_count >= 2  # At least 2 common English words
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags symbols (but keep the text)
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of a text using RoBERTa model
        
        Args:
            text: Text to analyze
        
        Returns:
            SentimentAnalysis object with scores
        """
        if not self.is_available():
            logger.warning("Sentiment model not available, returning neutral sentiment")
            return SentimentAnalysis(
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                dominant="neutral"
            )
        
        try:
            # Preprocess
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text:
                return SentimentAnalysis(
                    positive=0.33,
                    negative=0.33,
                    neutral=0.34,
                    dominant="neutral"
                )
            
            # Tokenize
            inputs = self.sentiment_tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to probabilities
            scores = scores[0].cpu().numpy()
            
            # Model outputs: negative, neutral, positive
            sentiment = SentimentAnalysis(
                negative=float(scores[0]),
                neutral=float(scores[1]),
                positive=float(scores[2]),
                dominant="negative" if scores[0] > max(scores[1], scores[2]) else 
                         "neutral" if scores[1] > scores[2] else "positive"
            )
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentAnalysis(
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                dominant="neutral"
            )
    
    async def analyze_batch_sentiments(self, texts: List[str]) -> List[SentimentAnalysis]:
        """
        Analyze sentiments for multiple texts (batch processing for efficiency)
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of SentimentAnalysis objects
        """
        if not texts:
            return []
        
        results = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                sentiment = await self.analyze_sentiment(text)
                results.append(sentiment)
        
        return results
    
    async def summarize_text(self, text: str, max_length: int = 130) -> str:
        """
        Generate a summary of the text using BART model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
        
        Returns:
            Summary text
        """
        if not self.is_available():
            logger.warning("Summarization model not available")
            return text[:200] + "..." if len(text) > 200 else text
        
        try:
            # Preprocess
            cleaned_text = self.preprocess_text(text)
            
            # Minimum text length for summarization
            if len(cleaned_text.split()) < 30:
                return cleaned_text
            
            # Generate summary
            summary = self.summarization_pipeline(
                cleaned_text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return summary[0]["summary_text"]
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    # TODO: Réactiver summarize_comments plus tard
    # async def summarize_comments(self, comments: List[str]) -> str:
    async def summarize_comments_DISABLED(self, comments: List[str]) -> str:
        """
        Generate a summary from multiple comments (English only)
        
        Args:
            comments: List of comment texts
        
        Returns:
            Summary of all comments in English
        """
        if not comments:
            return "No English comments available"
        
        # Filter for English comments only
        english_comments = [c for c in comments if self.is_english(c)]
        
        if not english_comments:
            logger.warning("No English comments found for summarization")
            return "No English comments available for summary"
        
        logger.info(f"Filtered {len(english_comments)}/{len(comments)} English comments")
        
        # Filter out very short comments (less than 10 words)
        meaningful_comments = [c for c in english_comments if len(c.split()) >= 10]
        
        if not meaningful_comments:
            meaningful_comments = english_comments  # Fallback to all English comments
        
        # Combine comments with separators for better context
        combined_text = " [COMMENT] ".join(meaningful_comments)
        
        # Limit combined text length (BART works better with moderate input)
        if len(combined_text) > 3000:
            # Take most relevant comments (sorted by length as proxy for detail)
            meaningful_comments.sort(key=lambda x: len(x), reverse=True)
            meaningful_comments = meaningful_comments[:10]  # Top 10 detailed comments
            combined_text = " [COMMENT] ".join(meaningful_comments)
        
        # Clean up the separator before summarization
        combined_text = combined_text.replace(" [COMMENT] ", ". ")
        
        return await self.summarize_text(combined_text, max_length=150)
    
    def _extract_keywords_from_comments(self, comments: List[str], top_k: int = 10) -> List[str]:
        """
        Extract meaningful keywords using NLP techniques (noun phrases, TF-IDF)
        
        ✨ QUALITY IMPROVEMENTS:
        - Blacklist étendue (mots conversationnels)
        - Support keyphrase model optionnel
        - Filtrage plus strict
        
        Args:
            comments: List of comment texts
            top_k: Number of top keywords to extract
        
        Returns:
            List of meaningful product-specific keywords
        """
        if not comments:
            return []
        
        nlp_model = get_nlp()
        
        # ✅ BLACKLIST ÉTENDUE: Mots conversationnels à exclure
        generic_words = {
            # Emotional words (not product features)
            'bad', 'good', 'great', 'terrible', 'awful', 'horrible', 'poor', 'excellent',
            'amazing', 'wonderful', 'fantastic', 'nice', 'best', 'worst', 'better', 'worse',
            # Generic descriptors
            'issue', 'problem', 'thing', 'stuff', 'way', 'time', 'day', 'year', 'people',
            'person', 'lot', 'bit', 'kind', 'sort', 'type', 'pretty', 'really', 'very',
            'quite', 'much', 'many', 'little', 'big', 'small', 'new', 'old', 'first',
            'last', 'next', 'previous', 'second', 'third', 'high', 'low', 'long', 'short',
            # ✨ NEW: Conversational noise
            'thanks', 'thank', 'edit', 'shot', 'update', 'area', 'today', 'yesterday',
            'everyone', 'guys', 'someone', 'anyone', 'people', 'thread', 'post',
            'reddit', 'subreddit', 'op', 'comment', 'comments', 'lol', 'haha', 'hehe',
            'yeah', 'yep', 'nope', 'maybe', 'probably', 'definitely', 'honestly',
            'literally', 'basically', 'actually', 'seriously', 'personally',
            # Platform-specific noise
            'twitter', 'tweet', 'retweet', 'follow', 'follower', 'like', 'share',
            # Time references (not product features)
            'week', 'month', 'hour', 'minute', 'morning', 'evening', 'night'
        }
        
        # ✅ TRY KEYPHRASE MODEL if enabled (optional advanced feature)
        if self.settings.enable_keyphrase_model:
            keywords = self._extract_with_keyphrase_model(comments, generic_words, top_k)
            if keywords:
                logger.info(f"✨ Extracted {len(keywords)} keywords using keyphrase model: {keywords}")
                return keywords
        
        # Standard extraction methods
        if nlp_model and nlp_model is not False:
            # Advanced extraction with spaCy (noun phrases + nouns)
            keywords = self._extract_with_spacy(comments, nlp_model, generic_words, top_k)
        else:
            # Fallback: TF-IDF based extraction
            keywords = self._extract_with_tfidf(comments, generic_words, top_k)
        
        logger.info(f"Extracted {len(keywords)} meaningful keywords: {keywords}")
        
        return keywords
    
    def _extract_with_keyphrase_model(
        self,
        comments: List[str],
        generic_words: set,
        top_k: int
    ) -> List[str]:
        """
        ✨ NEW: Extract keyphrases using HuggingFace keyphrase extraction model
        (Optional advanced feature, requires enable_keyphrase_model=True)
        
        Args:
            comments: List of comment texts
            generic_words: Set of generic words to exclude
            top_k: Number of top keyphrases to return
        
        Returns:
            List of extracted keyphrases
        """
        try:
            from transformers import pipeline as hf_pipeline
            
            # Load keyphrase extraction model
            keyphrase_extractor = hf_pipeline(
                "ner",  # Named Entity Recognition can be repurposed for keyphrase extraction
                model=self.settings.keyphrase_model,
                device=-1  # CPU only
            )
            
            # Combine comments
            combined_text = " ".join(comments[:50])  # Limit for performance
            
            # Extract keyphrases
            results = keyphrase_extractor(combined_text[:1000])
            
            # Filter and rank keyphrases
            keyphrases = Counter()
            for entity in results:
                phrase = entity['word'].strip().lower()
                # Filter generic words
                if phrase not in generic_words and len(phrase) >= 3:
                    keyphrases[phrase] += entity.get('score', 1.0)
            
            # Return top keyphrases
            top_phrases = [phrase for phrase, score in keyphrases.most_common(top_k)]
            
            logger.info(f"✨ Keyphrase model extracted: {top_phrases}")
            return top_phrases
            
        except Exception as e:
            logger.warning(f"⚠️ Keyphrase model extraction failed: {e}")
            return []
    
    def _extract_with_spacy(
        self, 
        comments: List[str], 
        nlp_model, 
        generic_words: set,
        top_k: int
    ) -> List[str]:
        """
        Extract keywords using spaCy (noun phrases + POS tagging)
        """
        noun_phrases = Counter()
        nouns = Counter()
        
        for comment in comments:
            try:
                doc = nlp_model(comment.lower())
                
                # Extract noun phrases (e.g., "response speed", "battery life")
                for chunk in doc.noun_chunks:
                    phrase = chunk.text.strip()
                    # Clean the phrase
                    phrase = re.sub(r'\b(the|a|an|my|your|this|that)\b', '', phrase).strip()
                    
                    # Filter conditions
                    if (len(phrase.split()) <= 3 and  # Max 3 words
                        len(phrase) >= 4 and  # Min 4 characters
                        not any(word in phrase for word in generic_words)):
                        noun_phrases[phrase] += 1
                
                # Extract individual nouns as backup
                for token in doc:
                    if token.pos_ in ('NOUN', 'PROPN') and len(token.text) >= 3:
                        if token.text not in generic_words:
                            nouns[token.lemma_] += 1
            
            except Exception as e:
                logger.error(f"Error processing comment with spaCy: {e}")
                continue
        
        # Prioritize noun phrases (more meaningful), then single nouns
        keywords = []
        
        # Add noun phrases with at least 2 mentions
        for phrase, count in noun_phrases.most_common(top_k):
            if count >= 2:
                keywords.append(phrase)
        
        # Fill remaining slots with high-frequency nouns
        if len(keywords) < top_k:
            for noun, count in nouns.most_common(top_k * 2):
                if count >= 2 and noun not in keywords and len(keywords) < top_k:
                    keywords.append(noun)
        
        return keywords[:top_k]
    
    def _extract_with_tfidf(
        self,
        comments: List[str],
        generic_words: set,
        top_k: int
    ) -> List[str]:
        """
        Fallback: Extract keywords using TF-IDF
        """
        try:
            # Stop words combining generic + standard English
            stop_words = generic_words.union({
                'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'should', 'could', 'can', 'may', 'might', 'a', 'an', 'and',
                'or', 'but', 'if', 'then', 'so', 'i', 'me', 'my', 'you', 'your',
                'he', 'she', 'it', 'we', 'they', 'them', 'this', 'that'
            })
            
            # Use TF-IDF with bigrams
            vectorizer = TfidfVectorizer(
                max_features=top_k * 3,
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words=list(stop_words),
                min_df=2,  # Must appear in at least 2 comments
                token_pattern=r'\b[a-z]{3,}\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(comments)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            # Get top keywords
            top_indices = avg_scores.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return keywords
        
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
            return []
    
    # TODO: Réactiver l'extraction de mots-clés plus tard
    # async def extract_aspects(
    #     self,
    #     comments: List[str],
    #     top_k: int = 5
    # ) -> List[AspectSentiment]:
    async def extract_aspects_DISABLED(
        self,
        comments: List[str],
        top_k: int = 5
    ) -> List[AspectSentiment]:
        """
        ✨ IMPROVED: Extract product aspects DYNAMICALLY with quality scoring
        
        NEW FEATURES:
        - Filtrage par longueur minimale (≥ 8-10 mots)
        - Fréquence minimale configurable (min_aspect_mentions)
        - Scoring par fréquence × variance de sentiments
        - Exclusion des aspects "neutres partout" (peu d'opinion)
        
        Args:
            comments: List of comment texts
            top_k: Number of top aspects to return
        
        Returns:
            List of AspectSentiment objects with high-quality aspects
        """
        if not comments:
            return []
        
        # ✅ STEP 1: Filter comments by length (meaningful comments only)
        meaningful_comments = []
        for comment in comments:
            # Detect language and keep only English
            if not self.is_english(comment):
                continue
            # Filter by length (≥ 8 words)
            if len(comment.split()) >= 8:
                meaningful_comments.append(comment)
        
        if not meaningful_comments:
            logger.warning("No meaningful English comments found for aspect extraction")
            return []
        
        logger.info(f"Filtered {len(meaningful_comments)}/{len(comments)} meaningful English comments")
        
        # ✅ STEP 2: Extract candidate keywords (large pool)
        keywords = self._extract_keywords_from_comments(meaningful_comments, top_k=20)
        
        if not keywords:
            logger.warning("No keywords extracted from comments")
            return []
        
        logger.info(f"Extracted {len(keywords)} candidate aspects: {keywords}")
        
        # ✅ STEP 3: Collect comments mentioning each aspect
        aspect_mentions = defaultdict(list)
        
        for comment in meaningful_comments:
            comment_lower = comment.lower()
            for keyword in keywords:
                # Check if keyword appears in comment (exact or contains)
                if keyword in comment_lower or any(word in comment_lower for word in keyword.split()):
                    aspect_mentions[keyword].append(comment)
        
        # ✅ STEP 4: Analyze sentiment + score each aspect
        aspect_candidates = []
        
        for aspect, comments_list in aspect_mentions.items():
            # Filter: minimum mentions (configurable)
            if len(comments_list) < self.settings.min_aspect_mentions:
                logger.debug(f"FILTER aspect '{aspect}': only {len(comments_list)} mentions (< {self.settings.min_aspect_mentions})")
                continue
            
            try:
                # Analyze sentiment for all comments mentioning this aspect
                sentiments = await self.analyze_batch_sentiments(comments_list)
                
                # Calculate sentiment statistics
                avg_positive = sum(s.positive for s in sentiments) / len(sentiments)
                avg_negative = sum(s.negative for s in sentiments) / len(sentiments)
                avg_neutral = sum(s.neutral for s in sentiments) / len(sentiments)
                
                # ✅ Calculate sentiment variance (polarization)
                # High variance = strong opinions (good!)
                # Low variance = mostly neutral (boring, filter out)
                sentiment_variance = abs(avg_positive - avg_negative)
                
                # Filter: minimum sentiment variance
                if sentiment_variance < self.settings.min_aspect_sentiment_variance:
                    logger.debug(f"FILTER aspect '{aspect}': low sentiment variance ({sentiment_variance:.2f} < {self.settings.min_aspect_sentiment_variance})")
                    continue
                
                # ✅ Calculate aspect quality score
                # Score = frequency × sentiment_polarization
                # Prioritizes aspects with:
                # - Many mentions (popular)
                # - Strong opinions (not neutral)
                aspect_score = len(comments_list) * (sentiment_variance + 0.1)
                
                dominant = "negative" if avg_negative > max(avg_positive, avg_neutral) else \
                          "neutral" if avg_neutral > avg_positive else "positive"
                
                aspect_candidates.append({
                    'aspect': aspect,
                    'sentiment': dominant,
                    'mentions': len(comments_list),
                    'score': aspect_score,
                    'avg_positive': avg_positive,
                    'avg_negative': avg_negative,
                    'avg_neutral': avg_neutral,
                    'sentiments': sentiments  # Keep for later use
                })
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for aspect '{aspect}': {e}")
                continue
        
        # ✅ STEP 5: Sort by quality score and return top aspects
        aspect_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for candidate in aspect_candidates[:top_k]:
            results.append(AspectSentiment(
                aspect=candidate['aspect'],
                sentiment=candidate['sentiment'],
                mentions=candidate['mentions']
            ))
        
        logger.info(
            f"✨ Extracted {len(results)} high-quality aspects: "
            f"{[(r.aspect, r.mentions) for r in results]}"
        )
        
        return results
    
    def calculate_overall_sentiment(
        self,
        sentiments: List[SentimentAnalysis]
    ) -> SentimentAnalysis:
        """
        Calculate overall sentiment from multiple sentiment analyses
        
        Args:
            sentiments: List of SentimentAnalysis objects
        
        Returns:
            Aggregated SentimentAnalysis
        """
        if not sentiments:
            return SentimentAnalysis(
                positive=0.33,
                negative=0.33,
                neutral=0.34,
                dominant="neutral"
            )
        
        avg_positive = sum(s.positive for s in sentiments) / len(sentiments)
        avg_negative = sum(s.negative for s in sentiments) / len(sentiments)
        avg_neutral = sum(s.neutral for s in sentiments) / len(sentiments)
        
        dominant = "negative" if avg_negative > max(avg_positive, avg_neutral) else \
                  "neutral" if avg_neutral > avg_positive else "positive"
        
        return SentimentAnalysis(
            positive=avg_positive,
            negative=avg_negative,
            neutral=avg_neutral,
            dominant=dominant
        )
    
    # TODO: Réactiver le build_aspect_summary plus tard
    # async def build_aspect_summary(
    #     self,
    #     aspect: str,
    #     sentiments: List[SentimentAnalysis],
    #     comments: List[str]
    # ) -> str:
    async def build_aspect_summary_DISABLED(
        self,
        aspect: str,
        sentiments: List[SentimentAnalysis],
        comments: List[str]
    ) -> str:
        """
        ✨ NEW: Build informative aspect summary based on statistics + typical phrases
        
        Instead of generic "positive feedback from users", generates:
        "Mostly positive (65% of comments). Users mention 'lasts all day' and 
        'better than 13 Pro' but a few complain about 'drains fast with games'."
        
        Args:
            aspect: Aspect name (e.g., "battery")
            sentiments: List of SentimentAnalysis for this aspect
            comments: List of comments mentioning this aspect
        
        Returns:
            Informative summary string
        """
        if not sentiments or not comments:
            return "Limited feedback available"
        
        # ✅ STEP 1: Calculate sentiment statistics
        avg_positive = sum(s.positive for s in sentiments) / len(sentiments)
        avg_negative = sum(s.negative for s in sentiments) / len(sentiments)
        avg_neutral = sum(s.neutral for s in sentiments) / len(sentiments)
        
        # Round percentages
        pct_positive = round(avg_positive * 100)
        pct_negative = round(avg_negative * 100)
        pct_neutral = round(avg_neutral * 100)
        
        # Dominant sentiment
        if pct_positive > max(pct_negative, pct_neutral):
            sentiment_label = "Mostly positive"
        elif pct_negative > pct_neutral:
            sentiment_label = "Mostly negative"
        else:
            sentiment_label = "Mixed opinions"
        
        # ✅ STEP 2: Extract typical phrases using TF-IDF / bigrams
        typical_phrases = self._extract_typical_phrases(comments, top_k=3)
        
        # ✅ STEP 3: Build summary
        summary_parts = [f"{sentiment_label} ({pct_positive}% positive)"]
        
        if typical_phrases:
            # Add typical phrases
            phrases_str = ", ".join([f"'{phrase}'" for phrase in typical_phrases])
            summary_parts.append(f"Users mention {phrases_str}")
        
        # Add negative feedback if significant
        if pct_negative >= 20:
            summary_parts.append(f"but {pct_negative}% report issues")
        
        final_summary = ". ".join(summary_parts) + "."
        
        return final_summary
    
    def _extract_typical_phrases(self, comments: List[str], top_k: int = 3) -> List[str]:
        """
        Extract typical/frequent phrases from comments using bigrams/trigrams
        
        Args:
            comments: List of comments
            top_k: Number of top phrases to return
        
        Returns:
            List of typical phrases
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Use bigrams and trigrams
            vectorizer = CountVectorizer(
                ngram_range=(2, 4),  # 2-4 word phrases
                max_features=20,
                stop_words='english',
                token_pattern=r'\b[a-z]{3,}\b'
            )
            
            # Combine comments
            combined = " ".join(comments[:20]).lower()  # Top 20 comments
            
            # Extract phrases
            try:
                vectorizer.fit([combined])
                phrases = vectorizer.get_feature_names_out()
                
                # Filter out generic phrases
                filtered_phrases = []
                for phrase in phrases:
                    # Skip if too generic
                    if not any(word in phrase for word in ['really', 'very', 'much', 'pretty', 'quite']):
                        filtered_phrases.append(phrase)
                
                return filtered_phrases[:top_k]
            except:
                # Fallback: extract manually
                return self._extract_phrases_manual(comments, top_k)
                
        except Exception as e:
            logger.error(f"Error extracting typical phrases: {e}")
            return []
    
    def _extract_phrases_manual(self, comments: List[str], top_k: int = 3) -> List[str]:
        """
        Manual phrase extraction fallback
        
        Args:
            comments: List of comments
            top_k: Number of phrases to return
        
        Returns:
            List of common phrases
        """
        # Combine and split into words
        words = " ".join(comments[:20]).lower().split()
        
        # Extract bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        # Count frequency
        bigram_counts = Counter(bigrams)
        
        # Filter and return top
        common_bigrams = []
        for bigram, count in bigram_counts.most_common(top_k * 2):
            if count >= 2 and len(bigram) >= 6:  # At least 2 occurrences, reasonable length
                common_bigrams.append(bigram)
                if len(common_bigrams) >= top_k:
                    break
        
        return common_bigrams
    
    # TODO: Réactiver generate_product_summary_by_aspects plus tard
    # async def generate_product_summary_by_aspects(
    #     self,
    #     posts: List,
    #     keyword: str
    # ) -> str:
    async def generate_product_summary_by_aspects_DISABLED(
        self,
        posts: List,
        keyword: str
    ) -> str:
        """
        ✨ IMPROVED: Generate structured product summary with informative aspect descriptions
        
        Uses build_aspect_summary() for rich, statistics-based summaries
        instead of generic templates.
        
        Args:
            posts: List of Post objects with comments
            keyword: Search keyword (product name)
        
        Returns:
            Structured summary by aspects with detailed insights
        """
        if not posts:
            return f"No information available about {keyword}"
        
        try:
            # Collect all English comments
            all_comments = []
            for post in posts:
                if post.comments:
                    english_comments = [c.text for c in post.comments if self.is_english(c.text)]
                    all_comments.extend(english_comments)
            
            if not all_comments:
                logger.warning(f"No English comments found for {keyword}")
                return f"No English comments available for {keyword}"
            
            logger.info(f"Analyzing {len(all_comments)} English comments for aspect-based summary")
            
            # Extract aspects and their sentiments
            aspects = await self.extract_aspects(all_comments, top_k=10)
            
            if not aspects:
                return f"Unable to identify key aspects for {keyword}"
            
            # ✅ Build informative summaries for each aspect
            summary_parts = []
            
            for aspect_obj in aspects[:4]:  # Limit to 4-5 aspects for readability
                # Collect comments mentioning this aspect
                aspect_comments = []
                aspect_sentiments = []
                
                for comment in all_comments:
                    if any(kw in comment.lower() for kw in self._get_aspect_keywords(aspect_obj.aspect)):
                        aspect_comments.append(comment)
                
                if not aspect_comments:
                    continue
                
                # Analyze sentiments for these comments
                aspect_sentiments = await self.analyze_batch_sentiments(aspect_comments)
                
                # ✨ Use new build_aspect_summary for rich insights
                aspect_summary = await self.build_aspect_summary(
                    aspect_obj.aspect,
                    aspect_sentiments,
                    aspect_comments
                )
                
                # Format: "Battery: Mostly positive (65% positive). Users mention 'lasts all day'..."
                summary_parts.append(f"**{aspect_obj.aspect.title()}**: {aspect_summary}")
            
            if not summary_parts:
                return f"Unable to generate aspect summary for {keyword}"
            
            # Combine all aspect summaries
            final_summary = " | ".join(summary_parts)
            
            logger.info(f"✨ Generated informative summary with {len(summary_parts)} aspects")
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Error generating aspect-based summary: {e}")
            return f"Unable to generate summary for {keyword}"
    
    def _get_aspect_keywords(self, aspect: str) -> List[str]:
        """
        Get keywords for a specific aspect (now dynamic, just returns the aspect itself)
        Since aspects are extracted dynamically from comments, we use the aspect as-is
        """
        # Return the aspect itself and common variations
        base_keyword = aspect.lower()
        keywords = [base_keyword]
        
        # Add plural/singular variations
        if base_keyword.endswith('s'):
            keywords.append(base_keyword[:-1])  # singular
        else:
            keywords.append(base_keyword + 's')  # plural
        
        # Add common variations
        if base_keyword.endswith('y'):
            keywords.append(base_keyword[:-1] + 'ies')  # e.g., battery -> batteries
        
        return keywords
    
    async def _generate_aspect_description(
        self,
        aspect: str,
        comments: List[str],
        sentiment: str
    ) -> str:
        """
        Generate a short description for an aspect based on comments
        
        Args:
            aspect: The aspect name (e.g., "camera")
            comments: Comments mentioning this aspect
            sentiment: Overall sentiment (positive/negative/neutral)
        
        Returns:
            Short description (e.g., "excellent quality, works well in low light")
        """
        if not comments:
            return f"{sentiment} feedback"
        
        # Combine relevant comments
        combined = " ".join(comments[:5])  # Use top 5 comments
        
        # Limit length
        if len(combined) > 500:
            combined = combined[:500]
        
        try:
            # Generate very short summary (max 50 words)
            summary = await self.summarize_text(
                combined,
                max_length=50,
                min_length=10
            )
            
            # Clean up and make it concise
            summary = summary.replace(aspect.title(), "").strip()
            summary = summary.replace(aspect.lower(), "").strip()
            
            # Remove starting "is" or "are" if present
            if summary.lower().startswith("is "):
                summary = summary[3:]
            elif summary.lower().startswith("are "):
                summary = summary[4:]
            
            # Lowercase first letter for consistency
            if summary:
                summary = summary[0].lower() + summary[1:]
            
            return summary if summary else f"{sentiment} feedback overall"
            
        except Exception as e:
            logger.error(f"Error generating aspect description: {e}")
            # Fallback: extract key phrases
            return self._extract_key_phrases(comments, sentiment)
    
    def _extract_key_phrases(self, comments: List[str], sentiment: str) -> str:
        """
        Extract key phrases from comments as fallback
        
        Args:
            comments: List of comments
            sentiment: Sentiment (positive/negative/neutral)
        
        Returns:
            Key phrases summary
        """
        # Common positive/negative words
        positive_words = ['excellent', 'great', 'amazing', 'good', 'love', 'perfect', 'best']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'disappointing', 'worst', 'issue']
        
        # Count frequency
        words_lower = " ".join(comments).lower()
        
        key_words = []
        if sentiment == "positive":
            key_words = [w for w in positive_words if w in words_lower]
        elif sentiment == "negative":
            key_words = [w for w in negative_words if w in words_lower]
        
        if key_words:
            return f"{', '.join(key_words[:3])} according to users"
        else:
            return f"{sentiment} feedback from users"
    
    async def generate_product_summary(
        self,
        posts: List,
        keyword: str
    ) -> str:
        """
        Generate product summary - uses aspect-based approach
        
        Args:
            posts: List of Post objects with comments
            keyword: Search keyword (product name)
        
        Returns:
            Structured summary by aspects
        """
        return await self.generate_product_summary_by_aspects(posts, keyword)
    
    async def extract_key_points(
        self,
        posts: List,
        aspects: List[AspectSentiment]
    ) -> List[str]:
        """
        Extract key points from posts based on aspects and sentiments
        
        Args:
            posts: List of Post objects
            aspects: List of identified aspects
        
        Returns:
            List of key points (bullet points)
        """
        key_points = []
        
        try:
            # 1. Overall sentiment key point
            all_sentiments = []
            for post in posts:
                if post.sentiment:
                    all_sentiments.append(post.sentiment)
                for comment in post.comments:
                    if comment.sentiment:
                        all_sentiments.append(comment.sentiment)
            
            if all_sentiments:
                overall = self.calculate_overall_sentiment(all_sentiments)
                sentiment_distribution = f"{overall.positive*100:.0f}% positive, {overall.negative*100:.0f}% negative, {overall.neutral*100:.0f}% neutral"
                key_points.append(f"Overall sentiment: {overall.dominant.capitalize()} ({sentiment_distribution})")
            
            # 2. Top aspects mentioned
            if aspects:
                top_aspects = aspects[:3]  # Top 3 aspects
                for aspect in top_aspects:
                    key_points.append(
                        f"{aspect.aspect.capitalize()}: {aspect.sentiment.capitalize()} sentiment ({aspect.mentions} mentions)"
                    )
            
            # 3. Community engagement
            total_comments = sum(len(post.comments) for post in posts)
            avg_score = sum(post.score or 0 for post in posts) / len(posts) if posts else 0
            key_points.append(f"Community engagement: {total_comments} comments, average score {avg_score:.1f}")
            
            # 4. Most discussed topics (from post titles)
            if posts:
                titles_text = " ".join([post.title or "" for post in posts])
                keywords = self.preprocess_text(titles_text).split()
                keyword_freq = Counter(keywords)
                
                # Get top 3 most frequent words (excluding very common ones)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are'}
                filtered_keywords = [(k, v) for k, v in keyword_freq.items() 
                                    if len(k) > 3 and k.lower() not in stop_words]
                
                if filtered_keywords:
                    top_keywords = sorted(filtered_keywords, key=lambda x: x[1], reverse=True)[:3]
                    keywords_str = ", ".join([k for k, v in top_keywords])
                    key_points.append(f"Most discussed topics: {keywords_str}")
            
            return key_points[:5]  # Return max 5 key points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return ["Analysis in progress..."]


# Singleton instance
sentiment_service = SentimentService()

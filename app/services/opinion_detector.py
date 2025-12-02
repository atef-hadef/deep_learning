"""
Opinion Detector Service
========================

Deep Learning service for detecting opinionated content (reviews, subjective posts)
using a trained Keras model.

This replaces the old heuristic-based and zero-shot classification approaches with
a fine-tuned neural network specifically trained to identify subjective opinions.

Model Architecture:
- Keras Sequential model (LSTM/GRU + Dense layers)
- Trained on labeled opinion vs non-opinion dataset
- Binary classification: opinion (1) vs non-opinion (0)
- Input: tokenized text sequences (max_len=60)

Usage:
    from app.services.opinion_detector import is_opinion, opinion_score
    
    text = "I love this product, the battery life is amazing!"
    if is_opinion(text):
        # Process as opinionated content
        sentiment = analyze_sentiment(text)
    else:
        # Skip non-opinionated content (news, announcements, etc.)
        pass
"""

import logging
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple
import joblib

logger = logging.getLogger(__name__)

# Configuration
MAX_LEN = 60
BASE_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = BASE_DIR / "opinion_detector_model.h5"
TOKENIZER_PATH = BASE_DIR / "opinion_tokenizer.pkl"


@lru_cache(maxsize=1)
def get_opinion_model():
    """
    Load the Keras opinion detection model (cached)
    
    Returns:
        Keras model for opinion detection
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        # Import tensorflow here to avoid loading it at module import time
        from tensorflow.keras.models import load_model
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Opinion detector model not found at {MODEL_PATH}. "
                f"Please ensure the model file is present."
            )
        
        logger.info(f"Loading opinion detector model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        logger.info(f"✅ Opinion detector model loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load opinion detector model: {e}")
        raise


@lru_cache(maxsize=1)
def get_opinion_tokenizer():
    """
    Load the tokenizer for opinion detection (cached)
    
    Returns:
        Keras tokenizer
        
    Raises:
        FileNotFoundError: If tokenizer file doesn't exist
        Exception: If tokenizer loading fails
    """
    try:
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Opinion tokenizer not found at {TOKENIZER_PATH}. "
                f"Please ensure the tokenizer file is present."
            )
        
        logger.info(f"Loading opinion tokenizer from {TOKENIZER_PATH}")
        tokenizer = joblib.load(TOKENIZER_PATH)
        logger.info(f"✅ Opinion tokenizer loaded successfully")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load opinion tokenizer: {e}")
        raise


def opinion_score(text: str) -> float:
    """
    Calculate the probability that a text contains an opinion (subjective content)
    
    This function uses a deep learning model to classify text as opinionated or not.
    Higher scores indicate more subjective/opinionated content (reviews, experiences).
    Lower scores indicate objective content (news, announcements, facts).
    
    Args:
        text: Input text to analyze
        
    Returns:
        Float between 0 and 1 representing opinion probability
        - 0.0 = definitely NOT an opinion (objective)
        - 0.5 = uncertain
        - 1.0 = definitely an opinion (subjective)
        
    Examples:
        >>> opinion_score("I love this phone, battery is amazing!")
        0.92  # High opinion score
        
        >>> opinion_score("New iPhone 15 released today")
        0.15  # Low opinion score (news)
    """
    if not text or len(text.strip()) < 5:
        logger.debug("Text too short for opinion detection")
        return 0.0
    
    try:
        # Import tensorflow here to avoid loading at module level
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Load model and tokenizer
        model = get_opinion_model()
        tokenizer = get_opinion_tokenizer()
        
        # Tokenize and pad sequence
        seq = tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        
        # Predict opinion probability
        raw_proba = float(model.predict(X, verbose=0)[0][0])
        
        # ⚠️ IMPORTANT: Model was trained with inverted labels (0=opinion, 1=non-opinion)
        # So we need to invert the prediction to get correct opinion probability
        proba = 1.0 - raw_proba
        
        logger.debug(f"Opinion score: {proba:.3f} (raw={raw_proba:.3f}) for text: '{text[:50]}...'")
        
        return proba
        
    except Exception as e:
        logger.error(f"Error computing opinion score: {e}")
        # Return neutral score on error (don't break the pipeline)
        return 0.5


def is_opinion(text: str, threshold: float = 0.5) -> bool:
    """
    Check if a text contains an opinion (binary decision)
    
    Args:
        text: Input text to analyze
        threshold: Decision threshold (default: 0.5)
                  - Lower threshold (e.g., 0.3) = more permissive (keep more posts)
                  - Higher threshold (e.g., 0.7) = more strict (filter more posts)
        
    Returns:
        True if text is likely an opinion, False otherwise
        
    Examples:
        >>> is_opinion("I recommend this product!", threshold=0.5)
        True
        
        >>> is_opinion("Product X announced today", threshold=0.5)
        False
    """
    score = opinion_score(text)
    result = score >= threshold
    
    logger.debug(
        f"Opinion detection: {result} (score={score:.3f}, threshold={threshold}) "
        f"for text: '{text[:50]}...'"
    )
    
    return result


def filter_opinion_posts(
    posts: List,
    threshold: float = 0.5,
    return_scores: bool = False
) -> Tuple[List, dict]:
    """
    Filter a list of posts to keep only opinionated content
    
    This replaces the old review_filter_service and relevance_service approaches.
    
    Args:
        posts: List of Post objects (with 'text' and/or 'title' attributes)
        threshold: Opinion detection threshold (default: 0.5)
        return_scores: If True, attach opinion scores to posts
        
    Returns:
        Tuple of (filtered_posts, stats)
        - filtered_posts: Posts that passed the opinion filter
        - stats: Dictionary with filtering statistics
        
    Example:
        >>> posts = [post1, post2, post3]
        >>> opinion_posts, stats = filter_opinion_posts(posts, threshold=0.5)
        >>> print(f"Kept {stats['kept']}/{stats['total']} posts")
    """
    if not posts:
        return [], {
            'total': 0,
            'kept': 0,
            'filtered': 0,
            'percentage': 0.0,
            'avg_score': 0.0
        }
    
    logger.info(
        f"🧠 Filtering {len(posts)} posts with OpinionDetector "
        f"(threshold={threshold})"
    )
    
    opinion_posts = []
    all_scores = []
    
    for post in posts:
        try:
            # Combine title and text for analysis
            text = ""
            if hasattr(post, 'title') and post.title:
                text += post.title + " "
            if hasattr(post, 'text') and post.text:
                text += post.text
            
            if not text.strip():
                logger.debug(f"Skipping post {post.id}: empty text")
                continue
            
            # Calculate opinion score
            score = opinion_score(text)
            all_scores.append(score)
            
            # Apply threshold
            if score >= threshold:
                # Keep opinionated post
                # Note: We don't attach opinion_score to the post object
                # because Pydantic models don't allow arbitrary attributes
                # The score is available in the stats dictionary instead
                opinion_posts.append(post)
                
                logger.debug(
                    f"✅ KEPT post {post.id[:8]}... "
                    f"(opinion_score={score:.3f})"
                )
            else:
                # Filter out non-opinionated post
                logger.debug(
                    f"❌ FILTERED post {post.id[:8]}... "
                    f"(opinion_score={score:.3f} < {threshold})"
                )
                
        except Exception as e:
            logger.error(f"Error filtering post {getattr(post, 'id', '?')}: {e}")
            # On error, skip the post (conservative approach)
            continue
    
    # Calculate statistics
    total = len(posts)
    kept = len(opinion_posts)
    filtered = total - kept
    percentage = (kept / total * 100) if total > 0 else 0.0
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    stats = {
        'total': total,
        'kept': kept,
        'filtered': filtered,
        'percentage': percentage,
        'avg_score': round(avg_score, 3),
        'min_score': round(min(all_scores), 3) if all_scores else 0.0,
        'max_score': round(max(all_scores), 3) if all_scores else 0.0
    }
    
    logger.info(
        f"✅ OpinionDetector filtering complete: "
        f"{kept}/{total} posts kept ({percentage:.1f}%), "
        f"avg_score={avg_score:.3f}"
    )
    
    return opinion_posts, stats


def is_service_available() -> bool:
    """
    Check if the opinion detection service is available
    
    Returns:
        True if model and tokenizer can be loaded, False otherwise
    """
    try:
        get_opinion_model()
        get_opinion_tokenizer()
        return True
    except Exception as e:
        logger.warning(f"Opinion detection service not available: {e}")
        return False


# Module-level convenience function for quick checks
def check_models_exist() -> dict:
    """
    Check if required model files exist
    
    Returns:
        Dictionary with file existence status
    """
    return {
        'model_exists': MODEL_PATH.exists(),
        'tokenizer_exists': TOKENIZER_PATH.exists(),
        'model_path': str(MODEL_PATH),
        'tokenizer_path': str(TOKENIZER_PATH)
    }


if __name__ == "__main__":
    # Quick test of the service
    logging.basicConfig(level=logging.INFO)
    
    print("Opinion Detector Service - Quick Test")
    print("=" * 50)
    
    # Check files
    status = check_models_exist()
    print(f"\nModel file exists: {status['model_exists']}")
    print(f"Tokenizer file exists: {status['tokenizer_exists']}")
    
    if status['model_exists'] and status['tokenizer_exists']:
        # Test examples
        test_texts = [
            "I love this product, the battery life is amazing!",
            "New iPhone 15 released today with USB-C",
            "After using it for 2 weeks, I'm very disappointed",
            "Company announces Q4 earnings report",
            "Best purchase I made this year, highly recommend!",
            "Is this phone worth buying?"
        ]
        
        print("\nTesting opinion detection:")
        print("-" * 50)
        
        for text in test_texts:
            score = opinion_score(text)
            is_op = is_opinion(text, threshold=0.5)
            print(f"Score: {score:.3f} | Opinion: {is_op:5} | {text}")
    else:
        print("\n⚠️ Model files not found. Please ensure:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {TOKENIZER_PATH}")

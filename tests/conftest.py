"""
Configuration pytest
"""
import pytest
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_post():
    """Fixture pour un post d'exemple"""
    return {
        "id": "test123",
        "platform": "reddit",
        "title": "Test Post",
        "text": "This is a test post",
        "author": "testuser",
        "created_at": "2024-01-15T10:00:00Z",
        "score": 100,
        "url": "https://reddit.com/test",
        "num_comments": 5,
        "comments": []
    }


@pytest.fixture
def sample_sentiment():
    """Fixture pour un sentiment d'exemple"""
    return {
        "positive": 0.6,
        "negative": 0.2,
        "neutral": 0.2,
        "dominant": "positive"
    }

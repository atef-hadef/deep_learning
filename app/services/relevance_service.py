"""
Relevance Detection Service

This service filters non-opinionated posts before sentiment analysis.
It detects whether a post contains meaningful opinions/reviews or just news/announcements.

Examples:
- Relevant: "I love this phone, battery is amazing!" ✅
- Not relevant: "New iPhone released today" ❌
"""

import re
from typing import List, Dict, Tuple
import logging
from app.schemas.responses import Post

logger = logging.getLogger(__name__)


class RelevanceService:
    """Service for detecting opinion-rich, relevant posts"""
    
    def __init__(self):
        # Opinion keywords (strongly opinionated language)
        self.opinion_keywords = {
            # Positive opinions
            'love', 'amazing', 'excellent', 'fantastic', 'perfect', 'awesome',
            'great', 'wonderful', 'impressed', 'satisfied', 'happy', 'glad',
            'recommend', 'best', 'favorite', 'favourite', 'incredible', 'brilliant',
            'outstanding', 'superb', 'enjoy', 'enjoying', 'enjoyed',
            
            # Negative opinions
            'hate', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
            'disappointing', 'frustrating', 'frustrated', 'annoying', 'annoyed',
            'regret', 'regretting', 'useless', 'waste', 'poor', 'bad',
            'disgusted', 'angry', 'furious', 'pathetic', 'garbage', 'trash',
            
            # Comparative/evaluative
            'better', 'worse', 'overrated', 'underrated', 'overhyped',
            'worth', 'worthless', 'overpriced', 'expensive', 'cheap',
            
            # Emotional reactions
            'shocked', 'surprised', 'amazed', 'stunned', 'blown away',
            'disappointed', 'letdown', 'satisfied', 'pleased', 'delighted',
        }
        
        # Strong opinion phrases
        self.opinion_phrases = [
            r'\bi (love|hate|like|dislike|enjoy|regret)',
            r'\bwould (not )?(recommend|buy|suggest)',
            r'\b(highly|strongly) recommend',
            r'\b(don\'t|do not) (buy|waste|bother)',
            r'\b(best|worst) (purchase|decision|choice)',
            r'\b(so|very|extremely|really) (good|bad|disappointed|happy|satisfied)',
            r'\bmoney well spent',
            r'\bwaste of money',
            r'\bworth (it|every penny|the price)',
            r'\bnot worth',
            r'\bmy (favorite|favourite|least favorite)',
            r'\b(glad|happy|sad|angry|frustrated) (i|that)',
        ]
        
        # Personal experience indicators
        self.experience_indicators = [
            r'\bi (bought|purchased|got|received|tried|tested|used)',
            r'\bmy (experience|opinion|review|thoughts)',
            r'\bafter (buying|using|trying|testing)',
            r'\b(have|has) been using',
            r'\bfor (\d+) (days?|weeks?|months?|years?)',
            r'\bowned (it|this) for',
        ]
        
        # Non-opinion patterns (news, announcements, facts)
        self.non_opinion_patterns = [
            r'\b(announced|announces|launching|released|releases|available|coming)',
            r'\b(new|upcoming) (product|model|version|release)',
            r'\bwill (be|feature|include|have|cost)',
            r'\b(specs|specifications|features)[:;]',
            r'\bpriced at',
            r'\bstarting at \$',
            r'\brumor[s]?[:;]',
            r'\b(leaked|leak)[:;]',
            r'\baccording to (reports?|sources?)',
            r'\bofficially confirmed',
        ]
        
        # Question patterns (often not opinions)
        self.question_patterns = [
            r'\bshould i (buy|get|purchase)',
            r'\b(is|are) (it|they|this) (worth|good|any good)',
            r'\bwhat (do you|does everyone) think',
            r'\banyone (tried|tested|used)',
            r'\bany recommendations',
        ]
    
    def calculate_relevance_score(self, text: str) -> Dict[str, any]:
        """
        Calculate relevance score for a post
        
        Args:
            text: Post text
        
        Returns:
            Dictionary with score and reasoning
        """
        if not text or len(text.strip()) < 10:
            return {
                'score': 0.0,
                'is_relevant': False,
                'reason': 'Text too short'
            }
        
        text_lower = text.lower()
        score = 0.0
        reasons = []
        
        # 1. Opinion keywords (strong signal)
        opinion_count = sum(1 for keyword in self.opinion_keywords if keyword in text_lower)
        if opinion_count > 0:
            keyword_score = min(opinion_count * 0.2, 0.6)
            score += keyword_score
            reasons.append(f"{opinion_count} opinion keywords")
        
        # 2. Opinion phrases (very strong signal)
        phrase_matches = 0
        for pattern in self.opinion_phrases:
            if re.search(pattern, text_lower):
                phrase_matches += 1
        
        if phrase_matches > 0:
            phrase_score = min(phrase_matches * 0.25, 0.7)
            score += phrase_score
            reasons.append(f"{phrase_matches} opinion phrases")
        
        # 3. Personal experience indicators (VERY strong signal for client reviews)
        experience_matches = 0
        for pattern in self.experience_indicators:
            if re.search(pattern, text_lower):
                experience_matches += 1
        
        if experience_matches > 0:
            # ✅ Increased weight: client reviews mention their experience
            experience_score = min(experience_matches * 0.3, 0.7)
            score += experience_score
            reasons.append(f"{experience_matches} experience indicators")
        
        # 4. First-person pronouns (important for client reviews)
        first_person_count = len(re.findall(r'\b(i|my|mine|me|myself)\b', text_lower))
        if first_person_count >= 2:
            # ✅ Increased weight: client reviews are personal
            first_person_score = min(first_person_count * 0.08, 0.4)
            score += first_person_score
            reasons.append(f"{first_person_count} first-person pronouns")
        elif first_person_count == 0:
            # ✅ Penalty if no personal pronouns (likely not a client review)
            score -= 0.15
            reasons.append("-no personal pronouns")
        
        # 5. Exclamation marks (weak positive signal)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            exclamation_score = min(exclamation_count * 0.05, 0.2)
            score += exclamation_score
            reasons.append(f"{exclamation_count} exclamations")
        
        # PENALTIES
        
        # 6. Non-opinion patterns (strong negative)
        non_opinion_matches = 0
        for pattern in self.non_opinion_patterns:
            if re.search(pattern, text_lower):
                non_opinion_matches += 1
        
        if non_opinion_matches > 0:
            penalty = min(non_opinion_matches * 0.3, 0.8)
            score -= penalty
            reasons.append(f"-{non_opinion_matches} news/announcement patterns")
        
        # 7. Questions only (moderate negative)
        question_matches = 0
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower):
                question_matches += 1
        
        if question_matches > 0 and '?' in text:
            penalty = min(question_matches * 0.2, 0.4)
            score -= penalty
            reasons.append(f"-{question_matches} question patterns")
        
        # 8. Short posts without substance (stronger penalty)
        if len(text) < 50:
            score -= 0.2  # ✅ Increased penalty
            reasons.append("-short text")
        
        # 9. ✅ BONUS: Real client review (opinion + experience + personal)
        if opinion_count > 0 and experience_matches > 0 and first_person_count >= 2:
            score += 0.2
            reasons.append("+client review bonus")
        
        # 10. ✅ Minimum length for quality reviews
        if len(text) < 30:
            score -= 0.3
            reasons.append("-too short for review")
        
        # Normalize score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Determine relevance (dynamic threshold based on min_score parameter)
        is_relevant = score >= 0.4  # Default threshold for client reviews
        
        return {
            'score': round(score, 3),
            'is_relevant': is_relevant,
            'reason': ' | '.join(reasons) if reasons else 'No clear signals'
        }
    
    def filter_relevant_posts(
        self,
        posts: List[Post],
        min_score: float = 0.3
    ) -> Tuple[List[Post], List[Dict]]:
        """
        Filter posts by relevance, keeping only opinion-rich content
        
        Args:
            posts: List of posts to filter
            min_score: Minimum relevance score (0-1)
        
        Returns:
            Tuple of (relevant_posts, filtering_stats)
        """
        if not posts:
            return [], []
        
        relevant_posts = []
        filtered_posts = []
        stats = {
            'total': len(posts),
            'relevant': 0,
            'filtered': 0,
            'scores': []
        }
        
        for post in posts:
            # Combine title and text for analysis
            full_text = (post.title or "") + " " + post.text
            
            # Calculate relevance
            relevance = self.calculate_relevance_score(full_text)
            
            # Store score
            stats['scores'].append(relevance['score'])
            
            if relevance['is_relevant'] and relevance['score'] >= min_score:
                # Keep relevant post
                post.relevance_score = relevance['score']
                relevant_posts.append(post)
                stats['relevant'] += 1
                
                logger.debug(
                    f"✅ KEPT post {post.id[:8]}... "
                    f"(score={relevance['score']:.2f}): {relevance['reason']}"
                )
            else:
                # Filter out irrelevant post
                filtered_posts.append({
                    'post_id': post.id,
                    'platform': post.platform,
                    'text_preview': full_text[:100],
                    'score': relevance['score'],
                    'reason': relevance['reason']
                })
                stats['filtered'] += 1
                
                logger.debug(
                    f"❌ FILTERED post {post.id[:8]}... "
                    f"(score={relevance['score']:.2f}): {relevance['reason']}"
                )
        
        # Calculate statistics
        if stats['scores']:
            stats['avg_score'] = round(sum(stats['scores']) / len(stats['scores']), 3)
            stats['max_score'] = round(max(stats['scores']), 3)
            stats['min_score'] = round(min(stats['scores']), 3)
        
        logger.info(
            f"📊 Relevance filtering: {stats['relevant']}/{stats['total']} posts kept "
            f"({stats['filtered']} filtered, avg_score={stats.get('avg_score', 0):.2f})"
        )
        
        return relevant_posts, filtered_posts
    
    def get_filtering_summary(self, filtered_posts: List[Dict]) -> Dict:
        """
        Generate summary of filtered posts
        
        Args:
            filtered_posts: List of filtered post info
        
        Returns:
            Summary dictionary
        """
        if not filtered_posts:
            return {
                'count': 0,
                'reasons': {},
                'examples': []
            }
        
        # Count reasons
        reason_counts = {}
        for post in filtered_posts:
            reason = post.get('reason', 'Unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Get examples (top 3)
        examples = [
            {
                'text': post['text_preview'],
                'score': post['score'],
                'reason': post['reason']
            }
            for post in filtered_posts[:3]
        ]
        
        return {
            'count': len(filtered_posts),
            'reasons': reason_counts,
            'examples': examples
        }


# Singleton instance
relevance_service = RelevanceService()

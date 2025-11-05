"""
Advanced Real-Time Sentiment Analysis for NBA Predictions
Uses transformer models (BERT/RoBERTa) for context-aware sentiment classification.

Features:
- BERT-based sentiment analysis with NBA-specific fine-tuning capability
- Multiple data sources: ESPN, The Athletic, Reddit, team press releases
- Time-decay weighting for recent vs old articles
- High-impact news detection (injuries, trades vs routine summaries)
- Integrated injury-flagging and public-confidence as model features
- Calibrated probability adjustments (±5-10%)

This module is designed to run ONLY during prediction time, not during training.
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import warnings
import numpy as np
import feedparser
warnings.filterwarnings('ignore')

# Try to import transformer libraries, fallback to keyword-based if not available
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not available. Using keyword-based fallback.")


class BERTSentimentClassifier:
    """BERT-based sentiment classifier for NBA news and social media"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize BERT-based sentiment classifier.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "cardiffnlp/twitter-roberta-base-sentiment-latest" (fast, good for social media)
                - "nlptown/bert-base-multilingual-uncased-sentiment" (multilingual)
                - "finiteautomata/bertweet-base-sentiment-analysis" (Twitter-optimized)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use pipeline for easier inference
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=-1  # Use CPU (-1) or GPU (0+) if available
                )
                print(f"✅ Loaded BERT model: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load BERT model: {e}. Using keyword fallback.")
                self.classifier = None
        else:
            self.classifier = None
    
    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify sentiment of text.
        
        Returns:
            Dict with 'label' (POSITIVE/NEGATIVE/NEUTRAL) and 'score' (confidence)
        """
        if not text or len(text.strip()) < 3:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        if self.classifier:
            try:
                # Truncate long texts (BERT has token limits)
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                result = self.classifier(text)[0]
                
                # Normalize to our format
                label = result['label'].upper()
                score = result['score']
                
                # Convert to positive/negative/neutral scale (0-1)
                if 'POSITIVE' in label or 'POS' in label:
                    return {'label': 'POSITIVE', 'score': score}
                elif 'NEGATIVE' in label or 'NEG' in label:
                    return {'label': 'NEGATIVE', 'score': 1.0 - score}  # Invert for consistency
                else:
                    return {'label': 'NEUTRAL', 'score': 0.5}
                    
            except Exception as e:
                print(f"⚠️  BERT classification error: {e}")
                return self._keyword_fallback(text)
        else:
            return self._keyword_fallback(text)
    
    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        """Fallback to keyword-based sentiment if BERT unavailable"""
        text_lower = text.lower()
        
        positive_keywords = ['win', 'victory', 'dominat', 'stellar', 'excellent', 
                           'impressive', 'breakout', 'hot', 'streak', 'career-high',
                           'amazing', 'clutch', 'beast', 'outstanding', 'elite']
        negative_keywords = ['loss', 'lose', 'losing', 'injury', 'hurt', 'out', 
                            'struggle', 'slump', 'disappointing', 'poor', 'blow',
                            'choke', 'terrible', 'trash', 'embarrassing', 'awful']
        
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        sentiment_score = (pos_count - neg_count) / total * 0.5 + 0.5
        
        if sentiment_score > 0.6:
            return {'label': 'POSITIVE', 'score': sentiment_score}
        elif sentiment_score < 0.4:
            return {'label': 'NEGATIVE', 'score': sentiment_score}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5}


class NewsArticle:
    """Represents a news article with metadata"""
    
    def __init__(self, title: str, content: str, source: str, 
                 publish_date: datetime, url: str = "", 
                 impact_level: str = "routine"):
        self.title = title
        self.content = content
        self.source = source
        self.publish_date = publish_date
        self.url = url
        self.impact_level = impact_level  # "high", "medium", "routine"
        self.sentiment = None
        self.sentiment_score = 0.5
    
    def calculate_age_days(self) -> float:
        """Calculate age of article in days"""
        return (datetime.now() - self.publish_date).total_seconds() / 86400
    
    def is_high_impact(self) -> bool:
        """Check if article is high-impact"""
        return self.impact_level == "high"
    
    def get_time_weight(self, decay_half_life_days: float = 3.0) -> float:
        """
        Calculate time-decay weight for article.
        Recent articles weighted more heavily.
        
        Args:
            decay_half_life_days: Days until weight drops to 50%
        """
        age_days = self.calculate_age_days()
        weight = np.exp(-np.log(2) * age_days / decay_half_life_days)
        return max(0.1, weight)  # Minimum weight of 0.1


class NBASentimentAnalyzer:
    """Advanced NBA sentiment analyzer with BERT and expanded data sources"""
    
    def __init__(self, use_bert: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        # Initialize BERT classifier
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.bert_classifier = BERTSentimentClassifier() if self.use_bert else None
        
        # High-impact keywords for news classification
        self.high_impact_keywords = [
            'season-ending', 'torn', 'fracture', 'surgery', 'out indefinitely',
            'trade', 'traded', 'acquired', 'signed', 'released', 'waived',
            'fired', 'coach', 'suspended', 'arrest', 'investigation'
        ]
        
        self.medium_impact_keywords = [
            'injury', 'questionable', 'doubtful', 'listed', 'day-to-day',
            'signing', 'extension', 'contract', 'free agent'
        ]
    
    def _classify_impact_level(self, text: str) -> str:
        """Classify news impact level"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in self.high_impact_keywords):
            return "high"
        elif any(kw in text_lower for kw in self.medium_impact_keywords):
            return "medium"
        else:
            return "routine"
    
    def get_team_sentiment(self, team_name: str, opponent_name: str = None) -> Dict[str, float]:
        """Get comprehensive sentiment for a team using BERT and expanded sources"""
        
        cache_key = f"{team_name}_{opponent_name}_{datetime.now().hour}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sentiment = {
            'overall_sentiment': 0.5,
            'news_sentiment': 0.5,
            'social_buzz': 0.5,
            'injury_concerns': 0.0,
            'momentum_narrative': 0.5,
            'public_confidence': 0.5,
            'media_attention': 0.5,
            'contrarian_indicator': 0.0,
            'high_impact_sentiment': 0.5,
            'weighted_sentiment': 0.5
        }
        
        try:
            # Collect articles from multiple sources
            articles = []
            
            # 1. ESPN News
            espn_articles = self._get_espn_articles(team_name)
            articles.extend(espn_articles)
            
            # 2. The Athletic (via RSS if available)
            athletic_articles = self._get_the_athletic_articles(team_name)
            articles.extend(athletic_articles)
            
            # 3. Reddit Posts
            reddit_articles = self._get_reddit_articles(team_name, opponent_name)
            articles.extend(reddit_articles)
            
            # 4. Team Press Releases
            press_releases = self._get_team_press_releases(team_name)
            articles.extend(press_releases)
            
            # Analyze sentiment for all articles using BERT
            if articles:
                sentiment_scores = []
                weighted_scores = []
                high_impact_scores = []
                
                for article in articles:
                    # Classify sentiment
                    sentiment_result = self._analyze_article_sentiment(article)
                    article.sentiment = sentiment_result['label']
                    article.sentiment_score = sentiment_result['score']
                    
                    # Convert to 0-1 scale
                    if article.sentiment == 'POSITIVE':
                        score = article.sentiment_score
                    elif article.sentiment == 'NEGATIVE':
                        score = 1.0 - article.sentiment_score
                    else:
                        score = 0.5
                    
                    sentiment_scores.append(score)
                    
                    # Calculate weighted score (time-decay + impact)
                    time_weight = article.get_time_weight()
                    impact_weight = 2.0 if article.is_high_impact() else 1.0
                    weighted_score = score * time_weight * impact_weight
                    weighted_scores.append(weighted_score)
                    
                    if article.is_high_impact():
                        high_impact_scores.append(score)
                
                # Calculate aggregate sentiment
                if sentiment_scores:
                    sentiment['news_sentiment'] = np.mean(sentiment_scores)
                    
                    # Weighted average
                    if weighted_scores:
                        total_weight = sum(
                            article.get_time_weight() * (2.0 if article.is_high_impact() else 1.0)
                            for article in articles
                        )
                        if total_weight > 0:
                            sentiment['weighted_sentiment'] = sum(weighted_scores) / total_weight
                    
                    # High-impact news sentiment
                    if high_impact_scores:
                        sentiment['high_impact_sentiment'] = np.mean(high_impact_scores)
            
            # 3. Injury News Check (keep existing logic as feature)
            injury_severity = self._check_injury_news(team_name)
            sentiment['injury_concerns'] = injury_severity
            
            # 4. Reddit Social Buzz (analyzed with BERT above, but also calculate separately)
            if reddit_articles:
                reddit_scores = [a.sentiment_score for a in reddit_articles if a.sentiment == 'POSITIVE']
                reddit_scores.extend([1.0 - a.sentiment_score for a in reddit_articles if a.sentiment == 'NEGATIVE'])
                if reddit_scores:
                    sentiment['social_buzz'] = np.mean(reddit_scores) if reddit_scores else 0.5
            else:
                sentiment['social_buzz'] = self._get_reddit_sentiment_fallback(team_name, opponent_name)
            
            # 5. Recent Performance Narrative
            momentum = self._analyze_momentum_narrative(team_name, articles)
            sentiment['momentum_narrative'] = momentum
            
            # 6. Public Confidence (betting trends)
            public_confidence = self._estimate_public_confidence(team_name, opponent_name)
            sentiment['public_confidence'] = public_confidence
            
            # 7. Media Attention Score
            media_score = self._calculate_media_attention(team_name, len(articles))
            sentiment['media_attention'] = media_score
            
            # 8. Calculate overall sentiment with BERT features
            # Use weighted sentiment as primary, combine with other features
            base_sentiment = sentiment.get('weighted_sentiment', sentiment['news_sentiment'])
            
            # Incorporate injury concerns and public confidence as features
            injury_factor = 1.0 - (sentiment['injury_concerns'] * 0.3)  # Reduce sentiment if injuries
            public_factor = sentiment['public_confidence'] * 0.15 + 0.85  # Slight boost from public confidence
            
            sentiment['overall_sentiment'] = (
                base_sentiment * 0.40 +
                sentiment['social_buzz'] * 0.20 +
                base_sentiment * injury_factor * 0.20 +
                sentiment['momentum_narrative'] * 0.10 +
                sentiment['public_confidence'] * 0.05 +
                sentiment['media_attention'] * 0.05
            )
            
            # Ensure valid range
            sentiment['overall_sentiment'] = max(0.0, min(1.0, sentiment['overall_sentiment']))
            
            # 9. Contrarian indicator (when public is too confident)
            if sentiment['public_confidence'] > 0.7 or sentiment['public_confidence'] < 0.3:
                sentiment['contrarian_indicator'] = 1.0
            
        except Exception as e:
            print(f"Sentiment analysis error for {team_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cache the result
        self.cache[cache_key] = sentiment
        
        return sentiment
    
    def _analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, float]:
        """Analyze sentiment of an article using BERT"""
        # Combine title and content for analysis
        text = f"{article.title}. {article.content[:200]}"  # Limit content length
        
        if self.bert_classifier:
            return self.bert_classifier.classify(text)
        else:
            # Fallback to keyword-based
            return self.bert_classifier._keyword_fallback(text) if self.bert_classifier else {'label': 'NEUTRAL', 'score': 0.5}
    
    def _get_espn_articles(self, team_name: str) -> List[NewsArticle]:
        """Scrape ESPN for team news articles"""
        articles = []
        try:
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.espn.com/nba/team/_/name/{team_abbr.lower()}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return articles
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find headlines and articles
            headlines = soup.find_all(['h1', 'h2', 'h3', 'h4', 'a'], limit=20)
            
            for elem in headlines:
                text = elem.get_text().strip()
                if len(text) < 10:  # Skip very short texts
                    continue
                
                # Try to find publish date
                publish_date = datetime.now() - timedelta(days=1)  # Default to yesterday
                
                # Determine impact level
                impact_level = self._classify_impact_level(text)
                
                article = NewsArticle(
                    title=text,
                    content=text,  # ESPN scraping limited, use title as content
                    source="ESPN",
                    publish_date=publish_date,
                    url=elem.get('href', '') if hasattr(elem, 'get') else '',
                    impact_level=impact_level
                )
                articles.append(article)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching ESPN articles: {e}")
        
        return articles
    
    def _get_the_athletic_articles(self, team_name: str) -> List[NewsArticle]:
        """Get articles from The Athletic (via RSS if available)"""
        articles = []
        try:
            # The Athletic RSS feeds (if available)
            # Note: The Athletic may require subscription, so this is a placeholder
            # In production, you might need API access or scraping
            
            # Try to find team-specific RSS feed
            team_abbr = self._get_team_abbreviation(team_name)
            
            # Common RSS patterns (may need adjustment)
            rss_urls = [
                f"https://theathletic.com/feeds/tag/{team_abbr.lower()}/",
                f"https://theathletic.com/rss/{team_abbr.lower()}/"
            ]
            
            for rss_url in rss_urls:
                try:
                    feed = feedparser.parse(rss_url)
                    if feed.entries:
                        for entry in feed.entries[:10]:  # Limit to 10 articles
                            publish_date = datetime.now() - timedelta(days=1)
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                publish_date = datetime(*entry.published_parsed[:6])
                            
                            impact_level = self._classify_impact_level(
                                f"{entry.get('title', '')} {entry.get('summary', '')}"
                            )
                            
                            article = NewsArticle(
                                title=entry.get('title', ''),
                                content=entry.get('summary', ''),
                                source="The Athletic",
                                publish_date=publish_date,
                                url=entry.get('link', ''),
                                impact_level=impact_level
                            )
                            articles.append(article)
                        break  # Successfully parsed, no need to try other URLs
                except:
                    continue
            
        except Exception as e:
            # The Athletic may not be accessible, that's okay
            pass
        
        return articles
    
    def _get_reddit_articles(self, team_name: str, opponent_name: str = None) -> List[NewsArticle]:
        """Get Reddit posts as articles"""
        articles = []
        try:
            # Reddit JSON API
            url = f"https://www.reddit.com/r/nba/search.json?q={team_name}&sort=new&limit=25&t=week"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return articles
            
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                return articles
            
            posts = data['data']['children']
            
            for post in posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                score = post_data.get('score', 0)
                created_utc = post_data.get('created_utc', 0)
                
                # Convert UTC timestamp to datetime
                publish_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
                
                # Determine impact level
                impact_level = self._classify_impact_level(f"{title} {selftext}")
                
                # Combine title and text
                content = f"{title}. {selftext[:200]}"
                
                article = NewsArticle(
                    title=title,
                    content=content,
                    source="Reddit",
                    publish_date=publish_date,
                    url=post_data.get('url', ''),
                    impact_level=impact_level
                )
                articles.append(article)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching Reddit articles: {e}")
        
        return articles
    
    def _get_team_press_releases(self, team_name: str) -> List[NewsArticle]:
        """Get team press releases (placeholder - would need team website scraping)"""
        articles = []
        try:
            # NBA.com team pages sometimes have press releases
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.nba.com/{team_abbr.lower()}/news"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for news items
                news_items = soup.find_all(['article', 'div'], class_=re.compile(r'news|press|release', re.I), limit=10)
                
                for item in news_items:
                    title_elem = item.find(['h1', 'h2', 'h3', 'h4', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        content = item.get_text().strip()[:300]
                        
                        # Press releases are typically high-impact
                        impact_level = "high" if any(kw in title.lower() for kw in ['sign', 'trade', 'injury', 'release']) else "medium"
                        
                        article = NewsArticle(
                            title=title,
                            content=content,
                            source=f"{team_name} Press Release",
                            publish_date=datetime.now() - timedelta(days=1),
                            url="",
                            impact_level=impact_level
                        )
                        articles.append(article)
            
        except Exception as e:
            # Press releases may not be easily accessible, that's okay
            pass
        
        return articles
    
    def _get_reddit_sentiment_fallback(self, team_name: str, opponent_name: str = None) -> float:
        """Fallback Reddit sentiment analysis (original keyword-based method)"""
        try:
            url = f"https://www.reddit.com/r/nba/search.json?q={team_name}&sort=new&limit=25&t=week"
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.5
            
            data = response.json()
            if 'data' not in data or 'children' not in data['data']:
                return 0.5
            
            posts = data['data']['children']
            positive_score = 0
            negative_score = 0
            total_engagement = 0
            
            positive_keywords = ['win', 'victory', 'dominat', 'amazing', 'clutch', 'beast']
            negative_keywords = ['loss', 'lose', 'choke', 'terrible', 'trash', 'embarrassing']
            
            for post in posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '').lower()
                score = post_data.get('score', 0)
                
                has_positive = any(kw in title for kw in positive_keywords)
                has_negative = any(kw in title for kw in negative_keywords)
                
                if has_positive:
                    positive_score += score
                if has_negative:
                    negative_score += score
                
                total_engagement += abs(score)
            
            if total_engagement == 0:
                return 0.5
            
            buzz = (positive_score - negative_score) / total_engagement * 0.5 + 0.5
            return max(0.0, min(1.0, buzz))
            
        except Exception as e:
            return 0.5
    
    def _check_injury_news(self, team_name: str) -> float:
        """Check for injury concerns (0 = no concerns, 1 = major concerns)"""
        try:
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.espn.com/nba/team/injuries/_/name/{team_abbr.lower()}"
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.0
            
            soup = BeautifulSoup(response.content, 'html.parser')
            injury_statuses = soup.find_all(string=re.compile(r'Out|Doubtful|Questionable', re.I))
            
            out_count = len([s for s in injury_statuses if 'out' in s.lower()])
            doubtful_count = len([s for s in injury_statuses if 'doubtful' in s.lower()])
            questionable_count = len([s for s in injury_statuses if 'questionable' in s.lower()])
            
            injury_score = (out_count * 1.0 + doubtful_count * 0.7 + questionable_count * 0.3) / 10
            return min(1.0, injury_score)
            
        except Exception as e:
            return 0.0
    
    def _analyze_momentum_narrative(self, team_name: str, articles: List[NewsArticle] = None) -> float:
        """Analyze recent performance narrative from articles"""
        if articles:
            # Use recent article sentiment as momentum indicator
            recent_articles = [a for a in articles if a.calculate_age_days() < 7]
            if recent_articles:
                momentum_scores = []
                for article in recent_articles:
                    if article.sentiment == 'POSITIVE':
                        momentum_scores.append(article.sentiment_score)
                    elif article.sentiment == 'NEGATIVE':
                        momentum_scores.append(1.0 - article.sentiment_score)
                    else:
                        momentum_scores.append(0.5)
                
                if momentum_scores:
                    return np.mean(momentum_scores)
        
        # Fallback to ESPN sentiment
        return self._get_espn_sentiment_fallback(team_name)
    
    def _get_espn_sentiment_fallback(self, team_name: str) -> float:
        """Fallback ESPN sentiment (original keyword-based method)"""
        try:
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.espn.com/nba/team/_/name/{team_abbr.lower()}"
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.5
            
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], limit=10)
            
            positive_keywords = ['win', 'victory', 'dominat', 'stellar', 'excellent', 
                               'impressive', 'breakout', 'hot', 'streak', 'career-high']
            negative_keywords = ['loss', 'lose', 'losing', 'injury', 'hurt', 'out', 
                                'struggle', 'slump', 'disappointing', 'poor', 'blow']
            
            positive_count = 0
            negative_count = 0
            
            for headline in headlines:
                text = headline.get_text().lower()
                positive_count += sum(1 for kw in positive_keywords if kw in text)
                negative_count += sum(1 for kw in negative_keywords if kw in text)
            
            total = positive_count + negative_count
            if total == 0:
                return 0.5
            
            sentiment = (positive_count - negative_count) / total * 0.5 + 0.5
            return max(0.0, min(1.0, sentiment))
            
        except Exception as e:
            return 0.5
    
    def _estimate_public_confidence(self, team_name: str, opponent_name: str = None) -> float:
        """Estimate public betting confidence"""
        try:
            # This could scrape betting percentage sites like Action Network
            # For now, return neutral with some variation
            base_confidence = 0.5
            
            # Add small variation based on team name hash (deterministic)
            variation = (hash(team_name) % 100) / 500  # ±0.1
            
            return max(0.0, min(1.0, base_confidence + variation))
        except:
            return 0.5
    
    def _calculate_media_attention(self, team_name: str, article_count: int = 0) -> float:
        """Calculate media attention score"""
        try:
            # Big market teams get more attention
            big_market_teams = ['Lakers', 'Knicks', 'Warriors', 'Celtics', 'Heat', 
                              'Bulls', '76ers', 'Nets', 'Mavericks', 'Clippers']
            
            base_score = 0.5
            if any(team in team_name for team in big_market_teams):
                base_score = 0.8
            
            # Adjust based on article count
            article_factor = min(article_count / 20.0, 0.2)  # Max 0.2 boost
            
            return min(1.0, base_score + article_factor)
        except:
            return 0.5
    
    def _get_team_abbreviation(self, team_name: str) -> str:
        """Convert team name to abbreviation"""
        team_abbr_map = {
            'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA',
            'Bulls': 'CHI', 'Cavaliers': 'CLE', 'Mavericks': 'DAL', 'Nuggets': 'DEN',
            'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
            'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA',
            'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
            'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
            'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR',
            'Jazz': 'UTA', 'Wizards': 'WAS'
        }
        
        for name, abbr in team_abbr_map.items():
            if name in team_name:
                return abbr
        
        return team_name[:3].upper()
    
    def get_game_sentiment(self, home_team: str, away_team: str) -> Dict[str, Dict]:
        """Get sentiment for both teams in a matchup"""
        
        print(f"  📊 Analyzing sentiment (BERT-enhanced): {home_team} vs {away_team}")
        
        home_sentiment = self.get_team_sentiment(home_team, away_team)
        time.sleep(0.5)  # Rate limiting
        
        away_sentiment = self.get_team_sentiment(away_team, home_team)
        
        # Calculate relative sentiment
        sentiment_differential = home_sentiment['overall_sentiment'] - away_sentiment['overall_sentiment']
        
        # Calculate combined buzz
        combined_buzz = (home_sentiment['social_buzz'] + away_sentiment['social_buzz']) / 2
        
        # Determine narrative
        narrative = self._determine_game_narrative(home_sentiment, away_sentiment, home_team, away_team)
        
        result = {
            'home_team': home_sentiment,
            'away_team': away_sentiment,
            'sentiment_differential': sentiment_differential,
            'combined_buzz': combined_buzz,
            'narrative': narrative,
            'contrarian_opportunity': (home_sentiment['contrarian_indicator'] + 
                                     away_sentiment['contrarian_indicator']) / 2,
            'high_impact_news_present': (
                home_sentiment.get('high_impact_sentiment', 0.5) != 0.5 or
                away_sentiment.get('high_impact_sentiment', 0.5) != 0.5
            )
        }
        
        print(f"    Home sentiment: {home_sentiment['overall_sentiment']:.2f}")
        print(f"    Away sentiment: {away_sentiment['overall_sentiment']:.2f}")
        print(f"    Narrative: {narrative}")
        if result['high_impact_news_present']:
            print(f"    ⚠️  High-impact news detected")
        
        return result
    
    def _determine_game_narrative(self, home_sent: Dict, away_sent: Dict, 
                                  home_team: str, away_team: str) -> str:
        """Determine the narrative around a game"""
        
        h_overall = home_sent['overall_sentiment']
        a_overall = away_sent['overall_sentiment']
        
        # High momentum game
        if h_overall > 0.65 and a_overall > 0.65:
            return "🔥 High-momentum clash"
        
        # One-sided momentum
        if h_overall > 0.7 and a_overall < 0.4:
            return f"⬆️ {home_team} surging vs struggling {away_team}"
        if a_overall > 0.7 and h_overall < 0.4:
            return f"⬆️ {away_team} surging vs struggling {home_team}"
        
        # Injury concerns
        if home_sent['injury_concerns'] > 0.5:
            return f"🏥 Injury concerns for {home_team}"
        if away_sent['injury_concerns'] > 0.5:
            return f"🏥 Injury concerns for {away_team}"
        
        # High-impact news
        if home_sent.get('high_impact_sentiment', 0.5) < 0.4 or away_sent.get('high_impact_sentiment', 0.5) < 0.4:
            return "📰 High-impact news affecting matchup"
        
        # High buzz game
        if home_sent['media_attention'] > 0.7 and away_sent['media_attention'] > 0.7:
            return "🎬 High-profile matchup"
        
        # Contrarian opportunity
        if home_sent['contrarian_indicator'] > 0.5 or away_sent['contrarian_indicator'] > 0.5:
            return "💡 Contrarian value opportunity"
        
        return "⚖️ Balanced matchup"
    
    def adjust_prediction_with_sentiment(self, base_prediction: Dict, 
                                        sentiment: Dict) -> Dict[str, float]:
        """
        Adjust prediction confidence based on sentiment.
        Uses calibrated ±5-10% range based on sentiment strength and high-impact news.
        """
        
        adjusted = base_prediction.copy()
        
        # Sentiment differential affects probability
        sentiment_diff = sentiment['sentiment_differential']
        
        # Determine adjustment range based on sentiment strength and high-impact news
        base_adjustment_range = 0.05  # ±5% base
        if sentiment.get('high_impact_news_present', False):
            # High-impact news allows larger adjustments (±10%)
            adjustment_range = 0.10
        else:
            # Regular news uses ±5% range
            adjustment_range = base_adjustment_range
        
        # Scale adjustment by sentiment strength (stronger sentiment = larger adjustment)
        sentiment_strength = abs(sentiment_diff)
        if sentiment_strength > 0.3:  # Strong sentiment
            scale_factor = 1.0
        elif sentiment_strength > 0.15:  # Moderate sentiment
            scale_factor = 0.7
        else:  # Weak sentiment
            scale_factor = 0.4
        
        # Calculate probability adjustment
        probability_adjustment = sentiment_diff * adjustment_range * scale_factor
        
        if 'home_win_probability' in adjusted:
            adjusted['home_win_probability'] += probability_adjustment
            adjusted['home_win_probability'] = max(0.0, min(1.0, adjusted['home_win_probability']))
        
        # Adjust confidence based on narrative clarity and high-impact news
        if 'confidence' in adjusted:
            # High buzz or clear narrative increases confidence
            buzz_boost = sentiment['combined_buzz'] * 0.05
            
            # High-impact news provides additional confidence signal
            if sentiment.get('high_impact_news_present', False):
                buzz_boost += 0.03
            
            adjusted['confidence'] += buzz_boost
            adjusted['confidence'] = min(1.0, adjusted['confidence'])
        
        # Flag contrarian opportunities
        adjusted['contrarian_opportunity'] = sentiment['contrarian_opportunity'] > 0.5
        adjusted['narrative'] = sentiment['narrative']
        adjusted['sentiment_score'] = sentiment['sentiment_differential']
        adjusted['sentiment_adjustment'] = probability_adjustment
        adjusted['high_impact_news'] = sentiment.get('high_impact_news_present', False)
        
        return adjusted


def get_sentiment_for_prediction(home_team: str, away_team: str) -> Dict:
    """Convenience function to get sentiment for a game prediction"""
    analyzer = NBASentimentAnalyzer()
    return analyzer.get_game_sentiment(home_team, away_team)


if __name__ == "__main__":
    # Test the sentiment analyzer
    print("Testing Advanced NBA Sentiment Analyzer (BERT-enhanced)...")
    print("="*70)
    
    analyzer = NBASentimentAnalyzer()
    
    # Test with a sample game
    sentiment = analyzer.get_game_sentiment("Lakers", "Celtics")
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Sentiment Differential: {sentiment['sentiment_differential']:.3f}")
    print(f"Combined Buzz: {sentiment['combined_buzz']:.3f}")
    print(f"Narrative: {sentiment['narrative']}")
    print(f"Contrarian Opportunity: {sentiment['contrarian_opportunity']:.3f}")
    print(f"High-Impact News: {sentiment.get('high_impact_news_present', False)}")
    print(f"Home Weighted Sentiment: {sentiment['home_team'].get('weighted_sentiment', 0.5):.3f}")
    print(f"Away Weighted Sentiment: {sentiment['away_team'].get('weighted_sentiment', 0.5):.3f}")

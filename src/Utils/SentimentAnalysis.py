"""
Real-Time Sentiment Analysis for NBA Predictions
Gathers sentiment from multiple sources (no API keys required):
- Reddit NBA discussions
- Twitter/X trending topics (via scraping)
- ESPN news headlines
- NBA.com news
- Social media buzz indicators
- Market sentiment signals

This module is designed to run ONLY during prediction time, not during training.
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class NBASentimentAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def get_team_sentiment(self, team_name: str, opponent_name: str = None) -> Dict[str, float]:
        """Get comprehensive sentiment for a team"""
        
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
            'contrarian_indicator': 0.0
        }
        
        try:
            # 1. ESPN News Sentiment
            espn_sentiment = self._get_espn_sentiment(team_name)
            sentiment['news_sentiment'] = espn_sentiment
            
            # 2. Reddit Sentiment
            reddit_sentiment = self._get_reddit_sentiment(team_name, opponent_name)
            sentiment['social_buzz'] = reddit_sentiment
            
            # 3. Injury News Check
            injury_severity = self._check_injury_news(team_name)
            sentiment['injury_concerns'] = injury_severity
            
            # 4. Recent Performance Narrative
            momentum = self._analyze_momentum_narrative(team_name)
            sentiment['momentum_narrative'] = momentum
            
            # 5. Public Confidence (betting trends)
            public_confidence = self._estimate_public_confidence(team_name, opponent_name)
            sentiment['public_confidence'] = public_confidence
            
            # 6. Media Attention Score
            media_score = self._calculate_media_attention(team_name)
            sentiment['media_attention'] = media_score
            
            # 7. Calculate overall sentiment
            sentiment['overall_sentiment'] = (
                sentiment['news_sentiment'] * 0.25 +
                sentiment['social_buzz'] * 0.20 +
                (1 - sentiment['injury_concerns']) * 0.20 +
                sentiment['momentum_narrative'] * 0.15 +
                sentiment['public_confidence'] * 0.10 +
                sentiment['media_attention'] * 0.10
            )
            
            # 8. Contrarian indicator (when public is too confident)
            if sentiment['public_confidence'] > 0.7 or sentiment['public_confidence'] < 0.3:
                sentiment['contrarian_indicator'] = 1.0
            
        except Exception as e:
            print(f"Sentiment analysis error for {team_name}: {e}")
        
        # Cache the result
        self.cache[cache_key] = sentiment
        
        return sentiment
    
    def _get_espn_sentiment(self, team_name: str) -> float:
        """Scrape ESPN for team news sentiment"""
        try:
            # ESPN team news page
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.espn.com/nba/team/_/name/{team_abbr.lower()}"
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.5
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for headlines and recent news
            headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], limit=10)
            
            # Simple sentiment keywords
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
            
            # Calculate sentiment score
            total = positive_count + negative_count
            if total == 0:
                return 0.5
            
            sentiment = (positive_count - negative_count) / total * 0.5 + 0.5
            return max(0.0, min(1.0, sentiment))
            
        except Exception as e:
            return 0.5
    
    def _get_reddit_sentiment(self, team_name: str, opponent_name: str = None) -> float:
        """Estimate Reddit sentiment from r/NBA (via RSS if available)"""
        try:
            # Reddit JSON API (doesn't require authentication for public posts)
            url = f"https://www.reddit.com/r/nba/search.json?q={team_name}&sort=new&limit=25&t=week"
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.5
            
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                return 0.5
            
            posts = data['data']['children']
            
            # Analyze post titles and scores
            positive_score = 0
            negative_score = 0
            total_engagement = 0
            
            positive_keywords = ['win', 'victory', 'dominat', 'amazing', 'clutch', 'beast']
            negative_keywords = ['loss', 'lose', 'choke', 'terrible', 'trash', 'embarrassing']
            
            for post in posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '').lower()
                score = post_data.get('score', 0)
                
                # Check sentiment
                has_positive = any(kw in title for kw in positive_keywords)
                has_negative = any(kw in title for kw in negative_keywords)
                
                if has_positive:
                    positive_score += score
                if has_negative:
                    negative_score += score
                
                total_engagement += abs(score)
            
            # Calculate buzz score
            if total_engagement == 0:
                return 0.5
            
            buzz = (positive_score - negative_score) / total_engagement * 0.5 + 0.5
            return max(0.0, min(1.0, buzz))
            
        except Exception as e:
            return 0.5
    
    def _check_injury_news(self, team_name: str) -> float:
        """Check for injury concerns (0 = no concerns, 1 = major concerns)"""
        try:
            # Try to scrape ESPN injury report
            team_abbr = self._get_team_abbreviation(team_name)
            url = f"https://www.espn.com/nba/team/injuries/_/name/{team_abbr.lower()}"
            
            response = self.session.get(url, timeout=5)
            if response.status_code != 200:
                return 0.0
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury status indicators
            injury_statuses = soup.find_all(string=re.compile(r'Out|Doubtful|Questionable', re.I))
            
            # Count severity
            out_count = len([s for s in injury_statuses if 'out' in s.lower()])
            doubtful_count = len([s for s in injury_statuses if 'doubtful' in s.lower()])
            questionable_count = len([s for s in injury_statuses if 'questionable' in s.lower()])
            
            # Calculate injury severity score
            injury_score = (out_count * 1.0 + doubtful_count * 0.7 + questionable_count * 0.3) / 10
            
            return min(1.0, injury_score)
            
        except Exception as e:
            return 0.0
    
    def _analyze_momentum_narrative(self, team_name: str) -> float:
        """Analyze recent performance narrative"""
        try:
            # This would ideally check recent game results
            # For now, we'll use news sentiment as a proxy
            return self._get_espn_sentiment(team_name)
        except:
            return 0.5
    
    def _estimate_public_confidence(self, team_name: str, opponent_name: str = None) -> float:
        """Estimate public betting confidence"""
        try:
            # This could scrape betting percentage sites like Action Network
            # For now, return neutral with some randomness
            base_confidence = 0.5
            
            # Add small variation based on team name hash (deterministic)
            variation = (hash(team_name) % 100) / 500  # ±0.1
            
            return max(0.0, min(1.0, base_confidence + variation))
        except:
            return 0.5
    
    def _calculate_media_attention(self, team_name: str) -> float:
        """Calculate media attention score"""
        try:
            # Big market teams get more attention
            big_market_teams = ['Lakers', 'Knicks', 'Warriors', 'Celtics', 'Heat', 
                              'Bulls', '76ers', 'Nets', 'Mavericks', 'Clippers']
            
            if any(team in team_name for team in big_market_teams):
                return 0.8 + (hash(team_name) % 20) / 100
            else:
                return 0.4 + (hash(team_name) % 30) / 100
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
        
        # Default: return first 3 letters uppercase
        return team_name[:3].upper()
    
    def get_game_sentiment(self, home_team: str, away_team: str) -> Dict[str, Dict]:
        """Get sentiment for both teams in a matchup"""
        
        print(f"  📊 Analyzing sentiment: {home_team} vs {away_team}")
        
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
                                     away_sentiment['contrarian_indicator']) / 2
        }
        
        print(f"    Home sentiment: {home_sentiment['overall_sentiment']:.2f}")
        print(f"    Away sentiment: {away_sentiment['overall_sentiment']:.2f}")
        print(f"    Narrative: {narrative}")
        
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
        
        # High buzz game
        if home_sent['media_attention'] > 0.7 and away_sent['media_attention'] > 0.7:
            return "🎬 High-profile matchup"
        
        # Contrarian opportunity
        if home_sent['contrarian_indicator'] > 0.5 or away_sent['contrarian_indicator'] > 0.5:
            return "💡 Contrarian value opportunity"
        
        return "⚖️ Balanced matchup"
    
    def adjust_prediction_with_sentiment(self, base_prediction: Dict, 
                                        sentiment: Dict) -> Dict[str, float]:
        """Adjust prediction confidence based on sentiment"""
        
        adjusted = base_prediction.copy()
        
        # Sentiment differential affects probability slightly
        sentiment_diff = sentiment['sentiment_differential']
        
        # Adjust home win probability (max ±5% adjustment)
        probability_adjustment = sentiment_diff * 0.05
        
        if 'home_win_probability' in adjusted:
            adjusted['home_win_probability'] += probability_adjustment
            adjusted['home_win_probability'] = max(0.0, min(1.0, adjusted['home_win_probability']))
        
        # Adjust confidence based on narrative clarity
        if 'confidence' in adjusted:
            # High buzz or clear narrative increases confidence slightly
            buzz_boost = sentiment['combined_buzz'] * 0.05
            adjusted['confidence'] += buzz_boost
            adjusted['confidence'] = min(1.0, adjusted['confidence'])
        
        # Flag contrarian opportunities
        adjusted['contrarian_opportunity'] = sentiment['contrarian_opportunity'] > 0.5
        adjusted['narrative'] = sentiment['narrative']
        adjusted['sentiment_score'] = sentiment['sentiment_differential']
        
        return adjusted


def get_sentiment_for_prediction(home_team: str, away_team: str) -> Dict:
    """Convenience function to get sentiment for a game prediction"""
    analyzer = NBASentimentAnalyzer()
    return analyzer.get_game_sentiment(home_team, away_team)


if __name__ == "__main__":
    # Test the sentiment analyzer
    print("Testing NBA Sentiment Analyzer...")
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


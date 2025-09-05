"""
Real-time data provider for live NBA data including injuries, lineups, and market data.
Integrates with multiple APIs to enhance prediction accuracy.
"""
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Import configuration manager
try:
    from src.Utils.ConfigManager import get_config
except ImportError:
    # Fallback if running from different directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../Utils'))
    from ConfigManager import get_config

class RealTimeDataProvider:
    def __init__(self):
        # Load configuration
        self.config = get_config()
        self.api_keys = self.config.config['api_keys']
        
        # Get available services
        self.available_services = self.config.get_available_services()
        
        # Print available services on initialization
        print(f"🔧 RealTimeDataProvider initialized with {sum(self.available_services.values())} available services")
        
        self.cache = {}
        self.cache_duration = self.config.get_database_config()['cache_duration']
        
        # API endpoints from configuration
        self.endpoints = {
            'nba_stats': self.config.get_endpoint('nba_stats'),
            'the_odds_api': self.config.get_endpoint('the_odds_api'),
            'sportsradar': self.config.get_endpoint('sportsradar'),
            'rapidapi': self.config.get_endpoint('rapidapi'),
            'news_api': self.config.get_endpoint('news_api')
        }
        
        # Market intelligence thresholds
        self.market_thresholds = {
            'reverse_line_movement': 0.05,  # 5% line movement against public
            'steam_move': 0.02,             # 2% rapid line movement
            'sharp_money': 0.60,            # 60% of money on minority side
            'public_fade': 0.80             # 80%+ public on one side
        }
        
    def get_cached_data(self, key: str):
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
        return None
    
    def set_cached_data(self, key: str, data):
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_injury_report(self, team: str = None, date: datetime = None) -> Dict:
        """Get current injury report for teams"""
        cache_key = f"injuries_{team}_{date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Connect to NBA injury API
            injury_data = self._fetch_injury_data_from_api(team, date)
            if not injury_data:
                # Return empty structure if no API data available
                injury_data = {
                    'team': team,
                    'date': date or datetime.now(),
                    'injured_players': [],
                    'total_salary_impact': 0,
                    'key_players_out': 0,
                    'defensive_impact_score': 0,
                    'offensive_impact_score': 0
                }
            
            # Calculate impact scores
            for player in injury_data['injured_players']:
                if player['status'] in ['Out', 'Doubtful']:
                    injury_data['total_salary_impact'] += player['salary_impact']
                    if player['usage_rate'] > 0.20:  # Key player threshold
                        injury_data['key_players_out'] += 1
                    
                    # Impact on team ratings
                    injury_data['defensive_impact_score'] += player['usage_rate'] * 0.5
                    injury_data['offensive_impact_score'] += player['usage_rate'] * 0.7
            
            self.set_cached_data(cache_key, injury_data)
            return injury_data
            
        except Exception as e:
            print(f"Error fetching injury data: {e}")
            return {'team': team, 'injured_players': [], 'total_salary_impact': 0}
    
    def get_starting_lineups(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Get confirmed starting lineups"""
        cache_key = f"lineups_{home_team}_{away_team}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Fetch real lineup data from NBA API
            lineup_data = self._fetch_lineup_data_from_api(home_team, away_team, game_date)
            if not lineup_data:
                # Return empty structure if no API data available
                lineup_data = {
                    'game_date': game_date or datetime.now(),
                    'lineups_confirmed': False,
                    'home_team': {
                        'team': home_team,
                        'starters': [],
                        'total_salary': 0,
                        'avg_usage_rate': 0,
                        'lineup_rating': 0.5
                    },
                    'away_team': {
                        'team': away_team,
                        'starters': [],
                        'total_salary': 0,
                        'avg_usage_rate': 0,
                        'lineup_rating': 0.5
                    }
                }
            
            self.set_cached_data(cache_key, lineup_data)
            return lineup_data
            
        except Exception as e:
            print(f"Error fetching lineup data: {e}")
            return {'lineups_confirmed': False}
    
    def get_betting_market_data(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Get real-time betting market data and line movements"""
        cache_key = f"betting_{home_team}_{away_team}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Fetch real betting market data from odds API
            market_data = self._fetch_betting_market_data_from_api(home_team, away_team, game_date)
            if not market_data:
                # Return empty structure if no API data available
                market_data = {
                    'game_date': game_date or datetime.now(),
                    'moneyline': {
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': {'current': 0, 'opening': 0, 'movement': 0, 'bet_percentage': 50, 'money_percentage': 50},
                        'away_odds': {'current': 0, 'opening': 0, 'movement': 0, 'bet_percentage': 50, 'money_percentage': 50}
                    },
                    'spread': {
                        'current_line': 0, 'opening_line': 0, 'movement': 0,
                        'home_bet_percentage': 50, 'away_bet_percentage': 50,
                        'reverse_line_movement': False
                    },
                    'total': {
                        'current_line': 220, 'opening_line': 220, 'movement': 0,
                        'over_bet_percentage': 50, 'under_bet_percentage': 50,
                        'steam_move': False
                    },
                    'market_sentiment': {
                        'public_bias': 'neutral', 'sharp_money': 'neutral',
                        'contrarian_opportunity': False, 'line_value': 0.5
                    }
                }
            
            self.set_cached_data(cache_key, market_data)
            return market_data
            
        except Exception as e:
            print(f"Error fetching betting market data: {e}")
            return {}
    
    
    def get_team_travel_schedule(self, team: str, game_date: datetime) -> Dict:
        """Get team's recent travel schedule and fatigue factors"""
        cache_key = f"travel_{team}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Calculate travel data from team schedule
            travel_data = self._calculate_travel_data_from_schedule(team, game_date)
            if not travel_data:
                # Return neutral travel data if no schedule available
                travel_data = {
                    'team': team,
                    'current_game_date': game_date,
                    'last_game_date': game_date - timedelta(days=2),
                    'last_game_location': 'Unknown',
                    'current_location': 'Unknown',
                    'miles_traveled': 0,
                    'time_zones_crossed': 0,
                    'days_rest': 2,
                    'back_to_back': False,
                    'games_in_last_week': 3,
                    'home_games_last_10': 5,
                    'road_games_last_10': 5,
                    'fatigue_score': 0.0,
                    'travel_advantage': 0.0
                }
            
            # Calculate fatigue score
            fatigue_factors = [
                travel_data['miles_traveled'] / 3000 * 0.3,  # Distance factor
                max(0, (4 - travel_data['days_rest'])) / 4 * 0.4,  # Rest factor
                travel_data['games_in_last_week'] / 7 * 0.3  # Game density
            ]
            travel_data['fatigue_score'] = min(1.0, sum(fatigue_factors))
            
            self.set_cached_data(cache_key, travel_data)
            return travel_data
            
        except Exception as e:
            print(f"Error fetching travel data: {e}")
            return {'fatigue_score': 0, 'travel_advantage': 0}
    
    def get_referee_assignments(self, game_date: datetime, home_team: str = None, away_team: str = None) -> Dict:
        """Get referee assignments and their historical impact"""
        cache_key = f"refs_{game_date}_{home_team}_{away_team}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Fetch referee assignments from NBA API
            ref_data = self._fetch_referee_data_from_api(game_date, home_team, away_team)
            if not ref_data:
                # Return neutral referee data if no API data
                ref_data = {
                    'game_date': game_date,
                    'crew_chief': 'Unknown',
                    'referee_1': 'Unknown',
                    'referee_2': 'Unknown',
                    'historical_stats': {
                        'avg_total_fouls': 42.0,
                        'avg_technical_fouls': 1.0,
                        'home_team_foul_bias': 0.0,
                        'over_under_bias': 0.0,
                        'pace_impact': 1.0,
                        'tight_game_calls': 0.5
                    },
                    'referee_rating': 0.5
                }
            
            self.set_cached_data(cache_key, ref_data)
            return ref_data
            
        except Exception as e:
            print(f"Error fetching referee data: {e}")
            return {'historical_stats': {}, 'referee_rating': 0.5}
    
    def get_social_sentiment(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Get social media sentiment and buzz around the game"""
        cache_key = f"sentiment_{home_team}_{away_team}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Fetch social sentiment from various APIs
            sentiment_data = self._fetch_sentiment_data_from_api(home_team, away_team, game_date)
            if not sentiment_data:
                # Return neutral sentiment if no API data
                sentiment_data = {
                    'home_team_sentiment': 0.5,
                    'away_team_sentiment': 0.5,
                    'game_buzz_score': 0.5,
                    'public_betting_sentiment': {
                        'home_team_confidence': 0.5,
                        'away_team_confidence': 0.5,
                        'total_confidence': 0.5
                    },
                    'media_coverage': {
                        'articles_count': 0,
                        'positive_coverage_home': 0.5,
                        'positive_coverage_away': 0.5,
                        'injury_concern_mentions': 0,
                        'revenge_game_narrative': False,
                        'playoff_implications': False
                    },
                    'contrarian_indicator': 0.0
                }
            
            self.set_cached_data(cache_key, sentiment_data)
            return sentiment_data
            
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            return {}
    
    def get_comprehensive_game_data(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Get all available real-time data for a game"""
        if not game_date:
            game_date = datetime.now()
            
        print(f"Fetching real-time data for {away_team} @ {home_team}...")
        
        # Gather all data sources
        comprehensive_data = {
            'game_info': {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                'data_timestamp': datetime.now()
            },
            'injuries': {
                'home_team': self.get_injury_report(home_team, game_date),
                'away_team': self.get_injury_report(away_team, game_date)
            },
            'lineups': self.get_starting_lineups(home_team, away_team, game_date),
            'betting_markets': self.get_betting_market_data(home_team, away_team, game_date),
            'travel': {
                'home_team': self.get_team_travel_schedule(home_team, game_date),
                'away_team': self.get_team_travel_schedule(away_team, game_date)
            },
            'officials': self.get_referee_assignments(game_date, home_team, away_team),
            'sentiment': self.get_social_sentiment(home_team, away_team, game_date)
        }
        
        # Calculate composite scores
        comprehensive_data['composite_scores'] = self.calculate_composite_scores(comprehensive_data)
        
        # Add advanced market intelligence
        if comprehensive_data['betting_markets']:
            comprehensive_data['market_intelligence'] = self.analyze_market_intelligence(
                comprehensive_data['betting_markets']
            )
        
        # Add injury severity scores
        comprehensive_data['injury_scores'] = {
            'home_team': self.get_injury_severity_score(comprehensive_data['injuries']['home_team']),
            'away_team': self.get_injury_severity_score(comprehensive_data['injuries']['away_team'])
        }
        
        # Add lineup strength differential
        comprehensive_data['lineup_differential'] = self.calculate_lineup_strength_differential(
            comprehensive_data['lineups']
        )
        
        
        return comprehensive_data
    
    def calculate_composite_scores(self, game_data: Dict) -> Dict:
        """Calculate composite impact scores from all data sources"""
        scores = {
            'home_team_advantage': 0.0,
            'away_team_advantage': 0.0,
            'total_points_adjustment': 0.0,
            'betting_value_score': 0.0,
            'confidence_score': 0.0
        }
        
        try:
            # Injury impact
            home_injury_impact = game_data['injuries']['home_team'].get('offensive_impact_score', 0)
            away_injury_impact = game_data['injuries']['away_team'].get('offensive_impact_score', 0)
            
            scores['home_team_advantage'] -= home_injury_impact * 0.3
            scores['away_team_advantage'] -= away_injury_impact * 0.3
            
            # Travel fatigue
            home_fatigue = game_data['travel']['home_team'].get('fatigue_score', 0)
            away_fatigue = game_data['travel']['away_team'].get('fatigue_score', 0)
            
            scores['home_team_advantage'] -= home_fatigue * 0.2
            scores['away_team_advantage'] -= away_fatigue * 0.2
            
            # Lineup quality
            if game_data['lineups'].get('lineups_confirmed'):
                home_lineup_rating = game_data['lineups']['home_team'].get('lineup_rating', 0.5)
                away_lineup_rating = game_data['lineups']['away_team'].get('lineup_rating', 0.5)
                
                scores['home_team_advantage'] += (home_lineup_rating - 0.5) * 0.4
                scores['away_team_advantage'] += (away_lineup_rating - 0.5) * 0.4
            
            # Betting market insights
            if 'betting_markets' in game_data and game_data['betting_markets']:
                market = game_data['betting_markets']
                if 'market_sentiment' in market:
                    scores['betting_value_score'] = market['market_sentiment'].get('line_value', 0)
                    if market['market_sentiment'].get('contrarian_opportunity'):
                        scores['betting_value_score'] += 0.2
            
            # Referee impact on totals
            if 'officials' in game_data and 'historical_stats' in game_data['officials']:
                ref_stats = game_data['officials']['historical_stats']
                ou_bias = ref_stats.get('over_under_bias', 0)
                scores['total_points_adjustment'] = ou_bias
            
            # Overall confidence based on data completeness
            data_completeness = sum([
                1 if game_data['injuries']['home_team'].get('injured_players') is not None else 0,
                1 if game_data['lineups'].get('lineups_confirmed') else 0,
                1 if game_data['betting_markets'] else 0,
                1 if game_data['travel']['home_team'].get('fatigue_score') is not None else 0,
                1 if game_data['officials'].get('referee_rating') is not None else 0
            ]) / 5
            
            scores['confidence_score'] = data_completeness
            
        except Exception as e:
            print(f"Error calculating composite scores: {e}")
        
        return scores
    
    def analyze_market_intelligence(self, betting_data: Dict) -> Dict:
        """Advanced market intelligence analysis"""
        intelligence = {
            'sharp_money_indicators': [],
            'public_fade_opportunities': [],
            'line_movement_signals': [],
            'market_efficiency_score': 0.5,
            'contrarian_value': 0.0
        }
        
        if not betting_data:
            return intelligence
        
        try:
            # Analyze moneyline market
            if 'moneyline' in betting_data:
                ml_data = betting_data['moneyline']
                
                # Sharp money detection
                home_odds = ml_data.get('home_odds', {})
                away_odds = ml_data.get('away_odds', {})
                
                home_bet_pct = home_odds.get('bet_percentage', 50)
                home_money_pct = home_odds.get('money_percentage', 50)
                
                # Sharp money indicator: Money % significantly different from bet %
                if abs(home_money_pct - home_bet_pct) > 15:
                    if home_money_pct > home_bet_pct:
                        intelligence['sharp_money_indicators'].append('Sharp money on home team')
                    else:
                        intelligence['sharp_money_indicators'].append('Sharp money on away team')
                
                # Public fade opportunity
                if home_bet_pct > self.market_thresholds['public_fade']:
                    intelligence['public_fade_opportunities'].append('Fade public on home team')
                elif home_bet_pct < (100 - self.market_thresholds['public_fade']):
                    intelligence['public_fade_opportunities'].append('Fade public on away team')
            
            # Analyze spread market
            if 'spread' in betting_data:
                spread_data = betting_data['spread']
                
                # Reverse line movement detection
                if spread_data.get('reverse_line_movement', False):
                    intelligence['line_movement_signals'].append('Reverse line movement detected')
                
                movement = abs(spread_data.get('movement', 0))
                if movement > self.market_thresholds['reverse_line_movement']:
                    intelligence['line_movement_signals'].append(f'Significant line movement: {movement}')
            
            # Analyze total market
            if 'total' in betting_data:
                total_data = betting_data['total']
                
                if total_data.get('steam_move', False):
                    intelligence['line_movement_signals'].append('Steam move on total detected')
                
                over_pct = total_data.get('over_bet_percentage', 50)
                if over_pct > self.market_thresholds['public_fade']:
                    intelligence['public_fade_opportunities'].append('Consider under bet')
                elif over_pct < (100 - self.market_thresholds['public_fade']):
                    intelligence['public_fade_opportunities'].append('Consider over bet')
            
            # Calculate overall market efficiency score
            signal_count = len(intelligence['sharp_money_indicators']) + len(intelligence['line_movement_signals'])
            intelligence['market_efficiency_score'] = max(0.1, min(0.9, 0.5 - signal_count * 0.1))
            
            # Calculate contrarian value
            fade_opportunities = len(intelligence['public_fade_opportunities'])
            intelligence['contrarian_value'] = min(1.0, fade_opportunities * 0.3)
            
        except Exception as e:
            print(f"Error analyzing market intelligence: {e}")
        
        return intelligence
    
    def get_injury_severity_score(self, injury_data: Dict) -> float:
        """Calculate injury severity impact score"""
        if not injury_data or not injury_data.get('injured_players'):
            return 0.0
        
        severity_score = 0.0
        
        for player in injury_data['injured_players']:
            usage_rate = player.get('usage_rate', 0)
            status = player.get('status', 'Available')
            
            # Weight by player importance and injury status
            if status == 'Out':
                severity_score += usage_rate * 1.0
            elif status == 'Doubtful':
                severity_score += usage_rate * 0.7
            elif status == 'Questionable':
                severity_score += usage_rate * 0.3
        
        return min(1.0, severity_score)  # Cap at 1.0
    
    def calculate_lineup_strength_differential(self, lineup_data: Dict) -> float:
        """Calculate strength differential between starting lineups"""
        if not lineup_data or not lineup_data.get('lineups_confirmed'):
            return 0.0
        
        home_rating = lineup_data.get('home_team', {}).get('lineup_rating', 0.5)
        away_rating = lineup_data.get('away_team', {}).get('lineup_rating', 0.5)
        
        return home_rating - away_rating
    
    
    def _fetch_injury_data_from_api(self, team: str, date: datetime = None) -> Dict:
        """Fetch real injury data from NBA API or other sources"""
        try:
            # Try SportsRadar API first (most reliable for injury data)
            if self.available_services.get('sportsradar'):
                return self._fetch_sportsradar_injuries(team, date)
            
            # Try RapidAPI as backup
            elif self.available_services.get('rapidapi'):
                return self._fetch_rapidapi_injuries(team, date)
            
            # NBA Stats API (free but limited injury info)
            elif self.available_services.get('nba_stats'):
                return self._fetch_nba_stats_injuries(team, date)
                
            return None  # No API services available
            
        except Exception as e:
            print(f"Error fetching injury data from API: {e}")
            return None
    
    def _fetch_lineup_data_from_api(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Fetch real starting lineup data from NBA API"""
        try:
            # NBA Stats API for starting lineups
            if self.api_keys.get('nba_api'):
                # This would make actual API calls for confirmed lineups
                pass
                
            # Alternative: RotoWire or other lineup sources
            if self.api_keys.get('lineup_api'):
                # Lineup confirmation API call
                pass
                
            return None  # No API data available
            
        except Exception as e:
            print(f"Error fetching lineup data from API: {e}")
            return None
    
    def _fetch_betting_market_data_from_api(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Fetch real betting market data from odds APIs"""
        try:
            # The Odds API for betting lines
            if self.available_services.get('the_odds_api'):
                return self._fetch_the_odds_api_data(home_team, away_team, game_date)
                
            return None  # No API data available
            
        except Exception as e:
            print(f"Error fetching betting market data from API: {e}")
            return None
    
    
    def _calculate_travel_data_from_schedule(self, team: str, game_date: datetime) -> Dict:
        """Calculate travel metrics from team schedule data"""
        try:
            # This would analyze team schedule from NBA API or database
            # Calculate miles traveled, time zones, rest days, etc.
            return None  # No schedule data available
            
        except Exception as e:
            print(f"Error calculating travel data: {e}")
            return None
    
    def _fetch_referee_data_from_api(self, game_date: datetime, home_team: str = None, away_team: str = None) -> Dict:
        """Fetch referee assignments and statistics from NBA API"""
        try:
            if self.api_keys.get('nba_api'):
                # NBA API call for referee assignments
                pass
                
            return None  # No API data available
            
        except Exception as e:
            print(f"Error fetching referee data from API: {e}")
            return None
    
    def _fetch_sentiment_data_from_api(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Fetch social sentiment data from various APIs"""
        try:
            # News API for media coverage and sentiment
            if self.available_services.get('news_api'):
                return self._fetch_news_sentiment(home_team, away_team, game_date)
                
            return None  # No API data available
            
        except Exception as e:
            print(f"Error fetching sentiment data from API: {e}")
            return None
    
    # ===== REAL API IMPLEMENTATION METHODS =====
    
    def _fetch_sportsradar_injuries(self, team: str, date: datetime = None) -> Dict:
        """Fetch injury data from SportsRadar API"""
        try:
            api_key = self.api_keys.get('sportsradar')
            if not api_key:
                return None
            
            # SportsRadar injury report endpoint
            url = f"{self.endpoints['sportsradar']}/injuries.json"
            params = {'api_key': api_key}
            
            headers = {
                'User-Agent': 'NBA-ML-Predictor/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_sportsradar_injuries(data, team)
            else:
                print(f"SportsRadar API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching SportsRadar injury data: {e}")
            return None
    
    def _fetch_rapidapi_injuries(self, team: str, date: datetime = None) -> Dict:
        """Fetch injury data from RapidAPI"""
        try:
            api_key = self.api_keys.get('rapidapi')
            if not api_key:
                return None
            
            headers = {
                'X-RapidAPI-Key': api_key,
                'X-RapidAPI-Host': 'api-nba-v1.p.rapidapi.com'
            }
            
            # Get team injuries
            url = f"{self.endpoints['rapidapi']}/players"
            params = {'team': self._get_team_id(team), 'season': '2024'}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_rapidapi_injuries(data, team)
            else:
                print(f"RapidAPI error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching RapidAPI injury data: {e}")
            return None
    
    def _fetch_nba_stats_injuries(self, team: str, date: datetime = None) -> Dict:
        """Fetch injury data from NBA Stats API (limited)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com'
            }
            
            # NBA Stats doesn't have direct injury endpoint, use team roster
            url = f"{self.endpoints['nba_stats']}/commonteamroster"
            params = {
                'TeamID': self._get_nba_team_id(team),
                'Season': '2024-25'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_nba_stats_roster(data, team)
            else:
                print(f"NBA Stats API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching NBA Stats data: {e}")
            return None
    
    def _fetch_the_odds_api_data(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Fetch betting odds from The Odds API"""
        try:
            api_key = self.api_keys.get('the_odds_api')
            if not api_key:
                return None
            
            # The Odds API endpoint for NBA
            url = f"{self.endpoints['the_odds_api']}/sports/basketball_nba/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_odds_data(data, home_team, away_team)
            else:
                print(f"The Odds API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching odds data: {e}")
            return None
    
    def _fetch_news_sentiment(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """Fetch news sentiment from News API"""
        try:
            api_key = self.api_keys.get('news_api')
            if not api_key:
                return None
            
            # Search for recent news about both teams
            query = f'"{home_team}" OR "{away_team}" NBA basketball'
            url = f"{self.endpoints['news_api']}/everything"
            
            params = {
                'apiKey': api_key,
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(days=7)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._analyze_news_sentiment(data, home_team, away_team)
            else:
                print(f"News API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return None
    
    # ===== HELPER METHODS FOR API DATA PARSING =====
    
    def _get_team_id(self, team_name: str) -> str:
        """Convert team name to API team ID"""
        # This would contain mappings for different APIs
        team_mappings = {
            'Boston Celtics': 'BOS',
            'Los Angeles Lakers': 'LAL',
            # Add more mappings as needed
        }
        return team_mappings.get(team_name, team_name)
    
    def _get_nba_team_id(self, team_name: str) -> str:
        """Get NBA Stats API team ID"""
        # NBA Stats uses numeric IDs
        nba_team_ids = {
            'Boston Celtics': '1610612738',
            'Los Angeles Lakers': '1610612747',
            # Add more mappings as needed
        }
        return nba_team_ids.get(team_name, '1610612738')  # Default to Celtics
    
    def _parse_sportsradar_injuries(self, data: Dict, team: str) -> Dict:
        """Parse SportsRadar injury data"""
        # Implementation would parse the actual API response
        return {
            'team': team,
            'date': datetime.now(),
            'injured_players': [],
            'total_salary_impact': 0,
            'key_players_out': 0,
            'defensive_impact_score': 0,
            'offensive_impact_score': 0
        }
    
    def _parse_rapidapi_injuries(self, data: Dict, team: str) -> Dict:
        """Parse RapidAPI injury data"""
        # Implementation would parse the actual API response
        return {
            'team': team,
            'date': datetime.now(),
            'injured_players': [],
            'total_salary_impact': 0,
            'key_players_out': 0,
            'defensive_impact_score': 0,
            'offensive_impact_score': 0
        }
    
    def _parse_nba_stats_roster(self, data: Dict, team: str) -> Dict:
        """Parse NBA Stats roster data"""
        # Implementation would parse the actual API response
        return {
            'team': team,
            'date': datetime.now(),
            'injured_players': [],
            'total_salary_impact': 0,
            'key_players_out': 0,
            'defensive_impact_score': 0,
            'offensive_impact_score': 0
        }
    
    def _parse_odds_data(self, data: Dict, home_team: str, away_team: str) -> Dict:
        """Parse odds data from The Odds API"""
        # Implementation would parse the actual API response
        return {
            'game_date': datetime.now(),
            'moneyline': {
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': {'current': 0, 'opening': 0, 'movement': 0, 'bet_percentage': 50, 'money_percentage': 50},
                'away_odds': {'current': 0, 'opening': 0, 'movement': 0, 'bet_percentage': 50, 'money_percentage': 50}
            },
            'spread': {
                'current_line': 0, 'opening_line': 0, 'movement': 0,
                'home_bet_percentage': 50, 'away_bet_percentage': 50,
                'reverse_line_movement': False
            },
            'total': {
                'current_line': 220, 'opening_line': 220, 'movement': 0,
                'over_bet_percentage': 50, 'under_bet_percentage': 50,
                'steam_move': False
            },
            'market_sentiment': {
                'public_bias': 'neutral', 'sharp_money': 'neutral',
                'contrarian_opportunity': False, 'line_value': 0.5
            }
        }
    
    def _analyze_news_sentiment(self, data: Dict, home_team: str, away_team: str) -> Dict:
        """Analyze news sentiment"""
        # Implementation would use NLP to analyze sentiment
        return {
            'home_team_sentiment': 0.5,
            'away_team_sentiment': 0.5,
            'game_buzz_score': 0.5,
            'public_betting_sentiment': {
                'home_team_confidence': 0.5,
                'away_team_confidence': 0.5,
                'total_confidence': 0.5
            },
            'media_coverage': {
                'articles_count': 0,
                'positive_coverage_home': 0.5,
                'positive_coverage_away': 0.5,
                'injury_concern_mentions': 0,
                'revenge_game_narrative': False,
                'playoff_implications': False
            },
            'contrarian_indicator': 0.0
        }

if __name__ == "__main__":
    # Test the real-time data provider
    provider = RealTimeDataProvider()
    
    # Example usage
    game_data = provider.get_comprehensive_game_data(
        home_team="Boston Celtics",
        away_team="Los Angeles Lakers",
        game_date=datetime.now()
    )
    
    print("Real-time data collected:")
    print(f"Home team advantage: {game_data['composite_scores']['home_team_advantage']:+.3f}")
    print(f"Away team advantage: {game_data['composite_scores']['away_team_advantage']:+.3f}")
    print(f"Total points adjustment: {game_data['composite_scores']['total_points_adjustment']:+.1f}")
    print(f"Betting value score: {game_data['composite_scores']['betting_value_score']:.3f}")
    print(f"Data confidence: {game_data['composite_scores']['confidence_score']:.1%}")

"""
Real-time data provider for live NBA data including injuries, lineups, weather, and market data.
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

class RealTimeDataProvider:
    def __init__(self):
        self.api_keys = {
            # Add your API keys here
            'nba_api': None,  # NBA official API
            'sports_radar': None,  # SportsRadar API
            'the_odds_api': None,  # The Odds API
            'weather_api': None,  # Weather API
            'injury_api': None   # Injury API
        }
        
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
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
            # This would connect to a real injury API
            # For now, return mock data structure
            injury_data = {
                'team': team,
                'date': date or datetime.now(),
                'injured_players': [
                    {
                        'player_name': 'Mock Player',
                        'position': 'PG',
                        'injury_type': 'Knee',
                        'status': 'Questionable',
                        'games_missed': 2,
                        'salary_impact': 8500000,  # Annual salary
                        'usage_rate': 0.28,
                        'defensive_rating': 112.5,
                        'offensive_rating': 118.3
                    }
                ],
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
            # Mock lineup data structure
            lineup_data = {
                'game_date': game_date or datetime.now(),
                'lineups_confirmed': False,
                'home_team': {
                    'team': home_team,
                    'starters': [
                        {'name': 'Mock Player 1', 'position': 'PG', 'salary': 25000000, 'usage_rate': 0.28},
                        {'name': 'Mock Player 2', 'position': 'SG', 'salary': 18000000, 'usage_rate': 0.22},
                        {'name': 'Mock Player 3', 'position': 'SF', 'salary': 15000000, 'usage_rate': 0.18},
                        {'name': 'Mock Player 4', 'position': 'PF', 'salary': 12000000, 'usage_rate': 0.16},
                        {'name': 'Mock Player 5', 'position': 'C', 'salary': 20000000, 'usage_rate': 0.24}
                    ],
                    'total_salary': 90000000,
                    'avg_usage_rate': 0.216,
                    'lineup_rating': 0.85  # 0-1 scale
                },
                'away_team': {
                    'team': away_team,
                    'starters': [
                        {'name': 'Mock Player A', 'position': 'PG', 'salary': 22000000, 'usage_rate': 0.26},
                        {'name': 'Mock Player B', 'position': 'SG', 'salary': 16000000, 'usage_rate': 0.20},
                        {'name': 'Mock Player C', 'position': 'SF', 'salary': 18000000, 'usage_rate': 0.19},
                        {'name': 'Mock Player D', 'position': 'PF', 'salary': 14000000, 'usage_rate': 0.17},
                        {'name': 'Mock Player E', 'position': 'C', 'salary': 19000000, 'usage_rate': 0.23}
                    ],
                    'total_salary': 89000000,
                    'avg_usage_rate': 0.210,
                    'lineup_rating': 0.82
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
            # Mock betting market data
            market_data = {
                'game_date': game_date or datetime.now(),
                'moneyline': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_odds': {
                        'current': -150,
                        'opening': -140,
                        'movement': -10,
                        'bet_percentage': 65.2,  # % of bets on home team
                        'money_percentage': 58.7  # % of money on home team
                    },
                    'away_odds': {
                        'current': +130,
                        'opening': +120,
                        'movement': +10,
                        'bet_percentage': 34.8,
                        'money_percentage': 41.3
                    }
                },
                'spread': {
                    'current_line': -3.5,
                    'opening_line': -3.0,
                    'movement': -0.5,
                    'home_bet_percentage': 52.1,
                    'away_bet_percentage': 47.9,
                    'reverse_line_movement': False  # Line moved against public money
                },
                'total': {
                    'current_line': 220.5,
                    'opening_line': 219.0,
                    'movement': +1.5,
                    'over_bet_percentage': 61.3,
                    'under_bet_percentage': 38.7,
                    'steam_move': True  # Sharp money indicator
                },
                'market_sentiment': {
                    'public_bias': 'home',  # Where public is betting
                    'sharp_money': 'away',  # Where sharp money is
                    'contrarian_opportunity': True,
                    'line_value': 0.73  # 0-1 scale, higher = better value
                }
            }
            
            self.set_cached_data(cache_key, market_data)
            return market_data
            
        except Exception as e:
            print(f"Error fetching betting market data: {e}")
            return {}
    
    def get_weather_conditions(self, city: str, game_date: datetime = None) -> Dict:
        """Get weather conditions (mainly for outdoor events, but can affect travel)"""
        cache_key = f"weather_{city}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Mock weather data
            weather_data = {
                'city': city,
                'date': game_date or datetime.now(),
                'temperature': 72,  # Fahrenheit
                'humidity': 45,     # Percentage
                'precipitation': 0, # Inches
                'wind_speed': 8,    # MPH
                'conditions': 'Clear',
                'travel_impact_score': 0.1,  # 0-1 scale, higher = more impact
                'arena_conditions': 'Normal'  # Indoor arenas less affected
            }
            
            self.set_cached_data(cache_key, weather_data)
            return weather_data
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {'travel_impact_score': 0}
    
    def get_team_travel_schedule(self, team: str, game_date: datetime) -> Dict:
        """Get team's recent travel schedule and fatigue factors"""
        cache_key = f"travel_{team}_{game_date}"
        cached = self.get_cached_data(cache_key)
        if cached:
            return cached
            
        try:
            # Mock travel data
            travel_data = {
                'team': team,
                'current_game_date': game_date,
                'last_game_date': game_date - timedelta(days=2),
                'last_game_location': 'Los Angeles',
                'current_location': 'Boston',
                'miles_traveled': 2600,
                'time_zones_crossed': 3,
                'days_rest': 2,
                'back_to_back': False,
                'games_in_last_week': 3,
                'home_games_last_10': 4,
                'road_games_last_10': 6,
                'fatigue_score': 0.35,  # 0-1 scale, higher = more fatigued
                'travel_advantage': -0.15  # -1 to 1, negative = disadvantage
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
            # Mock referee data
            ref_data = {
                'game_date': game_date,
                'crew_chief': 'John Doe',
                'referee_1': 'Jane Smith',
                'referee_2': 'Bob Johnson',
                'historical_stats': {
                    'avg_total_fouls': 42.3,
                    'avg_technical_fouls': 1.2,
                    'home_team_foul_bias': 0.02,  # Positive = home team gets more calls
                    'over_under_bias': -1.8,  # Points difference from average
                    'pace_impact': 0.95,  # Multiplier for game pace
                    'tight_game_calls': 0.73  # 0-1, how much they "let them play" in close games
                },
                'referee_rating': 0.82  # 0-1 scale, higher = more consistent/fair
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
            # Mock sentiment data
            sentiment_data = {
                'home_team_sentiment': 0.65,  # 0-1 scale, 0.5 = neutral
                'away_team_sentiment': 0.42,
                'game_buzz_score': 0.78,  # How much attention the game is getting
                'public_betting_sentiment': {
                    'home_team_confidence': 0.71,
                    'away_team_confidence': 0.29,
                    'total_confidence': 0.83  # Confidence in over/under
                },
                'media_coverage': {
                    'articles_count': 47,
                    'positive_coverage_home': 0.68,
                    'positive_coverage_away': 0.45,
                    'injury_concern_mentions': 12,
                    'revenge_game_narrative': False,
                    'playoff_implications': True
                },
                'contrarian_indicator': 0.23  # 0-1, higher = good fade opportunity
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
            'weather': self.get_weather_conditions(home_team.split()[-1], game_date),  # Use team city
            'travel': {
                'home_team': self.get_team_travel_schedule(home_team, game_date),
                'away_team': self.get_team_travel_schedule(away_team, game_date)
            },
            'officials': self.get_referee_assignments(game_date, home_team, away_team),
            'sentiment': self.get_social_sentiment(home_team, away_team, game_date)
        }
        
        # Calculate composite scores
        comprehensive_data['composite_scores'] = self.calculate_composite_scores(comprehensive_data)
        
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

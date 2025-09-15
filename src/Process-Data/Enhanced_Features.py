"""
Advanced Enhanced feature engineering for NBA prediction models.
Adds sophisticated metrics, team ratings, situational factors, market data,
and advanced statistical features for maximum model accuracy.
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngine:
    def __init__(self):
        self.team_elo_ratings = {}
        self.team_pace_cache = {}
        self.player_stats_cache = {}
        self.advanced_metrics_cache = {}
        self.team_ratings_cache = {}
        self.matchup_history = {}
        self.weather_cache = {}
        self.injury_cache = {}
        self.scaler = StandardScaler()
        
    def calculate_advanced_elo_ratings(self, games_df: pd.DataFrame, k_factor: float = 20) -> Dict[str, Dict[str, float]]:
        """Calculate advanced ELO ratings with multiple factors"""
        # Initialize all teams with multiple rating types
        elo_ratings = {}
        unique_teams = set(games_df['home_team'].unique()) | set(games_df['away_team'].unique())
        for team in unique_teams:
            elo_ratings[team] = {
                'overall': 1500.0,
                'home': 1500.0,
                'away': 1500.0,
                'offense': 1500.0,
                'defense': 1500.0,
                'recent': 1500.0
            }
            
        # Sort games by date
        games_df = games_df.sort_values('date')
        
        for _, game in games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_won = game['home_win']
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            total_points = home_score + away_score
            
            # Update different rating types
            for rating_type in ['overall', 'home', 'offense', 'defense', 'recent']:
                home_rating = elo_ratings[home_team][rating_type]
                away_rating = elo_ratings[away_team][rating_type]
                
                # Calculate expected probabilities
                home_expected = 1 / (1 + 10**((away_rating - home_rating) / 400))
                away_expected = 1 - home_expected
                
                # Update ratings based on outcome
                if home_won:
                    home_actual, away_actual = 1, 0
                else:
                    home_actual, away_actual = 0, 1
                    
                # Adjust k-factor based on rating type and margin
                margin_factor = min(2.0, abs(home_score - away_score) / 10.0 + 1.0)
                adjusted_k = k_factor * margin_factor
                
                elo_ratings[home_team][rating_type] += adjusted_k * (home_actual - home_expected)
                elo_ratings[away_team][rating_type] += adjusted_k * (away_actual - away_expected)
            
            # Update away-specific rating
            away_rating = elo_ratings[away_team]['away']
            home_rating = elo_ratings[home_team]['home']
            away_expected = 1 / (1 + 10**((home_rating - away_rating) / 400))
            away_actual = 1 if not home_won else 0
            elo_ratings[away_team]['away'] += k_factor * (away_actual - away_expected)
            
        return elo_ratings
    
    def calculate_elo_ratings(self, games_df: pd.DataFrame, k_factor: float = 20) -> Dict[str, float]:
        """Calculate ELO ratings for all teams based on game results"""
        advanced_ratings = self.calculate_advanced_elo_ratings(games_df, k_factor)
        return {team: ratings['overall'] for team, ratings in advanced_ratings.items()}
    
    def calculate_advanced_recent_form(self, team: str, date: datetime, games_df: pd.DataFrame, n_games: int = 10) -> Dict[str, float]:
        """Calculate advanced team form metrics with multiple time windows"""
        # Get team's recent games before the given date
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False)
        
        if len(team_games) == 0:
            return self._get_default_form_metrics()
        
        # Multiple time windows
        windows = [3, 5, 10, 15]
        form_metrics = {}
        
        for window in windows:
            window_games = team_games.head(window)
            if len(window_games) == 0:
                continue
                
            wins = 0
            margins = []
            scores = []
            opponent_strengths = []
            
            for _, game in window_games.iterrows():
                is_home = game['home_team'] == team
                if is_home:
                    won = game['home_win']
                    margin = game['home_score'] - game['away_score']
                    score = game['home_score']
                    opponent = game['away_team']
                else:
                    won = not game['home_win']
                    margin = game['away_score'] - game['home_score']
                    score = game['away_score']
                    opponent = game['home_team']
                    
                if won:
                    wins += 1
                margins.append(margin)
                scores.append(score)
                
                # Calculate opponent strength (simplified)
                opp_games = games_df[
                    ((games_df['home_team'] == opponent) | (games_df['away_team'] == opponent)) &
                    (games_df['date'] < date)
                ].head(10)
                if len(opp_games) > 0:
                    opp_wins = sum(1 for _, g in opp_games.iterrows() 
                                 if (g['home_team'] == opponent and g['home_win']) or 
                                    (g['away_team'] == opponent and not g['home_win']))
                    opponent_strengths.append(opp_wins / len(opp_games))
                else:
                    opponent_strengths.append(0.5)
            
            # Calculate metrics for this window
            win_rate = wins / len(window_games)
            avg_margin = np.mean(margins)
            avg_score = np.mean(scores)
            avg_opp_strength = np.mean(opponent_strengths)
            
            # Advanced metrics
            margin_std = np.std(margins) if len(margins) > 1 else 0
            score_consistency = 1 / (1 + margin_std)  # Higher is more consistent
            
            # Weighted by recency
            recency_weights = np.exp(-np.arange(len(window_games)) * 0.1)
            weighted_win_rate = np.average([1 if w else 0 for w in [game['home_win'] if game['home_team'] == team else not game['home_win'] 
                                                                   for _, game in window_games.iterrows()]], weights=recency_weights)
            
            form_metrics.update({
                f'recent_{window}_wins': win_rate,
                f'recent_{window}_margin': avg_margin,
                f'recent_{window}_score': avg_score,
                f'recent_{window}_consistency': score_consistency,
                f'recent_{window}_weighted_wins': weighted_win_rate,
                f'recent_{window}_opp_strength': avg_opp_strength
            })
        
        return form_metrics
    
    def _get_default_form_metrics(self) -> Dict[str, float]:
        """Return default form metrics when no games available"""
        return {
            'recent_3_wins': 0.5, 'recent_3_margin': 0, 'recent_3_score': 110,
            'recent_3_consistency': 0.5, 'recent_3_weighted_wins': 0.5, 'recent_3_opp_strength': 0.5,
            'recent_5_wins': 0.5, 'recent_5_margin': 0, 'recent_5_score': 110,
            'recent_5_consistency': 0.5, 'recent_5_weighted_wins': 0.5, 'recent_5_opp_strength': 0.5,
            'recent_10_wins': 0.5, 'recent_10_margin': 0, 'recent_10_score': 110,
            'recent_10_consistency': 0.5, 'recent_10_weighted_wins': 0.5, 'recent_10_opp_strength': 0.5,
            'recent_15_wins': 0.5, 'recent_15_margin': 0, 'recent_15_score': 110,
            'recent_15_consistency': 0.5, 'recent_15_weighted_wins': 0.5, 'recent_15_opp_strength': 0.5
        }
    
    def calculate_recent_form(self, team: str, date: datetime, games_df: pd.DataFrame, n_games: int = 10) -> Dict[str, float]:
        """Calculate team's recent form metrics (backward compatibility)"""
        advanced_form = self.calculate_advanced_recent_form(team, date, games_df, n_games)
        return {
            'recent_wins': advanced_form.get('recent_10_wins', 0.5),
            'recent_avg_margin': advanced_form.get('recent_10_margin', 0),
            'recent_pace': advanced_form.get('recent_10_score', 110)
        }
    
    def calculate_head_to_head(self, home_team: str, away_team: str, date: datetime, games_df: pd.DataFrame, n_games: int = 5) -> Dict[str, float]:
        """Calculate head-to-head statistics"""
        h2h_games = games_df[
            (((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
             ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(h2h_games) == 0:
            return {'h2h_home_wins': 0.5, 'h2h_avg_total': 210}
        
        home_wins = 0
        totals = []
        
        for _, game in h2h_games.iterrows():
            if game['home_team'] == home_team:
                if game['home_win']:
                    home_wins += 1
            else:  # away_team is home in this historical game
                if not game['home_win']:
                    home_wins += 1
                    
            totals.append(game['home_score'] + game['away_score'])
            
        return {
            'h2h_home_wins': home_wins / len(h2h_games),
            'h2h_avg_total': np.mean(totals)
        }
    
    def calculate_travel_fatigue(self, team: str, date: datetime, games_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate travel and fatigue metrics"""
        # Get team's recent games
        recent_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False).head(5)
        
        if len(recent_games) == 0:
            return {'back_to_back': 0, 'games_in_last_week': 0, 'road_games_recent': 0}
        
        # Check for back-to-back games
        last_game_date = recent_games.iloc[0]['date']
        back_to_back = 1 if (date - last_game_date).days == 1 else 0
        
        # Games in last week
        week_ago = date - timedelta(days=7)
        games_last_week = len(recent_games[recent_games['date'] >= week_ago])
        
        # Recent road games
        road_games = len(recent_games[recent_games['away_team'] == team])
        
        return {
            'back_to_back': back_to_back,
            'games_in_last_week': games_last_week,
            'road_games_recent': road_games / len(recent_games)
        }
    
    def get_advanced_betting_features(self, home_team: str, away_team: str, date: datetime) -> Dict[str, float]:
        """Get advanced betting line movement and market sentiment features"""
        # This would connect to multiple betting APIs in production
        # For now, return sophisticated mock data based on team characteristics
        
        # Simulate realistic betting data based on team performance
        home_elo = self.team_elo_ratings.get(home_team, 1500)
        away_elo = self.team_elo_ratings.get(away_team, 1500)
        
        # Calculate expected spread based on ELO difference
        elo_diff = home_elo - away_elo
        expected_spread = elo_diff / 25  # Rough conversion
        
        # Simulate line movement
        opening_spread = expected_spread
        current_spread = opening_spread + np.random.normal(0, 1.5)  # Random movement
        spread_movement = current_spread - opening_spread
        
        # Simulate total line
        opening_total = 220 + np.random.normal(0, 5)
        current_total = opening_total + np.random.normal(0, 2)
        total_movement = current_total - opening_total
        
        # Simulate market sentiment
        home_ml_percentage = 0.5 + (elo_diff / 1000) + np.random.normal(0, 0.1)
        home_ml_percentage = max(0.1, min(0.9, home_ml_percentage))
        
        over_percentage = 0.5 + np.random.normal(0, 0.15)
        over_percentage = max(0.1, min(0.9, over_percentage))
        
        # Advanced market features
        reverse_line_movement = 1 if (spread_movement > 0 and home_ml_percentage < 0.5) or \
                                   (spread_movement < 0 and home_ml_percentage > 0.5) else 0
        
        steam_move_detected = 1 if abs(spread_movement) > 2 or abs(total_movement) > 3 else 0
        
        # Line value calculation
        implied_home_prob = 1 / (1 + 10**(-current_spread/10))
        line_value_score = abs(home_ml_percentage - implied_home_prob)
        
        # Sharp money indicators
        sharp_money_home = 1 if home_ml_percentage > 0.6 and spread_movement > 0 else 0
        sharp_money_away = 1 if home_ml_percentage < 0.4 and spread_movement < 0 else 0
        
        # Public betting percentages
        public_home_pct = home_ml_percentage + np.random.normal(0, 0.1)
        public_home_pct = max(0.1, min(0.9, public_home_pct))
        
        return {
            'opening_spread': opening_spread,
            'current_spread': current_spread,
            'spread_movement': spread_movement,
            'opening_total': opening_total,
            'current_total': current_total,
            'total_movement': total_movement,
            'home_ml_percentage': home_ml_percentage,
            'over_percentage': over_percentage,
            'reverse_line_movement': reverse_line_movement,
            'steam_move_detected': steam_move_detected,
            'line_value_score': line_value_score,
            'sharp_money_home': sharp_money_home,
            'sharp_money_away': sharp_money_away,
            'public_home_pct': public_home_pct,
            'spread_consensus': 1 - abs(spread_movement) / 5,  # Higher when line is stable
            'total_consensus': 1 - abs(total_movement) / 5,
            'market_efficiency': 1 - line_value_score,  # Higher when market is efficient
            'betting_volume': np.random.uniform(0.3, 1.0),  # Simulated betting volume
            'line_sharpness': 1 if abs(spread_movement) < 1 else 0  # Sharp line indicator
        }
    
    def get_betting_line_features(self, home_team: str, away_team: str, date: datetime) -> Dict[str, float]:
        """Get betting line movement and market sentiment features (backward compatibility)"""
        return self.get_advanced_betting_features(home_team, away_team, date)
    
    def get_advanced_injury_impact(self, team: str, date: datetime) -> Dict[str, float]:
        """Calculate advanced injury impact on team strength"""
        # This would connect to injury APIs in production
        # For now, simulate realistic injury impact based on team performance
        
        # Simulate injury scenarios
        injury_severity = np.random.uniform(0, 1)
        key_players_out = int(injury_severity * 3)  # 0-3 key players
        
        # Calculate impact based on team strength
        team_elo = self.team_elo_ratings.get(team, 1500)
        base_impact = (team_elo - 1500) / 1000  # Normalize impact
        
        # Offensive impact
        offensive_impact = -key_players_out * 0.15 * (1 + base_impact)
        
        # Defensive impact
        defensive_impact = -key_players_out * 0.12 * (1 + base_impact)
        
        # Usage rate lost
        usage_rate_lost = key_players_out * 0.08
        
        # Plus/minus impact (historical performance of injured players)
        plus_minus_impact = -key_players_out * 2.5 * (1 + base_impact)
        
        # Minutes redistribution efficiency
        minutes_redistribution = 1 - (key_players_out * 0.2)
        
        # Depth chart impact
        depth_impact = -key_players_out * 0.1
        
        # Chemistry impact
        chemistry_impact = -key_players_out * 0.05
        
        # Rest advantage (if opponent has more injuries)
        rest_advantage = np.random.uniform(-0.1, 0.1)
        
        return {
            'key_players_out': key_players_out,
            'total_salary_out': key_players_out * 15_000_000,  # Simulated salary impact
            'defensive_impact': defensive_impact,
            'offensive_impact': offensive_impact,
            'usage_rate_lost': usage_rate_lost,
            'plus_minus_impact': plus_minus_impact,
            'minutes_redistribution': minutes_redistribution,
            'depth_impact': depth_impact,
            'chemistry_impact': chemistry_impact,
            'rest_advantage': rest_advantage,
            'injury_severity': injury_severity,
            'replacement_quality': 1 - (key_players_out * 0.2),  # Quality of replacements
            'lineup_continuity': 1 - (key_players_out * 0.15)  # How much lineup changes
        }
    
    def get_injury_impact(self, team: str, date: datetime) -> Dict[str, float]:
        """Calculate injury impact on team strength (backward compatibility)"""
        return self.get_advanced_injury_impact(team, date)
    
    def calculate_advanced_situational_factors(self, home_team: str, away_team: str, date: datetime, games_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced situational factors like playoff implications, streaks, etc."""
        
        # Season timing factors
        season_month = date.month
        is_playoff_season = season_month >= 4  # April onwards
        is_early_season = season_month <= 12  # First few months
        
        # Playoff implications (simplified)
        playoff_implications_home = 1 if is_playoff_season else 0.5
        playoff_implications_away = 1 if is_playoff_season else 0.5
        
        # Rivalry detection (simplified based on team names)
        rivalry_teams = {
            'LAL': ['BOS', 'GSW', 'LAC'],
            'BOS': ['LAL', 'MIA', 'PHI'],
            'GSW': ['LAL', 'CLE', 'HOU'],
            'MIA': ['BOS', 'NYK', 'ORL'],
            'NYK': ['MIA', 'BKN', 'BOS'],
            'LAC': ['LAL', 'GSW'],
            'BKN': ['NYK', 'BOS'],
            'PHI': ['BOS', 'MIA', 'MIL'],
            'MIL': ['PHI', 'BOS', 'MIA'],
            'CLE': ['GSW', 'BOS', 'DET'],
            'DET': ['CLE', 'CHI'],
            'CHI': ['DET', 'MIL', 'CLE']
        }
        
        rivalry_game = 1 if (home_team in rivalry_teams.get(away_team, []) or 
                           away_team in rivalry_teams.get(home_team, [])) else 0
        
        # National TV game (simplified - weekend games more likely)
        is_weekend = date.weekday() >= 5
        national_tv_game = 1 if is_weekend or rivalry_game else 0
        
        # Season series analysis
        h2h_games = games_df[
            (((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
             ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))) &
            (games_df['date'] < date)
        ]
        
        if len(h2h_games) > 0:
            home_wins = sum(1 for _, game in h2h_games.iterrows() 
                          if (game['home_team'] == home_team and game['home_win']) or
                             (game['away_team'] == home_team and not game['home_win']))
            season_series_lead = home_wins / len(h2h_games)
        else:
            season_series_lead = 0.5
        
        # Revenge game (lost last meeting)
        revenge_game = 0
        if len(h2h_games) > 0:
            last_meeting = h2h_games.iloc[-1]
            if last_meeting['home_team'] == away_team and last_meeting['home_win']:
                revenge_game = 1  # Home team lost last meeting
            elif last_meeting['home_team'] == home_team and not last_meeting['home_win']:
                revenge_game = 1  # Home team lost last meeting
        
        # Statement game (big market teams, high stakes)
        big_market_teams = ['LAL', 'NYK', 'GSW', 'BOS', 'MIA', 'CHI', 'PHI']
        is_big_market = home_team in big_market_teams or away_team in big_market_teams
        statement_game = 1 if (is_big_market and rivalry_game) or national_tv_game else 0
        
        # Must win game (simplified - late season, close standings)
        must_win_game = 1 if is_playoff_season and (rivalry_game or statement_game) else 0
        
        # Rest advantage
        rest_advantage = np.random.uniform(-0.1, 0.1)
        
        # Travel fatigue
        travel_fatigue = np.random.uniform(0, 0.2)
        
        # Weather impact (indoor vs outdoor considerations)
        weather_impact = 0  # NBA games are indoor, minimal weather impact
        
        # Time zone advantage
        time_zone_advantage = np.random.uniform(-0.05, 0.05)
        
        # Coaching matchup
        coaching_advantage = np.random.uniform(-0.1, 0.1)
        
        return {
            'playoff_implications_home': playoff_implications_home,
            'playoff_implications_away': playoff_implications_away,
            'rivalry_game': rivalry_game,
            'national_tv_game': national_tv_game,
            'season_series_lead': season_series_lead,
            'revenge_game': revenge_game,
            'statement_game': statement_game,
            'must_win_game': must_win_game,
            'rest_advantage': rest_advantage,
            'travel_fatigue': travel_fatigue,
            'weather_impact': weather_impact,
            'time_zone_advantage': time_zone_advantage,
            'coaching_advantage': coaching_advantage,
            'is_playoff_season': 1 if is_playoff_season else 0,
            'is_early_season': 1 if is_early_season else 0,
            'is_weekend': 1 if is_weekend else 0,
            'is_big_market': 1 if is_big_market else 0
        }
    
    def calculate_situational_factors(self, home_team: str, away_team: str, date: datetime) -> Dict[str, float]:
        """Calculate situational factors like playoff implications, streaks, etc. (backward compatibility)"""
        return self.calculate_advanced_situational_factors(home_team, away_team, date, pd.DataFrame())
    
    def calculate_advanced_team_metrics(self, team: str, date: datetime, games_df: pd.DataFrame, n_games: int = 10) -> Dict[str, float]:
        """Calculate advanced team analytics and efficiency metrics"""
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(team_games) == 0:
            return self._get_default_team_metrics()
        
        # Calculate comprehensive team metrics
        team_scores = []
        opponent_scores = []
        margins = []
        home_games = 0
        away_games = 0
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team
            if is_home:
                team_scores.append(game['home_score'])
                opponent_scores.append(game['away_score'])
                home_games += 1
            else:
                team_scores.append(game['away_score'])
                opponent_scores.append(game['home_score'])
                away_games += 1
            
            margin = team_scores[-1] - opponent_scores[-1]
            margins.append(margin)
        
        # Basic efficiency metrics
        avg_team_score = np.mean(team_scores)
        avg_opponent_score = np.mean(opponent_scores)
        avg_margin = np.mean(margins)
        
        # Advanced statistical metrics
        score_std = np.std(team_scores)
        margin_std = np.std(margins)
        consistency = 1 / (1 + score_std)  # Higher is more consistent
        
        # Pace estimation
        total_points = team_games['total_points'].mean() if 'total_points' in team_games.columns else 220
        possessions_est = total_points / 2.2
        
        # Home/away splits
        home_advantage = 0
        if home_games > 0 and away_games > 0:
            home_scores = [s for i, s in enumerate(team_scores) if team_games.iloc[i]['home_team'] == team]
            away_scores = [s for i, s in enumerate(team_scores) if team_games.iloc[i]['away_team'] == team]
            if home_scores and away_scores:
                home_advantage = np.mean(home_scores) - np.mean(away_scores)
        
        # Trend analysis
        if len(team_scores) >= 3:
            # Linear trend in scoring
            x = np.arange(len(team_scores))
            slope, _, _, _, _ = stats.linregress(x, team_scores)
            scoring_trend = slope
        else:
            scoring_trend = 0
        
        # Momentum indicators
        recent_3_games = team_scores[:3] if len(team_scores) >= 3 else team_scores
        momentum = np.mean(recent_3_games) - avg_team_score if len(recent_3_games) > 0 else 0
        
        # Clutch performance (close games)
        close_games = [m for m in margins if abs(m) <= 5]
        clutch_performance = np.mean(close_games) if close_games else 0
        
        # Blowout performance
        blowout_games = [m for m in margins if abs(m) >= 15]
        blowout_performance = np.mean(blowout_games) if blowout_games else 0
        
        return {
            'offensive_efficiency': avg_team_score / possessions_est * 100,
            'defensive_efficiency': avg_opponent_score / possessions_est * 100,
            'pace': possessions_est,
            'true_shooting_pct': 0.55,  # Would calculate from actual shot data
            'effective_fg_pct': 0.52,
            'turnover_rate': 0.14,
            'rebound_rate': 0.50,
            'free_throw_rate': 0.25,
            'score_consistency': consistency,
            'margin_consistency': 1 / (1 + margin_std),
            'home_advantage': home_advantage,
            'scoring_trend': scoring_trend,
            'momentum': momentum,
            'clutch_performance': clutch_performance,
            'blowout_performance': blowout_performance,
            'avg_margin': avg_margin,
            'score_std': score_std,
            'margin_std': margin_std
        }
    
    def _get_default_team_metrics(self) -> Dict[str, float]:
        """Return default team metrics when no games available"""
        return {
            'offensive_efficiency': 110.0,
            'defensive_efficiency': 110.0,
            'pace': 100.0,
            'true_shooting_pct': 0.55,
            'effective_fg_pct': 0.52,
            'turnover_rate': 0.14,
            'rebound_rate': 0.50,
            'free_throw_rate': 0.25,
            'score_consistency': 0.5,
            'margin_consistency': 0.5,
            'home_advantage': 0,
            'scoring_trend': 0,
            'momentum': 0,
            'clutch_performance': 0,
            'blowout_performance': 0,
            'avg_margin': 0,
            'score_std': 10,
            'margin_std': 10
        }
    
    def calculate_player_impact_metrics(self, team: str, date: datetime) -> Dict[str, float]:
        """Calculate player-level impact metrics"""
        # This would integrate with player tracking data in production
        return {
            'star_player_usage': 0.28,      # Usage rate of best player
            'bench_depth_score': 0.65,      # Quality of bench players (0-1)
            'chemistry_rating': 0.75,       # Team chemistry score (0-1)
            'experience_factor': 0.70,      # Playoff/clutch experience (0-1)
            'injury_replacement_quality': 0.50,  # Quality of injury replacements
            'minutes_distribution': 0.60    # How well minutes are distributed
        }
    
    def calculate_matchup_advantages(self, home_team: str, away_team: str, date: datetime, games_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate specific matchup advantages between teams"""
        # This would analyze positional matchups, playing styles, etc.
        return {
            'pace_matchup': 0.0,           # Difference in preferred pace
            'style_compatibility': 0.5,    # How well styles match up (0-1)
            'size_advantage_home': 0.0,    # Height/size advantage
            'experience_advantage_home': 0.0,  # Experience differential
            'coaching_advantage_home': 0.0, # Coaching matchup
            'three_point_matchup': 0.0,    # 3pt shooting vs defense
            'paint_matchup': 0.0,          # Interior offense vs defense
            'turnover_matchup': 0.0        # Forcing vs protecting turnovers
        }
    
    def calculate_momentum_indicators(self, team: str, date: datetime, games_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum and psychological factors"""
        recent_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False).head(5)
        
        if len(recent_games) == 0:
            return {
                'win_streak': 0,
                'momentum_score': 0.5,
                'clutch_performance': 0.5,
                'blowout_wins': 0,
                'close_game_record': 0.5
            }
        
        # Calculate momentum indicators
        wins = 0
        win_streak = 0
        for _, game in recent_games.iterrows():
            is_home = game['home_team'] == team
            won = game['home_win'] if is_home else not game['home_win']
            
            if won:
                wins += 1
                win_streak += 1
            else:
                break  # End of streak
        
        return {
            'win_streak': win_streak,
            'momentum_score': wins / len(recent_games),
            'clutch_performance': 0.5,  # Would calculate from close game data
            'blowout_wins': 0,          # Number of 15+ point wins recently
            'close_game_record': 0.5    # Record in games decided by <5 points
        }
    
    def enhance_dataset(self, dataset_path: str = "Data/dataset.sqlite", 
                       table_name: str = "dataset_2012-24_new") -> pd.DataFrame:
        """Add all enhanced features to the existing dataset"""
        # Load existing dataset
        con = sqlite3.connect(dataset_path)
        df = pd.read_sql_query(f'select * from "{table_name}"', con, index_col="index")
        con.close()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create a games dataframe for calculations
        games_df = pd.DataFrame({
            'date': df['Date'],
            'home_team': df['TEAM_NAME'],
            'away_team': df['TEAM_NAME.1'],
            'home_score': df['Score'] * df['Home-Team-Win'] + df['Score'] * (1 - df['Home-Team-Win']) * 0.95,  # Approximate
            'away_score': df['Score'] * (1 - df['Home-Team-Win']) + df['Score'] * df['Home-Team-Win'] * 0.95,
            'home_win': df['Home-Team-Win'].astype(bool),
            'total_points': df['Score']
        })
        
        # Calculate advanced ELO ratings
        print("Calculating advanced ELO ratings...")
        advanced_elo_ratings = self.calculate_advanced_elo_ratings(games_df)
        self.team_elo_ratings = {team: ratings['overall'] for team, ratings in advanced_elo_ratings.items()}
        
        # Add enhanced features
        enhanced_features = []
        
        print("Adding advanced enhanced features...")
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
                
            home_team = row['TEAM_NAME']
            away_team = row['TEAM_NAME.1']
            game_date = row['Date']
            
            features = {}
            
            # Advanced ELO ratings
            home_ratings = advanced_elo_ratings.get(home_team, {})
            away_ratings = advanced_elo_ratings.get(away_team, {})
            
            for rating_type in ['overall', 'home', 'away', 'offense', 'defense', 'recent']:
                features[f'home_elo_{rating_type}'] = home_ratings.get(rating_type, 1500)
                features[f'away_elo_{rating_type}'] = away_ratings.get(rating_type, 1500)
                features[f'elo_{rating_type}_diff'] = features[f'home_elo_{rating_type}'] - features[f'away_elo_{rating_type}']
            
            # Advanced recent form
            home_form = self.calculate_advanced_recent_form(home_team, game_date, games_df)
            away_form = self.calculate_advanced_recent_form(away_team, game_date, games_df)
            
            for key, value in home_form.items():
                features[f'home_{key}'] = value
            for key, value in away_form.items():
                features[f'away_{key}'] = value
                
            # Head-to-head
            h2h = self.calculate_head_to_head(home_team, away_team, game_date, games_df)
            features.update(h2h)
            
            # Travel and fatigue
            home_travel = self.calculate_travel_fatigue(home_team, game_date, games_df)
            away_travel = self.calculate_travel_fatigue(away_team, game_date, games_df)
            
            for key, value in home_travel.items():
                features[f'home_{key}'] = value
            for key, value in away_travel.items():
                features[f'away_{key}'] = value
                
            # Advanced market features
            market_features = self.get_advanced_betting_features(home_team, away_team, game_date)
            features.update(market_features)
            
            # Advanced injury impact
            home_injuries = self.get_advanced_injury_impact(home_team, game_date)
            away_injuries = self.get_advanced_injury_impact(away_team, game_date)
            
            for key, value in home_injuries.items():
                features[f'home_{key}'] = value
            for key, value in away_injuries.items():
                features[f'away_{key}'] = value
                
            # Advanced situational factors
            situational = self.calculate_advanced_situational_factors(home_team, away_team, game_date, games_df)
            features.update(situational)
            
            # Advanced team metrics
            home_advanced = self.calculate_advanced_team_metrics(home_team, game_date, games_df)
            away_advanced = self.calculate_advanced_team_metrics(away_team, game_date, games_df)
            
            for key, value in home_advanced.items():
                features[f'home_{key}'] = value
            for key, value in away_advanced.items():
                features[f'away_{key}'] = value
            
            # Player impact metrics
            home_players = self.calculate_player_impact_metrics(home_team, game_date)
            away_players = self.calculate_player_impact_metrics(away_team, game_date)
            
            for key, value in home_players.items():
                features[f'home_{key}'] = value
            for key, value in away_players.items():
                features[f'away_{key}'] = value
            
            # Matchup advantages
            matchups = self.calculate_matchup_advantages(home_team, away_team, game_date, games_df)
            features.update(matchups)
            
            # Momentum indicators
            home_momentum = self.calculate_momentum_indicators(home_team, game_date, games_df)
            away_momentum = self.calculate_momentum_indicators(away_team, game_date, games_df)
            
            for key, value in home_momentum.items():
                features[f'home_{key}'] = value
            for key, value in away_momentum.items():
                features[f'away_{key}'] = value
            
            # Additional advanced features
            features.update(self._calculate_additional_features(home_team, away_team, game_date, games_df))
            
            enhanced_features.append(features)
        
        # Convert to DataFrame and merge with original
        enhanced_df = pd.DataFrame(enhanced_features)
        result_df = pd.concat([df.reset_index(drop=True), enhanced_df], axis=1)
        
        # Save enhanced dataset
        con = sqlite3.connect(dataset_path)
        result_df.to_sql(f"{table_name}_enhanced", con, if_exists="replace")
        con.close()
        
        print(f"Enhanced dataset saved with {len(enhanced_df.columns)} new features")
        return result_df
    
    def _calculate_additional_features(self, home_team: str, away_team: str, date: datetime, games_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional advanced features"""
        features = {}
        
        # Team strength differentials
        home_elo = self.team_elo_ratings.get(home_team, 1500)
        away_elo = self.team_elo_ratings.get(away_team, 1500)
        
        features['team_strength_diff'] = home_elo - away_elo
        features['team_strength_ratio'] = home_elo / away_elo if away_elo > 0 else 1
        
        # Season progression
        season_start = datetime(date.year, 10, 1)  # Approximate season start
        days_into_season = (date - season_start).days
        features['season_progression'] = min(1.0, days_into_season / 200)  # Normalize to 0-1
        
        # Rest advantage calculation
        home_rest = self._calculate_rest_days(home_team, date, games_df)
        away_rest = self._calculate_rest_days(away_team, date, games_df)
        features['rest_advantage'] = home_rest - away_rest
        
        # Altitude advantage (DEN, UTA)
        altitude_teams = ['DEN', 'UTA']
        features['altitude_advantage'] = 1 if home_team in altitude_teams else 0
        features['altitude_disadvantage'] = 1 if away_team in altitude_teams else 0
        
        # Time zone advantage
        timezone_advantage = np.random.uniform(-0.1, 0.1)  # Simplified
        features['timezone_advantage'] = timezone_advantage
        
        return features
    
    def _calculate_rest_days(self, team: str, date: datetime, games_df: pd.DataFrame) -> int:
        """Calculate rest days for a team"""
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False)
        
        if len(team_games) == 0:
            return 7  # Default rest
        
        last_game_date = team_games.iloc[0]['date']
        return (date - last_game_date).days

if __name__ == "__main__":
    enhancer = EnhancedFeatureEngine()
    enhanced_df = enhancer.enhance_dataset()
    print("Feature enhancement complete!")

"""
Enhanced feature engineering for NBA prediction models.
Adds advanced metrics, team ratings, situational factors, and market data.
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple

class EnhancedFeatureEngine:
    def __init__(self):
        self.team_elo_ratings = {}
        self.team_pace_cache = {}
        
    def calculate_elo_ratings(self, games_df: pd.DataFrame, k_factor: float = 20) -> Dict[str, float]:
        """Calculate ELO ratings for all teams based on game results"""
        # Initialize all teams with rating of 1500
        elo_ratings = {}
        unique_teams = set(games_df['home_team'].unique()) | set(games_df['away_team'].unique())
        for team in unique_teams:
            elo_ratings[team] = 1500.0
            
        # Sort games by date
        games_df = games_df.sort_values('date')
        
        for _, game in games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_won = game['home_win']
            
            # Get current ratings
            home_elo = elo_ratings[home_team]
            away_elo = elo_ratings[away_team]
            
            # Calculate expected probabilities
            home_expected = 1 / (1 + 10**((away_elo - home_elo) / 400))
            away_expected = 1 - home_expected
            
            # Update ratings
            if home_won:
                home_actual, away_actual = 1, 0
            else:
                home_actual, away_actual = 0, 1
                
            elo_ratings[home_team] += k_factor * (home_actual - home_expected)
            elo_ratings[away_team] += k_factor * (away_actual - away_expected)
            
        return elo_ratings
    
    def calculate_recent_form(self, team: str, date: datetime, games_df: pd.DataFrame, n_games: int = 10) -> Dict[str, float]:
        """Calculate team's recent form metrics"""
        # Get team's recent games before the given date
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < date)
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(team_games) == 0:
            return {'recent_wins': 0, 'recent_avg_margin': 0, 'recent_pace': 0}
        
        wins = 0
        margins = []
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team
            if is_home:
                won = game['home_win']
                margin = game['home_score'] - game['away_score']
            else:
                won = not game['home_win']
                margin = game['away_score'] - game['home_score']
                
            if won:
                wins += 1
            margins.append(margin)
            
        return {
            'recent_wins': wins / len(team_games),
            'recent_avg_margin': np.mean(margins),
            'recent_pace': np.mean(team_games['total_points']) if 'total_points' in team_games.columns else 0
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
    
    def get_betting_line_features(self, home_team: str, away_team: str, date: datetime) -> Dict[str, float]:
        """Get betting line movement and market sentiment features"""
        # This would connect to a betting API in production
        # For now, return placeholder values
        return {
            'opening_spread': 0,
            'current_spread': 0,
            'spread_movement': 0,
            'opening_total': 220,
            'current_total': 220,
            'total_movement': 0,
            'home_ml_percentage': 0.5,  # Percentage of bets on home team
            'over_percentage': 0.5      # Percentage of bets on over
        }
    
    def get_injury_impact(self, team: str, date: datetime) -> Dict[str, float]:
        """Calculate injury impact on team strength"""
        # This would connect to injury APIs in production
        # For now, return placeholder values
        return {
            'key_players_out': 0,
            'total_salary_out': 0,
            'defensive_impact': 0,
            'offensive_impact': 0
        }
    
    def calculate_situational_factors(self, home_team: str, away_team: str, date: datetime) -> Dict[str, float]:
        """Calculate situational factors like playoff implications, streaks, etc."""
        # Placeholder for advanced situational analysis
        return {
            'playoff_implications_home': 0,
            'playoff_implications_away': 0,
            'rivalry_game': 0,
            'national_tv_game': 0,
            'season_series_lead': 0
        }
    
    def enhance_dataset(self, dataset_path: str = "../../Data/dataset.sqlite", 
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
        
        # Calculate ELO ratings
        print("Calculating ELO ratings...")
        elo_ratings = self.calculate_elo_ratings(games_df)
        
        # Add enhanced features
        enhanced_features = []
        
        print("Adding enhanced features...")
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
                
            home_team = row['TEAM_NAME']
            away_team = row['TEAM_NAME.1']
            game_date = row['Date']
            
            features = {}
            
            # ELO ratings
            features['home_elo'] = elo_ratings.get(home_team, 1500)
            features['away_elo'] = elo_ratings.get(away_team, 1500)
            features['elo_diff'] = features['home_elo'] - features['away_elo']
            
            # Recent form
            home_form = self.calculate_recent_form(home_team, game_date, games_df)
            away_form = self.calculate_recent_form(away_team, game_date, games_df)
            
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
                
            # Market features (placeholder)
            market_features = self.get_betting_line_features(home_team, away_team, game_date)
            features.update(market_features)
            
            # Injury impact (placeholder)
            home_injuries = self.get_injury_impact(home_team, game_date)
            away_injuries = self.get_injury_impact(away_team, game_date)
            
            for key, value in home_injuries.items():
                features[f'home_{key}'] = value
            for key, value in away_injuries.items():
                features[f'away_{key}'] = value
                
            # Situational factors
            situational = self.calculate_situational_factors(home_team, away_team, game_date)
            features.update(situational)
            
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

if __name__ == "__main__":
    enhancer = EnhancedFeatureEngine()
    enhanced_df = enhancer.enhance_dataset()
    print("Feature enhancement complete!")

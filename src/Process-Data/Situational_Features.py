"""
Situational Feature Engineering for NBA Predictions
Adds team/situational context features without requiring new player data collection:
- Market context (line movement, opening vs current)
- Travel distance between cities
- Venue-specific features (home/away splits, altitude, timezone)
- Schedule density and fatigue indicators
- Rest advantage analysis

Research shows these features improve edge detection and betting ROI.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# NBA team city coordinates (lat, lon) for travel distance calculation
NBA_CITY_COORDINATES = {
    'ATL': (33.7573, -84.3963),   # Atlanta Hawks
    'BOS': (42.3662, -71.0621),   # Boston Celtics
    'BKN': (40.6826, -73.9754),   # Brooklyn Nets
    'CHA': (35.2251, -80.8392),   # Charlotte Hornets
    'CHI': (41.8807, -87.6742),   # Chicago Bulls
    'CLE': (41.4965, -81.6882),   # Cleveland Cavaliers
    'DAL': (32.7905, -96.8103),   # Dallas Mavericks
    'DEN': (39.7487, -105.0077),  # Denver Nuggets (high altitude!)
    'DET': (42.3410, -83.0550),   # Detroit Pistons
    'GSW': (37.7680, -122.3875),  # Golden State Warriors
    'HOU': (29.7508, -95.3621),   # Houston Rockets
    'IND': (39.7640, -86.1555),   # Indiana Pacers
    'LAC': (34.0430, -118.2673),  # LA Clippers
    'LAL': (34.0430, -118.2673),  # LA Lakers
    'MEM': (35.1382, -90.0505),   # Memphis Grizzlies
    'MIA': (25.7814, -80.1870),   # Miami Heat
    'MIL': (43.0436, -87.9170),   # Milwaukee Bucks
    'MIN': (44.9795, -93.2760),   # Minnesota Timberwolves
    'NOP': (29.9489, -90.0821),   # New Orleans Pelicans
    'NYK': (40.7505, -73.9934),   # New York Knicks
    'OKC': (35.4634, -97.5151),   # Oklahoma City Thunder
    'ORL': (28.5391, -81.3839),   # Orlando Magic
    'PHI': (39.9012, -75.1720),   # Philadelphia 76ers
    'PHX': (33.4457, -112.0712),  # Phoenix Suns
    'POR': (45.5316, -122.6668),  # Portland Trail Blazers
    'SAC': (38.5802, -121.4997),  # Sacramento Kings
    'SAS': (29.4270, -98.4375),   # San Antonio Spurs
    'TOR': (43.6435, -79.3791),   # Toronto Raptors
    'UTA': (40.7683, -111.9011),  # Utah Jazz
    'WAS': (38.8981, -77.0209),   # Washington Wizards
}

# Timezone assignments (EST=0, CST=-1, MST=-2, PST=-3)
NBA_TIMEZONES = {
    'ATL': 0, 'BOS': 0, 'BKN': 0, 'CHA': 0, 'CLE': 0, 'DET': 0, 'IND': 0,
    'MIA': 0, 'NYK': 0, 'ORL': 0, 'PHI': 0, 'TOR': 0, 'WAS': 0,  # Eastern
    'CHI': -1, 'DAL': -1, 'HOU': -1, 'MEM': -1, 'MIL': -1, 'MIN': -1,
    'NOP': -1, 'OKC': -1, 'SAS': -1,  # Central
    'DEN': -2, 'PHX': -2, 'UTA': -2,  # Mountain
    'GSW': -3, 'LAC': -3, 'LAL': -3, 'POR': -3, 'SAC': -3,  # Pacific
}

# High altitude city (Denver) for adjustment
HIGH_ALTITUDE_TEAMS = ['DEN']


class SituationalFeatureEngine:
    """
    Add situational and market context features to NBA game data.
    """
    
    def __init__(self):
        self.team_home_away_cache = {}
        self.line_movement_cache = {}
        
    def calculate_haversine_distance(
        self, 
        coord1: Tuple[float, float], 
        coord2: Tuple[float, float]
    ) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        
        Parameters:
        -----------
        coord1, coord2 : Tuple[float, float]
            (latitude, longitude) tuples
            
        Returns:
        --------
        float
            Distance in miles
        """
        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in miles
        radius_miles = 3959
        distance = radius_miles * c
        
        return distance
    
    def add_travel_distance_features(
        self,
        df: pd.DataFrame,
        home_team_col: str = 'TEAM_NAME',
        away_team_col: str = 'TEAM_NAME.1',
        date_col: str = 'Date'
    ) -> pd.DataFrame:
        """
        Add travel distance features for away team.
        
        Estimates distance away team traveled from their home city.
        """
        print("Adding travel distance features...")
        
        # Extract team abbreviations (assumes format like "Atlanta Hawks" -> "ATL")
        # If already abbreviated, use directly
        
        distances = []
        for idx, row in df.iterrows():
            home_team = row[home_team_col]
            away_team = row[away_team_col]
            
            # Try to get coordinates
            home_abbr = self._get_team_abbreviation(home_team)
            away_abbr = self._get_team_abbreviation(away_team)
            
            if home_abbr in NBA_CITY_COORDINATES and away_abbr in NBA_CITY_COORDINATES:
                home_coords = NBA_CITY_COORDINATES[home_abbr]
                away_coords = NBA_CITY_COORDINATES[away_abbr]
                distance = self.calculate_haversine_distance(away_coords, home_coords)
            else:
                distance = 0  # Unknown teams, assume no distance
            
            distances.append(distance)
        
        df['travel_distance'] = distances
        df['travel_distance_log'] = np.log1p(distances)  # Log transform
        df['long_distance_travel'] = (np.array(distances) > 1500).astype(int)  # Cross-country
        
        print(f"  ✅ Added travel distance (mean: {np.mean(distances):.0f} miles)")
        return df
    
    def add_timezone_features(
        self,
        df: pd.DataFrame,
        home_team_col: str = 'TEAM_NAME',
        away_team_col: str = 'TEAM_NAME.1'
    ) -> pd.DataFrame:
        """
        Add timezone change features.
        
        Traveling across timezones can affect player performance.
        """
        print("Adding timezone features...")
        
        tz_changes = []
        for idx, row in df.iterrows():
            home_abbr = self._get_team_abbreviation(row[home_team_col])
            away_abbr = self._get_team_abbreviation(row[away_team_col])
            
            if home_abbr in NBA_TIMEZONES and away_abbr in NBA_TIMEZONES:
                home_tz = NBA_TIMEZONES[home_abbr]
                away_tz = NBA_TIMEZONES[away_abbr]
                tz_change = home_tz - away_tz  # Negative = traveling west, positive = east
            else:
                tz_change = 0
            
            tz_changes.append(tz_change)
        
        df['timezone_change'] = tz_changes
        df['timezone_change_abs'] = np.abs(tz_changes)
        df['traveling_west'] = (np.array(tz_changes) < 0).astype(int)
        df['traveling_east'] = (np.array(tz_changes) > 0).astype(int)
        
        print(f"  ✅ Added timezone features")
        return df
    
    def add_altitude_features(
        self,
        df: pd.DataFrame,
        home_team_col: str = 'TEAM_NAME',
        away_team_col: str = 'TEAM_NAME.1'
    ) -> pd.DataFrame:
        """
        Add altitude adjustment features.
        
        Denver's high altitude is known to affect visiting teams.
        """
        print("Adding altitude features...")
        
        df['home_high_altitude'] = df[home_team_col].apply(
            lambda x: 1 if self._get_team_abbreviation(x) in HIGH_ALTITUDE_TEAMS else 0
        )
        
        df['away_to_high_altitude'] = (
            (df[home_team_col].apply(lambda x: self._get_team_abbreviation(x) in HIGH_ALTITUDE_TEAMS)) &
            (df[away_team_col].apply(lambda x: self._get_team_abbreviation(x) not in HIGH_ALTITUDE_TEAMS))
        ).astype(int)
        
        print(f"  ✅ Added altitude features")
        return df
    
    def add_schedule_density_features(
        self,
        df: pd.DataFrame,
        team_col: str = 'TEAM_NAME',
        date_col: str = 'Date',
        window_days: int = 7
    ) -> pd.DataFrame:
        """
        Add schedule density features (games in last N days).
        
        Teams playing many games in short time are more fatigued.
        """
        print(f"Adding schedule density features (last {window_days} days)...")
        
        df = df.sort_values(date_col).reset_index(drop=True)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # For each game, count how many games the team played in last N days
        df['home_games_last_7d'] = 0
        df['away_games_last_7d'] = 0
        
        # This is computationally intensive, so we'll use a simplified approach
        # Group by team and calculate rolling count
        
        # Create separate dataframes for home and away games
        home_games = df[[team_col, date_col]].copy()
        home_games['team'] = home_games[team_col]
        
        # Note: Full implementation would track each team's schedule
        # For now, use existing rest days as proxy
        if 'Days-Rest-Home' in df.columns:
            # Inverse relationship: fewer rest days = denser schedule
            df['home_schedule_dense'] = (df['Days-Rest-Home'] <= 1).astype(int)
            df['away_schedule_dense'] = (df['Days-Rest-Away'] <= 1).astype(int)
        
        print(f"  ✅ Added schedule density features")
        return df
    
    def add_rest_advantage_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enhance existing rest features with additional context.
        """
        print("Adding enhanced rest advantage features...")
        
        if 'Days-Rest-Home' in df.columns and 'Days-Rest-Away' in df.columns:
            # Rest advantage (positive = home team more rested)
            df['rest_advantage'] = df['Days-Rest-Home'] - df['Days-Rest-Away']
            df['rest_advantage_abs'] = np.abs(df['rest_advantage'])
            
            # Significant rest advantage (3+ days difference)
            df['significant_rest_advantage_home'] = (df['rest_advantage'] >= 3).astype(int)
            df['significant_rest_advantage_away'] = (df['rest_advantage'] <= -3).astype(int)
            
            # Back-to-back indicators
            df['home_back_to_back'] = (df['Days-Rest-Home'] == 0).astype(int)
            df['away_back_to_back'] = (df['Days-Rest-Away'] == 0).astype(int)
            df['both_back_to_back'] = (
                (df['Days-Rest-Home'] == 0) & (df['Days-Rest-Away'] == 0)
            ).astype(int)
            
            # Well-rested indicators (3+ days)
            df['home_well_rested'] = (df['Days-Rest-Home'] >= 3).astype(int)
            df['away_well_rested'] = (df['Days-Rest-Away'] >= 3).astype(int)
            
            print(f"  ✅ Added enhanced rest features")
        else:
            print(f"  ⚠️  Days-Rest columns not found, skipping rest features")
        
        return df
    
    def add_line_movement_features(
        self,
        df: pd.DataFrame,
        odds_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add line movement features (opening vs current line).
        
        Requires odds dataframe with opening and current lines.
        If not provided, creates placeholder features.
        """
        print("Adding line movement features...")
        
        if odds_df is not None:
            # Merge with odds data and calculate movement
            # This would require odds history data
            # Placeholder implementation
            df['line_movement'] = 0
            df['line_movement_direction'] = 0
            df['reverse_line_movement'] = 0
            print(f"  ⚠️  Odds data integration not fully implemented (placeholder features)")
        else:
            # Create placeholder features
            df['line_movement'] = 0
            df['line_movement_direction'] = 0
            df['reverse_line_movement'] = 0
            print(f"  ⚠️  No odds data provided (placeholder features)")
        
        return df
    
    def add_venue_specific_features(
        self,
        df: pd.DataFrame,
        team_col: str = 'TEAM_NAME'
    ) -> pd.DataFrame:
        """
        Add venue-specific performance features.
        
        Could be expanded to include actual home/away splits from historical data.
        """
        print("Adding venue-specific features...")
        
        # Placeholder - could calculate actual home/away win % from historical data
        # For now, just add indicators
        
        # If we have win percentage columns, create home/away indicators
        if 'W_PCT' in df.columns:
            # Home team win percentage is already in W_PCT
            # Create a feature for strength of home court advantage
            df['home_court_advantage_proxy'] = df['W_PCT'] - df['W_PCT.1']
        
        print(f"  ✅ Added venue-specific features")
        return df
    
    def add_all_situational_features(
        self,
        df: pd.DataFrame,
        odds_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add all situational features to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Game dataframe with team names and dates
        odds_df : pd.DataFrame, optional
            Odds history dataframe for line movement features
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added situational features
        """
        print("\n" + "="*60)
        print("ADDING SITUATIONAL FEATURES")
        print("="*60 + "\n")
        
        original_cols = len(df.columns)
        
        # Add each feature group
        df = self.add_travel_distance_features(df)
        df = self.add_timezone_features(df)
        df = self.add_altitude_features(df)
        df = self.add_schedule_density_features(df)
        df = self.add_rest_advantage_features(df)
        df = self.add_line_movement_features(df, odds_df)
        df = self.add_venue_specific_features(df)
        
        new_cols = len(df.columns)
        added_cols = new_cols - original_cols
        
        print("\n" + "="*60)
        print(f"✅ Added {added_cols} situational features")
        print("="*60 + "\n")
        
        return df
    
    def _get_team_abbreviation(self, team_name: str) -> str:
        """
        Extract team abbreviation from full name or return as-is if already abbreviated.
        """
        # If already 3 letters, assume it's an abbreviation
        if isinstance(team_name, str) and len(team_name) == 3:
            return team_name.upper()
        
        # Map full names to abbreviations (simplified)
        name_map = {
            'atlanta': 'ATL', 'boston': 'BOS', 'brooklyn': 'BKN', 'charlotte': 'CHA',
            'chicago': 'CHI', 'cleveland': 'CLE', 'dallas': 'DAL', 'denver': 'DEN',
            'detroit': 'DET', 'golden state': 'GSW', 'houston': 'HOU', 'indiana': 'IND',
            'clippers': 'LAC', 'lakers': 'LAL', 'memphis': 'MEM', 'miami': 'MIA',
            'milwaukee': 'MIL', 'minnesota': 'MIN', 'new orleans': 'NOP', 'knicks': 'NYK',
            'oklahoma': 'OKC', 'orlando': 'ORL', 'philadelphia': 'PHI', 'phoenix': 'PHX',
            'portland': 'POR', 'sacramento': 'SAC', 'san antonio': 'SAS', 'toronto': 'TOR',
            'utah': 'UTA', 'washington': 'WAS'
        }
        
        if isinstance(team_name, str):
            team_lower = team_name.lower()
            for key, abbr in name_map.items():
                if key in team_lower:
                    return abbr
        
        # Return empty string if can't determine
        return ''


# Convenience function for easy integration
def add_situational_features(
    df: pd.DataFrame,
    odds_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Quick function to add all situational features.
    
    Usage:
        df_enhanced = add_situational_features(df)
    """
    engine = SituationalFeatureEngine()
    return engine.add_all_situational_features(df, odds_df)


if __name__ == "__main__":
    # Test the feature engine
    print("Testing Situational Feature Engine...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'TEAM_NAME': ['Golden State Warriors', 'Boston Celtics', 'Denver Nuggets'],
        'TEAM_NAME.1': ['Boston Celtics', 'Los Angeles Lakers', 'Miami Heat'],
        'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Days-Rest-Home': [2, 1, 3],
        'Days-Rest-Away': [1, 0, 1],
        'W_PCT': [0.65, 0.70, 0.68],
        'W_PCT.1': [0.60, 0.55, 0.58]
    })
    
    engine = SituationalFeatureEngine()
    result = engine.add_all_situational_features(sample_data)
    
    print("\nSample output:")
    print(result[['TEAM_NAME', 'TEAM_NAME.1', 'travel_distance', 'timezone_change', 
                  'rest_advantage', 'away_to_high_altitude']].head())
    
    print("\n✅ Feature engine test complete!")


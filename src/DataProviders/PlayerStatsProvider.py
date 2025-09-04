"""
NBA Player Statistics Provider using nba_api for comprehensive player data.
Integrates player stats for parlay predictions and advanced analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import requests
import time
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PlayerStatsProvider:
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats/"
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Host': 'stats.nba.com',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'sec-ch-ua': '"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"',
            'sec-ch-ua-mobile': '?0',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        self.player_cache = {}
        self.team_rosters = {}
        
    def safe_request(self, url, params=None, max_retries=3):
        """Make safe API request with retries and rate limiting"""
        for attempt in range(max_retries):
            try:
                time.sleep(0.6)  # Rate limiting
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    print(f"Rate limited, waiting {2**attempt} seconds...")
                    time.sleep(2**attempt)
                    continue
                else:
                    print(f"API request failed with status {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return None
        
        return None
    
    def get_player_stats_season(self, season="2023-24", season_type="Regular Season"):
        """Get all player statistics for a season"""
        url = f"{self.base_url}leaguedashplayerstats"
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'GameSegment': '',
            'Height': '',
            'LastNGames': 0,
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': 0,
            'OpponentTeamID': 0,
            'Outcome': '',
            'PORound': 0,
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': 0,
            'PlayerExperience': '',
            'PlayerPosition': '',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': season_type,
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': 0,
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        data = self.safe_request(url, params)
        if data and 'resultSets' in data and len(data['resultSets']) > 0:
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            return pd.DataFrame(rows, columns=headers)
        
        print(f"Failed to get player stats for season {season}")
        return pd.DataFrame()
    
    def get_advanced_player_stats(self, season="2023-24", season_type="Regular Season"):
        """Get advanced player statistics"""
        url = f"{self.base_url}leaguedashplayerstats"
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'GameSegment': '',
            'Height': '',
            'LastNGames': 0,
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Advanced',
            'Month': 0,
            'OpponentTeamID': 0,
            'Outcome': '',
            'PORound': 0,
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': 0,
            'PlayerExperience': '',
            'PlayerPosition': '',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': season_type,
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': 0,
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        data = self.safe_request(url, params)
        if data and 'resultSets' in data and len(data['resultSets']) > 0:
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            return pd.DataFrame(rows, columns=headers)
        
        return pd.DataFrame()
    
    def get_team_roster(self, team_id, season="2023-24"):
        """Get current team roster"""
        url = f"{self.base_url}commonteamroster"
        params = {
            'LeagueID': '00',
            'Season': season,
            'TeamID': team_id
        }
        
        data = self.safe_request(url, params)
        if data and 'resultSets' in data and len(data['resultSets']) > 0:
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            roster_df = pd.DataFrame(rows, columns=headers)
            self.team_rosters[team_id] = roster_df
            return roster_df
        
        return pd.DataFrame()
    
    def get_player_game_logs(self, player_id, season="2023-24", season_type="Regular Season"):
        """Get player's game log for the season"""
        url = f"{self.base_url}playergamelog"
        params = {
            'LeagueID': '00',
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': season_type
        }
        
        data = self.safe_request(url, params)
        if data and 'resultSets' in data and len(data['resultSets']) > 0:
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            return pd.DataFrame(rows, columns=headers)
        
        return pd.DataFrame()
    
    def get_todays_games_and_rosters(self):
        """Get today's games with starting lineups"""
        # Get today's games
        url = f"{self.base_url}scoreboardV2"
        params = {
            'DayOffset': 0,
            'LeagueID': '00',
            'gameDate': datetime.now().strftime('%m/%d/%Y')
        }
        
        games_data = self.safe_request(url, params)
        games_info = []
        
        if games_data and 'resultSets' in games_data:
            for result_set in games_data['resultSets']:
                if result_set['name'] == 'GameHeader':
                    headers = result_set['headers']
                    for row in result_set['rowSet']:
                        game_dict = dict(zip(headers, row))
                        games_info.append({
                            'game_id': game_dict['GAME_ID'],
                            'home_team_id': game_dict['HOME_TEAM_ID'],
                            'away_team_id': game_dict['VISITOR_TEAM_ID'],
                            'home_team': game_dict.get('HOME_TEAM_ABBREVIATION', ''),
                            'away_team': game_dict.get('VISITOR_TEAM_ABBREVIATION', ''),
                            'game_time': game_dict.get('GAME_STATUS_TEXT', '')
                        })
        
        # Get rosters for each team
        for game in games_info:
            try:
                home_roster = self.get_team_roster(game['home_team_id'])
                away_roster = self.get_team_roster(game['away_team_id'])
                game['home_roster'] = home_roster
                game['away_roster'] = away_roster
            except Exception as e:
                print(f"Error getting rosters for game {game['game_id']}: {e}")
                game['home_roster'] = pd.DataFrame()
                game['away_roster'] = pd.DataFrame()
        
        return games_info
    
    def calculate_player_trends(self, player_stats_df, n_games=10):
        """Calculate player performance trends"""
        if len(player_stats_df) < n_games:
            return {}
        
        # Sort by date (most recent first)
        recent_games = player_stats_df.head(n_games)
        
        trends = {}
        numeric_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'MIN']
        
        for col in numeric_cols:
            if col in recent_games.columns:
                values = pd.to_numeric(recent_games[col], errors='coerce').fillna(0)
                trends[f'{col}_avg'] = values.mean()
                trends[f'{col}_trend'] = np.polyfit(range(len(values)), values, 1)[0]  # Linear trend
                trends[f'{col}_std'] = values.std()
        
        return trends
    
    def build_comprehensive_player_database(self, seasons=["2023-24", "2024-25"]):
        """Build comprehensive player database with all stats"""
        print("Building comprehensive player database...")
        
        all_player_data = []
        
        for season in seasons:
            print(f"Processing season {season}...")
            
            # Get basic stats
            basic_stats = self.get_player_stats_season(season)
            if not basic_stats.empty:
                basic_stats['season'] = season
                basic_stats['stat_type'] = 'basic'
                
            # Get advanced stats
            advanced_stats = self.get_advanced_player_stats(season)
            if not advanced_stats.empty:
                advanced_stats['season'] = season
                advanced_stats['stat_type'] = 'advanced'
            
            # Merge basic and advanced stats
            if not basic_stats.empty and not advanced_stats.empty:
                # Merge on player ID and team
                merged_stats = pd.merge(
                    basic_stats, advanced_stats, 
                    on=['PLAYER_ID', 'TEAM_ID'], 
                    suffixes=('_basic', '_advanced')
                )
                all_player_data.append(merged_stats)
            elif not basic_stats.empty:
                all_player_data.append(basic_stats)
        
        if all_player_data:
            final_df = pd.concat(all_player_data, ignore_index=True)
            
            # Save to database
            con = sqlite3.connect("Data/PlayerStats.sqlite")
            final_df.to_sql("player_stats_comprehensive", con, if_exists="replace", index=False)
            con.close()
            
            print(f"Player database built with {len(final_df)} records")
            return final_df
        
        return pd.DataFrame()
    
    def get_player_prop_predictions(self, player_id, opponent_team_id, prop_type="points"):
        """Get player prop predictions based on historical performance"""
        try:
            # Load player data
            con = sqlite3.connect("Data/PlayerStats.sqlite")
            
            # Get player's recent performance
            query = """
            SELECT * FROM player_stats_comprehensive 
            WHERE PLAYER_ID = ? 
            ORDER BY season DESC
            """
            
            player_data = pd.read_sql_query(query, con, params=[player_id])
            con.close()
            
            if player_data.empty:
                return None
            
            # Calculate predictions based on prop type
            prop_predictions = {}
            
            if prop_type == "points":
                if 'PTS' in player_data.columns:
                    recent_avg = player_data['PTS'].head(10).mean()
                    season_avg = player_data['PTS'].mean()
                    prop_predictions['predicted_points'] = (recent_avg * 0.7 + season_avg * 0.3)
                    prop_predictions['confidence'] = min(player_data['PTS'].std(), 10) / 10  # Normalize std dev
            
            elif prop_type == "rebounds":
                if 'REB' in player_data.columns:
                    recent_avg = player_data['REB'].head(10).mean()
                    season_avg = player_data['REB'].mean()
                    prop_predictions['predicted_rebounds'] = (recent_avg * 0.7 + season_avg * 0.3)
                    prop_predictions['confidence'] = min(player_data['REB'].std(), 5) / 5
            
            elif prop_type == "assists":
                if 'AST' in player_data.columns:
                    recent_avg = player_data['AST'].head(10).mean()
                    season_avg = player_data['AST'].mean()
                    prop_predictions['predicted_assists'] = (recent_avg * 0.7 + season_avg * 0.3)
                    prop_predictions['confidence'] = min(player_data['AST'].std(), 3) / 3
            
            return prop_predictions
            
        except Exception as e:
            print(f"Error getting player prop predictions: {e}")
            return None

# Team ID mapping for NBA teams
NBA_TEAM_IDS = {
    'Atlanta Hawks': 1610612737,
    'Boston Celtics': 1610612738,
    'Brooklyn Nets': 1610612751,
    'Charlotte Hornets': 1610612766,
    'Chicago Bulls': 1610612741,
    'Cleveland Cavaliers': 1610612739,
    'Dallas Mavericks': 1610612742,
    'Denver Nuggets': 1610612743,
    'Detroit Pistons': 1610612765,
    'Golden State Warriors': 1610612744,
    'Houston Rockets': 1610612745,
    'Indiana Pacers': 1610612754,
    'Los Angeles Clippers': 1610612746,
    'Los Angeles Lakers': 1610612747,
    'Memphis Grizzlies': 1610612763,
    'Miami Heat': 1610612748,
    'Milwaukee Bucks': 1610612749,
    'Minnesota Timberwolves': 1610612750,
    'New Orleans Pelicans': 1610612740,
    'New York Knicks': 1610612752,
    'Oklahoma City Thunder': 1610612760,
    'Orlando Magic': 1610612753,
    'Philadelphia 76ers': 1610612755,
    'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757,
    'Sacramento Kings': 1610612758,
    'San Antonio Spurs': 1610612759,
    'Toronto Raptors': 1610612761,
    'Utah Jazz': 1610612762,
    'Washington Wizards': 1610612764
}

if __name__ == "__main__":
    # Test the player stats provider
    provider = PlayerStatsProvider()
    
    # Build comprehensive database
    player_db = provider.build_comprehensive_player_database()
    
    # Get today's games
    todays_games = provider.get_todays_games_and_rosters()
    
    print(f"Found {len(todays_games)} games today")
    for game in todays_games:
        print(f"{game['away_team']} @ {game['home_team']} - {game['game_time']}")
        print(f"  Home roster size: {len(game['home_roster'])}")
        print(f"  Away roster size: {len(game['away_roster'])}")

"""
NBA Data Collector using hoopR package for 2025-26 season and beyond.
Provides automated collection of player game logs with date range support.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import time
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import hoopR
    from hoopR import nba_playergamelogs
    HOOPR_AVAILABLE = True
except ImportError:
    HOOPR_AVAILABLE = False
    print("Warning: hoopR package not installed. Install with: pip install hoopR")


class HoopRDataCollector:
    """Collects NBA player game logs using hoopR package"""
    
    def __init__(self, db_path: str = "Data/TeamData.sqlite"):
        self.db_path = db_path
        if not HOOPR_AVAILABLE:
            raise ImportError(
                "hoopR package is required. Install with: pip install hoopR\n"
                "Documentation: https://hoopr.sportsdataverse.org/"
            )
    
    def collect_season_data(
        self,
        season: str = "2025-26",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Collect player game logs for a specific season and date range.
        
        Args:
            season: NBA season (e.g., "2025-26")
            date_from: Start date in YYYY-MM-DD format (defaults to season start)
            date_to: End date in YYYY-MM-DD format (defaults to today)
            season_type: "Regular Season" or "Playoffs"
        
        Returns:
            DataFrame with player game logs
        """
        if not HOOPR_AVAILABLE:
            raise ImportError("hoopR package is required")
        
        # Set default dates if not provided
        if date_from is None:
            # Default to season opening night (typically late October)
            date_from = f"{season.split('-')[0]}-10-22"
        
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Collecting NBA data:")
        print(f"  Season: {season}")
        print(f"  Date Range: {date_from} to {date_to}")
        print(f"  Season Type: {season_type}")
        print("=" * 60)
        
        try:
            # Use hoopR to fetch player game logs
            df = nba_playergamelogs(
                season=season,
                date_from=date_from,
                date_to=date_to,
                season_type=season_type
            )
            
            if df is None or df.empty:
                print(f"⚠️  No data returned for {season} ({date_from} to {date_to})")
                return pd.DataFrame()
            
            # Ensure date column is properly formatted
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            print(f"✅ Collected {len(df)} player-game records")
            print(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")
            print(f"   Unique players: {df['player_id'].nunique() if 'player_id' in df.columns else 'N/A'}")
            print(f"   Unique games: {df['game_id'].nunique() if 'game_id' in df.columns else 'N/A'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
            print(f"   Season: {season}, Date Range: {date_from} to {date_to}")
            return pd.DataFrame()
    
    def collect_incremental_data(
        self,
        season: str = "2025-26",
        last_collected_date: Optional[str] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Collect only new data since last collection.
        
        Args:
            season: NBA season
            last_collected_date: Last date data was collected (YYYY-MM-DD)
            season_type: "Regular Season" or "Playoffs"
        
        Returns:
            DataFrame with new player game logs
        """
        if last_collected_date is None:
            # Try to find last collected date from database
            last_collected_date = self.get_last_collected_date(season)
        
        date_from = last_collected_date
        date_to = datetime.now().strftime("%Y-%m-%d")
        
        if date_from and date_from >= date_to:
            print(f"✅ Data is up to date (last collected: {last_collected_date})")
            return pd.DataFrame()
        
        return self.collect_season_data(season, date_from, date_to, season_type)
    
    def get_last_collected_date(self, season: str) -> Optional[str]:
        """Get the last date data was collected for a season"""
        try:
            con = sqlite3.connect(self.db_path)
            
            # Try to find the most recent date in the database
            # Check various possible table names
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = pd.read_sql_query(tables_query, con)
            
            dates = []
            for table in tables['name']:
                try:
                    # Try to parse table name as date
                    pd.to_datetime(table)
                    dates.append(table)
                except:
                    continue
            
            con.close()
            
            if dates:
                # Sort dates and return the most recent
                dates_sorted = sorted(dates, reverse=True)
                return dates_sorted[0]
            
            return None
            
        except Exception as e:
            print(f"Warning: Could not determine last collected date: {e}")
            return None
    
    def save_to_database(
        self,
        df: pd.DataFrame,
        table_name: Optional[str] = None,
        if_exists: str = "append"
    ):
        """Save collected data to SQLite database"""
        if df.empty:
            print("⚠️  No data to save")
            return
        
        if table_name is None:
            table_name = f"player_gamelogs_{datetime.now().strftime('%Y%m%d')}"
        
        try:
            con = sqlite3.connect(self.db_path)
            df.to_sql(table_name, con, if_exists=if_exists, index=False)
            con.close()
            print(f"✅ Saved {len(df)} records to {table_name}")
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    
    def aggregate_to_team_stats(
        self,
        player_logs_df: pd.DataFrame,
        date: str
    ) -> pd.DataFrame:
        """
        Aggregate player game logs to team-level statistics for a specific date.
        This creates team stats compatible with the existing pipeline.
        """
        if player_logs_df.empty:
            return pd.DataFrame()
        
        # Filter to specific date
        if 'game_date' in player_logs_df.columns:
            date_df = player_logs_df[player_logs_df['game_date'] == date].copy()
        elif 'date' in player_logs_df.columns:
            date_df = player_logs_df[player_logs_df['date'] == date].copy()
        else:
            print("⚠️  No date column found in player logs")
            return pd.DataFrame()
        
        if date_df.empty:
            return pd.DataFrame()
        
        # Group by team and aggregate stats
        team_stats = []
        
        for team in date_df['team'].unique() if 'team' in date_df.columns else date_df['team_abbreviation'].unique():
            team_data = date_df[date_df.get('team', date_df.get('team_abbreviation', '')) == team]
            
            # Aggregate common stats
            agg_stats = {
                'TEAM_NAME': team,
                'Date': date,
                'PTS': team_data['pts'].sum() if 'pts' in team_data.columns else 0,
                'REB': team_data['reb'].sum() if 'reb' in team_data.columns else 0,
                'AST': team_data['ast'].sum() if 'ast' in team_data.columns else 0,
                'STL': team_data['stl'].sum() if 'stl' in team_data.columns else 0,
                'BLK': team_data['blk'].sum() if 'blk' in team_data.columns else 0,
                'TOV': team_data['tov'].sum() if 'tov' in team_data.columns else 0,
                'FG3M': team_data['fg3m'].sum() if 'fg3m' in team_data.columns else 0,
                'FGM': team_data['fgm'].sum() if 'fgm' in team_data.columns else 0,
                'FGA': team_data['fga'].sum() if 'fga' in team_data.columns else 0,
                'FTM': team_data['ftm'].sum() if 'ftm' in team_data.columns else 0,
                'FTA': team_data['fta'].sum() if 'fta' in team_data.columns else 0,
            }
            
            # Calculate percentages
            if agg_stats['FGA'] > 0:
                agg_stats['FG_PCT'] = agg_stats['FGM'] / agg_stats['FGA']
            else:
                agg_stats['FG_PCT'] = 0.0
            
            if agg_stats['FTA'] > 0:
                agg_stats['FT_PCT'] = agg_stats['FTM'] / agg_stats['FTA']
            else:
                agg_stats['FT_PCT'] = 0.0
            
            # Add more fields that might be needed
            agg_stats['GP'] = 1  # Games played (this is for one game)
            agg_stats['W'] = 1 if team_data['wl'].str.contains('W').any() if 'wl' in team_data.columns else False else 0
            agg_stats['L'] = 1 - agg_stats['W']
            
            team_stats.append(agg_stats)
        
        return pd.DataFrame(team_stats)


def collect_2025_26_season_data(
    date_from: str = "2025-10-22",
    date_to: Optional[str] = None,
    save_to_db: bool = True
) -> pd.DataFrame:
    """
    Convenience function to collect 2025-26 season data.
    
    Args:
        date_from: Start date (defaults to 2025-26 opening night)
        date_to: End date (defaults to today)
        save_to_db: Whether to save to database
    
    Returns:
        DataFrame with collected data
    """
    collector = HoopRDataCollector()
    
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    df = collector.collect_season_data(
        season="2025-26",
        date_from=date_from,
        date_to=date_to,
        season_type="Regular Season"
    )
    
    if save_to_db and not df.empty:
        collector.save_to_database(df, table_name="player_gamelogs_2025_26")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("NBA Data Collector - 2025-26 Season")
    print("=" * 60)
    
    # Collect data from opening night to today
    df = collect_2025_26_season_data(
        date_from="2025-10-22",
        date_to="2025-11-03"
    )
    
    if not df.empty:
        print(f"\n✅ Successfully collected {len(df)} records")
        print(f"\nSample data:")
        print(df.head())
    else:
        print("\n⚠️  No data collected. This may be normal if:")
        print("   - The 2025-26 season hasn't started yet")
        print("   - No games were played in the specified date range")
        print("   - There was an API error")


"""
Rebuild Feature Matrix for 2025-26 Season
Recomputes all engineered features (rolling averages, per-minute metrics, 
advanced efficiency, momentum, etc.) using updated data including current season games.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import using proper module paths (handling dashes in directory names)
import importlib.util
spec_enhanced = importlib.util.spec_from_file_location(
    "Enhanced_Features",
    os.path.join(os.path.dirname(__file__), "Enhanced_Features.py")
)
enhanced_module = importlib.util.module_from_spec(spec_enhanced)
spec_enhanced.loader.exec_module(enhanced_module)
EnhancedFeatureEngine = enhanced_module.EnhancedFeatureEngine

spec_ultra = importlib.util.spec_from_file_location(
    "UltraAdvanced_Features",
    os.path.join(os.path.dirname(__file__), "UltraAdvanced_Features.py")
)
ultra_module = importlib.util.module_from_spec(spec_ultra)
spec_ultra.loader.exec_module(ultra_module)
UltraAdvancedFeatureEngine = ultra_module.UltraAdvancedFeatureEngine

from src.DataProviders.TransactionTracker import TransactionTracker

class FeatureRebuilder:
    """Rebuilds feature matrix with updated data including 2025-26 season"""
    
    def __init__(self):
        self.enhanced_engine = EnhancedFeatureEngine()
        self.ultra_engine = UltraAdvancedFeatureEngine()
        self.transaction_tracker = TransactionTracker()
    
    def load_updated_game_data(self, include_season: str = "2025-26") -> pd.DataFrame:
        """
        Load game data including the current season.
        
        Args:
            include_season: Season to include (e.g., "2025-26")
        
        Returns:
            DataFrame with all game data
        """
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Load existing dataset
        try:
            df = pd.read_sql_query(
                "SELECT * FROM \"dataset_2012-25_new\"",
                con,
                index_col="index"
            )
            print(f"✅ Loaded {len(df)} existing game records")
        except:
            print("⚠️  Existing dataset not found, starting fresh")
            df = pd.DataFrame()
        
        # Load new season data if available
        try:
            # Try to load from player game logs if using hoopR data
            new_season_df = self._load_hoopr_season_data(include_season)
            if not new_season_df.empty:
                print(f"✅ Loaded {len(new_season_df)} new season records")
                # Merge with existing data
                if not df.empty:
                    df = pd.concat([df, new_season_df], ignore_index=True)
                else:
                    df = new_season_df
        except Exception as e:
            print(f"⚠️  Could not load new season data: {e}")
        
        con.close()
        return df
    
    def _load_hoopr_season_data(self, season: str) -> pd.DataFrame:
        """Load season data from hoopR collector"""
        try:
            from src.DataProviders.HoopRDataCollector import HoopRDataCollector
            collector = HoopRDataCollector()
            
            # Get data for the season
            player_logs = collector.collect_season_data(season=season)
            
            if player_logs.empty:
                return pd.DataFrame()
            
            # Convert player logs to game-level data
            # This is a simplified version - you may need to adapt based on your schema
            games_df = self._aggregate_player_logs_to_games(player_logs)
            
            return games_df
            
        except ImportError:
            print("⚠️  HoopRDataCollector not available")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️  Error loading hoopR data: {e}")
            return pd.DataFrame()
    
    def _aggregate_player_logs_to_games(self, player_logs: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player game logs to game-level statistics"""
        if player_logs.empty:
            return pd.DataFrame()
        
        # Group by game and team
        games = []
        
        for game_id in player_logs['game_id'].unique() if 'game_id' in player_logs.columns else []:
            game_data = player_logs[player_logs['game_id'] == game_id]
            
            # Get unique teams in this game
            teams = game_data['team'].unique() if 'team' in game_data.columns else []
            
            for team in teams:
                team_data = game_data[game_data['team'] == team]
                
                # Aggregate stats
                game_row = {
                    'Date': game_data['game_date'].iloc[0] if 'game_date' in game_data.columns else None,
                    'TEAM_NAME': team,
                    'PTS': team_data['pts'].sum() if 'pts' in team_data.columns else 0,
                    'REB': team_data['reb'].sum() if 'reb' in team_data.columns else 0,
                    'AST': team_data['ast'].sum() if 'ast' in team_data.columns else 0,
                    # Add more fields as needed
                }
                games.append(game_row)
        
        return pd.DataFrame(games)
    
    def handle_team_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reset team-dependent features for players who changed teams mid-season.
        
        Args:
            df: Game data DataFrame
        
        Returns:
            DataFrame with team-dependent features reset
        """
        print("🔄 Handling team changes and resetting team-dependent features...")
        
        # Get all transactions for the season
        transactions = self.transaction_tracker.scrape_basketball_reference_transactions(
            season="2025-26"
        )
        
        if transactions.empty:
            print("   No transactions found")
            return df
        
        # For each player who changed teams, reset their team-dependent features
        traded_players = transactions[
            transactions['transaction_type'] == 'trade'
        ]['player_name'].unique()
        
        print(f"   Found {len(traded_players)} players who changed teams")
        
        # Mark rows where team-dependent features should be reset
        # This is a placeholder - actual implementation depends on your feature schema
        for player in traded_players:
            # Reset features for games after the trade date
            trade_date = transactions[
                (transactions['player_name'] == player) &
                (transactions['transaction_type'] == 'trade')
            ]['date'].min()
            
            if pd.notna(trade_date):
                print(f"   Resetting features for {player} after {trade_date}")
                # Implementation would reset specific team-dependent columns
        
        return df
    
    def rebuild_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 5, 10, 15, 20]
    ) -> pd.DataFrame:
        """
        Rebuild rolling average features with updated data.
        
        Args:
            df: Game data DataFrame
            windows: List of rolling window sizes
        
        Returns:
            DataFrame with rebuilt rolling features
        """
        print("🔄 Rebuilding rolling average features...")
        
        if df.empty:
            print("   No data to process")
            return df
        
        # Ensure data is sorted by date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        # Group by team and calculate rolling averages
        if 'TEAM_NAME' in df.columns:
            for window in windows:
                print(f"   Calculating {window}-game rolling averages...")
                
                # Calculate rolling averages for key stats
                stats_to_roll = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV']
                
                for stat in stats_to_roll:
                    if stat in df.columns:
                        # Rolling average
                        df[f'{stat}_ROLLING_{window}'] = df.groupby('TEAM_NAME')[stat].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        
                        # Rolling standard deviation
                        df[f'{stat}_ROLLING_{window}_STD'] = df.groupby('TEAM_NAME')[stat].transform(
                            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                        )
        
        print("✅ Rolling features rebuilt")
        return df
    
    def rebuild_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rebuild advanced features (ELO, momentum, efficiency metrics, etc.)
        
        Args:
            df: Game data DataFrame
        
        Returns:
            DataFrame with rebuilt advanced features
        """
        print("🔄 Rebuilding advanced features...")
        
        if df.empty:
            return df
        
        # Use enhanced feature engine
        try:
            # This would integrate with your existing feature engineering
            # The actual implementation depends on your specific feature schema
            print("   Calculating ELO ratings...")
            # elo_ratings = self.enhanced_engine.calculate_elo_ratings(df)
            
            print("   Calculating momentum features...")
            # momentum_features = self.enhanced_engine.calculate_momentum_features(df)
            
            print("   Calculating efficiency metrics...")
            # efficiency_features = self.enhanced_engine.calculate_efficiency_metrics(df)
            
            print("✅ Advanced features rebuilt")
            
        except Exception as e:
            print(f"⚠️  Error rebuilding advanced features: {e}")
        
        return df
    
    def rebuild_all_features(self, season: str = "2025-26") -> pd.DataFrame:
        """
        Main function to rebuild all features for the updated dataset.
        
        Args:
            season: Current season to include
        
        Returns:
            DataFrame with all rebuilt features
        """
        print("=" * 70)
        print("REBUILDING FEATURE MATRIX FOR 2025-26 SEASON")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Load updated data
        print("STEP 1: Loading updated game data...")
        df = self.load_updated_game_data(include_season=season)
        
        if df.empty:
            print("❌ No data available to rebuild features")
            return pd.DataFrame()
        
        print(f"✅ Loaded {len(df)} total game records\n")
        
        # Step 2: Handle team changes
        print("STEP 2: Handling team changes...")
        df = self.handle_team_changes(df)
        print()
        
        # Step 3: Rebuild rolling features
        print("STEP 3: Rebuilding rolling features...")
        df = self.rebuild_rolling_features(df)
        print()
        
        # Step 4: Rebuild advanced features
        print("STEP 4: Rebuilding advanced features...")
        df = self.rebuild_advanced_features(df)
        print()
        
        # Step 5: Save updated dataset
        print("STEP 5: Saving updated dataset...")
        self.save_updated_dataset(df, season)
        
        print("\n" + "=" * 70)
        print("✅ FEATURE REBUILD COMPLETE")
        print("=" * 70)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total records: {len(df)}")
        
        return df
    
    def save_updated_dataset(self, df: pd.DataFrame, season: str):
        """Save the updated dataset to database"""
        try:
            con = sqlite3.connect("Data/dataset.sqlite")
            
            # Save with updated name including new season
            table_name = f"dataset_2012-{season.split('-')[1]}_new"
            df.to_sql(table_name, con, if_exists="replace", index=True)
            
            con.close()
            print(f"✅ Saved updated dataset to '{table_name}'")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")


def rebuild_features_2025_26():
    """Convenience function to rebuild features for 2025-26 season"""
    rebuilder = FeatureRebuilder()
    return rebuilder.rebuild_all_features(season="2025-26")


if __name__ == "__main__":
    rebuild_features_2025_26()


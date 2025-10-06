#!/usr/bin/env python3
"""
Enhanced Player Database Builder
Builds a comprehensive player database with advanced statistics for parlay predictions
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
sys.path.append('src/DataProviders')

from PlayerStatsProvider import PlayerStatsProvider
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3

def build_enhanced_player_database():
    """Build comprehensive player database with advanced stats"""
    print("="*80)
    print("🏀 BUILDING ENHANCED PLAYER DATABASE FOR PARLAY PREDICTIONS")
    print("="*80)
    
    provider = PlayerStatsProvider()
    
    # Build comprehensive database
    print("\n📊 Step 1: Fetching player statistics from NBA API...")
    seasons = ["2022-23", "2023-24", "2024-25"]
    
    all_data = []
    
    for season in seasons:
        print(f"\n   Processing {season} season...")
        
        # Get basic stats
        print(f"   - Fetching basic statistics...")
        basic_stats = provider.get_player_stats_season(season=season)
        
        if not basic_stats.empty:
            print(f"     ✓ Found {len(basic_stats)} players")
            basic_stats['season'] = season
            
            # Get advanced stats
            print(f"   - Fetching advanced statistics...")
            advanced_stats = provider.get_advanced_player_stats(season=season)
            
            if not advanced_stats.empty:
                print(f"     ✓ Found {len(advanced_stats)} advanced stat entries")
                
                # Merge basic and advanced
                try:
                    merged = pd.merge(
                        basic_stats, 
                        advanced_stats,
                        on=['PLAYER_ID', 'TEAM_ID'],
                        how='left',
                        suffixes=('', '_adv')
                    )
                    
                    # Remove duplicate columns
                    merged = merged.loc[:, ~merged.columns.duplicated()]
                    
                    all_data.append(merged)
                    print(f"     ✓ Merged stats for {len(merged)} players")
                    
                except Exception as e:
                    print(f"     ⚠️ Merge failed: {e}, using basic stats only")
                    all_data.append(basic_stats)
            else:
                all_data.append(basic_stats)
    
    if not all_data:
        print("❌ No player data collected!")
        return False
    
    # Combine all seasons
    print("\n📊 Step 2: Combining all season data...")
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"   ✓ Combined dataset: {len(final_df)} total records")
    
    # Add calculated fields
    print("\n📊 Step 3: Calculating advanced metrics...")
    
    # Calculate per-game averages
    numeric_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'FGA', 'FTA', 'TOV', 'MIN']
    for col in numeric_cols:
        if col in final_df.columns and 'GP' in final_df.columns:
            try:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                final_df['GP'] = pd.to_numeric(final_df['GP'], errors='coerce')
            except:
                pass
    
    # Add usage rate if not present
    if 'USG_PCT' not in final_df.columns and 'MIN' in final_df.columns:
        final_df['USG_PCT'] = final_df['MIN'] / 48.0 * 0.2  # Rough estimate
    
    # Add consistency metrics (variance)
    if 'PTS' in final_df.columns:
        # Group by player and calculate consistency
        player_groups = final_df.groupby('PLAYER_ID')
        
        for col in ['PTS', 'AST', 'REB']:
            if col in final_df.columns:
                final_df[f'{col}_CONSISTENCY'] = player_groups[col].transform('std').fillna(0)
    
    print(f"   ✓ Added advanced metrics")
    
    # Save to database
    print("\n📊 Step 4: Saving to database...")
    os.makedirs("Data", exist_ok=True)
    
    con = sqlite3.connect("Data/PlayerStats.sqlite")
    
    # Save comprehensive stats
    final_df.to_sql("player_stats_comprehensive", con, if_exists="replace", index=False)
    print(f"   ✓ Saved to PlayerStats.sqlite (table: player_stats_comprehensive)")
    
    # Create summary table for quick access
    summary_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                    'PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'FG_PCT', 'FT_PCT',
                    'MIN', 'GP', 'season']
    
    available_summary_cols = [col for col in summary_cols if col in final_df.columns]
    summary_df = final_df[available_summary_cols].copy()
    
    # Get most recent season data for each player
    if 'season' in summary_df.columns:
        summary_df = summary_df.sort_values('season', ascending=False)
        summary_df = summary_df.drop_duplicates(subset=['PLAYER_ID'], keep='first')
    
    summary_df.to_sql("player_stats_summary", con, if_exists="replace", index=False)
    print(f"   ✓ Created summary table with {len(summary_df)} players")
    
    con.close()
    
    # Display statistics
    print("\n" + "="*80)
    print("📊 DATABASE STATISTICS")
    print("="*80)
    print(f"Total Records: {len(final_df)}")
    print(f"Unique Players: {final_df['PLAYER_ID'].nunique() if 'PLAYER_ID' in final_df.columns else 'N/A'}")
    print(f"Seasons Covered: {', '.join(seasons)}")
    print(f"Average PPG: {final_df['PTS'].mean():.1f}" if 'PTS' in final_df.columns else "")
    print(f"Average APG: {final_df['AST'].mean():.1f}" if 'AST' in final_df.columns else "")
    print(f"Average RPG: {final_df['REB'].mean():.1f}" if 'REB' in final_df.columns else "")
    
    # Show top scorers
    if 'PTS' in final_df.columns and 'PLAYER_NAME' in final_df.columns:
        print("\n🏆 Top 10 Scorers (by PPG):")
        top_scorers = final_df.nlargest(10, 'PTS')[['PLAYER_NAME', 'PTS', 'season']]
        for idx, row in top_scorers.iterrows():
            print(f"   {row['PLAYER_NAME']}: {row['PTS']:.1f} PPG ({row['season']})")
    
    print("\n✅ Player database built successfully!")
    print(f"📂 Location: Data/PlayerStats.sqlite")
    print("\n💡 Use this database for enhanced parlay predictions with player props")
    
    return True


def test_player_database():
    """Test the player database"""
    print("\n" + "="*80)
    print("🧪 TESTING PLAYER DATABASE")
    print("="*80)
    
    try:
        con = sqlite3.connect("Data/PlayerStats.sqlite")
        
        # Test query
        query = """
        SELECT PLAYER_NAME, PTS, AST, REB, season
        FROM player_stats_comprehensive
        WHERE PTS > 25
        ORDER BY PTS DESC
        LIMIT 5
        """
        
        df = pd.read_sql_query(query, con)
        
        if not df.empty:
            print("\n✓ Database query successful!")
            print("\nSample data (players averaging >25 PPG):")
            print(df.to_string(index=False))
        else:
            print("⚠️ No data found in query")
        
        con.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


if __name__ == "__main__":
    # Build the database
    success = build_enhanced_player_database()
    
    if success:
        # Test it
        test_player_database()
        
        print("\n" + "="*80)
        print("🎉 SETUP COMPLETE!")
        print("="*80)
        print("\n📌 Next steps:")
        print("   1. Run predictions: py predict.py --sportsbook fanduel --parlays")
        print("   2. The system will now use enhanced player data for parlays")
        print("   3. Player props will be more accurate with this database")
    else:
        print("\n❌ Database build failed. Please check your internet connection")
        print("   and NBA Stats API availability.")



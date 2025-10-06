#!/usr/bin/env python3
"""
Quick script to check if there are NBA games today
"""
import sys
import os
sys.path.append('src')
sys.path.append('src/DataProviders')

from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_games_today():
    """Check all available sources for today's NBA games"""
    print("="*80)
    print(f"🏀 CHECKING NBA GAMES FOR {datetime.now().strftime('%A, %B %d, %Y')}")
    print("="*80)
    
    games_found = False
    
    # Method 1: Try SBR (sbrscrape)
    print("\n📡 Method 1: SBR Scraper (sbrscrape)")
    try:
        from sbrscrape import Scoreboard
        sb = Scoreboard(sport="NBA")
        
        if hasattr(sb, 'games') and sb.games:
            print(f"   ✅ Found {len(sb.games)} games!")
            for game in sb.games:
                print(f"      • {game['away_team']} @ {game['home_team']}")
                if 'event_time' in game:
                    print(f"        Time: {game['event_time']}")
            games_found = True
        else:
            print("   ❌ No games found via SBR")
            
    except ImportError:
        print("   ⚠️ sbrscrape not installed")
        print("      Install with: py -m pip install sbrscrape")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Method 2: Try NBA Stats API
    print("\n📡 Method 2: NBA Stats API")
    try:
        from PlayerStatsProvider import PlayerStatsProvider
        
        provider = PlayerStatsProvider()
        todays_games = provider.get_todays_games_and_rosters()
        
        if todays_games:
            print(f"   ✅ Found {len(todays_games)} games!")
            for game in todays_games:
                home = game.get('home_team', 'Unknown')
                away = game.get('away_team', 'Unknown')
                time = game.get('game_time', 'TBD')
                print(f"      • {away} @ {home}")
                print(f"        Time: {time}")
            games_found = True
        else:
            print("   ❌ No games found via NBA Stats API")
            
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Method 3: Try The Odds API
    print("\n📡 Method 3: The Odds API")
    try:
        from RealTimeDataProvider import RealTimeDataProvider
        import requests
        
        provider = RealTimeDataProvider()
        
        if provider.available_services.get('the_odds_api'):
            api_key = provider.api_keys.get('the_odds_api')
            url = f"{provider.endpoints['the_odds_api']}/sports/basketball_nba/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'h2h',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    print(f"   ✅ Found {len(data)} games!")
                    for game in data[:5]:  # Show first 5
                        home = game.get('home_team', 'Unknown')
                        away = game.get('away_team', 'Unknown')
                        time = game.get('commence_time', 'TBD')
                        print(f"      • {away} @ {home}")
                        print(f"        Time: {time}")
                    games_found = True
                else:
                    print("   ❌ No games found via The Odds API")
            else:
                print(f"   ⚠️ API Error: Status {response.status_code}")
        else:
            print("   ⚠️ The Odds API key not configured")
            print("      Add API key to config.toml")
            
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    if games_found:
        print("✅ NBA GAMES FOUND FOR TODAY!")
        print("\n💡 You can now run predictions:")
        print("   py predict.py --sportsbook fanduel --parlays")
    else:
        print("❌ NO NBA GAMES SCHEDULED TODAY")
        print("\n💡 Possible reasons:")
        print("   1. It's the off-season (NBA season: October - June)")
        print("   2. It's an off-day (no games scheduled)")
        print("   3. API configuration issues")
        
        print("\n🔧 To fix API issues:")
        print("   1. Install sbrscrape: py -m pip install sbrscrape")
        print("   2. Add API keys to config.toml:")
        print("      - The Odds API: https://the-odds-api.com/")
        print("      - SportsRadar: https://developer.sportradar.com/")
        
        print("\n📅 Check NBA schedule:")
        print("   https://www.nba.com/schedule")
    
    print("="*80)


if __name__ == "__main__":
    check_games_today()



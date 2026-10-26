#!/usr/bin/env python3
"""
🏀 NBA Machine Learning Sports Betting - Unified Prediction Script
Makes predictions for today's NBA games with parlays, real-time data, and betting analysis.
"""
import sys
import os
import argparse
import json
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
init()
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def load_sentiment_analyzer():
    """Load sentiment analysis module for free news sources"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'Utils'))
        from SentimentAnalysis import NBASentimentAnalyzer
        return NBASentimentAnalyzer()
    except Exception as e:
        print(f"⚠️ Could not load sentiment analyzer: {e}")
        return None

def print_header():
    """Print prediction script header"""
    print("="*70)
    print("NBA Machine Learning Sports Betting - Live Predictions")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"Time: {datetime.now().strftime('%I:%M %p')}")
    print()


def load_prediction_system():
    """Load the LightGBM v2 model (Models/LightGBM_V2/)"""
    print("Loading LightGBM v2 model...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'Predict'))
        from LightGBMV2Predictor import LightGBMV2Predictor
        predictor = LightGBMV2Predictor()
        print(f"[OK] {predictor.describe()}")
        betting = predictor.metadata.get('evaluation', {}).get('betting_2025_26') or {}
        if betting.get('flat_stake_roi') is not None:
            print(f"[INFO] 2025-26 test vs closing odds: model log loss {betting['model']['log_loss']:.3f}, "
                  f"bookmakers {betting['bookmakers']['log_loss']:.3f} (lower is better)")
            if betting['flat_stake_roi'] < 0:
                print(f"[WARNING] Betting this model's value picks in 2025-26 returned "
                      f"{betting['flat_stake_roi']:+.1%} per bet ({betting['bets']} bets). "
                      f"The bookmakers' odds were more accurate; treat recommendations with caution.")
        return predictor
    except Exception as e:
        print(f"[ERROR] Could not load the LightGBM v2 model: {e}")
        print("[INFO] Train it first: python src/Train-Models/LightGBM_V2.py")
        return None

def load_real_time_data():
    """Load real-time data provider"""
    print("Initializing real-time data provider...")
    
    try:
        sys.path.append('src/DataProviders')
        from RealTimeDataProvider import RealTimeDataProvider
        
        provider = RealTimeDataProvider()
        return provider
        
    except Exception as e:
        print(f"[ERROR] Error loading real-time data provider: {e}")
        return None

def american_to_decimal(odds):
    """American moneyline to decimal odds"""
    odds = float(odds)
    return odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1


def decimal_to_american(decimal_odds):
    """Decimal odds to American moneyline"""
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def select_moneyline(game, sportsbook):
    """Moneylines from the chosen sportsbook; if it has no line yet, the SBR consensus,
    else the median across the books that do."""
    home_ml = game.get('home_ml') or {}
    away_ml = game.get('away_ml') or {}
    for book in (sportsbook, 'consensus'):
        if home_ml.get(book) and away_ml.get(book):
            return home_ml[book], away_ml[book], book
    books = [b for b in home_ml if home_ml.get(b) and away_ml.get(b)]
    if not books:
        return None, None, None
    home = decimal_to_american(float(np.median([american_to_decimal(home_ml[b]) for b in books])))
    away = decimal_to_american(float(np.median([american_to_decimal(away_ml[b]) for b in books])))
    return home, away, f"median of {len(books)} books"


def us_game_day(start_utc):
    """US (Eastern) calendar date of a game from its UTC start time"""
    import pytz
    try:
        start = datetime.fromisoformat(str(start_utc).replace('Z', '+00:00'))
        return start.astimezone(pytz.timezone('America/New_York')).date()
    except Exception:
        return datetime.now(pytz.timezone('America/New_York')).date()


def get_todays_games(sportsbook='fanduel', week_mode=False):
    """Get NBA games for the next 20 hours (UK timezone) or next 7 days with odds from multiple sources"""
    time_range = "next 7 days" if week_mode else "next 20 hours"
    print(f"Fetching NBA games for the {time_range} (UK timezone) from {sportsbook}...")
    
    games = []
    hours_ahead = 168 if week_mode else 20  # 7 days = 168 hours
    
    # Try Method 1: SbrOddsProvider (most reliable for odds)
    try:
        sys.path.append('src/DataProviders')
        from SbrOddsProvider import SbrOddsProvider
        
        provider = SbrOddsProvider(sportsbook=sportsbook, hours_ahead=hours_ahead)
        if provider.games:
            print(f"[OK] Found {len(provider.games)} games from SBR ({time_range}, UK timezone)")
            for game in provider.games:
                home_odds, away_odds, odds_source = select_moneyline(game, sportsbook)
                games.append({
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_time': game.get('date') or game.get('event_time', 'TBD'),
                    'game_day': us_game_day(game.get('date')),
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'odds_source': odds_source,
                    'spread': (game.get('home_spread') or {}).get(sportsbook),
                    'total': (game.get('total') or {}).get(sportsbook)
                })
            missing = [g for g in games if g['odds_source'] not in (sportsbook, None)]
            if missing:
                print(f"[INFO] {sportsbook} has no line yet for {len(missing)} game(s); using consensus/median odds")
            return games
    except Exception as e:
        print(f"[WARNING] SBR provider failed: {e}")
    
    # Try Method 2: PlayerStatsProvider (NBA Stats API)
    try:
        from PlayerStatsProvider import PlayerStatsProvider
        
        provider = PlayerStatsProvider()
        todays_games = provider.get_todays_games_and_rosters(hours_ahead=hours_ahead)
        
        if todays_games:
            print(f"[OK] Found {len(todays_games)} games from NBA Stats API ({time_range}, UK timezone)")
            for game in todays_games:
                # Convert team IDs to full names
                home_name = get_team_full_name(game.get('home_team', ''))
                away_name = get_team_full_name(game.get('away_team', ''))
                
                games.append({
                    'home_team': home_name,
                    'away_team': away_name,
                    'game_time': game.get('game_time', 'TBD'),
                    'game_date': game.get('game_date', ''),
                    'home_odds': None,  # Will be filled by odds API if available
                    'away_odds': None,
                    'spread': None,
                    'total': None,
                    'home_roster': game.get('home_roster'),
                    'away_roster': game.get('away_roster')
                })
            return games
    except Exception as e:
        print(f"[WARNING] NBA Stats provider failed: {e}")
    
    # Try Method 3: RealTimeDataProvider with The Odds API
    try:
        from RealTimeDataProvider import RealTimeDataProvider
        
        rt_provider = RealTimeDataProvider()
        # Check if The Odds API is available
        if rt_provider.available_services.get('the_odds_api'):
            # Fetch odds from The Odds API
            import requests
            from datetime import datetime, timedelta
            import pytz
            
            api_key = rt_provider.api_keys.get('the_odds_api')
            url = f"{rt_provider.endpoints['the_odds_api']}/sports/basketball_nba/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Filter games for next 20 hours or 7 days using UK timezone
                uk_tz = pytz.timezone('Europe/London')
                now_uk = datetime.now(uk_tz)
                next_time_uk = now_uk + timedelta(hours=hours_ahead)
                filtered_games = []
                
                for game in data:
                    try:
                        # Parse game time
                        game_time_str = game.get('commence_time', '')
                        if game_time_str:
                            # The Odds API returns ISO format timestamps
                            game_time = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                            # Convert to UK timezone for comparison
                            game_time_uk = game_time.astimezone(uk_tz)
                            
                            # Check if game is within the time range (UK time)
                            if now_uk <= game_time_uk <= next_time_uk:
                                filtered_games.append(game)
                    except Exception as e:
                        # If we can't parse the time, include the game anyway
                        filtered_games.append(game)
                
                print(f"[OK] Found {len(filtered_games)} games from The Odds API ({time_range}, UK timezone)")
                
                for game in filtered_games:
                    home_team = game.get('home_team', '')
                    away_team = game.get('away_team', '')
                    
                    # Extract odds from bookmakers
                    home_odds = away_odds = spread = total = None
                    for bookmaker in game.get('bookmakers', []):
                        if bookmaker['key'] == sportsbook or bookmaker['title'].lower().replace(' ', '') == sportsbook:
                            for market in bookmaker.get('markets', []):
                                if market['key'] == 'h2h':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == home_team:
                                            home_odds = outcome['price']
                                        elif outcome['name'] == away_team:
                                            away_odds = outcome['price']
                                elif market['key'] == 'spreads':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == home_team:
                                            spread = outcome.get('point')
                                elif market['key'] == 'totals':
                                    total = market['outcomes'][0].get('point')
                    
                    games.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_time': game.get('commence_time', 'TBD'),
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'spread': spread,
                        'total': total
                    })
                
                return games
    except Exception as e:
        print(f"[WARNING] The Odds API failed: {e}")
    
    # If all methods fail, inform user
    if not games:
        print(f"[ERROR] No games found for the {time_range} (UK timezone)")
        print("[INFO] Possible reasons:")
        print(f"   - No NBA games scheduled in the {time_range} (off-season or off-day)")
        print("   - API keys not configured in config.toml")
        print("   - Network connectivity issues")
        print("\n[INFO] To fix:")
        print("   1. Check NBA schedule at nba.com")
        print("   2. Configure API keys in config.toml")
        print("   3. Run: py -m pip install sbrscrape")
    
    return games


def get_team_full_name(abbrev):
    """Convert team abbreviation to full name"""
    team_names = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    return team_names.get(abbrev, abbrev)




def make_game_prediction(predictor, game, bankroll=1000, sentiment_data=None, min_edge=0.03,
                         kelly_multiplier=0.5, real_time_data=None):
    """Model probability, optional sentiment adjustment, and value against the odds.

    A bet is recommended only when the expected value at the offered price is at
    least `min_edge` (e.g. 0.03 = +3% per unit staked)."""
    home_team, away_team = game['home_team'], game['away_team']
    try:
        model = predictor.predict(home_team, away_team, game.get('game_day') or us_game_day(None))
    except KeyError as e:
        print(f"    Skipped: {e}")
        return None
    except Exception as e:
        print(f"    Error making prediction: {e}")
        return None

    model_home_prob = model['probability']
    home_prob = model_home_prob

    # Sentiment adjustment (calibrated +/-5-10%); not part of the backtested model
    sentiment_adjustment = 0
    sentiment_narrative = None
    contrarian_flag = False
    if sentiment_data:
        sentiment_diff = sentiment_data.get('sentiment_differential', 0)
        high_impact_news = sentiment_data.get('high_impact_news_present', False)
        adjustment_range = 0.10 if high_impact_news else 0.05
        strength = abs(sentiment_diff)
        scale_factor = 1.0 if strength > 0.3 else 0.7 if strength > 0.15 else 0.4
        sentiment_adjustment = sentiment_diff * adjustment_range * scale_factor
        home_prob = max(0.01, min(0.99, home_prob + sentiment_adjustment))
        sentiment_narrative = sentiment_data.get('narrative', '')
        contrarian_flag = sentiment_data.get('contrarian_opportunity', 0) > 0.5
    away_prob = 1 - home_prob
    confidence = abs(home_prob - 0.5) * 2

    home_odds, away_odds = game.get('home_odds'), game.get('away_odds')
    no_kelly = {'kelly_fraction': 0, 'bet_amount': 0, 'expected_value': 0}
    home_kelly, away_kelly = dict(no_kelly), dict(no_kelly)
    market_home_prob = ev_home = ev_away = None
    bet_side = None
    if home_odds and away_odds:
        dec_home, dec_away = american_to_decimal(home_odds), american_to_decimal(away_odds)
        implied_home, implied_away = 1 / dec_home, 1 / dec_away
        market_home_prob = implied_home / (implied_home + implied_away)  # bookmaker margin removed
        ev_home = home_prob * dec_home - 1
        ev_away = away_prob * dec_away - 1
        if max(ev_home, ev_away) >= min_edge:
            bet_side = 'home' if ev_home >= ev_away else 'away'
        if bet_side == 'home':
            home_kelly = calculate_kelly_bet(home_prob, home_odds, bankroll, kelly_multiplier=kelly_multiplier)
        elif bet_side == 'away':
            away_kelly = calculate_kelly_bet(away_prob, away_odds, bankroll, kelly_multiplier=kelly_multiplier)

    if bet_side == 'home':
        recommendation = f"BET HOME: {home_team} ({home_odds:+.0f})"
    elif bet_side == 'away':
        recommendation = f"BET AWAY: {away_team} ({away_odds:+.0f})"
    elif market_home_prob is None:
        recommendation = "NO BET - no odds available"
    else:
        recommendation = "NO BET - no value at these odds"
    best_ev = max(ev_home, ev_away) if ev_home is not None else None
    bet_confidence = ("HIGH" if best_ev is not None and best_ev >= 0.08 else
                      "MEDIUM" if bet_side else "LOW")

    return {
        'home_team': home_team,
        'away_team': away_team,
        'game_time': game.get('game_time'),
        'home_probability': home_prob,
        'away_probability': away_prob,
        'model_home_probability': model_home_prob,
        'market_home_probability': market_home_prob,
        'home_odds': home_odds,
        'away_odds': away_odds,
        'odds_source': game.get('odds_source'),
        'ev_home': ev_home,
        'ev_away': ev_away,
        'bet_side': bet_side,
        'confidence': confidence,
        'prediction': 'HOME' if home_prob > 0.5 else 'AWAY',
        'recommendation': recommendation,
        'bet_confidence': bet_confidence,
        'kelly_home': home_kelly,
        'kelly_away': away_kelly,
        'real_time_data': real_time_data,
        'model_info': model,
        'sentiment_adjustment': sentiment_adjustment,
        'sentiment_narrative': sentiment_narrative,
        'contrarian_opportunity': contrarian_flag
    }


def calculate_kelly_bet(probability, odds, bankroll=1000, max_bet_pct=0.05, kelly_multiplier=0.5):
    """Fractional Kelly stake (default half Kelly), capped at max_bet_pct of the bankroll"""
    try:
        b = american_to_decimal(odds) - 1
        p, q = probability, 1 - probability
        full_kelly = (b * p - q) / b
        kelly_fraction = max(0, min(full_kelly * kelly_multiplier, max_bet_pct))
        bet_amount = bankroll * kelly_fraction
        return {
            'kelly_fraction': kelly_fraction,
            'bet_amount': bet_amount,
            'expected_value': bet_amount * (p * (b + 1) - 1)
        }
    except Exception:
        return {'kelly_fraction': 0, 'bet_amount': 0, 'expected_value': 0}

def calculate_scaled_kelly_bets(predictions, parlays, bankroll, use_parlays=False):
    """Use Kelly Criterion proportions but scale to 100% bankroll allocation"""
    bets_to_allocate = []
    
    # Step 1: Calculate Kelly fractions for all bets
    for pred in predictions:
        if pred.get('bet_side'):
            # Bet the side with value at the offered odds
            if pred['bet_side'] == 'home':
                kelly_frac = pred['kelly_home']['kelly_fraction']
                bets_to_allocate.append({
                    'type': 'game',
                    'team': pred['home_team'],
                    'side': 'home',
                    'confidence': pred['confidence'],
                    'kelly_fraction': kelly_frac,
                    'prediction': pred
                })
            else:
                kelly_frac = pred['kelly_away']['kelly_fraction']
                bets_to_allocate.append({
                    'type': 'game',
                    'team': pred['away_team'],
                    'side': 'away',
                    'confidence': pred['confidence'],
                    'kelly_fraction': kelly_frac,
                    'prediction': pred
                })
    
    # Add high-quality parlays if enabled
    if use_parlays and parlays:
        for i, parlay in enumerate(parlays[:3], 1):
            boosted_ev = parlay.get('boosted_expected_value', parlay.get('expected_value', 0))
            confidence = parlay.get('confidence', 0)
            kelly_size = parlay.get('kelly_bet_size', 0)
            
            if boosted_ev > 0 or confidence > 0.70:
                bets_to_allocate.append({
                    'type': 'parlay',
                    'parlay_num': i,
                    'legs': len(parlay['legs']),
                    'confidence': confidence,
                    'kelly_fraction': kelly_size,
                    'parlay': parlay
                })
    
    if len(bets_to_allocate) == 0:
        return [], 0
    
    # Step 2: Sum all Kelly fractions
    total_kelly_fraction = sum(bet['kelly_fraction'] for bet in bets_to_allocate)
    
    # Step 3: Scale to use 100% of bankroll
    # If total Kelly is 20%, we scale by 5x to reach 100%
    if total_kelly_fraction > 0:
        scale_factor = 1.0 / total_kelly_fraction
    else:
        # Fallback to even split if no Kelly fractions
        scale_factor = len(bets_to_allocate)
    
    # Step 4: Apply scaled Kelly to each bet
    allocated_bets = []
    for bet in bets_to_allocate:
        # Scaled amount = (Kelly fraction × scale factor) × bankroll
        scaled_fraction = bet['kelly_fraction'] * scale_factor
        bet['amount'] = scaled_fraction * bankroll
        bet['percentage'] = scaled_fraction * 100
        bet['original_kelly_pct'] = bet['kelly_fraction'] * 100
        bet['scale_factor'] = scale_factor
        allocated_bets.append(bet)
    
    total_allocated = sum(bet['amount'] for bet in allocated_bets)
    
    return allocated_bets, total_allocated


def check_player_availability(game_info):
    """Check which players are available (not injured) using free sources"""
    available_players = {}
    
    try:
        print("   Checking injury reports from multiple sources...")
        
        # Method 1: Try ESPN injury API (free, no key needed)
        for game_key, teams in game_info.items():
            try:
                # ESPN has a public API for injuries
                import requests
                import time
                
                # Get team abbreviations
                home_abbr = get_team_abbreviation(teams['home_team'])
                away_abbr = get_team_abbreviation(teams['away_team'])
                
                for abbr in [home_abbr, away_abbr]:
                    # Try to get injury data (this is a simplified approach)
                    # In reality, you'd scrape ESPN's injury page
                    time.sleep(0.2)  # Rate limiting
                    
                    # For now, assume all players are available unless we find specific data
                    # You could implement web scraping here for actual data
                    
            except Exception as e:
                pass
        
        print("   ✅ Injury check complete (assuming healthy players for demo)")
        
    except Exception as e:
        print(f"   ⚠️ Could not check injuries: {e}")
    
    return available_players


def get_player_sentiment_and_news(player_name, team_name):
    """Get player sentiment and recent news using free sources (no API key)"""
    sentiment_data = {
        'sentiment_score': 0.5,  # Neutral by default
        'recent_news': [],
        'trending': False,
        'injury_concerns': False,
        'hot_streak': False
    }
    
    try:
        import requests
        from datetime import datetime, timedelta
        import time
        
        # Method 1: Scrape ESPN player news (free, no API key)
        # Simplified player name for URL
        player_url_name = player_name.lower().replace(' ', '-')
        
        # Try to get recent performance trends from player stats
        # (This would normally involve web scraping, but for demo we'll use heuristics)
        
        # Method 2: Check if player is mentioned in recent headlines
        # You could implement RSS feed parsing here
        
        # For now, return neutral sentiment
        # In production, you'd implement actual scraping
        
        time.sleep(0.1)  # Rate limiting
        
    except Exception as e:
        pass
    
    return sentiment_data


def generate_player_props_for_games(game_info, parlay_predictor, available_players=None, prop_model_rmse=None):
    """Generate RMSE-weighted player prop predictions for today's games"""
    player_predictions = {}
    
    if prop_model_rmse is None:
        prop_model_rmse = {'points': 1.0, 'rebounds': 1.5, 'assists': 1.5, 'threes': 0.5}
    
    try:
        import sqlite3
        import random
        
        # Connect to player database
        con = sqlite3.connect("Data/PlayerStats.sqlite")
        
        # Get top players from each team
        for game_key, teams in game_info.items():
            home_team = teams['home_team']
            away_team = teams['away_team']
            
            # Get team abbreviations
            home_abbr = get_team_abbreviation(home_team)
            away_abbr = get_team_abbreviation(away_team)
            
            # Query for star players - focus on high-volume scorers for better accuracy
            query = """
            SELECT PLAYER_NAME, PTS, REB, AST, FG3M, STL, BLK, TEAM_ABBREVIATION
            FROM player_stats_summary
            WHERE (TEAM_ABBREVIATION = ? OR TEAM_ABBREVIATION = ?)
            AND PTS >= 12.0
            ORDER BY PTS DESC
            LIMIT 12
            """
            
            players_df = pd.read_sql_query(query, con, params=[home_abbr, away_abbr])
            
            if not players_df.empty:
                print(f"   Found {len(players_df)} star players for {game_key}")
                
                # Generate props for each player
                for _, player_row in players_df.iterrows():
                    player_name = player_row['PLAYER_NAME']
                    
                    # Skip if player is known to be unavailable
                    if available_players and player_name in available_players.get('injured', []):
                        print(f"      ⚠️ Skipping {player_name} (injury concern)")
                        continue
                    
                    # Get player sentiment and news
                    sentiment = get_player_sentiment_and_news(player_name, player_row['TEAM_ABBREVIATION'])
                    
                    # Adjust confidence based on sentiment
                    sentiment_boost = (sentiment['sentiment_score'] - 0.5) * 0.1  # -0.05 to +0.05
                    
                    # Create prop predictions with realistic lines
                    props = {}
                    
                    # Points prop with RMSE-weighted accuracy
                    if player_row['PTS'] > 0:
                        import random
                        line_adjustment = random.choice([-2.5, -1.5, -0.5, 0.5, 1.5])
                        pts_line = player_row['PTS'] + line_adjustment
                        pts_prediction = player_row['PTS']
                        edge = pts_prediction - pts_line
                        
                        # Calculate accuracy factor based on RMSE (lower RMSE = higher accuracy)
                        pts_rmse = prop_model_rmse.get('points', 1.0)
                        accuracy_factor = max(0.3, 1.0 - (pts_rmse / 5.0))  # Scale RMSE to 0.3-1.0 range
                        
                        # Weight confidence by both edge and model accuracy
                        base_confidence = 0.55 + (abs(edge) * 0.08) + sentiment_boost
                        weighted_confidence = base_confidence * accuracy_factor
                        
                        props['points'] = {
                            'prediction': pts_prediction,
                            'line': pts_line,
                            'edge': edge,
                            'confidence': min(weighted_confidence, 0.85),
                            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                            'uncertainty': pts_rmse / 10.0,  # Use RMSE for uncertainty
                            'market_odds': 0,
                            'public_percentage': 0.5,
                            'sharp_money': 0,
                            'sentiment': sentiment['sentiment_score'],
                            'hot_streak': sentiment.get('hot_streak', False),
                            'rmse': pts_rmse,
                            'accuracy_factor': accuracy_factor
                        }
                    
                    # Rebounds prop with RMSE-weighted accuracy
                    if player_row['REB'] > 0:
                        line_adjustment = random.choice([-1.5, -0.5, 0.5, 1.5])
                        reb_line = player_row['REB'] + line_adjustment
                        reb_prediction = player_row['REB']
                        edge = reb_prediction - reb_line
                        
                        reb_rmse = prop_model_rmse.get('rebounds', 1.5)
                        accuracy_factor = max(0.3, 1.0 - (reb_rmse / 5.0))
                        
                        base_confidence = 0.52 + (abs(edge) * 0.08) + sentiment_boost
                        weighted_confidence = base_confidence * accuracy_factor
                        
                        props['rebounds'] = {
                            'prediction': reb_prediction,
                            'line': reb_line,
                            'edge': edge,
                            'confidence': min(weighted_confidence, 0.80),
                            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                            'uncertainty': reb_rmse / 10.0,
                            'market_odds': 0,
                            'public_percentage': 0.5,
                            'sharp_money': 0,
                            'sentiment': sentiment['sentiment_score'],
                            'rmse': reb_rmse,
                            'accuracy_factor': accuracy_factor
                        }
                    
                    # Assists prop with RMSE-weighted accuracy
                    if player_row['AST'] > 0:
                        line_adjustment = random.choice([-1.5, -0.5, 0.5, 1.5])
                        ast_line = player_row['AST'] + line_adjustment
                        ast_prediction = player_row['AST']
                        edge = ast_prediction - ast_line
                        
                        ast_rmse = prop_model_rmse.get('assists', 1.5)
                        accuracy_factor = max(0.3, 1.0 - (ast_rmse / 5.0))
                        
                        base_confidence = 0.50 + (abs(edge) * 0.08) + sentiment_boost
                        weighted_confidence = base_confidence * accuracy_factor
                        
                        props['assists'] = {
                            'prediction': ast_prediction,
                            'line': ast_line,
                            'edge': edge,
                            'confidence': min(weighted_confidence, 0.78),
                            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                            'uncertainty': ast_rmse / 10.0,
                            'market_odds': 0,
                            'public_percentage': 0.5,
                            'sharp_money': 0,
                            'sentiment': sentiment['sentiment_score'],
                            'rmse': ast_rmse,
                            'accuracy_factor': accuracy_factor
                        }
                    
                    # Three-pointers prop with RMSE-weighted accuracy (most accurate!)
                    if player_row['FG3M'] > 0:
                        line_adjustment = random.choice([-1.0, -0.5, 0.5, 1.0])
                        threes_line = player_row['FG3M'] + line_adjustment
                        threes_prediction = player_row['FG3M']
                        edge = threes_prediction - threes_line
                        
                        threes_rmse = prop_model_rmse.get('threes', 0.5)  # Best RMSE!
                        accuracy_factor = max(0.3, 1.0 - (threes_rmse / 5.0))  # Will be ~0.9!
                        
                        base_confidence = 0.54 + (abs(edge) * 0.08) + sentiment_boost
                        weighted_confidence = base_confidence * accuracy_factor
                        
                        props['threes'] = {
                            'prediction': threes_prediction,
                            'line': threes_line,
                            'edge': edge,
                            'confidence': min(weighted_confidence, 0.82),
                            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                            'uncertainty': threes_rmse / 10.0,  # Very low uncertainty!
                            'market_odds': 0,
                            'public_percentage': 0.5,
                            'sharp_money': 0,
                            'sentiment': sentiment['sentiment_score'],
                            'rmse': threes_rmse,
                            'accuracy_factor': accuracy_factor
                        }
                    
                    # Add steals + blocks combo prop (lower confidence due to volatility)
                    if player_row['STL'] > 0 or player_row['BLK'] > 0:
                        stl_blk_total = player_row['STL'] + player_row['BLK']
                        line_adjustment = random.choice([-0.5, 0.5, 1.5])
                        stl_blk_line = stl_blk_total + line_adjustment
                        edge = stl_blk_total - stl_blk_line
                        
                        # Defensive stats are more volatile, use lower confidence
                        stl_blk_rmse = 1.8  # Estimate higher uncertainty
                        accuracy_factor = max(0.3, 1.0 - (stl_blk_rmse / 5.0))
                        
                        base_confidence = 0.48 + (abs(edge) * 0.08) + sentiment_boost
                        weighted_confidence = base_confidence * accuracy_factor
                        
                        props['steals_blocks'] = {
                            'prediction': stl_blk_total,
                            'line': stl_blk_line,
                            'edge': edge,
                            'confidence': min(weighted_confidence, 0.75),
                            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                            'uncertainty': stl_blk_rmse / 10.0,
                            'market_odds': 0,
                            'public_percentage': 0.5,
                            'sharp_money': 0,
                            'sentiment': sentiment['sentiment_score'],
                            'rmse': stl_blk_rmse,
                            'accuracy_factor': accuracy_factor
                        }
                    
                    # Only add high-quality props (confidence > 0.42 for more variety while staying accurate)
                    strong_props = {k: v for k, v in props.items() 
                                  if v['recommendation'] != 'PASS' and v['confidence'] > 0.42}
                    
                    if strong_props:
                        # Only store strong props to reduce combinations
                        player_predictions[player_name] = strong_props
                        
                        # Show props with accuracy indicators
                        hot_indicator = "🔥" if sentiment.get('hot_streak') else ""
                        avg_accuracy = np.mean([p.get('accuracy_factor', 0.5) for p in strong_props.values()])
                        accuracy_indicator = "⭐" if avg_accuracy > 0.75 else ""
                        
                        print(f"      • {player_name}{hot_indicator}{accuracy_indicator}: {len(strong_props)} props ({', '.join(strong_props.keys())})")
        
        con.close()
        
        print(f"   ✅ Generated props for {len(player_predictions)} players")
        
    except Exception as e:
        print(f"   ⚠️ Error generating player props: {e}")
        import traceback
        traceback.print_exc()
    
    return player_predictions


def get_team_abbreviation(team_name):
    """Convert full team name to abbreviation"""
    abbrev_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }
    
    # Try exact match first
    if team_name in abbrev_map:
        return abbrev_map[team_name]
    
    # Try partial match
    for full_name, abbr in abbrev_map.items():
        if team_name in full_name or full_name in team_name:
            return abbr
    
    # Default: use first 3 letters
    return team_name[:3].upper()

def generate_parlays(predictions, min_confidence=0.3, max_legs=6):
    """Generate AI-powered parlay combinations with enhanced player props"""
    print(f"\n🎲 Generating AI-powered parlays...")
    
    try:
        # Load parlay predictor
        sys.path.append('src/Predict')
        from ParlayPredictor import AdvancedParlayPredictor
        
        parlay_predictor = AdvancedParlayPredictor()
        
        # Load player data for prop predictions
        player_data = parlay_predictor.load_player_data()
        
        prop_model_rmse = {}  # Store RMSE for accuracy weighting
        
        if not player_data.empty:
            print("✅ Loaded player database for prop predictions")
            # Calculate correlations
            parlay_predictor.calculate_advanced_correlations(player_data)
            # Train prop models and capture RMSE scores
            parlay_predictor.train_player_prop_models(player_data)
            
            # Extract RMSE scores for accuracy weighting
            for prop_type, model_info in parlay_predictor.prop_models.items():
                if model_info and 'rmse' in model_info:
                    prop_model_rmse[prop_type] = model_info['rmse']
                    print(f"   {prop_type.title()} model accuracy: RMSE={model_info['rmse']:.3f}")
        else:
            print("⚠️ Player data unavailable, using game predictions only")
        
        # Convert predictions to game predictions format
        game_predictions = {}
        game_info = {}  # Store game info for player prop generation
        
        for i, pred in enumerate(predictions):
            if pred:
                game_key = f"{pred['away_team']} @ {pred['home_team']}"
                game_predictions[game_key] = {
                    'probability': pred['home_probability'],
                    'confidence': pred['confidence'],
                    'edge': pred['home_probability'] - 0.5,
                    'recommendation': 'ML Home' if pred['home_probability'] > 0.5 else 'ML Away',
                    'uncertainty': 0.1,
                    'market_odds': pred.get('kelly_home', {}).get('kelly_fraction', 0),
                    'public_percentage': 0.5,
                    'sharp_money': 0
                }
                
                # Store game info for player props
                game_info[game_key] = {
                    'home_team': pred['home_team'],
                    'away_team': pred['away_team']
                }
        
        # Check player availability and injuries first
        print("🏥 Checking player availability and injuries...")
        available_players = check_player_availability(game_info)
        
        # Generate ACTUAL player prop predictions from database
        print("🏀 Generating player prop predictions (RMSE-weighted)...")
        player_predictions = generate_player_props_for_games(game_info, parlay_predictor, available_players, prop_model_rmse)
        
        # Filter by confidence threshold - be more flexible
        high_conf_games = {k: v for k, v in game_predictions.items() if v['confidence'] > min_confidence}
        
        if len(high_conf_games) < 2:
            print(f"⚠️ Not enough high-confidence games for parlays (found {len(high_conf_games)}, need 2+)")
            print(f"💡 Using all available games for parlays (lowered threshold)")
            
            # Use all available games if we have at least 2
            if len(game_predictions) >= 2:
                print("🔄 Creating parlays with all available predictions...")
                high_conf_games = game_predictions
            else:
                print(f"❌ Only {len(game_predictions)} game(s) available - need at least 2 for parlays")
                return []
        
        if len(high_conf_games) < 2:
            return []
        
        # Generate advanced parlay combinations
        parlays = parlay_predictor.generate_advanced_parlay_combinations(
            high_conf_games, 
            player_predictions,
            max_legs=max_legs,
            min_confidence=min_confidence * 0.8  # Lower threshold for individual legs
        )
        
        print(f"✅ Generated {len(parlays)} parlay combinations")
        return parlays
        
    except Exception as e:
        print(f"⚠️ Parlay generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def _print_game(i, pred, show_details):
    """Print one game's prediction"""
    print(f"\nGAME {i}: {pred['away_team']} @ {pred['home_team']}")
    print("-" * 50)
    winner = pred['home_team'] if pred['home_probability'] > 0.5 else pred['away_team']
    prob = max(pred['home_probability'], pred['away_probability'])
    print(f"PREDICTED WINNER: {winner} ({prob:.1%})")
    model_home = pred.get('model_home_probability')
    market_home = pred.get('market_home_probability')
    if market_home is not None:
        print(f"MODEL vs MARKET: {pred['home_team']} {model_home:.1%} model / {market_home:.1%} bookmakers")
        print(f"ODDS ({pred.get('odds_source')}): {pred['home_team']} {pred['home_odds']:+.0f}, "
              f"{pred['away_team']} {pred['away_odds']:+.0f}")
        print(f"EXPECTED VALUE: {pred['home_team']} {pred['ev_home']:+.1%}, {pred['away_team']} {pred['ev_away']:+.1%}")
    else:
        print(f"MODEL: {pred['home_team']} {model_home:.1%} (no odds available)")

    if pred.get('sentiment_adjustment') and abs(pred['sentiment_adjustment']) > 0.01:
        adj_pct = pred['sentiment_adjustment'] * 100
        adj_direction = "+" if adj_pct > 0 else ""
        print(f"SENTIMENT ADJUST: {adj_direction}{adj_pct:.1f}% (from news/social)")
        if pred.get('sentiment_narrative'):
            print(f"    {pred['sentiment_narrative']}")
        if pred.get('contrarian_opportunity'):
            print(f"    WARNING: CONTRARIAN OPPORTUNITY (public overconfident)")

    print(f"RECOMMENDATION: {pred['recommendation']}")
    if pred['kelly_home']['bet_amount'] > 0:
        print(f"KELLY BET (HOME): ${pred['kelly_home']['bet_amount']:.0f} ({pred['kelly_home']['kelly_fraction']:.1%})")
    if pred['kelly_away']['bet_amount'] > 0:
        print(f"KELLY BET (AWAY): ${pred['kelly_away']['bet_amount']:.0f} ({pred['kelly_away']['kelly_fraction']:.1%})")

    if show_details and pred.get('real_time_data'):
        rt_data = pred['real_time_data']
        if 'injury_scores' in rt_data:
            home_inj = rt_data['injury_scores']['home_team']
            away_inj = rt_data['injury_scores']['away_team']
            if home_inj > 0 or away_inj > 0:
                print(f"INJURY IMPACT: Home {home_inj:.2f}, Away {away_inj:.2f}")


def display_predictions(predictions, show_details=True, week_mode=False):
    """Display predictions in a formatted way"""
    time_range = "NEXT 7 DAYS" if week_mode else "NEXT 20 HOURS"
    print(f"\nNBA PREDICTIONS ({time_range} - UK TIMEZONE)")
    print("="*70)
    predictions = [p for p in predictions if p]
    if not week_mode:
        for i, pred in enumerate(predictions, 1):
            _print_game(i, pred, show_details)
        return
    import pytz
    games_by_date = {}
    for pred in predictions:
        try:
            game_time = datetime.fromisoformat(str(pred['game_time']).replace('Z', '+00:00'))
            date_key = game_time.astimezone(pytz.timezone('Europe/London')).strftime('%A, %B %d, %Y')
        except Exception:
            date_key = "TBD"
        games_by_date.setdefault(date_key, []).append(pred)
    for date_key, date_games in games_by_date.items():
        print(f"\n📅 {date_key}")
        print("="*50)
        for i, pred in enumerate(date_games, 1):
            _print_game(i, pred, show_details)

def display_parlays(parlays):
    """Display parlay recommendations"""
    if not parlays:
        return
    
    print(f"\n🎲 AI-POWERED PARLAY RECOMMENDATIONS")
    print("="*70)
    
    for i, parlay in enumerate(parlays[:5], 1):  # Show top 5
        print(f"\n🎯 PARLAY {i}:")
        
        # Show both original and boosted EV
        original_ev = parlay.get('original_expected_value', parlay.get('expected_value', 0))
        boosted_ev = parlay.get('boosted_expected_value', original_ev)
        
        if abs(boosted_ev - original_ev) > 0.001:
            print(f"💰 Expected Value: {original_ev:+.3f} → {boosted_ev:+.3f} (boosted)")
        else:
            print(f"💰 Expected Value: {boosted_ev:+.3f}")
            
        print(f"🎲 American Odds: {parlay.get('american_odds', 0):+.0f}")
        print(f"📊 Win Probability: {parlay.get('adjusted_probability', parlay.get('combined_probability', 0)):.1%}")
        print(f"🎯 Confidence: {parlay.get('confidence', 0):.1%}")
        print(f"⚠️ Risk Score: {parlay.get('risk_score', 0):.2f}")
        print(f"💎 Advanced Score: {parlay.get('advanced_score', 0):.1f}")
        
        kelly_size = parlay.get('kelly_bet_size', 0)
        if kelly_size > 0:
            print(f"💸 Kelly Bet Size: {kelly_size:.1%} of bankroll ✅")
        else:
            print(f"💸 Kelly Bet Size: 0.0% (MONITOR ONLY - No edge detected) ⚠️")
        
        print("🏀 Legs:")
        for j, leg in enumerate(parlay.get('legs', []), 1):
            if isinstance(leg, dict):
                print(f"   {j}. {leg.get('description', 'Unknown bet')}")
            else:
                print(f"   {j}. {leg}")

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='NBA ML Prediction Script')
    parser.add_argument('--sportsbook', default='fanduel', 
                       choices=['fanduel', 'draftkings', 'betmgm', 'caesars'],
                       help='Sportsbook for odds')
    parser.add_argument('--parlays', action='store_true', help='Generate parlay recommendations')
    parser.add_argument('--real-time', action='store_true', help='Use real-time data')
    parser.add_argument('--sentiment', action='store_true', default=True, help='Include sentiment from free news sources (default: enabled)')
    parser.add_argument('--no-sentiment', action='store_false', dest='sentiment', help='Disable sentiment analysis')
    parser.add_argument('--confidence', type=float, default=0.25, help='Minimum confidence for bets (default: 0.25)')
    parser.add_argument('--bankroll', type=float, default=1000, help='Total bankroll amount (default: $1000)')
    parser.add_argument('--kc', '--kelly', action='store_true', dest='kelly_criterion',
                       help='Kelly Criterion bet sizing (the default; kept for compatibility)')
    parser.add_argument('--scaled-kelly', action='store_true',
                       help='Aggressive mode: scale Kelly stakes so 100%% of the bankroll is bet')
    parser.add_argument('--kelly-fraction', type=float, default=0.5,
                       help='Fraction of full Kelly to stake (default: 0.5 = half Kelly)')
    parser.add_argument('--min-edge', type=float, default=0.03,
                       help='Minimum expected value at the offered odds to recommend a bet (default: 0.03 = +3%%)')
    parser.add_argument('--no-details', action='store_true', help='Hide detailed analysis')
    parser.add_argument('--list-models', '--model-info', action='store_true', dest='list_models',
                       help='Show the prediction model and its test results, then exit')
    parser.add_argument('--week', action='store_true', help='Get predictions for the entire week (next 7 days) instead of just today')
    
    args = parser.parse_args()
    
    print_header()
    
    # Handle model info
    if args.list_models:
        predictor = load_prediction_system()
        if predictor:
            print(json.dumps(predictor.metadata.get('evaluation', {}), indent=2))
        return bool(predictor)
    
    # Display bet sizing mode
    print("\nBET SIZING CONFIGURATION")
    print("="*70)
    if not args.scaled_kelly:
        print(f"Mode: Kelly Criterion ({args.kelly_fraction:g}x Kelly, max 5% per bet)")
        print(f"   • Only bets with at least {args.min_edge:+.0%} expected value at the offered odds")
        print("   • Reduces variance and drawdowns")
        print("   • Capital preservation priority")
    else:
        print("Mode: Scaled Kelly (Aggressive - 100% Allocation)")
        print("   • Uses Kelly Criterion proportions (confidence-based sizing)")
        print("   • Scaled to deploy 100% of bankroll")
        print("   • Higher confidence bets get larger allocations")
        print("   • Uses all capital while respecting relative edges")
        print("   • WARNING: Full bankroll at risk - use with caution!")
    print(f"Bankroll: ${args.bankroll:,.2f}")
    print("="*70 + "\n")
    
    # Load prediction system
    predictor = load_prediction_system()
    if not predictor:
        return False
    
    # Load real-time data provider
    real_time_provider = None
    if args.real_time:
        real_time_provider = load_real_time_data()
        if real_time_provider:
            print("[OK] Real-time data provider loaded")
        else:
            print("[WARNING] Real-time data unavailable, using base predictions")
    
    # Load sentiment analyzer
    sentiment_analyzer = None
    if args.sentiment:
        print("Loading sentiment analyzer (ESPN, Reddit, injury news)...")
        sentiment_analyzer = load_sentiment_analyzer()
        if sentiment_analyzer:
            print("[OK] Sentiment analysis enabled (free news sources)")
        else:
            print("[WARNING] Sentiment analysis unavailable, using base predictions")
    
    # Get games for the next 20 hours or 7 days (UK timezone)
    games = get_todays_games(args.sportsbook, week_mode=args.week)
    if not games:
        time_range = "next 7 days" if args.week else "next 20 hours"
        print(f"[ERROR] No games found for the {time_range} (UK timezone)")
        print("[INFO] Check your internet connection or try a different sportsbook")
        return False
    
    # Make predictions for each game
    print(f"Making predictions for {len(games)} games...")
    predictions = []
    
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        
        print(f"  Analyzing: {away_team} @ {home_team}...")
        
        # Get sentiment analysis if enabled
        sentiment_data = None
        if sentiment_analyzer:
            try:
                sentiment_data = sentiment_analyzer.get_game_sentiment(home_team, away_team)
                if sentiment_data and abs(sentiment_data.get('sentiment_differential', 0)) > 0.1:
                    print(f"    Sentiment: {sentiment_data.get('narrative', 'Neutral')}")
            except Exception as e:
                pass  # Silently continue if sentiment fails
        
        # Real-time injury/market info (display only)
        real_time_data = None
        if real_time_provider:
            try:
                real_time_data = real_time_provider.get_comprehensive_game_data(home_team, away_team, datetime.now())
            except Exception:
                pass
        
        # Model prediction, sentiment adjustment and value against the odds
        prediction = make_game_prediction(
            predictor, game, bankroll=args.bankroll, sentiment_data=sentiment_data,
            min_edge=args.min_edge, kelly_multiplier=args.kelly_fraction, real_time_data=real_time_data
        )
        
        if prediction:
            predictions.append(prediction)
    
    # Display predictions
    if predictions:
        display_predictions(predictions, show_details=not args.no_details, week_mode=args.week)
        
        # Generate parlays if requested
        if args.parlays:
            parlays = generate_parlays(predictions, min_confidence=args.confidence)
            display_parlays(parlays)
        
        # Summary statistics
        print(f"\nPREDICTION SUMMARY")
        print("="*70)
        
        high_conf_count = sum(1 for p in predictions if p['confidence'] > args.confidence)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        recommended_bets = sum(1 for p in predictions if p.get('bet_side'))
        
        print(f"Total Games Analyzed: {len(predictions)}")
        print(f"High Confidence Games: {high_conf_count}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Recommended Bets: {recommended_bets}")
        
        # Determine allocation mode
        if not args.scaled_kelly:
            # KELLY CRITERION MODE (Conservative, optimal bet sizing)
            print(f"\n💰 BET SIZING MODE: Kelly Criterion (Conservative)")
            print("-" * 70)
            
            if recommended_bets > 0:
                total_kelly = sum([
                    max(p['kelly_home']['bet_amount'], p['kelly_away']['bet_amount'])
                    for p in predictions
                ])
                print(f"Total Kelly Bet Amount: ${total_kelly:.2f}")
                print(f"Bankroll Utilization: {total_kelly/args.bankroll:.1%}")
                print(f"Remaining Bankroll: ${max(0, args.bankroll - total_kelly):.2f}")
                
                # Display bankroll allocation breakdown
                print(f"\nBANKROLL ALLOCATION (Total: ${args.bankroll:.2f})")
                print("="*70)
                
                bet_num = 1
                total_kelly = 0
                
                for pred in predictions:
                    if pred['kelly_home']['bet_amount'] > 0:
                        amount = pred['kelly_home']['bet_amount']
                        pct = (amount / args.bankroll) * 100
                        print(f"  {bet_num}. {pred['home_team']} ML: ${amount:.2f} ({pct:.1f}%)")
                        bet_num += 1
                        total_kelly += amount
                    if pred['kelly_away']['bet_amount'] > 0:
                        amount = pred['kelly_away']['bet_amount']
                        pct = (amount / args.bankroll) * 100
                        print(f"  {bet_num}. {pred['away_team']} ML: ${amount:.2f} ({pct:.1f}%)")
                        bet_num += 1
                        total_kelly += amount
                
                # Add top parlays to allocation (show all high-quality parlays)
                if args.parlays and parlays:
                    print(f"\n  💎 Top Parlays:")
                    parlay_count = 0
                    for i, parlay in enumerate(parlays[:5], 1):
                        # Show parlays with positive boosted EV or high confidence
                        boosted_ev = parlay.get('boosted_expected_value', parlay.get('expected_value', 0))
                        confidence = parlay.get('confidence', 0)
                        
                        if parlay['kelly_bet_size'] > 0:
                            amount = args.bankroll * parlay['kelly_bet_size']
                            pct = parlay['kelly_bet_size'] * 100
                            print(f"  {bet_num}. Parlay #{i} ({len(parlay['legs'])} legs): ${amount:.2f} ({pct:.1f}%) ✅")
                            bet_num += 1
                            parlay_count += 1
                            total_kelly += amount
                        elif boosted_ev > -0.02 and confidence > 0.65:
                            # Show high-confidence parlays even with 0% Kelly (monitor bets)
                            print(f"  {bet_num}. Parlay #{i} ({len(parlay['legs'])} legs): $0.00 (0.0%) ⚠️ MONITOR ONLY")
                            bet_num += 1
                            parlay_count += 1
                    
                    if parlay_count == 0:
                        print(f"  ⚠️ No parlays meet Kelly criteria (all 0% sizing)")
                
                print(f"\n  {'='*66}")
                print(f"  TOTAL ALLOCATED: ${total_kelly:.2f} ({total_kelly/args.bankroll:.1%})")
                print(f"  REMAINING: ${max(0, args.bankroll - total_kelly):.2f}")
        else:
            # SCALED KELLY MODE (uses Kelly proportions, scaled to 100%)
            if recommended_bets > 0:
                parlays_to_include = parlays if args.parlays else []
                allocated_bets, total_allocated = calculate_scaled_kelly_bets(
                    predictions, parlays_to_include, args.bankroll, use_parlays=args.parlays
                )
                
                # Display bankroll allocation breakdown
                print(f"\nBANKROLL ALLOCATION (Total: ${args.bankroll:.2f})")
                print("="*70)
                
                bet_num = 1
                for bet in allocated_bets:
                    if bet['type'] == 'game':
                        team = bet['team']
                        amount = bet['amount']
                        pct = bet['percentage']
                        original_kelly = bet['original_kelly_pct']
                        
                        # Show original Kelly → scaled percentage
                        print(f"  {bet_num}. {team} ML: ${amount:.2f} ({pct:.1f}%) [Kelly: {original_kelly:.1f}% -> {pct:.1f}%]")
                        bet_num += 1
                    elif bet['type'] == 'parlay':
                        parlay_num = bet['parlay_num']
                        legs = bet['legs']
                        amount = bet['amount']
                        pct = bet['percentage']
                        original_kelly = bet['original_kelly_pct']
                        
                        print(f"  {bet_num}. Parlay #{parlay_num} ({legs} legs): ${amount:.2f} ({pct:.1f}%) [Kelly: {original_kelly:.1f}% -> {pct:.1f}%]")
                        bet_num += 1
                
                # Show parlays section if enabled
                if args.parlays and parlays:
                    parlay_in_allocation = sum(1 for b in allocated_bets if b['type'] == 'parlay')
                    if parlay_in_allocation > 0:
                        print(f"\n  💎 Parlays Included: {parlay_in_allocation} (confidence-weighted)")
                
                print(f"\n  {'='*66}")
                print(f"  TOTAL ALLOCATED: ${total_allocated:.2f} (100.0%)")
                print(f"  REMAINING: $0.00")
    
    else:
        print("[ERROR] No predictions could be generated")
        return False
    
    # Save predictions to Excel with parlays and bankroll allocation
    save_predictions_to_excel(predictions, parlays if args.parlays else [], args.sportsbook, args.bankroll, week_mode=args.week)
    
    if sentiment_analyzer:
        print("\nNote: the model was backtested without the sentiment adjustment; that part is untested.")
    print(f"\n🎉 PREDICTION ANALYSIS COMPLETE!")
    print("💡 Remember: Bet responsibly and within your means!")
    
    return True

def save_predictions_to_excel(predictions, parlays, sportsbook, bankroll, week_mode=False):
    """Save predictions to formatted Excel file with multiple sheets"""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        os.makedirs("Predictions", exist_ok=True)
        time_suffix = "week" if week_mode else "daily"
        filename = f"Predictions/predictions_{time_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Game Predictions
            pred_data = []
            for pred in predictions:
                market = pred.get('market_home_probability')
                pred_data.append({
                    'Game': f"{pred['away_team']} @ {pred['home_team']}",
                    'Predicted Winner': pred['home_team'] if pred['prediction'] == 'HOME' else pred['away_team'],
                    'Win Probability': f"{max(pred['home_probability'], pred['away_probability']):.1%}",
                    'Model Home Win %': f"{pred['model_home_probability']:.1%}",
                    'Bookmaker Home Win %': f"{market:.1%}" if market is not None else "-",
                    'Odds (Home/Away)': (f"{pred['home_odds']:+.0f} / {pred['away_odds']:+.0f}"
                                         if market is not None else "-"),
                    'Odds Source': pred.get('odds_source') or "-",
                    'EV (Home/Away)': (f"{pred['ev_home']:+.1%} / {pred['ev_away']:+.1%}"
                                       if market is not None else "-"),
                    'Confidence': f"{pred['confidence']:.1%}",
                    'Recommendation': pred['recommendation'],
                    'Kelly Bet (Home)': f"${pred['kelly_home']['bet_amount']:.2f}" if pred['kelly_home']['bet_amount'] > 0 else "-",
                    'Kelly Bet (Away)': f"${pred['kelly_away']['bet_amount']:.2f}" if pred['kelly_away']['bet_amount'] > 0 else "-",
                })
            
            df_games = pd.DataFrame(pred_data)
            df_games.to_excel(writer, sheet_name='Game Predictions', index=False)
            
            # Format Game Predictions sheet
            ws_games = writer.sheets['Game Predictions']
            ws_games.column_dimensions['A'].width = 35
            ws_games.column_dimensions['B'].width = 25
            ws_games.column_dimensions['C'].width = 18
            ws_games.column_dimensions['D'].width = 15
            ws_games.column_dimensions['E'].width = 30
            ws_games.column_dimensions['F'].width = 18
            ws_games.column_dimensions['G'].width = 18
            
            # Style headers
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=12)
            for cell in ws_games[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Sheet 2: Parlays
            if parlays:
                parlay_data = []
                for i, parlay in enumerate(parlays[:10], 1):  # Top 10 parlays
                    legs_text = "\n".join([f"{j}. {leg}" for j, leg in enumerate(parlay['legs'], 1)])
                    parlay_data.append({
                        'Parlay #': i,
                        'Legs': legs_text,
                        'American Odds': f"{parlay['american_odds']:+.0f}",
                        'Win Probability': f"{parlay.get('adjusted_probability', parlay['combined_probability']):.1%}",
                        'Confidence': f"{parlay['confidence']:.1%}",
                        'Expected Value': f"{parlay['expected_value']:+.3f}",
                        'Risk Score': f"{parlay.get('risk_score', 0):.2f}",
                        'Kelly Bet Size': f"{parlay['kelly_bet_size']:.1%}"
                    })
                
                df_parlays = pd.DataFrame(parlay_data)
                df_parlays.to_excel(writer, sheet_name='Parlays', index=False)
                
                # Format Parlays sheet
                ws_parlays = writer.sheets['Parlays']
                ws_parlays.column_dimensions['A'].width = 10
                ws_parlays.column_dimensions['B'].width = 50
                ws_parlays.column_dimensions['C'].width = 15
                ws_parlays.column_dimensions['D'].width = 18
                ws_parlays.column_dimensions['E'].width = 15
                ws_parlays.column_dimensions['F'].width = 18
                ws_parlays.column_dimensions['G'].width = 15
                ws_parlays.column_dimensions['H'].width = 18
                
                for cell in ws_parlays[1]:
                    cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
                    cell.font = Font(bold=True, color="FFFFFF", size=12)
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Wrap text for legs column
                for row in range(2, len(parlay_data) + 2):
                    ws_parlays[f'B{row}'].alignment = Alignment(wrap_text=True, vertical='top')
                    ws_parlays.row_dimensions[row].height = 60
            
            # Sheet 3: Bankroll Allocation
            allocation_data = []
            
            # Add game bets
            total_allocated = 0
            for pred in predictions:
                if pred['kelly_home']['bet_amount'] > 0:
                    allocation_data.append({
                        'Bet Type': 'Game ML',
                        'Bet': f"{pred['home_team']} ML",
                        'Amount': pred['kelly_home']['bet_amount'],
                        'Percentage': f"{(pred['kelly_home']['bet_amount']/bankroll)*100:.1f}%"
                    })
                    total_allocated += pred['kelly_home']['bet_amount']
                
                if pred['kelly_away']['bet_amount'] > 0:
                    allocation_data.append({
                        'Bet Type': 'Game ML',
                        'Bet': f"{pred['away_team']} ML",
                        'Amount': pred['kelly_away']['bet_amount'],
                        'Percentage': f"{(pred['kelly_away']['bet_amount']/bankroll)*100:.1f}%"
                    })
                    total_allocated += pred['kelly_away']['bet_amount']
            
            # Add parlay bets (if any have positive Kelly size)
            if parlays:
                for i, parlay in enumerate(parlays[:5], 1):
                    if parlay['kelly_bet_size'] > 0:
                        parlay_amount = bankroll * parlay['kelly_bet_size']
                        allocation_data.append({
                            'Bet Type': 'Parlay',
                            'Bet': f"Parlay #{i} ({len(parlay['legs'])} legs)",
                            'Amount': parlay_amount,
                            'Percentage': f"{parlay['kelly_bet_size']*100:.1f}%"
                        })
                        total_allocated += parlay_amount
            
            # Add summary rows
            allocation_data.append({'Bet Type': '', 'Bet': '', 'Amount': '', 'Percentage': ''})
            allocation_data.append({
                'Bet Type': 'TOTAL',
                'Bet': 'Total Allocated',
                'Amount': total_allocated,
                'Percentage': f"{(total_allocated/bankroll)*100:.1f}%"
            })
            allocation_data.append({
                'Bet Type': 'REMAINING',
                'Bet': 'Remaining Bankroll',
                'Amount': max(0, bankroll - total_allocated),
                'Percentage': f"{max(0, (bankroll - total_allocated)/bankroll)*100:.1f}%"
            })
            
            df_allocation = pd.DataFrame(allocation_data)
            df_allocation.to_excel(writer, sheet_name='Bankroll Allocation', index=False)
            
            # Format Bankroll Allocation sheet
            ws_alloc = writer.sheets['Bankroll Allocation']
            ws_alloc.column_dimensions['A'].width = 15
            ws_alloc.column_dimensions['B'].width = 35
            ws_alloc.column_dimensions['C'].width = 15
            ws_alloc.column_dimensions['D'].width = 15
            
            for cell in ws_alloc[1]:
                cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF", size=12)
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Format amounts as currency
            for row in range(2, len(allocation_data) + 2):
                if ws_alloc[f'C{row}'].value and ws_alloc[f'C{row}'].value != '':
                    try:
                        ws_alloc[f'C{row}'].number_format = '$#,##0.00'
                    except:
                        pass
            
            # Highlight totals
            total_row = len(allocation_data) - 1
            for col in ['A', 'B', 'C', 'D']:
                ws_alloc[f'{col}{total_row}'].fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
                ws_alloc[f'{col}{total_row}'].font = Font(bold=True)
            
            # Sheet 4: Summary
            summary_data = [{
                'Metric': 'Total Bankroll',
                'Value': f"${bankroll:.2f}"
            }, {
                'Metric': 'Date',
                'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, {
                'Metric': 'Sportsbook',
                'Value': sportsbook
            }, {
                'Metric': 'Games Analyzed',
                'Value': len(predictions)
            }, {
                'Metric': 'Parlays Generated',
                'Value': len(parlays) if parlays else 0
            }, {
                'Metric': 'Total Allocated',
                'Value': f"${total_allocated:.2f}"
            }, {
                'Metric': 'Bankroll Utilization',
                'Value': f"{(total_allocated/bankroll)*100:.1f}%"
            }, {
                'Metric': 'Remaining',
                'Value': f"${max(0, bankroll - total_allocated):.2f}"
            }]
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format Summary sheet
            ws_summary = writer.sheets['Summary']
            ws_summary.column_dimensions['A'].width = 25
            ws_summary.column_dimensions['B'].width = 25
            
            for cell in ws_summary[1]:
                cell.fill = PatternFill(start_color="203764", end_color="203764", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF", size=12)
                cell.alignment = Alignment(horizontal='center', vertical='center')
        
        print(f"Predictions saved to Excel: {filename}")
        return filename
        
    except Exception as e:
        print(f"[WARNING] Could not save to Excel: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

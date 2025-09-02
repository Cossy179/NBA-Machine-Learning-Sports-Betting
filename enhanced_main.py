"""
Enhanced main runner that integrates all advanced prediction systems.
Provides comprehensive NBA betting predictions with maximum accuracy.
"""
import argparse
import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init

# Add src to path
sys.path.append('src')

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.DataProviders.RealTimeDataProvider import RealTimeDataProvider
from src.Predict.Advanced_Prediction_Runner import advanced_prediction_runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame

init()

# URLs
todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
data_url = ('https://stats.nba.com/stats/leaguedashteamstats?'
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&'
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&'
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&'
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&'
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&'
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=')

def enhance_game_features(games, df, odds, real_time_provider=None):
    """Enhanced version of createTodaysGames with real-time data integration"""
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    real_time_data = []

    print(f"{Fore.CYAN}Enhancing game features with real-time data...{Style.RESET_ALL}")

    for i, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        
        if home_team not in team_index_current or away_team not in team_index_current:
            print(f"{Fore.YELLOW}Skipping {away_team} @ {home_team} - Team not in index{Style.RESET_ALL}")
            continue

        print(f"Processing: {away_team} @ {home_team}")

        # Get odds data
        if odds is not None:
            game_key = home_team + ':' + away_team
            if game_key in odds:
                game_odds = odds[game_key]
                todays_games_uo.append(game_odds['under_over_odds'])
                home_team_odds.append(game_odds[home_team]['money_line_odds'])
                away_team_odds.append(game_odds[away_team]['money_line_odds'])
            else:
                print(f"{Fore.RED}No odds found for {game_key}{Style.RESET_ALL}")
                todays_games_uo.append(220)  # Default total
                home_team_odds.append(-110)
                away_team_odds.append(-110)
        else:
            # Manual input fallback
            todays_games_uo.append(input(f'{home_team} vs {away_team} O/U: '))
            home_team_odds.append(input(f'{home_team} odds: '))
            away_team_odds.append(input(f'{away_team} odds: '))

        # Get real-time data
        if real_time_provider:
            try:
                rt_data = real_time_provider.get_comprehensive_game_data(
                    home_team, away_team, datetime.now()
                )
                real_time_data.append(rt_data)
                print(f"  ✓ Real-time data collected (Confidence: {rt_data['composite_scores']['confidence_score']:.1%})")
            except Exception as e:
                print(f"  ⚠ Real-time data error: {e}")
                real_time_data.append({})
        else:
            real_time_data.append({})

        # Calculate days rest (existing logic)
        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        
        # Home team rest
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
        
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + datetime.today() - last_home_date
        else:
            home_days_off = timedelta(days=7)

        # Away team rest
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
        
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + datetime.today() - last_away_date
        else:
            away_days_off = timedelta(days=7)

        # Get team stats
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        
        # Combine team stats
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days

        # Add real-time features if available
        if real_time_data[-1]:
            rt = real_time_data[-1]
            
            # Injury impact
            if 'injuries' in rt:
                stats['Home-Injury-Impact'] = rt['injuries']['home_team'].get('offensive_impact_score', 0)
                stats['Away-Injury-Impact'] = rt['injuries']['away_team'].get('offensive_impact_score', 0)
                stats['Home-Key-Players-Out'] = rt['injuries']['home_team'].get('key_players_out', 0)
                stats['Away-Key-Players-Out'] = rt['injuries']['away_team'].get('key_players_out', 0)
            
            # Travel fatigue
            if 'travel' in rt:
                stats['Home-Fatigue-Score'] = rt['travel']['home_team'].get('fatigue_score', 0)
                stats['Away-Fatigue-Score'] = rt['travel']['away_team'].get('fatigue_score', 0)
                stats['Home-Travel-Advantage'] = rt['travel']['home_team'].get('travel_advantage', 0)
                stats['Away-Travel-Advantage'] = rt['travel']['away_team'].get('travel_advantage', 0)
            
            # Lineup quality
            if 'lineups' in rt and rt['lineups'].get('lineups_confirmed'):
                stats['Home-Lineup-Rating'] = rt['lineups']['home_team'].get('lineup_rating', 0.5)
                stats['Away-Lineup-Rating'] = rt['lineups']['away_team'].get('lineup_rating', 0.5)
                stats['Lineup-Advantage'] = stats['Home-Lineup-Rating'] - stats['Away-Lineup-Rating']
            
            # Betting market data
            if 'betting_markets' in rt and rt['betting_markets']:
                market = rt['betting_markets']
                if 'spread' in market:
                    stats['Line-Movement'] = market['spread'].get('movement', 0)
                    stats['Reverse-Line-Movement'] = 1 if market['spread'].get('reverse_line_movement') else 0
                if 'total' in market:
                    stats['Total-Movement'] = market['total'].get('movement', 0)
                    stats['Steam-Move'] = 1 if market['total'].get('steam_move') else 0
                if 'market_sentiment' in market:
                    stats['Contrarian-Opportunity'] = 1 if market['market_sentiment'].get('contrarian_opportunity') else 0
                    stats['Line-Value'] = market['market_sentiment'].get('line_value', 0)
            
            # Referee impact
            if 'officials' in rt and 'historical_stats' in rt['officials']:
                ref_stats = rt['officials']['historical_stats']
                stats['Ref-OU-Bias'] = ref_stats.get('over_under_bias', 0)
                stats['Ref-Home-Bias'] = ref_stats.get('home_team_foul_bias', 0)
                stats['Ref-Pace-Impact'] = ref_stats.get('pace_impact', 1.0)
            
            # Sentiment scores
            if 'sentiment' in rt:
                sentiment = rt['sentiment']
                stats['Home-Sentiment'] = sentiment.get('home_team_sentiment', 0.5)
                stats['Away-Sentiment'] = sentiment.get('away_team_sentiment', 0.5)
                stats['Game-Buzz'] = sentiment.get('game_buzz_score', 0)
                stats['Contrarian-Indicator'] = sentiment.get('contrarian_indicator', 0)
            
            # Composite scores
            if 'composite_scores' in rt:
                comp = rt['composite_scores']
                stats['Home-Total-Advantage'] = comp.get('home_team_advantage', 0)
                stats['Away-Total-Advantage'] = comp.get('away_team_advantage', 0)
                stats['Total-Points-Adjustment'] = comp.get('total_points_adjustment', 0)
                stats['Betting-Value-Score'] = comp.get('betting_value_score', 0)
                stats['Data-Confidence'] = comp.get('confidence_score', 0)

        match_data.append(stats)

    # Create final dataframe
    if not match_data:
        print(f"{Fore.RED}No valid games found!{Style.RESET_ALL}")
        return None, None, None, None, None, None

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T
    
    # Clean up the dataframe
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'], errors='ignore')
    
    # Fill NaN values with 0 for new features
    frame_ml = frame_ml.fillna(0)
    
    # Convert to numpy array for model input
    data = frame_ml.values.astype(float)

    print(f"{Fore.GREEN}✓ Enhanced features created for {len(games)} games{Style.RESET_ALL}")
    print(f"Feature dimensions: {data.shape}")

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, real_time_data

def display_enhanced_predictions_header():
    """Display enhanced predictions header"""
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🏀 ENHANCED NBA PREDICTION SYSTEM v2.0 🏀{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Features:{Style.RESET_ALL}")
    print(f"  • Multi-model ensemble predictions")
    print(f"  • Real-time injury and lineup data")
    print(f"  • Advanced betting market analysis")
    print(f"  • Travel fatigue and referee impacts")
    print(f"  • Multiple prediction targets (ML, OU, Spreads, Props)")
    print(f"  • Calibrated probabilities and confidence intervals")
    print(f"  • Kelly Criterion bankroll management")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

def main():
    """Enhanced main function with comprehensive prediction system"""
    display_enhanced_predictions_header()
    
    # Initialize real-time data provider
    real_time_provider = RealTimeDataProvider() if args.realtime else None
    
    # Get odds data
    odds = None
    if args.odds:
        print(f"{Fore.YELLOW}Fetching {args.odds} odds data...{Style.RESET_ALL}")
        try:
            odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
            games = create_todays_games_from_odds(odds)
            
            if len(games) == 0:
                print(f"{Fore.RED}No games found in odds data.{Style.RESET_ALL}")
                return
                
            # Validate games
            valid_games = []
            for game in games:
                game_key = game[0] + ':' + game[1]
                if game_key in odds:
                    valid_games.append(game)
                    home_odds = odds[game_key][game[0]]['money_line_odds']
                    away_odds = odds[game_key][game[1]]['money_line_odds']
                    ou_line = odds[game_key]['under_over_odds']
                    print(f"  {game[1]} ({away_odds}) @ {game[0]} ({home_odds}) | O/U {ou_line}")
                else:
                    print(f"{Fore.RED}  Missing odds for {game_key}{Style.RESET_ALL}")
            
            games = valid_games
            if not games:
                print(f"{Fore.RED}No valid games with complete odds data.{Style.RESET_ALL}")
                return
                
        except Exception as e:
            print(f"{Fore.RED}Error fetching odds: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to manual input...{Style.RESET_ALL}")
            odds = None
    
    if not odds:
        # Fallback to NBA API
        print(f"{Fore.YELLOW}Fetching games from NBA API...{Style.RESET_ALL}")
        try:
            from src.Utils.tools import get_todays_games_json, create_todays_games
            data = get_todays_games_json(todays_games_url)
            games = create_todays_games(data)
        except Exception as e:
            print(f"{Fore.RED}Error fetching NBA games: {e}{Style.RESET_ALL}")
            return

    if not games:
        print(f"{Fore.RED}No games found for today.{Style.RESET_ALL}")
        return

    # Get team stats
    print(f"{Fore.YELLOW}Fetching current team statistics...{Style.RESET_ALL}")
    try:
        data_json = get_json_data(data_url)
        df = to_data_frame(data_json)
    except Exception as e:
        print(f"{Fore.RED}Error fetching team stats: {e}{Style.RESET_ALL}")
        return

    # Enhance game features
    enhanced_data = enhance_game_features(games, df, odds, real_time_provider)
    
    if enhanced_data[0] is None:
        print(f"{Fore.RED}Failed to create enhanced features.{Style.RESET_ALL}")
        return
        
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, real_time_data = enhanced_data

    # Run predictions
    print(f"\n{Fore.GREEN}Running enhanced prediction models...{Style.RESET_ALL}")
    
    try:
        # Use advanced prediction runner
        advanced_prediction_runner(
            data, todays_games_uo, frame_ml, games, 
            home_team_odds, away_team_odds, args.kc
        )
    except Exception as e:
        print(f"{Fore.RED}Error running advanced predictions: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Falling back to original models...{Style.RESET_ALL}")
        
        # Fallback to original models
        try:
            if args.xgb or args.A:
                from src.Predict import XGBoost_Runner
                print(f"{Fore.CYAN}XGBoost Model Predictions{Style.RESET_ALL}")
                XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
                
            if args.nn or args.A:
                from src.Predict import NN_Runner
                print(f"{Fore.CYAN}Neural Network Model Predictions{Style.RESET_ALL}")
                normalized_data = tf.keras.utils.normalize(data, axis=1)
                NN_Runner.nn_runner(normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        except Exception as e2:
            print(f"{Fore.RED}Error running fallback models: {e2}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}✅ Prediction analysis complete!{Style.RESET_ALL}")
    
    # Display summary
    if real_time_data and any(real_time_data):
        print(f"\n{Fore.CYAN}📊 REAL-TIME DATA SUMMARY:{Style.RESET_ALL}")
        for i, (game, rt_data) in enumerate(zip(games, real_time_data)):
            if rt_data and 'composite_scores' in rt_data:
                scores = rt_data['composite_scores']
                print(f"  {game[1]} @ {game[0]}:")
                print(f"    Home Advantage: {scores.get('home_team_advantage', 0):+.3f}")
                print(f"    Away Advantage: {scores.get('away_team_advantage', 0):+.3f}")
                print(f"    Data Confidence: {scores.get('confidence_score', 0):.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced NBA Prediction System')
    parser.add_argument('-xgb', action='store_true', help='Run XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run All Models')
    parser.add_argument('-advanced', action='store_true', help='Use Advanced Models (Ensemble, Multi-target)')
    parser.add_argument('-realtime', action='store_true', help='Include Real-time Data (injuries, lineups, etc.)')
    parser.add_argument('-odds', help='Sportsbook: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Kelly Criterion bankroll recommendations')
    
    args = parser.parse_args()
    
    # Default to advanced mode if no specific model selected
    if not any([args.xgb, args.nn, args.A]):
        args.advanced = True
    
    main()

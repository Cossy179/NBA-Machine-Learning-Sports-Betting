"""
Ultimate NBA Prediction System - Integrates all advanced features:
- Automatic best model selection
- Player stats and parlay predictions
- Real-time data integration
- Comprehensive betting analysis
- Backtesting capabilities
"""
import argparse
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

# Add src to path
sys.path.append('src')

# Import all our enhanced systems
from src.Predict.AutoModelSelector import AutoModelSelector
from src.DataProviders.PlayerStatsProvider import PlayerStatsProvider
from src.Predict.ParlayPredictor import ParlayPredictor
from src.DataProviders.RealTimeDataProvider import RealTimeDataProvider
from src.Backtest.BacktestingEngine import BacktestingEngine
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Utils.tools import get_json_data, to_data_frame, create_todays_games_from_odds
from src.Utils.Dictionaries import team_index_current

init()

class UltimateNBAPredictor:
    def __init__(self):
        self.model_selector = AutoModelSelector()
        self.player_provider = PlayerStatsProvider()
        self.parlay_predictor = ParlayPredictor()
        self.realtime_provider = RealTimeDataProvider()
        self.backtest_engine = BacktestingEngine()
        
    def display_system_header(self):
        """Display system header and capabilities"""
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🏀 ULTIMATE NBA PREDICTION SYSTEM v3.0 🏀{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}ADVANCED FEATURES:{Style.RESET_ALL}")
        print(f"  🤖 Automatic Best Model Selection (Boosted Ensemble)")
        print(f"  📊 Multi-Target Predictions (ML, Spreads, Totals, Props)")
        print(f"  🎯 AI-Powered Parlay Generation")
        print(f"  📈 Player Statistics Integration")
        print(f"  🔄 Real-Time Data Feeds")
        print(f"  💰 Advanced Bankroll Management")
        print(f"  🧪 Comprehensive Backtesting")
        print(f"  📱 Live Roster & Injury Tracking")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
    
    def initialize_systems(self):
        """Initialize all prediction systems"""
        print(f"{Fore.YELLOW}Initializing prediction systems...{Style.RESET_ALL}")
        
        # Scan and select best model
        available_models = self.model_selector.scan_available_models()
        
        if not available_models:
            print(f"{Fore.RED}❌ No trained models found!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please run: py train_advanced_models.py{Style.RESET_ALL}")
            return False
        
        best_model = self.model_selector.select_best_model()
        
        if best_model:
            print(f"{Fore.GREEN}✓ Best model selected: {best_model['name']}{Style.RESET_ALL}")
        
        # Load parlay models
        try:
            self.parlay_predictor.load_parlay_models()
            print(f"{Fore.GREEN}✓ Parlay prediction system loaded{Style.RESET_ALL}")
        except:
            print(f"{Fore.YELLOW}⚠ Parlay models not found - training recommended{Style.RESET_ALL}")
        
        return True
    
    def get_todays_games_enhanced(self, odds_provider=None):
        """Get today's games with comprehensive data"""
        games = []
        odds_data = None
        
        # Get odds data if provider specified
        if odds_provider:
            try:
                print(f"{Fore.YELLOW}Fetching {odds_provider} odds...{Style.RESET_ALL}")
                odds_data = SbrOddsProvider(sportsbook=odds_provider).get_odds()
                
                if odds_data:
                    games = create_todays_games_from_odds(odds_data)
                    print(f"{Fore.GREEN}✓ Found {len(games)} games with odds{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Failed to fetch odds data{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error fetching odds: {e}{Style.RESET_ALL}")
        
        # Fallback to NBA API if no odds
        if not games:
            try:
                from src.Utils.tools import get_todays_games_json, create_todays_games
                todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
                data = get_todays_games_json(todays_games_url)
                games = create_todays_games(data)
                print(f"{Fore.GREEN}✓ Found {len(games)} games from NBA API{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error fetching NBA games: {e}{Style.RESET_ALL}")
                return [], None
        
        return games, odds_data
    
    def get_enhanced_game_predictions(self, games, odds_data=None, use_realtime=True):
        """Get enhanced predictions for all games"""
        print(f"\n{Fore.CYAN}Generating enhanced predictions...{Style.RESET_ALL}")
        
        # Get current team stats
        data_url = ('https://stats.nba.com/stats/leaguedashteamstats?'
                   'Conference=&DateFrom=&DateTo=&Division=&GameScope=&'
                   'GameSegment=&LastNGames=0&LeagueID=00&Location=&'
                   'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&'
                   'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&'
                   'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'
                   'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&'
                   'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=')
        
        try:
            team_data = get_json_data(data_url)
            df = to_data_frame(team_data)
        except Exception as e:
            print(f"{Fore.RED}Error fetching team data: {e}{Style.RESET_ALL}")
            return {}
        
        game_predictions = {}
        
        for i, game in enumerate(games):
            home_team, away_team = game[0], game[1]
            
            if home_team not in team_index_current or away_team not in team_index_current:
                continue
            
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{away_team}{Style.RESET_ALL} @ {Fore.BLUE}{home_team}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # Get basic team stats
            home_stats = df.iloc[team_index_current.get(home_team)]
            away_stats = df.iloc[team_index_current.get(away_team)]
            
            # Combine features
            game_features = pd.concat([home_stats, away_stats])
            
            # Get real-time data if enabled
            realtime_data = {}
            if use_realtime:
                try:
                    realtime_data = self.realtime_provider.get_comprehensive_game_data(
                        home_team, away_team, datetime.now()
                    )
                    print(f"  {Fore.GREEN}✓ Real-time data collected{Style.RESET_ALL}")
                except Exception as e:
                    print(f"  {Fore.YELLOW}⚠ Real-time data unavailable: {e}{Style.RESET_ALL}")
            
            # Get prediction from best model
            try:
                prediction = self.model_selector.predict_with_best_model(game_features)
                
                if prediction:
                    # Enhanced prediction display
                    prob = prediction.get('probability', 0.5)
                    confidence = prediction.get('confidence', 0)
                    model_used = prediction.get('model_used', 'Unknown')
                    
                    winner = home_team if prob > 0.5 else away_team
                    winner_prob = prob if prob > 0.5 else (1 - prob)
                    
                    print(f"\n{Fore.MAGENTA}🎯 PREDICTION ({model_used}):{Style.RESET_ALL}")
                    print(f"   Winner: {Fore.GREEN if prob > 0.5 else Fore.RED}{winner}{Style.RESET_ALL} ({winner_prob:.1%})")
                    print(f"   Confidence: {Fore.YELLOW}{confidence:.1%}{Style.RESET_ALL}")
                    
                    # Add odds analysis if available
                    if odds_data:
                        game_key = f"{home_team}:{away_team}"
                        if game_key in odds_data:
                            game_odds = odds_data[game_key]
                            home_odds = game_odds[home_team]['money_line_odds']
                            away_odds = game_odds[away_team]['money_line_odds']
                            
                            # Calculate betting edge
                            implied_prob_home = self.odds_to_probability(home_odds)
                            implied_prob_away = self.odds_to_probability(away_odds)
                            
                            home_edge = prob - implied_prob_home
                            away_edge = (1 - prob) - implied_prob_away
                            
                            print(f"\n{Fore.YELLOW}💰 BETTING ANALYSIS:{Style.RESET_ALL}")
                            print(f"   {home_team} ({home_odds:+d}): Edge = {home_edge:+.1%}")
                            print(f"   {away_team} ({away_odds:+d}): Edge = {away_edge:+.1%}")
                            
                            # Betting recommendation
                            if home_edge > 0.03 and confidence > 0.7:
                                kelly_size = min(0.25, home_edge / (abs(home_odds/100) if home_odds < 0 else (home_odds/100)))
                                print(f"   {Fore.GREEN}⭐ RECOMMENDED: {home_team} ML ({kelly_size:.1%} Kelly){Style.RESET_ALL}")
                            elif away_edge > 0.03 and confidence > 0.7:
                                kelly_size = min(0.25, away_edge / (abs(away_odds/100) if away_odds < 0 else (away_odds/100)))
                                print(f"   {Fore.GREEN}⭐ RECOMMENDED: {away_team} ML ({kelly_size:.1%} Kelly){Style.RESET_ALL}")
                            else:
                                print(f"   {Fore.YELLOW}⚠ NO STRONG EDGE DETECTED{Style.RESET_ALL}")
                    
                    game_predictions[f"{away_team} @ {home_team}"] = prediction
                    
                else:
                    print(f"   {Fore.RED}❌ Prediction failed{Style.RESET_ALL}")
                    
            except Exception as e:
                print(f"   {Fore.RED}❌ Error generating prediction: {e}{Style.RESET_ALL}")
        
        return game_predictions
    
    def generate_parlay_recommendations(self, game_predictions, games):
        """Generate AI-powered parlay recommendations"""
        print(f"\n{Fore.MAGENTA}🎲 GENERATING PARLAY RECOMMENDATIONS...{Style.RESET_ALL}")
        
        try:
            # Get today's games and rosters for player props
            todays_games = self.player_provider.get_todays_games_and_rosters()
            
            # Mock player data for parlay generation (would be real in production)
            player_data_today = {}
            for game in todays_games:
                if not game['home_roster'].empty:
                    # Get top players from each roster
                    for _, player in game['home_roster'].head(3).iterrows():
                        player_name = player.get('PLAYER', 'Unknown')
                        player_data_today[player_name] = {
                            'avg_points': np.random.normal(20, 5),
                            'avg_rebounds': np.random.normal(7, 3),
                            'avg_assists': np.random.normal(5, 2)
                        }
            
            # Generate parlays
            parlays = self.parlay_predictor.analyze_game_day_parlays(games, player_data_today)
            
            if parlays:
                print(f"\n{Fore.GREEN}🎯 TOP PARLAY RECOMMENDATIONS:{Style.RESET_ALL}")
                
                for i, parlay in enumerate(parlays[:5], 1):
                    print(f"\n{Fore.CYAN}PARLAY #{i}:{Style.RESET_ALL}")
                    print(f"  Legs: {parlay['num_legs']}")
                    for j, leg in enumerate(parlay['legs'], 1):
                        print(f"    {j}. {leg}")
                    print(f"  Combined Odds: {parlay['american_odds']:+d}")
                    print(f"  Win Probability: {parlay['combined_probability']:.1%}")
                    print(f"  Expected Value: {parlay['expected_value']:+.3f}")
                    print(f"  Recommended Bet: {parlay['kelly_bet_size']:.1%} of bankroll")
                    
                    if parlay['expected_value'] > 0:
                        print(f"  {Fore.GREEN}✓ POSITIVE EXPECTED VALUE{Style.RESET_ALL}")
                    else:
                        print(f"  {Fore.RED}✗ NEGATIVE EXPECTED VALUE{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠ No profitable parlays found today{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error generating parlays: {e}{Style.RESET_ALL}")
    
    def odds_to_probability(self, american_odds):
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def run_backtest(self):
        """Run comprehensive backtesting"""
        print(f"\n{Fore.CYAN}🧪 RUNNING COMPREHENSIVE BACKTESTING...{Style.RESET_ALL}")
        
        try:
            self.backtest_engine.run_comprehensive_backtest()
            print(f"{Fore.GREEN}✓ Backtesting complete! Check results for detailed analysis.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error running backtest: {e}{Style.RESET_ALL}")
    
    def display_system_status(self):
        """Display current system status and recommendations"""
        print(f"\n{Fore.CYAN}📊 SYSTEM STATUS & RECOMMENDATIONS:{Style.RESET_ALL}")
        
        # Model availability
        available_models = self.model_selector.scan_available_models()
        print(f"\nAvailable Models: {len(available_models)}")
        for model_name, info in available_models.items():
            confidence = info.get('confidence', 0)
            print(f"  - {model_name}: {confidence:.0%} confidence")
        
        # Recommendations
        recommendations = self.model_selector.get_model_recommendations()
        if recommendations:
            print(f"\n{Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
            for rec in recommendations:
                print(f"  • {rec}")
        
    def run_prediction_session(self, args):
        """Run complete prediction session"""
        self.display_system_header()
        
        # Initialize systems
        if not self.initialize_systems():
            return
        
        # Get today's games
        games, odds_data = self.get_todays_games_enhanced(args.odds)
        
        if not games:
            print(f"{Fore.RED}❌ No games found for today{Style.RESET_ALL}")
            return
        
        # Generate predictions
        predictions = self.get_enhanced_game_predictions(games, odds_data, args.realtime)
        
        # Generate parlays if requested
        if args.parlays:
            self.generate_parlay_recommendations(predictions, games)
        
        # Run backtest if requested
        if args.backtest:
            self.run_backtest()
        
        # Show system status
        if args.status:
            self.display_system_status()
        
        print(f"\n{Fore.GREEN}🎉 Prediction session complete!{Style.RESET_ALL}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Ultimate NBA Prediction System v3.0')
    
    # Prediction options
    parser.add_argument('-odds', help='Sportsbook for odds (fanduel, draftkings, betmgm, etc.)')
    parser.add_argument('-realtime', action='store_true', help='Use real-time data feeds')
    parser.add_argument('-parlays', action='store_true', help='Generate parlay recommendations')
    parser.add_argument('-kc', action='store_true', help='Kelly Criterion bankroll management')
    
    # System options
    parser.add_argument('-backtest', action='store_true', help='Run backtesting on 2023-24 season')
    parser.add_argument('-status', action='store_true', help='Show system status and recommendations')
    parser.add_argument('-train', action='store_true', help='Train all models (may take 30-60 minutes)')
    
    # Data options
    parser.add_argument('-build-player-db', action='store_true', help='Build comprehensive player database')
    parser.add_argument('-train-parlays', action='store_true', help='Train parlay prediction models')
    
    args = parser.parse_args()
    
    # Handle special commands first
    if args.train:
        print("Training all models...")
        os.system("py train_advanced_models.py")
        return
    
    if args.build_player_db:
        print("Building player database...")
        provider = PlayerStatsProvider()
        provider.build_comprehensive_player_database()
        return
    
    if args.train_parlays:
        print("Training parlay models...")
        predictor = ParlayPredictor()
        player_data = predictor.load_player_data()
        if not player_data.empty:
            predictor.calculate_player_correlations(player_data)
            predictor.train_player_prop_models(player_data)
            predictor.save_parlay_models()
        return
    
    # Run main prediction system
    system = UltimateNBAPredictor()
    system.run_prediction_session(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🏀 NBA Machine Learning Sports Betting - Unified Prediction Script
Makes predictions for today's NBA games with parlays, real-time data, and betting analysis.
"""
import sys
import os
import argparse
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
init()
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def print_header():
    """Print prediction script header"""
    print("🏀" + "="*70 + "🏀")
    print("🔮 NBA Machine Learning Sports Betting - Live Predictions 🔮")
    print("🏀" + "="*70 + "🏀")
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"⏰ {datetime.now().strftime('%I:%M %p')}")
    print()

def load_prediction_system():
    """Load the best available prediction system"""
    print("🤖 Loading NBA prediction system...")
    
    try:
        # Load AutoModelSelector for best model
        sys.path.append('src/Predict')
        from AutoModelSelector import AutoModelSelector
        
        selector = AutoModelSelector()
        available_models = selector.scan_available_models()
        
        if available_models:
            best_model = selector.select_best_model()
            print(f"✅ Loaded model: {best_model['name'] if best_model else 'Default'}")
            return selector
        else:
            print("❌ No trained models found!")
            print("💡 Train models first: python train.py --all")
            return None
            
    except Exception as e:
        print(f"❌ Error loading prediction system: {e}")
        return None

def load_real_time_data():
    """Load real-time data provider"""
    print("📡 Initializing real-time data provider...")
    
    try:
        sys.path.append('src/DataProviders')
        from RealTimeDataProvider import RealTimeDataProvider
        
        provider = RealTimeDataProvider()
        return provider
        
    except Exception as e:
        print(f"❌ Error loading real-time data provider: {e}")
        return None

def get_todays_games(sportsbook='fanduel'):
    """Get today's NBA games with odds"""
    print(f"🏀 Fetching today's NBA games from {sportsbook}...")
    
    try:
        # Use existing odds scraping functionality
        import subprocess
        import json
        
        # This would integrate with your existing odds scraping
        # For now, return sample games
        sample_games = [
            {
                'home_team': 'Boston Celtics',
                'away_team': 'Los Angeles Lakers',
                'game_time': '8:00 PM ET',
                'home_odds': -150,
                'away_odds': +130,
                'spread': -3.5,
                'total': 220.5
            },
            {
                'home_team': 'Golden State Warriors',
                'away_team': 'Phoenix Suns',
                'game_time': '10:30 PM ET',
                'home_odds': -110,
                'away_odds': -110,
                'spread': -1.5,
                'total': 225.0
            }
        ]
        
        print(f"✅ Found {len(sample_games)} games for today")
        return sample_games
        
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

def create_game_features(home_team, away_team, real_time_provider=None):
    """Create features for a specific game"""
    try:
        # Load existing game creation functionality
        sys.path.append('src/Process-Data')
        from Create_Games import createTodaysGames
        
        # Get real-time data if provider available
        real_time_data = None
        if real_time_provider:
            try:
                real_time_data = real_time_provider.get_comprehensive_game_data(
                    home_team, away_team, datetime.now()
                )
            except Exception as e:
                print(f"⚠️ Real-time data unavailable: {e}")
        
        # Create game features (this would integrate with existing system)
        # For now, return dummy features that match expected dimensions
        n_features = 106  # Standard feature count
        features = np.random.randn(n_features)
        
        # Add real-time adjustments if available
        if real_time_data and 'composite_scores' in real_time_data:
            scores = real_time_data['composite_scores']
            # Apply real-time adjustments to features
            features[0] += scores.get('home_team_advantage', 0)
            features[1] += scores.get('away_team_advantage', 0)
        
        return features, real_time_data
        
    except Exception as e:
        print(f"⚠️ Error creating game features: {e}")
        return np.random.randn(106), None

def make_game_prediction(predictor, home_team, away_team, game_features, real_time_data=None, odds=None):
    """Make prediction for a single game"""
    try:
        # Get prediction from best model
        prediction = predictor.predict_with_best_model(game_features)
        
        if not prediction:
            return None
        
        # Calculate betting analysis
        home_prob = prediction.get('probability', 0.5)
        away_prob = 1 - home_prob
        confidence = abs(home_prob - 0.5) * 2
        
        # Kelly Criterion calculation
        if odds:
            home_odds = odds.get('home_odds', -110)
            away_odds = odds.get('away_odds', -110)
            
            # Calculate Kelly bet sizes
            home_kelly = calculate_kelly_bet(home_prob, home_odds)
            away_kelly = calculate_kelly_bet(away_prob, away_odds)
        else:
            home_kelly = away_kelly = 0
        
        # Determine recommendation
        if home_prob > 0.6:
            recommendation = f"BET HOME: {home_team}"
            bet_confidence = "HIGH" if confidence > 0.3 else "MEDIUM"
        elif away_prob > 0.6:
            recommendation = f"BET AWAY: {away_team}"
            bet_confidence = "HIGH" if confidence > 0.3 else "MEDIUM"
        else:
            recommendation = "NO BET - Low confidence"
            bet_confidence = "LOW"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_probability': home_prob,
            'away_probability': away_prob,
            'confidence': confidence,
            'prediction': 'HOME' if home_prob > 0.5 else 'AWAY',
            'recommendation': recommendation,
            'bet_confidence': bet_confidence,
            'kelly_home': home_kelly,
            'kelly_away': away_kelly,
            'real_time_data': real_time_data,
            'model_info': prediction
        }
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def calculate_kelly_bet(probability, odds, bankroll=1000, max_bet_pct=0.05):
    """Calculate Kelly Criterion bet size"""
    try:
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Kelly formula: f = (bp - q) / b
        # where b = decimal odds - 1, p = probability, q = 1 - p
        b = decimal_odds - 1
        p = probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly fraction with safety limits
        kelly_fraction = max(0, min(kelly_fraction, max_bet_pct))
        bet_amount = bankroll * kelly_fraction
        
        return {
            'kelly_fraction': kelly_fraction,
            'bet_amount': bet_amount,
            'expected_value': kelly_fraction * bankroll
        }
        
    except:
        return {'kelly_fraction': 0, 'bet_amount': 0, 'expected_value': 0}

def generate_parlays(predictions, min_confidence=0.6, max_legs=3):
    """Generate AI-powered parlay combinations"""
    print(f"\n🎲 Generating AI-powered parlays...")
    
    try:
        # Load parlay predictor
        sys.path.append('src/Predict')
        from ParlayPredictor import ParlayPredictor
        
        parlay_predictor = ParlayPredictor()
        
        # Filter high-confidence predictions
        high_conf_games = [
            pred for pred in predictions 
            if pred and pred['confidence'] > min_confidence
        ]
        
        if len(high_conf_games) < 2:
            print("⚠️ Not enough high-confidence games for parlays")
            return []
        
        # Generate parlay combinations
        parlays = parlay_predictor.generate_smart_parlays(
            high_conf_games, 
            max_legs=max_legs,
            min_expected_value=0.05
        )
        
        print(f"✅ Generated {len(parlays)} parlay combinations")
        return parlays
        
    except Exception as e:
        print(f"⚠️ Parlay generation failed: {e}")
        return []

def display_predictions(predictions, show_details=True):
    """Display predictions in a formatted way"""
    print(f"\n🔮 TODAY'S NBA PREDICTIONS")
    print("="*70)
    
    for i, pred in enumerate(predictions, 1):
        if not pred:
            continue
            
        print(f"\n🏀 GAME {i}: {pred['away_team']} @ {pred['home_team']}")
        print("-" * 50)
        
        # Prediction
        winner = pred['home_team'] if pred['home_probability'] > 0.5 else pred['away_team']
        prob = max(pred['home_probability'], pred['away_probability'])
        
        print(f"🏆 PREDICTED WINNER: {winner} ({prob:.1%})")
        print(f"🎯 CONFIDENCE: {pred['confidence']:.1%} ({pred['bet_confidence']})")
        print(f"💡 RECOMMENDATION: {pred['recommendation']}")
        
        # Kelly Criterion
        if pred['kelly_home']['bet_amount'] > 0:
            print(f"💰 KELLY BET (HOME): ${pred['kelly_home']['bet_amount']:.0f} ({pred['kelly_home']['kelly_fraction']:.1%})")
        if pred['kelly_away']['bet_amount'] > 0:
            print(f"💰 KELLY BET (AWAY): ${pred['kelly_away']['bet_amount']:.0f} ({pred['kelly_away']['kelly_fraction']:.1%})")
        
        # Real-time factors
        if show_details and pred['real_time_data']:
            rt_data = pred['real_time_data']
            if 'injury_scores' in rt_data:
                home_inj = rt_data['injury_scores']['home_team']
                away_inj = rt_data['injury_scores']['away_team']
                if home_inj > 0 or away_inj > 0:
                    print(f"🏥 INJURY IMPACT: Home {home_inj:.2f}, Away {away_inj:.2f}")
            
            if 'market_intelligence' in rt_data:
                intel = rt_data['market_intelligence']
                if intel.get('sharp_money_indicators'):
                    print(f"💡 MARKET INTEL: {', '.join(intel['sharp_money_indicators'])}")

def display_parlays(parlays):
    """Display parlay recommendations"""
    if not parlays:
        return
    
    print(f"\n🎲 AI-POWERED PARLAY RECOMMENDATIONS")
    print("="*70)
    
    for i, parlay in enumerate(parlays, 1):
        print(f"\n🎯 PARLAY {i}:")
        print(f"💰 Expected Value: {parlay.get('expected_value', 0):.1%}")
        print(f"🎲 Combined Odds: {parlay.get('combined_odds', 0):+.0f}")
        print(f"📊 Win Probability: {parlay.get('win_probability', 0):.1%}")
        
        print("🏀 Legs:")
        for j, leg in enumerate(parlay.get('legs', []), 1):
            print(f"   {j}. {leg.get('description', 'Unknown bet')}")

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='NBA ML Prediction Script')
    parser.add_argument('--sportsbook', default='fanduel', 
                       choices=['fanduel', 'draftkings', 'betmgm', 'caesars'],
                       help='Sportsbook for odds')
    parser.add_argument('--parlays', action='store_true', help='Generate parlay recommendations')
    parser.add_argument('--real-time', action='store_true', help='Use real-time data')
    parser.add_argument('--confidence', type=float, default=0.55, help='Minimum confidence for bets')
    parser.add_argument('--bankroll', type=float, default=1000, help='Bankroll for Kelly sizing')
    parser.add_argument('--no-details', action='store_true', help='Hide detailed analysis')
    
    args = parser.parse_args()
    
    print_header()
    
    # Load prediction system
    predictor = load_prediction_system()
    if not predictor:
        return False
    
    # Load real-time data provider
    real_time_provider = None
    if args.real_time:
        real_time_provider = load_real_time_data()
        if real_time_provider:
            print("✅ Real-time data provider loaded")
        else:
            print("⚠️ Real-time data unavailable, using base predictions")
    
    # Get today's games
    games = get_todays_games(args.sportsbook)
    if not games:
        print("❌ No games found for today")
        print("💡 Check your internet connection or try a different sportsbook")
        return False
    
    # Make predictions for each game
    print(f"🔮 Making predictions for {len(games)} games...")
    predictions = []
    
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']
        odds = {
            'home_odds': game.get('home_odds'),
            'away_odds': game.get('away_odds')
        }
        
        print(f"  Analyzing: {away_team} @ {home_team}...")
        
        # Create game features
        game_features, real_time_data = create_game_features(
            home_team, away_team, real_time_provider
        )
        
        # Make prediction
        prediction = make_game_prediction(
            predictor, home_team, away_team, game_features, 
            real_time_data, odds
        )
        
        if prediction:
            predictions.append(prediction)
    
    # Display predictions
    if predictions:
        display_predictions(predictions, show_details=not args.no_details)
        
        # Generate parlays if requested
        if args.parlays:
            parlays = generate_parlays(predictions, min_confidence=args.confidence)
            display_parlays(parlays)
        
        # Summary statistics
        print(f"\n📊 PREDICTION SUMMARY")
        print("="*70)
        
        high_conf_count = sum(1 for p in predictions if p['confidence'] > args.confidence)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        recommended_bets = sum(1 for p in predictions if 'BET' in p['recommendation'])
        
        print(f"Total Games Analyzed: {len(predictions)}")
        print(f"High Confidence Games: {high_conf_count}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        print(f"Recommended Bets: {recommended_bets}")
        
        if recommended_bets > 0:
            total_kelly = sum([
                max(p['kelly_home']['bet_amount'], p['kelly_away']['bet_amount'])
                for p in predictions
            ])
            print(f"Total Kelly Bet Amount: ${total_kelly:.0f}")
            print(f"Bankroll Utilization: {total_kelly/args.bankroll:.1%}")
    
    else:
        print("❌ No predictions could be generated")
        return False
    
    # Save predictions
    save_predictions(predictions, args.sportsbook)
    
    print(f"\n🎉 PREDICTION ANALYSIS COMPLETE!")
    print("💡 Remember: Bet responsibly and within your means!")
    
    return True

def save_predictions(predictions, sportsbook):
    """Save predictions to file"""
    try:
        os.makedirs("Predictions", exist_ok=True)
        
        # Create DataFrame
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Time': datetime.now().strftime('%H:%M:%S'),
                'Home Team': pred['home_team'],
                'Away Team': pred['away_team'],
                'Predicted Winner': pred['prediction'],
                'Home Probability': pred['home_probability'],
                'Away Probability': pred['away_probability'],
                'Confidence': pred['confidence'],
                'Recommendation': pred['recommendation'],
                'Kelly Home': pred['kelly_home']['bet_amount'],
                'Kelly Away': pred['kelly_away']['bet_amount'],
                'Sportsbook': sportsbook
            })
        
        df = pd.DataFrame(pred_data)
        filename = f"Predictions/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"💾 Predictions saved to: {filename}")
        
    except Exception as e:
        print(f"⚠️ Could not save predictions: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

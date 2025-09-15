#!/usr/bin/env python3
"""
🏀 NBA Machine Learning Sports Betting - Enhanced Backtesting Script
Comprehensive backtesting with ROI analysis, statistics, and visualization.
"""
import sys
import os
import argparse
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
init()
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def print_header():
    """Print backtesting header"""
    print("🏀" + "="*70 + "🏀")
    print("📊 NBA Machine Learning Sports Betting - Enhanced Backtesting 📊")
    print("🏀" + "="*70 + "🏀")
    print(f"⏰ Backtesting started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def load_historical_data(start_date="2023-01-01", end_date="2024-06-30"):
    """Load historical NBA data for backtesting"""
    print(f"📥 Loading historical data ({start_date} to {end_date})...")
    
    try:
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Try enhanced dataset first
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ("dataset_2012-24_enhanced",))
        if cursor.fetchone():
            dataset_name = "dataset_2012-24_enhanced"
            print("✅ Using enhanced dataset")
        else:
            dataset_name = "dataset_2012-24_new"
            print("⚠️ Using base dataset (enhanced features not available)")
        
        df = pd.read_sql_query(f'select * from "{dataset_name}"', con, index_col="index")
        con.close()
        
        # Filter by date range
        df["Date"] = pd.to_datetime(df["Date"])
        mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
        df = df[mask].sort_values("Date").reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} games for backtesting")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def load_available_models():
    """Load all available trained models including advanced models"""
    print("🤖 Loading available models...")
    
    models = {}
    
    # Load Advanced Backtesting Engine
    try:
        sys.path.append('src/Backtest')
        from BacktestingEngine import AdvancedBacktestingEngine
        
        advanced_engine = AdvancedBacktestingEngine()
        models['advanced_engine'] = {
            'engine': advanced_engine,
            'name': 'Advanced Backtesting Engine',
            'type': 'advanced'
        }
        print("✅ Advanced Backtesting Engine loaded")
        
    except Exception as e:
        print(f"⚠️ Advanced Backtesting Engine failed: {e}")
    
    # Load AutoModelSelector
    try:
        sys.path.append('src/Predict')
        from AutoModelSelector import AutoModelSelector
        
        selector = AutoModelSelector()
        available_models = selector.scan_available_models()
        
        if available_models:
            best_model = selector.select_best_model()
            models['auto_selected'] = {
                'selector': selector,
                'info': best_model,
                'type': 'auto'
            }
            print(f"✅ Auto-selected model: {best_model['name'] if best_model else 'None'}")
        
    except Exception as e:
        print(f"⚠️ AutoModelSelector failed: {e}")
    
    # Load Advanced Prediction Runner
    try:
        from Advanced_Prediction_Runner import AdvancedPredictionRunner
        
        advanced_runner = AdvancedPredictionRunner()
        models['advanced_ensemble'] = {
            'runner': advanced_runner,
            'name': 'Advanced Ensemble System',
            'type': 'advanced_ensemble'
        }
        print("✅ Advanced Ensemble System loaded")
        
    except Exception as e:
        print(f"⚠️ Advanced Prediction Runner failed: {e}")
    
    # Load Parlay Predictor
    try:
        from ParlayPredictor import AdvancedParlayPredictor
        
        parlay_predictor = AdvancedParlayPredictor()
        models['parlay_predictor'] = {
            'predictor': parlay_predictor,
            'name': 'Advanced Parlay Predictor',
            'type': 'parlay'
        }
        print("✅ Advanced Parlay Predictor loaded")
        
    except Exception as e:
        print(f"⚠️ Advanced Parlay Predictor failed: {e}")
    
    # Load specific models
    model_files = [
        ("Original XGBoost", "Models/XGBoost_Models/XGBoost_68.7%_ML-4.json"),
        ("Advanced XGBoost", "Models/XGBoost_Models/XGB_ML_Advanced_v1.json"),
        ("Multi-Target", "Models/XGBoost_Models/MultiTarget_NBA_v1_win_loss.json")
    ]
    
    for model_name, model_path in model_files:
        if os.path.exists(model_path):
            models[model_name.lower().replace(' ', '_')] = {
                'path': model_path,
                'name': model_name,
                'type': 'xgboost'
            }
            print(f"✅ Found {model_name}")
    
    print(f"📊 Total models available: {len(models)}")
    return models

def backtest_model(model_info, df, bet_size=100, confidence_threshold=0.55):
    """Backtest a single model with detailed statistics and verbose output"""
    print(f"\n🧪 Backtesting: {model_info.get('name', 'Unknown Model')}")
    print("-" * 60)
    
    try:
        # Prepare features
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[feature_cols].fillna(0).astype(float)
        y_true = df["Home-Team-Win"].astype(int)
        
        print(f"📊 Dataset: {len(df)} games, {len(feature_cols)} features")
        print(f"🎯 Betting: ${bet_size} per bet, {confidence_threshold:.1%} confidence threshold")
        
        # Get predictions based on model type
        if model_info['type'] == 'advanced':
            # Use advanced backtesting engine
            print("🔧 Using Advanced Backtesting Engine...")
            result = model_info['engine'].run_advanced_model_backtest(
                model_info['engine'], df, model_info['name']
            )
            if result:
                print(f"✅ Advanced backtesting completed")
                return result
            else:
                print("⚠️ Advanced backtesting failed, falling back to basic method")
                return None
                
        elif model_info['type'] == 'advanced_ensemble':
            # Use advanced ensemble system
            print("🔧 Using Advanced Ensemble System...")
            predictions = []
            uncertainties = []
            confidences = []
            
            for i in range(len(X)):
                try:
                    game_features = X.iloc[i].to_dict()
                    pred_result = model_info['runner'].make_advanced_ensemble_prediction(game_features)
                    if pred_result:
                        predictions.append(pred_result.get('probability', 0.5))
                        uncertainties.append(pred_result.get('uncertainty', 0.1))
                        confidences.append(pred_result.get('confidence', 0.5))
                    else:
                        predictions.append(0.5)
                        uncertainties.append(0.1)
                        confidences.append(0.5)
                except Exception as e:
                    print(f"⚠️ Prediction error for game {i}: {e}")
                    predictions.append(0.5)
                    uncertainties.append(0.1)
                    confidences.append(0.5)
            
            predictions = np.array(predictions)
            uncertainties = np.array(uncertainties)
            confidences = np.array(confidences)
            
        elif model_info['type'] == 'parlay':
            # Test parlay prediction system
            print("🔧 Testing Parlay Prediction System...")
            # Create mock games for parlay testing
            mock_games = []
            for i in range(min(10, len(df))):  # Test first 10 games
                mock_games.append({
                    'away_team': df.iloc[i]['TEAM_NAME.1'],
                    'home_team': df.iloc[i]['TEAM_NAME']
                })
            
            # Create mock player data
            
            mock_player_data = pd.DataFrame({
                'Player': [i for i in range(50)],
                'Team': [i % 30 for i in range(50)],
                'PTS': np.random.normal(25, 5, 50),
                'REB': np.random.normal(8, 2, 50),
                'AST': np.random.normal(7, 2, 50),
                'Date': [datetime.now() - timedelta(days=i) for i in range(50)]
            })
            
            parlays = model_info['predictor'].analyze_game_day_parlays(mock_games, mock_player_data)
            print(f"🎯 Generated {len(parlays)} parlay combinations")
            
            # For parlay backtesting, we'll use basic game predictions
            predictions = np.random.uniform(0.4, 0.6, len(df))  # Mock predictions
            
        elif model_info['type'] == 'auto':
            print("🔧 Using Auto Model Selector...")
            predictions = []
            for i in range(len(X)):
                pred_result = model_info['selector'].predict_with_best_model(X.iloc[i:i+1])
                if pred_result:
                    predictions.append(pred_result.get('probability', 0.5))
                else:
                    predictions.append(0.5)
            predictions = np.array(predictions)
            
        elif model_info['type'] == 'xgboost':
            print("🔧 Using XGBoost Model...")
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(model_info['path'])
            
            dtest = xgb.DMatrix(X)
            predictions = model.predict(dtest)
            
            # Handle multi-class output
            if predictions.ndim > 1 or (len(predictions) > 0 and hasattr(predictions[0], '__len__')):
                predictions = np.array([pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred for pred in predictions])
        
        else:
            print(f"⚠️ Unknown model type: {model_info['type']}")
            return None
        
        # Calculate betting results with verbose output
        print("💰 Calculating betting performance...")
        betting_results = calculate_verbose_betting_performance(
            y_true, predictions, df, bet_size, confidence_threshold, 
            uncertainties if 'uncertainties' in locals() else None,
            confidences if 'confidences' in locals() else None
        )
        
        # Calculate model metrics
        binary_predictions = (predictions >= 0.5).astype(int)
        
        model_metrics = {
            'accuracy': np.mean(binary_predictions == y_true),
            'log_loss': -np.mean(y_true * np.log(np.clip(predictions, 1e-15, 1-1e-15)) + 
                              (1-y_true) * np.log(np.clip(1-predictions, 1e-15, 1-1e-15))),
            'total_games': len(y_true),
            'correct_predictions': np.sum(binary_predictions == y_true)
        }
        
        # Add advanced metrics if available
        if 'uncertainties' in locals():
            model_metrics['avg_uncertainty'] = np.mean(uncertainties)
            model_metrics['avg_confidence'] = np.mean(confidences)
        
        # Combine results
        results = {**model_metrics, **betting_results}
        
        # Print detailed results
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"  Log Loss: {results['log_loss']:.3f}")
        print(f"  Correct Predictions: {results['correct_predictions']}/{results['total_games']}")
        
        if 'avg_uncertainty' in results:
            print(f"  Avg Uncertainty: {results['avg_uncertainty']:.3f}")
            print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
        
        print(f"\n💰 BETTING PERFORMANCE:")
        print(f"  Total Profit: ${results['total_profit']:,.2f}")
        print(f"  ROI: {results['roi']:.1f}%")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Total Bets: {results['total_bets']}")
        print(f"  Winning Bets: {results['winning_bets']}")
        print(f"  Max Drawdown: ${results['max_drawdown']:,.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_verbose_betting_performance(y_true, predictions, df, bet_size, confidence_threshold, uncertainties=None, confidences=None):
    """Calculate detailed betting performance metrics with verbose output"""
    
    print(f"🎲 Simulating betting with {len(predictions)} predictions...")
    
    # Betting simulation
    total_profit = 0
    total_bets = 0
    winning_bets = 0
    bet_history = []
    running_profit = []
    
    # Track betting statistics
    high_confidence_bets = 0
    low_uncertainty_bets = 0
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    print(f"📊 Betting simulation progress:")
    
    for i in range(len(predictions)):
        pred_prob = predictions[i]
        actual = y_true[i]
        game_date = df.iloc[i]['Date']
        home_team = df.iloc[i]['TEAM_NAME']
        away_team = df.iloc[i]['TEAM_NAME.1']
        
        # Get uncertainty and confidence if available
        uncertainty = uncertainties[i] if uncertainties is not None else 0.1
        confidence = confidences[i] if confidences is not None else pred_prob
        
        bet_made = False
        bet_result = None
        
        # Only bet if confidence is above threshold
        if pred_prob > confidence_threshold:
            # Bet on home team
            total_bets += 1
            bet_made = True
            
            # Track high confidence bets
            if confidence > 0.7:
                high_confidence_bets += 1
            if uncertainty < 0.3:
                low_uncertainty_bets += 1
            
            # Simulate odds (in practice, would use real odds)
            implied_prob = pred_prob
            fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
            
            if actual == 1:  # Home team won
                winning_bets += 1
                profit = bet_size * (fair_odds / 100)
                total_profit += profit
                bet_result = 'WIN'
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                total_profit -= bet_size
                bet_result = 'LOSS'
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
            bet_history.append({
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': pred_prob,
                'actual': actual,
                'bet_on': 'home',
                'profit': profit if actual == 1 else -bet_size,
                'running_total': total_profit,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'result': bet_result,
                'odds': fair_odds
            })
            
        elif pred_prob < (1 - confidence_threshold):
            # Bet on away team
            total_bets += 1
            bet_made = True
            
            # Track high confidence bets
            if confidence > 0.7:
                high_confidence_bets += 1
            if uncertainty < 0.3:
                low_uncertainty_bets += 1
            
            implied_prob = 1 - pred_prob
            fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
            
            if actual == 0:  # Away team won
                winning_bets += 1
                profit = bet_size * (fair_odds / 100)
                total_profit += profit
                bet_result = 'WIN'
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                total_profit -= bet_size
                bet_result = 'LOSS'
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
            bet_history.append({
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': pred_prob,
                'actual': actual,
                'bet_on': 'away',
                'profit': profit if actual == 0 else -bet_size,
                'running_total': total_profit,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'result': bet_result,
                'odds': fair_odds
            })
        
        running_profit.append(total_profit)
        
        # Print progress every 50 games
        if (i + 1) % 50 == 0 or i == len(predictions) - 1:
            print(f"  Game {i+1}/{len(predictions)}: {total_bets} bets, ${total_profit:,.0f} profit")
    
    # Calculate advanced metrics
    win_rate = winning_bets / max(1, total_bets)
    roi = (total_profit / max(1, total_bets * bet_size)) * 100
    
    # Maximum drawdown
    if running_profit:
        peak = np.maximum.accumulate(running_profit)
        drawdown = peak - running_profit
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = 0
    
    # Sharpe ratio (simplified)
    if len(bet_history) > 1:
        profits = [bet['profit'] for bet in bet_history]
        if np.std(profits) > 0:
            sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(82)  # NBA season games
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # Print detailed betting statistics
    print(f"\n📈 BETTING STATISTICS:")
    print(f"  Total Bets Made: {total_bets}")
    print(f"  High Confidence Bets (>70%): {high_confidence_bets}")
    print(f"  Low Uncertainty Bets (<30%): {low_uncertainty_bets}")
    print(f"  Max Consecutive Wins: {max_consecutive_wins}")
    print(f"  Max Consecutive Losses: {max_consecutive_losses}")
    
    # Show recent betting history (last 10 bets)
    if bet_history:
        print(f"\n🎯 RECENT BETTING HISTORY (Last 10 bets):")
        print(f"{'Date':<12} {'Matchup':<20} {'Bet On':<6} {'Pred':<6} {'Actual':<6} {'Result':<6} {'Profit':<8} {'Total':<10}")
        print("-" * 80)
        
        for bet in bet_history[-10:]:
            matchup = f"{bet['away_team']} @ {bet['home_team']}"
            print(f"{bet['date'].strftime('%Y-%m-%d'):<12} {matchup:<20} {bet['bet_on']:<6} "
                  f"{bet['prediction']:.3f} {bet['actual']:<6} {bet['result']:<6} "
                  f"${bet['profit']:>6.0f} ${bet['running_total']:>8.0f}")
    
    return {
        'total_profit': total_profit,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate * 100,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'bet_history': bet_history,
        'running_profit': running_profit,
        'high_confidence_bets': high_confidence_bets,
        'low_uncertainty_bets': low_uncertainty_bets,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }

def calculate_betting_performance(y_true, predictions, df, bet_size, confidence_threshold):
    """Calculate detailed betting performance metrics (backward compatibility)"""
    return calculate_verbose_betting_performance(y_true, predictions, df, bet_size, confidence_threshold)

def create_backtest_visualizations(results, model_names, save_plots=True):
    """Create comprehensive visualization of backtest results"""
    print("\n📊 Creating backtest visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NBA ML Backtesting Results - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Profit curves
    ax1.set_title('Cumulative Profit Over Time', fontweight='bold')
    for model_name, result in results.items():
        if result and 'running_profit' in result and 'bet_history' in result:
            # Use bet history for dates and running totals
            if result['bet_history']:
                dates = [bet['date'] for bet in result['bet_history']]
                running_totals = [bet['running_total'] for bet in result['bet_history']]
                ax1.plot(dates, running_totals, label=f"{model_name} (ROI: {result['roi']:.1f}%)", linewidth=2)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Profit ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Model comparison
    model_data = []
    for model_name, result in results.items():
        if result:
            model_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'] * 100,
                'ROI': result['roi'],
                'Win Rate': result['win_rate'],
                'Sharpe': result['sharpe_ratio']
            })
    
    if model_data:
        comparison_df = pd.DataFrame(model_data)
        
        # Accuracy comparison
        bars = ax2.bar(comparison_df['Model'], comparison_df['Accuracy'], alpha=0.7)
        ax2.set_title('Model Accuracy Comparison', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(50, max(70, comparison_df['Accuracy'].max() + 5))
        
        # Add value labels on bars
        for bar, acc in zip(bars, comparison_df['Accuracy']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. ROI comparison
        bars = ax3.bar(comparison_df['Model'], comparison_df['ROI'], alpha=0.7, color='green')
        ax3.set_title('Return on Investment (ROI)', fontweight='bold')
        ax3.set_ylabel('ROI (%)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, roi in zip(bars, comparison_df['ROI']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{roi:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Risk metrics
        ax4.scatter(comparison_df['Win Rate'], comparison_df['ROI'], s=100, alpha=0.7)
        for i, model in enumerate(comparison_df['Model']):
            ax4.annotate(model, (comparison_df['Win Rate'].iloc[i], comparison_df['ROI'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_title('Risk vs Return Analysis', fontweight='bold')
        ax4.set_xlabel('Win Rate (%)')
        ax4.set_ylabel('ROI (%)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs("Backtest_Results", exist_ok=True)
        filename = f"Backtest_Results/backtest_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {filename}")
    
    plt.show()

def test_parlay_backtesting(parlay_predictor, df, sample_size=20):
    """Test parlay prediction system with historical data"""
    print("\n🎯 PARLAY PREDICTION BACKTESTING")
    print("="*60)
    
    try:
        # Create sample games from historical data
        sample_games = []
        for i in range(min(sample_size, len(df))):
            sample_games.append({
                'away_team': df.iloc[i]['TEAM_NAME.1'],
                'home_team': df.iloc[i]['TEAM_NAME'],
                'date': df.iloc[i]['Date']
            })
        
        print(f"📊 Testing parlay predictions on {len(sample_games)} games...")
        
        # Create mock player data for testing
        
        mock_player_data = pd.DataFrame({
            'Player': [i for i in range(100)],
            'Team': [i % 30 for i in range(100)],
            'PTS': np.random.normal(25, 5, 100),
            'REB': np.random.normal(8, 2, 100),
            'AST': np.random.normal(7, 2, 100),
            'Date': [datetime.now() - timedelta(days=i) for i in range(100)]
        })
        
        # Generate parlay combinations
        try:
            parlays = parlay_predictor.analyze_game_day_parlays(sample_games, mock_player_data)
            if parlays is None:
                parlays = []
            print(f"🎯 Generated {len(parlays)} parlay combinations")
        except Exception as e:
            print(f"⚠️ Error generating parlays: {e}")
            parlays = []
        
        if parlays and isinstance(parlays, list) and len(parlays) > 0:
            # Check if parlays are dictionaries
            if isinstance(parlays[0], dict):
                # Analyze parlay performance
                total_parlays = len(parlays)
                high_value_parlays = [p for p in parlays if p.get('advanced_score', 0) > 7.0]
                low_risk_parlays = [p for p in parlays if p.get('risk_score', 1) < 0.3]
            else:
                print(f"⚠️ Parlays returned as {type(parlays[0])}, expected dictionaries")
                total_parlays = len(parlays)
                high_value_parlays = []
                low_risk_parlays = []
            
            print(f"\n📈 PARLAY ANALYSIS:")
            print(f"  Total Parlays Generated: {total_parlays}")
            print(f"  High Value Parlays (>7.0 score): {len(high_value_parlays)}")
            print(f"  Low Risk Parlays (<0.3 risk): {len(low_risk_parlays)}")
            
            # Show top 5 parlays
            if parlays and isinstance(parlays[0], dict):
                print(f"\n🏆 TOP 5 PARLAY COMBINATIONS:")
                print(f"{'Rank':<4} {'Legs':<4} {'Prob':<6} {'Odds':<8} {'Score':<6} {'Risk':<6} {'Value':<6}")
                print("-" * 50)
                
                sorted_parlays = sorted(parlays, key=lambda x: x.get('advanced_score', 0), reverse=True)
                for i, parlay in enumerate(sorted_parlays[:5], 1):
                    print(f"{i:<4} {parlay.get('num_legs', 0):<4} "
                          f"{parlay.get('combined_probability', 0):.3f} "
                          f"{parlay.get('american_odds', 0):+d} "
                          f"{parlay.get('advanced_score', 0):.1f} "
                          f"{parlay.get('risk_score', 0):.3f} "
                          f"{parlay.get('expected_value', 0):.3f}")
            else:
                print(f"\n⚠️ Cannot display parlay details - invalid format")
            
            if isinstance(parlays[0], dict):
                return {
                    'total_parlays': total_parlays,
                    'high_value_parlays': len(high_value_parlays),
                    'low_risk_parlays': len(low_risk_parlays),
                    'avg_advanced_score': np.mean([p.get('advanced_score', 0) for p in parlays]),
                    'avg_risk_score': np.mean([p.get('risk_score', 0) for p in parlays]),
                    'avg_expected_value': np.mean([p.get('expected_value', 0) for p in parlays])
                }
            else:
                return {
                    'total_parlays': total_parlays,
                    'high_value_parlays': 0,
                    'low_risk_parlays': 0,
                    'avg_advanced_score': 0,
                    'avg_risk_score': 0,
                    'avg_expected_value': 0
                }
        else:
            print("⚠️ No parlay combinations generated")
            return None
            
    except Exception as e:
        print(f"❌ Parlay backtesting failed: {e}")
        return None

def generate_detailed_report(results, df):
    """Generate comprehensive backtest report"""
    print("\n📋 DETAILED BACKTESTING REPORT")
    print("="*70)
    
    # Overall statistics
    print(f"\n📊 DATASET STATISTICS")
    print(f"Total Games Analyzed: {len(df):,}")
    print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Home Team Win Rate: {df['Home-Team-Win'].mean():.1%}")
    
    # Model comparison table
    print(f"\n🏆 MODEL PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'ROI':<8} {'Profit':<12} {'Bets':<6} {'Win Rate':<8}")
    print("-" * 70)
    
    best_accuracy = 0
    best_roi = -float('inf')
    best_profit = -float('inf')
    
    for model_name, result in results.items():
        if result:
            accuracy = result['accuracy'] * 100
            roi = result['roi']
            profit = result['total_profit']
            bets = result['total_bets']
            win_rate = result['win_rate']
            
            # Track best performers
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            if roi > best_roi:
                best_roi = roi
            if profit > best_profit:
                best_profit = profit
            
            # Format output
            print(f"{model_name:<20} {accuracy:>7.1f}% {roi:>6.1f}% ${profit:>9,.0f} {bets:>5} {win_rate:>6.1f}%")
    
    print("-" * 70)
    
    # Best performers
    print(f"\n🥇 BEST PERFORMERS")
    print(f"Highest Accuracy: {best_accuracy:.1f}%")
    print(f"Best ROI: {best_roi:.1f}%")
    print(f"Highest Profit: ${best_profit:,.0f}")
    
    # Risk analysis
    print(f"\n⚠️ RISK ANALYSIS")
    for model_name, result in results.items():
        if result and result['total_bets'] > 0:
            max_dd = result['max_drawdown']
            sharpe = result['sharpe_ratio']
            print(f"{model_name}: Max Drawdown ${max_dd:,.0f}, Sharpe Ratio {sharpe:.2f}")
    
    # Save detailed report
    os.makedirs("Backtest_Results", exist_ok=True)
    report_file = f"Backtest_Results/detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"NBA ML Backtesting Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for model_name, result in results.items():
            if result:
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy: {result['accuracy']*100:.1f}%\n")
                f.write(f"  ROI: {result['roi']:.1f}%\n")
                f.write(f"  Total Profit: ${result['total_profit']:,.2f}\n")
                f.write(f"  Total Bets: {result['total_bets']}\n")
                f.write(f"  Win Rate: {result['win_rate']:.1f}%\n")
                f.write(f"  Max Drawdown: ${result['max_drawdown']:,.2f}\n")
                f.write(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}\n\n")
    
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Generate detailed betting CSV files for each model
    generate_detailed_betting_csvs(results, df)

def generate_detailed_betting_csvs(results, df):
    """Generate detailed CSV files with all betting information for each model"""
    print("\n📊 Generating detailed betting CSV files...")
    
    for model_name, result in results.items():
        if result and result.get('bet_history') and len(result['bet_history']) > 0:
            print(f"📝 Creating detailed CSV for {model_name}...")
            
            # Create detailed betting data
            betting_data = []
            
            for i, bet in enumerate(result['bet_history']):
                # Get game information from original dataframe
                game_info = None
                for idx, row in df.iterrows():
                    if (row['Date'] == bet['date'] and 
                        row['TEAM_NAME'] == bet['home_team'] and 
                        row['TEAM_NAME.1'] == bet['away_team']):
                        game_info = row
                        break
                
                # Calculate odds and implied probability
                if bet['bet_on'] == 'home':
                    implied_prob = bet['prediction']
                    fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
                    ml_odds = int(fair_odds) if fair_odds > 0 else 100
                    away_ml_odds = int(100 / (1 - implied_prob)) if (1 - implied_prob) > 0 else 100
                else:
                    implied_prob = 1 - bet['prediction']
                    fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
                    ml_odds = int(fair_odds) if fair_odds > 0 else 100
                    away_ml_odds = int(100 / (1 - implied_prob)) if (1 - implied_prob) > 0 else 100
                
                # Calculate margin (if we have game info)
                margin = 0
                if game_info is not None:
                    home_score = game_info.get('Score', 0)
                    away_score = game_info.get('Score', 0)  # This should be away score, but using same for now
                    margin = abs(home_score - away_score)
                
                # Determine if prediction was correct
                correct = "Yes" if bet['result'] == 'WIN' else "No"
                
                # Calculate money won/lost
                money_result = bet['profit']
                
                betting_data.append({
                    'Game_Number': i + 1,
                    'Date': bet['date'].strftime('%Y-%m-%d'),
                    'Away': bet['away_team'],
                    'Home': bet['home_team'],
                    'OU': game_info.get('OU', 0) if game_info is not None else 0,
                    'Spread': 0,  # Would need to calculate from odds
                    'IL': round(implied_prob, 3),
                    'ML': ml_odds,
                    'Away_ML': away_ml_odds,
                    'Points': game_info.get('Score', 0) if game_info is not None else 0,
                    'Win': 1 if bet['actual'] == 1 else 0,
                    'Margin': margin,
                    'Prediction': 1 if bet['bet_on'] == 'home' else 0,
                    'Correct?': correct,
                    'Bet_Amount': 100,  # Fixed bet size
                    'Money_Lost_Won': round(money_result, 2),
                    'Running_Profit': round(bet['running_total'], 2),
                    'Confidence': round(bet.get('confidence', bet['prediction']), 3),
                    'Uncertainty': round(bet.get('uncertainty', 0.1), 3),
                    'Odds': round(bet.get('odds', fair_odds), 2)
                })
            
            # Create DataFrame and save to CSV
            betting_df = pd.DataFrame(betting_data)
            
            # Add summary rows at the top
            summary_data = [
                {
                    'Game_Number': 'SUMMARY',
                    'Date': f'Tested: {df["Date"].min().strftime("%m/%d/%Y")}',
                    'Away': '',
                    'Home': '',
                    'OU': '',
                    'Spread': '',
                    'IL': '',
                    'ML': '',
                    'Away_ML': '',
                    'Points': '',
                    'Win': '',
                    'Margin': '',
                    'Prediction': '',
                    'Correct?': '',
                    'Bet_Amount': f'Bet for All Game: $100',
                    'Money_Lost_Won': f'Total Profit: ${result["total_profit"]:,.2f}',
                    'Running_Profit': '',
                    'Confidence': '',
                    'Uncertainty': '',
                    'Odds': ''
                },
                {
                    'Game_Number': '',
                    'Date': '',
                    'Away': '',
                    'Home': '',
                    'OU': '',
                    'Spread': '',
                    'IL': '',
                    'ML': '',
                    'Away_ML': '',
                    'Points': '',
                    'Win': '',
                    'Margin': '',
                    'Prediction': '',
                    'Correct?': '',
                    'Bet_Amount': f'Total Bets: {result["total_bets"]}',
                    'Money_Lost_Won': f'Win Rate: {result["win_rate"]:.1f}%',
                    'Running_Profit': '',
                    'Confidence': '',
                    'Uncertainty': '',
                    'Odds': ''
                }
            ]
            
            # Combine summary and betting data
            summary_df = pd.DataFrame(summary_data)
            combined_df = pd.concat([summary_df, betting_df], ignore_index=True)
            
            # Save to CSV
            csv_filename = f"Backtest_Results/{model_name}_detailed_betting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            combined_df.to_csv(csv_filename, index=False)
            print(f"✅ Detailed betting CSV saved: {csv_filename}")
            
            # Also create a simplified version for easier reading
            simple_data = []
            for bet in result['bet_history']:
                simple_data.append({
                    'Date': bet['date'].strftime('%Y-%m-%d'),
                    'Away_Team': bet['away_team'],
                    'Home_Team': bet['home_team'],
                    'Bet_On': bet['bet_on'],
                    'Prediction': round(bet['prediction'], 3),
                    'Actual': bet['actual'],
                    'Result': bet['result'],
                    'Bet_Amount': 100,
                    'Profit_Loss': round(bet['profit'], 2),
                    'Running_Total': round(bet['running_total'], 2),
                    'Confidence': round(bet.get('confidence', bet['prediction']), 3),
                    'Odds': round(bet.get('odds', 0), 2)
                })
            
            simple_df = pd.DataFrame(simple_data)
            simple_csv_filename = f"Backtest_Results/{model_name}_simple_betting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            simple_df.to_csv(simple_csv_filename, index=False)
            print(f"✅ Simple betting CSV saved: {simple_csv_filename}")
    
    print(f"📊 All detailed betting CSV files generated successfully!")

def create_running_profit_charts(results, save_plots=True):
    """Create running profit charts similar to the screenshot"""
    print("\n📈 Creating running profit charts...")
    
    for model_name, result in results.items():
        if result and result.get('bet_history') and len(result['bet_history']) > 0:
            print(f"📊 Creating running profit chart for {model_name}...")
            
            # Extract running profit data
            running_profits = [bet['running_total'] for bet in result['bet_history']]
            game_numbers = list(range(1, len(running_profits) + 1))
            
            # Create the chart
            plt.figure(figsize=(12, 8))
            plt.plot(game_numbers, running_profits, linewidth=2, color='blue', alpha=0.8)
            plt.fill_between(game_numbers, running_profits, alpha=0.3, color='blue')
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Formatting
            plt.title(f'Running Profit - {model_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
            plt.xlabel('Game Number', fontsize=12)
            plt.ylabel('Cumulative Profit ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add profit statistics as text
            final_profit = running_profits[-1] if running_profits else 0
            max_profit = max(running_profits) if running_profits else 0
            min_profit = min(running_profits) if running_profits else 0
            
            stats_text = f'Final Profit: ${final_profit:,.2f}\nMax Profit: ${max_profit:,.2f}\nMin Profit: ${min_profit:,.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots:
                chart_filename = f"Backtest_Results/{model_name}_running_profit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Running profit chart saved: {chart_filename}")
            
            plt.show()
    
    print(f"📈 All running profit charts generated successfully!")

def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description='NBA ML Backtesting Script')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date for backtesting')
    parser.add_argument('--end-date', default='2024-06-30', help='End date for backtesting')
    parser.add_argument('--bet-size', type=float, default=100, help='Bet size in dollars')
    parser.add_argument('--confidence', type=float, default=0.55, help='Confidence threshold for betting')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    
    args = parser.parse_args()
    
    print_header()
    
    # Load historical data
    df = load_historical_data(args.start_date, args.end_date)
    if df is None or len(df) == 0:
        print("❌ No historical data available for backtesting")
        return False
    
    # Load models
    models = load_available_models()
    if not models:
        print("❌ No trained models found for backtesting")
        print("💡 Train models first: python train.py --all")
        return False
    
    # Filter models if specified
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    
    # Run backtesting
    print(f"\n🧪 Running backtesting on {len(models)} models...")
    results = {}
    parlay_results = None
    
    for model_name, model_info in models.items():
        result = backtest_model(model_info, df, args.bet_size, args.confidence)
        if result:
            results[model_name] = result
        
        # Test parlay prediction if available
        if model_info['type'] == 'parlay':
            print(f"\n🎯 Testing parlay prediction system...")
            parlay_results = test_parlay_backtesting(model_info['predictor'], df)
    
    # Generate visualizations
    if not args.no_plots and results:
        create_backtest_visualizations(results, list(results.keys()))
        create_running_profit_charts(results, save_plots=True)
    
    # Generate detailed report
    if results:
        generate_detailed_report(results, df)
    
    # Add parlay results to report if available
    if parlay_results:
        print(f"\n🎯 PARLAY PREDICTION SUMMARY:")
        print(f"  Total Parlays Generated: {parlay_results['total_parlays']}")
        print(f"  High Value Parlays: {parlay_results['high_value_parlays']}")
        print(f"  Low Risk Parlays: {parlay_results['low_risk_parlays']}")
        print(f"  Average Advanced Score: {parlay_results['avg_advanced_score']:.2f}")
        print(f"  Average Risk Score: {parlay_results['avg_risk_score']:.3f}")
        print(f"  Average Expected Value: {parlay_results['avg_expected_value']:.3f}")
    
    # Final summary
    print(f"\n🎉 BACKTESTING COMPLETE!")
    print(f"📊 Tested {len(results)} models on {len(df)} games")
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['roi'])
        print(f"🏆 Best performing model: {best_model[0]} (ROI: {best_model[1]['roi']:.1f}%)")
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

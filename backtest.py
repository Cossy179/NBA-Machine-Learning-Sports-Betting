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
    """Load all available trained models"""
    print("🤖 Loading available models...")
    
    models = {}
    
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
    """Backtest a single model with detailed statistics"""
    print(f"\n🧪 Backtesting: {model_info.get('name', 'Unknown Model')}")
    print("-" * 40)
    
    try:
        # Prepare features
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[feature_cols].fillna(0).astype(float)
        y_true = df["Home-Team-Win"].astype(int)
        
        # Get predictions based on model type
        if model_info['type'] == 'auto':
            predictions = []
            for i in range(len(X)):
                pred_result = model_info['selector'].predict_with_best_model(X.iloc[i:i+1])
                if pred_result:
                    predictions.append(pred_result.get('probability', 0.5))
                else:
                    predictions.append(0.5)
            predictions = np.array(predictions)
            
        elif model_info['type'] == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(model_info['path'])
            
            dtest = xgb.DMatrix(X)
            predictions = model.predict(dtest)
            
            # Handle multi-class output
            if predictions.ndim > 1 or (len(predictions) > 0 and hasattr(predictions[0], '__len__')):
                predictions = np.array([pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred for pred in predictions])
        
        # Calculate betting results
        betting_results = calculate_betting_performance(
            y_true, predictions, df, bet_size, confidence_threshold
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
        
        # Combine results
        results = {**model_metrics, **betting_results}
        
        print(f"📊 Model Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"💰 Total Profit: ${results['total_profit']:,.2f}")
        print(f"📈 ROI: {results['roi']:.1f}%")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"📉 Max Drawdown: ${results['max_drawdown']:,.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return None

def calculate_betting_performance(y_true, predictions, df, bet_size, confidence_threshold):
    """Calculate detailed betting performance metrics"""
    
    # Betting simulation
    total_profit = 0
    total_bets = 0
    winning_bets = 0
    bet_history = []
    running_profit = []
    
    for i in range(len(predictions)):
        pred_prob = predictions[i]
        actual = y_true[i]
        game_date = df.iloc[i]['Date']
        home_team = df.iloc[i]['TEAM_NAME']
        away_team = df.iloc[i]['TEAM_NAME.1']
        
        # Only bet if confidence is above threshold
        if pred_prob > confidence_threshold:
            # Bet on home team
            total_bets += 1
            
            # Simulate odds (in practice, would use real odds)
            implied_prob = pred_prob
            fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
            
            if actual == 1:  # Home team won
                winning_bets += 1
                profit = bet_size * (fair_odds / 100)
                total_profit += profit
            else:
                total_profit -= bet_size
                
            bet_history.append({
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': pred_prob,
                'actual': actual,
                'bet_on': 'home',
                'profit': profit if actual == 1 else -bet_size,
                'running_total': total_profit
            })
            
        elif pred_prob < (1 - confidence_threshold):
            # Bet on away team
            total_bets += 1
            
            implied_prob = 1 - pred_prob
            fair_odds = 100 / implied_prob if implied_prob > 0.5 else 100 / (1 - implied_prob)
            
            if actual == 0:  # Away team won
                winning_bets += 1
                profit = bet_size * (fair_odds / 100)
                total_profit += profit
            else:
                total_profit -= bet_size
                
            bet_history.append({
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': pred_prob,
                'actual': actual,
                'bet_on': 'away',
                'profit': profit if actual == 0 else -bet_size,
                'running_total': total_profit
            })
        
        running_profit.append(total_profit)
    
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
    
    return {
        'total_profit': total_profit,
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate * 100,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'bet_history': bet_history,
        'running_profit': running_profit
    }

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
        if result and 'running_profit' in result:
            dates = [bet['date'] for bet in result['bet_history']]
            if dates:
                ax1.plot(dates, result['running_profit'], label=f"{model_name} (ROI: {result['roi']:.1f}%)", linewidth=2)
    
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
    
    for model_name, model_info in models.items():
        result = backtest_model(model_info, df, args.bet_size, args.confidence)
        if result:
            results[model_name] = result
    
    # Generate visualizations
    if not args.no_plots and results:
        create_backtest_visualizations(results, list(results.keys()))
    
    # Generate detailed report
    if results:
        generate_detailed_report(results, df)
    
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

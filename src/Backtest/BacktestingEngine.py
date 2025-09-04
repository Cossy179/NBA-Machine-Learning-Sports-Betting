"""
Comprehensive Backtesting Engine for NBA prediction models.
Tests models on historical data (2023-2024 season) with detailed performance metrics,
ROI analysis, and betting strategy evaluation.
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BacktestingEngine:
    def __init__(self):
        self.results = {}
        self.betting_results = []
        self.performance_metrics = {}
        self.roi_tracking = []
        
    def load_historical_data(self, start_date="2023-10-01", end_date="2024-06-30"):
        """Load historical game data for backtesting"""
        print(f"Loading historical data from {start_date} to {end_date}...")
        
        try:
            con = sqlite3.connect("Data/dataset.sqlite")
            
            # Load the enhanced dataset if available, otherwise base dataset
            query = '''
            SELECT * FROM "dataset_2012-24_new" 
            WHERE Date >= ? AND Date <= ?
            ORDER BY Date
            '''
            
            df = pd.read_sql_query(query, con, params=[start_date, end_date])
            con.close()
            
            if df.empty:
                print("No historical data found for the specified period")
                return pd.DataFrame()
            
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'])
            
            print(f"Loaded {len(df)} games for backtesting")
            return df
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def prepare_features_for_game(self, game_row, feature_cols):
        """Prepare features for a single game prediction"""
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        
        # Get available features
        available_features = [col for col in feature_cols if col in game_row.index and col not in exclude_cols]
        
        # Create feature vector
        features = {}
        for col in available_features:
            features[col] = game_row[col] if pd.notna(game_row[col]) else 0
        
        return features
    
    def simulate_betting_strategy(self, predictions, actual_outcomes, odds_data=None, strategy="kelly", bankroll=10000):
        """Simulate betting strategy with various approaches"""
        current_bankroll = bankroll
        bet_history = []
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_outcomes)):
            if pred is None:
                continue
                
            probability = pred.get('probability', 0.5)
            confidence = pred.get('confidence', 0)
            
            # Betting decision based on strategy
            should_bet = False
            bet_amount = 0
            
            if strategy == "kelly":
                # Kelly Criterion with confidence threshold
                if confidence > 0.6:  # Only bet on high confidence
                    # Assume -110 odds for simplicity (would use real odds in practice)
                    implied_odds = 1.91  # Decimal odds for -110
                    edge = probability - (1/implied_odds)
                    
                    if edge > 0:
                        kelly_fraction = edge / (implied_odds - 1)
                        bet_amount = current_bankroll * min(kelly_fraction, 0.25)  # Cap at 25%
                        should_bet = True
            
            elif strategy == "fixed_percentage":
                if confidence > 0.65:
                    bet_amount = current_bankroll * 0.02  # 2% of bankroll
                    should_bet = True
            
            elif strategy == "fixed_amount":
                if confidence > 0.6:
                    bet_amount = min(100, current_bankroll * 0.05)  # $100 or 5% max
                    should_bet = True
            
            if should_bet and bet_amount > 0 and current_bankroll > bet_amount:
                # Make the bet
                predicted_outcome = pred.get('prediction', 0)
                
                if predicted_outcome == actual:
                    # Win
                    profit = bet_amount * 0.91  # -110 odds profit
                    current_bankroll += profit
                    result = 'WIN'
                else:
                    # Loss
                    current_bankroll -= bet_amount
                    profit = -bet_amount
                    result = 'LOSS'
                
                bet_history.append({
                    'game_index': i,
                    'bet_amount': bet_amount,
                    'probability': probability,
                    'confidence': confidence,
                    'predicted': predicted_outcome,
                    'actual': actual,
                    'result': result,
                    'profit': profit,
                    'bankroll': current_bankroll
                })
        
        return bet_history, current_bankroll
    
    def run_model_backtest(self, model_predictor, historical_data, model_name="Unknown"):
        """Run backtest for a specific model"""
        print(f"Running backtest for {model_name}...")
        
        predictions = []
        actual_outcomes = []
        
        # Get feature columns (this would need to be adapted based on your model)
        feature_cols = [col for col in historical_data.columns 
                       if col not in ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1"]]
        
        for idx, game_row in historical_data.iterrows():
            try:
                # Prepare features
                game_features = self.prepare_features_for_game(game_row, feature_cols)
                
                # Make prediction
                if hasattr(model_predictor, 'predict_single_game'):
                    prediction = model_predictor.predict_single_game(game_features)
                elif callable(model_predictor):
                    prediction = model_predictor(game_features)
                else:
                    prediction = None
                
                predictions.append(prediction)
                actual_outcomes.append(int(game_row['Home-Team-Win']))
                
            except Exception as e:
                print(f"Error predicting game {idx}: {e}")
                predictions.append(None)
                actual_outcomes.append(int(game_row['Home-Team-Win']))
        
        # Calculate accuracy metrics
        valid_predictions = [(p, a) for p, a in zip(predictions, actual_outcomes) if p is not None]
        
        if not valid_predictions:
            print("No valid predictions generated")
            return {}
        
        pred_outcomes = [p.get('prediction', 0) for p, a in valid_predictions]
        actual_valid = [a for p, a in valid_predictions]
        
        accuracy = sum(1 for p, a in zip(pred_outcomes, actual_valid) if p == a) / len(valid_predictions)
        
        # Calculate probability metrics
        probabilities = [p.get('probability', 0.5) for p, a in valid_predictions]
        log_loss = self.calculate_log_loss(probabilities, actual_valid)
        brier_score = self.calculate_brier_score(probabilities, actual_valid)
        
        # Betting simulations
        betting_strategies = ['kelly', 'fixed_percentage', 'fixed_amount']
        betting_results = {}
        
        for strategy in betting_strategies:
            bet_history, final_bankroll = self.simulate_betting_strategy(
                predictions, actual_outcomes, strategy=strategy
            )
            
            if bet_history:
                roi = (final_bankroll - 10000) / 10000 * 100
                win_rate = sum(1 for bet in bet_history if bet['result'] == 'WIN') / len(bet_history)
                avg_bet = np.mean([bet['bet_amount'] for bet in bet_history])
                
                betting_results[strategy] = {
                    'final_bankroll': final_bankroll,
                    'roi': roi,
                    'total_bets': len(bet_history),
                    'win_rate': win_rate,
                    'avg_bet_size': avg_bet,
                    'bet_history': bet_history
                }
        
        results = {
            'model_name': model_name,
            'total_games': len(historical_data),
            'valid_predictions': len(valid_predictions),
            'accuracy': accuracy,
            'log_loss': log_loss,
            'brier_score': brier_score,
            'betting_results': betting_results
        }
        
        self.results[model_name] = results
        return results
    
    def calculate_log_loss(self, probabilities, actual_outcomes):
        """Calculate log loss"""
        epsilon = 1e-15  # Small value to avoid log(0)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        log_loss = 0
        for prob, actual in zip(probabilities, actual_outcomes):
            if actual == 1:
                log_loss += -np.log(prob)
            else:
                log_loss += -np.log(1 - prob)
        
        return log_loss / len(probabilities)
    
    def calculate_brier_score(self, probabilities, actual_outcomes):
        """Calculate Brier score"""
        brier_sum = sum((prob - actual) ** 2 for prob, actual in zip(probabilities, actual_outcomes))
        return brier_sum / len(probabilities)
    
    def run_comprehensive_backtest(self):
        """Run comprehensive backtest on all available models"""
        print("Starting comprehensive backtest...")
        
        # Load historical data
        historical_data = self.load_historical_data()
        
        if historical_data.empty:
            print("No historical data available")
            return
        
        # Test different models
        from src.Predict.AutoModelSelector import AutoModelSelector
        
        selector = AutoModelSelector()
        available_models = selector.scan_available_models()
        
        if not available_models:
            print("No trained models found for backtesting")
            return
        
        # Test each available model
        for model_name, model_info in available_models.items():
            try:
                if model_name == 'boosted_system':
                    model_predictor = selector.load_boosted_system()
                elif model_name == 'ensemble_system':
                    model_predictor = selector.load_ensemble_system()
                else:
                    continue  # Skip for now
                
                if model_predictor is not None:
                    self.run_model_backtest(model_predictor, historical_data, model_name)
                    
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Generate summary report
        self.generate_backtest_report()
    
    def generate_backtest_report(self):
        """Generate comprehensive backtest report"""
        if not self.results:
            print("No backtest results to report")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BACKTEST REPORT")
        print("="*80)
        
        # Model comparison table
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Log Loss':<10} {'Brier':<10} {'Best ROI':<10}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            best_roi = max([br['roi'] for br in results['betting_results'].values()]) if results['betting_results'] else 0
            
            print(f"{model_name:<20} {results['accuracy']:<10.3f} {results['log_loss']:<10.3f} "
                  f"{results['brier_score']:<10.3f} {best_roi:<10.1f}%")
        
        # Detailed betting strategy results
        print("\nBETTING STRATEGY RESULTS:")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            
            for strategy, bet_results in results['betting_results'].items():
                print(f"  {strategy.title()}:")
                print(f"    ROI: {bet_results['roi']:+.1f}%")
                print(f"    Win Rate: {bet_results['win_rate']:.1%}")
                print(f"    Total Bets: {bet_results['total_bets']}")
                print(f"    Avg Bet Size: ${bet_results['avg_bet_size']:.2f}")
        
        # Best overall model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBEST OVERALL MODEL: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
        
        # Best ROI model
        best_roi_model = None
        best_roi_value = -100
        
        for model_name, results in self.results.items():
            for strategy, bet_results in results['betting_results'].items():
                if bet_results['roi'] > best_roi_value:
                    best_roi_value = bet_results['roi']
                    best_roi_model = (model_name, strategy)
        
        if best_roi_model:
            print(f"BEST ROI STRATEGY: {best_roi_model[0]} with {best_roi_model[1]} ({best_roi_value:+.1f}%)")
    
    def plot_performance_charts(self):
        """Generate performance visualization charts"""
        if not self.results:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('NBA Model Backtesting Results', fontsize=16, fontweight='bold')
            
            # 1. Accuracy Comparison
            models = list(self.results.keys())
            accuracies = [self.results[model]['accuracy'] for model in models]
            
            axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. ROI Comparison
            roi_data = []
            strategy_labels = []
            
            for model in models:
                for strategy, results in self.results[model]['betting_results'].items():
                    roi_data.append(results['roi'])
                    strategy_labels.append(f"{model}\n{strategy}")
            
            axes[0, 1].bar(range(len(roi_data)), roi_data, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('ROI by Strategy')
            axes[0, 1].set_ylabel('ROI (%)')
            axes[0, 1].set_xticks(range(len(strategy_labels)))
            axes[0, 1].set_xticklabels(strategy_labels, rotation=45, ha='right')
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 3. Log Loss vs Accuracy
            log_losses = [self.results[model]['log_loss'] for model in models]
            
            axes[1, 0].scatter(log_losses, accuracies, s=100, alpha=0.7)
            for i, model in enumerate(models):
                axes[1, 0].annotate(model, (log_losses[i], accuracies[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 0].set_xlabel('Log Loss')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Accuracy vs Log Loss')
            
            # 4. Betting Volume vs Win Rate
            bet_volumes = []
            win_rates = []
            labels = []
            
            for model in models:
                for strategy, results in self.results[model]['betting_results'].items():
                    bet_volumes.append(results['total_bets'])
                    win_rates.append(results['win_rate'])
                    labels.append(f"{model}_{strategy}")
            
            scatter = axes[1, 1].scatter(bet_volumes, win_rates, s=100, alpha=0.7)
            axes[1, 1].set_xlabel('Total Bets')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].set_title('Betting Volume vs Win Rate')
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Break-even')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def save_detailed_results(self, filename="backtest_detailed_results.csv"):
        """Save detailed results to CSV"""
        try:
            all_bets = []
            
            for model_name, results in self.results.items():
                for strategy, bet_results in results['betting_results'].items():
                    for bet in bet_results['bet_history']:
                        bet_record = bet.copy()
                        bet_record['model'] = model_name
                        bet_record['strategy'] = strategy
                        all_bets.append(bet_record)
            
            if all_bets:
                df = pd.DataFrame(all_bets)
                df.to_csv(filename, index=False)
                print(f"Detailed results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

def run_full_backtest():
    """Run complete backtesting pipeline"""
    engine = BacktestingEngine()
    
    # Run comprehensive backtest
    engine.run_comprehensive_backtest()
    
    # Generate visualizations
    engine.plot_performance_charts()
    
    # Save detailed results
    engine.save_detailed_results()
    
    return engine

if __name__ == "__main__":
    print("Starting NBA Model Backtesting...")
    
    # Run full backtest
    backtest_engine = run_full_backtest()
    
    print("\nBacktesting complete! Check backtest_results.png and backtest_detailed_results.csv for detailed analysis.")

"""
Advanced Comprehensive Backtesting Engine for NBA prediction models.
Tests models on historical data with advanced performance metrics, uncertainty quantification,
ROI analysis, betting strategy evaluation, and market efficiency analysis.
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import json
warnings.filterwarnings('ignore')

class AdvancedBacktestingEngine:
    def __init__(self):
        self.results = {}
        self.betting_results = []
        self.performance_metrics = {}
        self.roi_tracking = []
        self.uncertainty_metrics = {}
        self.market_efficiency_metrics = {}
        self.calibration_metrics = {}
        self.advanced_metrics = {}
        self.feature_importance = {}
        self.model_comparison = {}
        self.risk_metrics = {}
        
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
    
    def run_advanced_model_backtest(self, model_predictor, historical_data, model_name="Unknown"):
        """Run advanced backtest for a specific model with comprehensive metrics"""
        print(f"Running advanced backtest for {model_name}...")
        
        predictions = []
        actual_outcomes = []
        probabilities = []
        uncertainties = []
        confidences = []
        
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
                
                # Extract advanced metrics
                if prediction and isinstance(prediction, dict):
                    probabilities.append(prediction.get('probability', 0.5))
                    uncertainties.append(prediction.get('uncertainty', 0.1))
                    confidences.append(prediction.get('confidence', 0.5))
                else:
                    probabilities.append(0.5)
                    uncertainties.append(0.1)
                    confidences.append(0.5)
                
            except Exception as e:
                print(f"Error predicting game {idx}: {e}")
                predictions.append(None)
                actual_outcomes.append(int(game_row['Home-Team-Win']))
                probabilities.append(0.5)
                uncertainties.append(0.1)
                confidences.append(0.5)
        
        # Calculate advanced metrics
        valid_predictions = [(p, a) for p, a in zip(predictions, actual_outcomes) if p is not None]
        
        if not valid_predictions:
            print("No valid predictions generated")
            return {}
        
        pred_outcomes = [p.get('prediction', 0) for p, a in valid_predictions]
        actual_valid = [a for p, a in valid_predictions]
        prob_valid = probabilities[:len(valid_predictions)]
        unc_valid = uncertainties[:len(valid_predictions)]
        
        # Calculate comprehensive metrics
        advanced_metrics = self.calculate_advanced_metrics(
            pred_outcomes, actual_valid, prob_valid, unc_valid
        )
        
        # Enhanced betting simulations with uncertainty
        betting_strategies = ['kelly', 'fixed_percentage', 'fixed_amount', 'uncertainty_adjusted']
        betting_results = {}
        
        for strategy in betting_strategies:
            bet_history, final_bankroll = self.simulate_advanced_betting_strategy(
                predictions, actual_outcomes, strategy=strategy, uncertainties=uncertainties
            )
            
            if bet_history:
                roi = (final_bankroll - 10000) / 10000 * 100
                win_rate = sum(1 for bet in bet_history if bet['result'] == 'WIN') / len(bet_history)
                avg_bet = np.mean([bet['bet_amount'] for bet in bet_history])
                
                # Calculate additional betting metrics
                profit_factor = self._calculate_profit_factor(bet_history)
                max_consecutive_wins = self._calculate_max_consecutive_wins(bet_history)
                max_consecutive_losses = self._calculate_max_consecutive_losses(bet_history)
                
                betting_results[strategy] = {
                    'final_bankroll': final_bankroll,
                    'roi': roi,
                    'total_bets': len(bet_history),
                    'win_rate': win_rate,
                    'avg_bet_size': avg_bet,
                    'profit_factor': profit_factor,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'bet_history': bet_history
                }
        
        # Feature importance analysis (if available)
        feature_importance = self._analyze_feature_importance(model_predictor, historical_data, feature_cols)
        
        results = {
            'model_name': model_name,
            'total_games': len(historical_data),
            'valid_predictions': len(valid_predictions),
            'advanced_metrics': advanced_metrics,
            'betting_results': betting_results,
            'feature_importance': feature_importance,
            'uncertainty_analysis': self._analyze_uncertainty(uncertainties, pred_outcomes, actual_valid),
            'calibration_analysis': self._analyze_calibration(probabilities, actual_valid)
        }
        
        self.results[model_name] = results
        self.advanced_metrics[model_name] = advanced_metrics
        return results
    
    def simulate_advanced_betting_strategy(self, predictions, actual_outcomes, strategy="kelly", 
                                         uncertainties=None, bankroll=10000):
        """Simulate advanced betting strategy with uncertainty consideration"""
        current_bankroll = bankroll
        bet_history = []
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_outcomes)):
            if pred is None:
                continue
                
            probability = pred.get('probability', 0.5)
            confidence = pred.get('confidence', 0)
            uncertainty = uncertainties[i] if uncertainties and i < len(uncertainties) else 0.1
            
            # Betting decision based on strategy
            should_bet = False
            bet_amount = 0
            
            if strategy == "kelly":
                # Kelly Criterion with confidence threshold
                if confidence > 0.6:
                    implied_odds = 1.91
                    edge = probability - (1/implied_odds)
                    
                    if edge > 0:
                        kelly_fraction = edge / (implied_odds - 1)
                        # Adjust for uncertainty
                        uncertainty_adjustment = 1 - uncertainty
                        bet_amount = current_bankroll * min(kelly_fraction * uncertainty_adjustment, 0.25)
                        should_bet = True
            
            elif strategy == "uncertainty_adjusted":
                # Only bet on low uncertainty predictions
                if uncertainty < 0.3 and confidence > 0.7:
                    implied_odds = 1.91
                    edge = probability - (1/implied_odds)
                    
                    if edge > 0:
                        bet_amount = current_bankroll * 0.02  # 2% of bankroll
                        should_bet = True
            
            elif strategy == "fixed_percentage":
                if confidence > 0.65 and uncertainty < 0.4:
                    bet_amount = current_bankroll * 0.02
                    should_bet = True
            
            elif strategy == "fixed_amount":
                if confidence > 0.6 and uncertainty < 0.5:
                    bet_amount = min(100, current_bankroll * 0.05)
                    should_bet = True
            
            if should_bet and bet_amount > 0 and current_bankroll > bet_amount:
                predicted_outcome = pred.get('prediction', 0)
                
                if predicted_outcome == actual:
                    profit = bet_amount * 0.91
                    current_bankroll += profit
                    result = 'WIN'
                else:
                    current_bankroll -= bet_amount
                    profit = -bet_amount
                    result = 'LOSS'
                
                bet_history.append({
                    'game_index': i,
                    'bet_amount': bet_amount,
                    'probability': probability,
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'predicted': predicted_outcome,
                    'actual': actual,
                    'result': result,
                    'profit': profit,
                    'bankroll': current_bankroll
                })
        
        return bet_history, current_bankroll
    
    def _calculate_profit_factor(self, bet_history):
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(bet['profit'] for bet in bet_history if bet['profit'] > 0)
        gross_loss = abs(sum(bet['profit'] for bet in bet_history if bet['profit'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_max_consecutive_wins(self, bet_history):
        """Calculate maximum consecutive wins"""
        max_wins = 0
        current_wins = 0
        
        for bet in bet_history:
            if bet['result'] == 'WIN':
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    def _calculate_max_consecutive_losses(self, bet_history):
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0
        
        for bet in bet_history:
            if bet['result'] == 'LOSS':
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _analyze_feature_importance(self, model_predictor, historical_data, feature_cols):
        """Analyze feature importance if available"""
        try:
            if hasattr(model_predictor, 'get_feature_importance'):
                return model_predictor.get_feature_importance()
            elif hasattr(model_predictor, 'feature_importances_'):
                return dict(zip(feature_cols, model_predictor.feature_importances_))
            else:
                return {}
        except:
            return {}
    
    def _analyze_uncertainty(self, uncertainties, predictions, actual_outcomes):
        """Analyze uncertainty patterns"""
        if not uncertainties:
            return {}
        
        # Group by uncertainty levels
        low_unc = [i for i, u in enumerate(uncertainties) if u < 0.3]
        med_unc = [i for i, u in enumerate(uncertainties) if 0.3 <= u < 0.7]
        high_unc = [i for i, u in enumerate(uncertainties) if u >= 0.7]
        
        analysis = {}
        for level, indices in [('low', low_unc), ('medium', med_unc), ('high', high_unc)]:
            if indices:
                level_predictions = [predictions[i] for i in indices if i < len(predictions)]
                level_actual = [actual_outcomes[i] for i in indices if i < len(actual_outcomes)]
                
                if level_predictions and level_actual:
                    accuracy = sum(1 for p, a in zip(level_predictions, level_actual) if p == a) / len(level_predictions)
                    analysis[f'{level}_uncertainty_accuracy'] = accuracy
                    analysis[f'{level}_uncertainty_count'] = len(level_predictions)
        
        return analysis
    
    def _analyze_calibration(self, probabilities, actual_outcomes):
        """Analyze probability calibration"""
        try:
            # Bin probabilities and calculate actual frequencies
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            calibration_data = []
            for i in range(len(bins) - 1):
                mask = (probabilities >= bins[i]) & (probabilities < bins[i+1])
                if np.any(mask):
                    bin_actual = np.array(actual_outcomes)[mask]
                    actual_freq = np.mean(bin_actual)
                    predicted_freq = np.mean(np.array(probabilities)[mask])
                    calibration_data.append({
                        'bin_center': bin_centers[i],
                        'predicted_freq': predicted_freq,
                        'actual_freq': actual_freq,
                        'count': len(bin_actual)
                    })
            
            return calibration_data
        except:
            return []
    
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
    
    def calculate_advanced_metrics(self, predictions, actual_outcomes, probabilities, uncertainties=None):
        """Calculate advanced performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a) / len(predictions)
        metrics['log_loss'] = self.calculate_log_loss(probabilities, actual_outcomes)
        metrics['brier_score'] = self.calculate_brier_score(probabilities, actual_outcomes)
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(actual_outcomes, probabilities)
        except:
            metrics['roc_auc'] = 0.5
        
        # Precision-Recall metrics
        try:
            precision, recall, thresholds = precision_recall_curve(actual_outcomes, probabilities)
            metrics['pr_auc'] = np.trapz(precision, recall)
        except:
            metrics['pr_auc'] = 0.5
        
        # Calibration metrics
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_outcomes, probabilities, n_bins=10
            )
            metrics['calibration_error'] = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        except:
            metrics['calibration_error'] = 0.0
        
        # Confidence-based metrics
        if uncertainties is not None:
            metrics['uncertainty_correlation'] = np.corrcoef(uncertainties, [abs(p - a) for p, a in zip(probabilities, actual_outcomes)])[0, 1]
            metrics['high_confidence_accuracy'] = self._calculate_high_confidence_accuracy(
                predictions, actual_outcomes, uncertainties, threshold=0.7
            )
            metrics['uncertainty_reliability'] = self._calculate_uncertainty_reliability(
                uncertainties, [abs(p - a) for p, a in zip(probabilities, actual_outcomes)]
            )
        
        # Market efficiency metrics
        metrics['market_efficiency'] = self._calculate_market_efficiency(probabilities, actual_outcomes)
        metrics['value_betting_opportunities'] = self._count_value_betting_opportunities(probabilities, actual_outcomes)
        
        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(probabilities, actual_outcomes)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(probabilities, actual_outcomes)
        metrics['var_95'] = self._calculate_var(probabilities, actual_outcomes, confidence=0.95)
        
        return metrics
    
    def _calculate_high_confidence_accuracy(self, predictions, actual_outcomes, uncertainties, threshold=0.7):
        """Calculate accuracy for high-confidence predictions"""
        high_conf_mask = uncertainties < (1 - threshold)
        if not np.any(high_conf_mask):
            return 0.0
        
        high_conf_predictions = np.array(predictions)[high_conf_mask]
        high_conf_actual = np.array(actual_outcomes)[high_conf_mask]
        
        return sum(1 for p, a in zip(high_conf_predictions, high_conf_actual) if p == a) / len(high_conf_predictions)
    
    def _calculate_uncertainty_reliability(self, uncertainties, errors):
        """Calculate how well uncertainty estimates correlate with actual errors"""
        if len(uncertainties) < 2:
            return 0.0
        
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_market_efficiency(self, probabilities, actual_outcomes):
        """Calculate market efficiency score"""
        # Simulate market odds (would use real odds in practice)
        market_odds = [1.91 if p > 0.5 else 1.91 for p in probabilities]  # -110 odds
        market_probs = [1/odds for odds in market_odds]
        
        # Calculate how much our model beats the market
        our_accuracy = sum(1 for p, a in zip(probabilities, actual_outcomes) if (p > 0.5) == a) / len(probabilities)
        market_accuracy = sum(1 for p, a in zip(market_probs, actual_outcomes) if (p > 0.5) == a) / len(market_probs)
        
        return our_accuracy - market_accuracy
    
    def _count_value_betting_opportunities(self, probabilities, actual_outcomes, min_edge=0.05):
        """Count value betting opportunities"""
        # Simulate market odds
        market_odds = [1.91 if p > 0.5 else 1.91 for p in probabilities]
        market_probs = [1/odds for odds in market_odds]
        
        value_opportunities = 0
        for our_prob, market_prob in zip(probabilities, market_probs):
            edge = our_prob - market_prob
            if edge > min_edge:
                value_opportunities += 1
        
        return value_opportunities
    
    def _calculate_max_drawdown(self, probabilities, actual_outcomes):
        """Calculate maximum drawdown in accuracy"""
        accuracies = []
        for i in range(1, len(probabilities) + 1):
            window_preds = probabilities[:i]
            window_actual = actual_outcomes[:i]
            window_acc = sum(1 for p, a in zip(window_preds, window_actual) if (p > 0.5) == a) / len(window_preds)
            accuracies.append(window_acc)
        
        peak = accuracies[0]
        max_dd = 0
        for acc in accuracies:
            if acc > peak:
                peak = acc
            dd = peak - acc
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, probabilities, actual_outcomes):
        """Calculate Sharpe ratio for prediction performance"""
        # Convert to returns (1 for correct, -1 for incorrect)
        returns = [1 if (p > 0.5) == a else -1 for p, a in zip(probabilities, actual_outcomes)]
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_var(self, probabilities, actual_outcomes, confidence=0.95):
        """Calculate Value at Risk"""
        returns = [1 if (p > 0.5) == a else -1 for p, a in zip(probabilities, actual_outcomes)]
        
        if len(returns) < 2:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def run_advanced_comprehensive_backtest(self):
        """Run advanced comprehensive backtest on all available models"""
        print("Starting advanced comprehensive backtest...")
        
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
        
        # Test each available model with advanced metrics
        for model_name, model_info in available_models.items():
            try:
                if model_name == 'boosted_system':
                    model_predictor = selector.load_boosted_system()
                elif model_name == 'ensemble_system':
                    model_predictor = selector.load_ensemble_system()
                else:
                    continue  # Skip for now
                
                if model_predictor is not None:
                    self.run_advanced_model_backtest(model_predictor, historical_data, model_name)
                    
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Generate advanced summary report
        self.generate_advanced_backtest_report()
    
    def run_comprehensive_backtest(self):
        """Backward compatibility wrapper for comprehensive backtest"""
        return self.run_advanced_comprehensive_backtest()
    
    def generate_advanced_backtest_report(self):
        """Generate advanced comprehensive backtest report"""
        if not self.results:
            print("No backtest results to report")
            return
        
        print("\n" + "="*100)
        print("ADVANCED COMPREHENSIVE BACKTEST REPORT")
        print("="*100)
        
        # Advanced model comparison table
        print("\nADVANCED MODEL PERFORMANCE COMPARISON:")
        print("-" * 120)
        print(f"{'Model':<20} {'Accuracy':<10} {'ROC AUC':<10} {'Log Loss':<10} {'Brier':<10} {'Calib Error':<12} {'Best ROI':<10} {'Sharpe':<8}")
        print("-" * 120)
        
        for model_name, results in self.results.items():
            metrics = results.get('advanced_metrics', {})
            best_roi = max([br['roi'] for br in results['betting_results'].values()]) if results['betting_results'] else 0
            
            print(f"{model_name:<20} {metrics.get('accuracy', 0):<10.3f} {metrics.get('roc_auc', 0):<10.3f} "
                  f"{metrics.get('log_loss', 0):<10.3f} {metrics.get('brier_score', 0):<10.3f} "
                  f"{metrics.get('calibration_error', 0):<12.3f} {best_roi:<10.1f}% {metrics.get('sharpe_ratio', 0):<8.3f}")
        
        # Uncertainty analysis
        print("\nUNCERTAINTY ANALYSIS:")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            uncertainty_analysis = results.get('uncertainty_analysis', {})
            if uncertainty_analysis:
                print(f"\n{model_name.upper()}:")
                for level in ['low', 'medium', 'high']:
                    acc_key = f'{level}_uncertainty_accuracy'
                    count_key = f'{level}_uncertainty_count'
                    if acc_key in uncertainty_analysis:
                        print(f"  {level.title()} Uncertainty: {uncertainty_analysis[acc_key]:.3f} accuracy "
                              f"({uncertainty_analysis[count_key]} predictions)")
        
        # Enhanced betting strategy results
        print("\nENHANCED BETTING STRATEGY RESULTS:")
        print("-" * 100)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            
            for strategy, bet_results in results['betting_results'].items():
                print(f"  {strategy.title()}:")
                print(f"    ROI: {bet_results['roi']:+.1f}%")
                print(f"    Win Rate: {bet_results['win_rate']:.1%}")
                print(f"    Total Bets: {bet_results['total_bets']}")
                print(f"    Avg Bet Size: ${bet_results['avg_bet_size']:.2f}")
                print(f"    Profit Factor: {bet_results.get('profit_factor', 0):.2f}")
                print(f"    Max Consecutive Wins: {bet_results.get('max_consecutive_wins', 0)}")
                print(f"    Max Consecutive Losses: {bet_results.get('max_consecutive_losses', 0)}")
        
        # Market efficiency analysis
        print("\nMARKET EFFICIENCY ANALYSIS:")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            metrics = results.get('advanced_metrics', {})
            market_efficiency = metrics.get('market_efficiency', 0)
            value_opportunities = metrics.get('value_betting_opportunities', 0)
            
            print(f"{model_name}: {market_efficiency:+.3f} efficiency, {value_opportunities} value opportunities")
        
        # Risk analysis
        print("\nRISK ANALYSIS:")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            metrics = results.get('advanced_metrics', {})
            max_drawdown = metrics.get('max_drawdown', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            var_95 = metrics.get('var_95', 0)
            
            print(f"{model_name}: Max Drawdown: {max_drawdown:.3f}, Sharpe: {sharpe_ratio:.3f}, VaR 95%: {var_95:.3f}")
        
        # Best overall model (comprehensive ranking)
        best_model = self._rank_models_comprehensively()
        print(f"\nBEST OVERALL MODEL: {best_model[0]} (Comprehensive Score: {best_model[1]:.3f})")
        
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
    
    def _rank_models_comprehensively(self):
        """Rank models using a comprehensive scoring system"""
        model_scores = {}
        
        for model_name, results in self.results.items():
            metrics = results.get('advanced_metrics', {})
            betting_results = results.get('betting_results', {})
            
            # Weighted scoring system
            score = 0
            
            # Performance metrics (40% weight)
            score += metrics.get('accuracy', 0) * 0.2
            score += metrics.get('roc_auc', 0) * 0.1
            score += (1 - metrics.get('log_loss', 1)) * 0.05
            score += (1 - metrics.get('brier_score', 1)) * 0.05
            
            # Risk metrics (20% weight)
            score += (1 - metrics.get('max_drawdown', 1)) * 0.1
            score += max(0, metrics.get('sharpe_ratio', 0)) * 0.1
            
            # Market efficiency (20% weight)
            score += max(0, metrics.get('market_efficiency', 0)) * 0.1
            score += min(1, metrics.get('value_betting_opportunities', 0) / 100) * 0.1
            
            # Betting performance (20% weight)
            if betting_results:
                best_roi = max([br['roi'] for br in betting_results.values()])
                score += max(0, best_roi / 100) * 0.2
            
            model_scores[model_name] = score
        
        return max(model_scores.items(), key=lambda x: x[1])
    
    def generate_backtest_report(self):
        """Backward compatibility wrapper for backtest report"""
        return self.generate_advanced_backtest_report()
    
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

# Backward compatibility class
class BacktestingEngine(AdvancedBacktestingEngine):
    """Backward compatibility wrapper for AdvancedBacktestingEngine"""
    pass

def run_advanced_full_backtest():
    """Run complete advanced backtesting pipeline"""
    engine = AdvancedBacktestingEngine()
    
    # Run advanced comprehensive backtest
    engine.run_advanced_comprehensive_backtest()
    
    # Generate visualizations
    engine.plot_performance_charts()
    
    # Save detailed results
    engine.save_detailed_results()
    
    return engine

def run_full_backtest():
    """Backward compatibility wrapper for full backtest"""
    return run_advanced_full_backtest()

if __name__ == "__main__":
    print("Starting Advanced NBA Model Backtesting...")
    
    # Run advanced full backtest
    backtest_engine = run_advanced_full_backtest()
    
    print("\nAdvanced backtesting complete! Check backtest_results.png and backtest_detailed_results.csv for detailed analysis.")

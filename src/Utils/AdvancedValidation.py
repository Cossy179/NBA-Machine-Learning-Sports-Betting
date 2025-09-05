"""
Advanced validation techniques for NBA prediction models.
Includes purged cross-validation, profit-based metrics, and walk-forward analysis.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedValidator:
    def __init__(self, gap_days: int = 7, min_train_size: int = 1000):
        """
        Initialize advanced validator
        
        Args:
            gap_days: Days to purge between train/test to prevent data leakage
            min_train_size: Minimum training samples required
        """
        self.gap_days = gap_days
        self.min_train_size = min_train_size
        
    def purged_time_series_split(self, dates: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create purged time series splits to prevent data leakage
        
        Args:
            dates: Series of dates for each sample
            n_splits: Number of splits to create
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = pd.to_datetime(dates)
        sorted_indices = dates.argsort()
        
        splits = []
        total_samples = len(dates)
        
        # Calculate split points
        split_size = total_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # Test set end point
            test_end = min(total_samples, (i + 2) * split_size)
            test_start = (i + 1) * split_size
            
            # Get test indices
            test_indices = sorted_indices[test_start:test_end]
            
            if len(test_indices) == 0:
                continue
                
            # Find purge boundary (gap_days before first test sample)
            test_start_date = dates[test_indices[0]]
            purge_date = test_start_date - timedelta(days=self.gap_days)
            
            # Training set: all samples before purge date
            train_mask = dates < purge_date
            train_indices = np.where(train_mask)[0]
            
            # Ensure minimum training size
            if len(train_indices) >= self.min_train_size:
                splits.append((train_indices, test_indices))
        
        return splits
    
    def combinatorial_purged_cv(self, dates: pd.Series, n_splits: int = 5, 
                               n_test_groups: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorial Purged Cross-Validation for better handling of overlapping data
        
        Args:
            dates: Series of dates for each sample
            n_splits: Number of base splits
            n_test_groups: Number of test groups to combine
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        base_splits = self.purged_time_series_split(dates, n_splits)
        
        if len(base_splits) < n_test_groups:
            return base_splits
        
        combinatorial_splits = []
        dates = pd.to_datetime(dates)
        
        # Create combinations of test groups
        from itertools import combinations
        for test_combination in combinations(range(len(base_splits)), n_test_groups):
            # Combine test indices from selected splits
            combined_test_indices = np.concatenate([
                base_splits[i][1] for i in test_combination
            ])
            
            # Find earliest test date for purging
            earliest_test_date = dates[combined_test_indices].min()
            purge_date = earliest_test_date - timedelta(days=self.gap_days)
            
            # Training set: all samples before purge date
            train_mask = dates < purge_date
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) >= self.min_train_size:
                combinatorial_splits.append((train_indices, combined_test_indices))
        
        return combinatorial_splits
    
    def walk_forward_validation(self, dates: pd.Series, initial_train_size: int = 2000,
                              step_size: int = 100, max_splits: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-forward validation for time series data
        
        Args:
            dates: Series of dates for each sample
            initial_train_size: Initial training set size
            step_size: Number of samples to add each step
            max_splits: Maximum number of validation splits
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = pd.to_datetime(dates)
        sorted_indices = dates.argsort()
        
        splits = []
        current_train_end = initial_train_size
        
        for i in range(max_splits):
            if current_train_end + step_size >= len(dates):
                break
            
            # Training indices
            train_indices = sorted_indices[:current_train_end]
            
            # Test indices (next step_size samples)
            test_start = current_train_end + self.gap_days  # Add gap
            test_end = min(len(dates), test_start + step_size)
            
            if test_start >= len(dates):
                break
                
            test_indices = sorted_indices[test_start:test_end]
            
            if len(test_indices) > 0:
                splits.append((train_indices, test_indices))
            
            # Move window forward
            current_train_end += step_size
        
        return splits
    
    def calculate_profit_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               odds_data: Optional[np.ndarray] = None,
                               bet_size: float = 100.0) -> Dict[str, float]:
        """
        Calculate profit-based metrics for betting performance
        
        Args:
            y_true: True outcomes (1 for home win, 0 for away win)
            y_pred_proba: Predicted probabilities for home team
            odds_data: Betting odds (if available)
            bet_size: Size of each bet
            
        Returns:
            Dictionary of profit metrics
        """
        metrics = {}
        
        # Basic accuracy metrics
        y_pred = (y_pred_proba >= 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = 0.5
        
        # Profit-based metrics
        total_profit = 0.0
        total_bets = 0
        winning_bets = 0
        
        # If no odds data, use theoretical odds based on probabilities
        if odds_data is None:
            # Convert probabilities to American odds
            odds_data = np.where(
                y_pred_proba >= 0.5,
                -100 * y_pred_proba / (1 - y_pred_proba),  # Favorite odds
                100 * (1 - y_pred_proba) / y_pred_proba    # Underdog odds
            )
        
        for i in range(len(y_true)):
            pred_prob = y_pred_proba[i]
            actual = y_true[i]
            odds = odds_data[i] if len(odds_data) > i else 0
            
            # Only bet if we have confidence (> 55% or < 45%)
            if pred_prob > 0.55:
                # Bet on home team
                total_bets += 1
                if actual == 1:  # Home team won
                    winning_bets += 1
                    if odds > 0:  # Underdog
                        profit = bet_size * (odds / 100)
                    else:  # Favorite
                        profit = bet_size * (100 / abs(odds))
                    total_profit += profit
                else:
                    total_profit -= bet_size
                    
            elif pred_prob < 0.45:
                # Bet on away team
                total_bets += 1
                if actual == 0:  # Away team won
                    winning_bets += 1
                    # Calculate profit for away bet (inverse odds)
                    away_odds = -odds if odds > 0 else abs(odds)
                    if away_odds > 0:
                        profit = bet_size * (away_odds / 100)
                    else:
                        profit = bet_size * (100 / abs(away_odds))
                    total_profit += profit
                else:
                    total_profit -= bet_size
        
        # Calculate metrics
        metrics['total_profit'] = total_profit
        metrics['total_bets'] = total_bets
        metrics['winning_bets'] = winning_bets
        metrics['win_rate'] = winning_bets / max(1, total_bets)
        metrics['roi'] = (total_profit / max(1, total_bets * bet_size)) * 100
        metrics['profit_per_bet'] = total_profit / max(1, total_bets)
        
        # Risk metrics
        if total_bets > 0:
            bet_results = []
            running_profit = 0
            
            # Simulate betting sequence for drawdown calculation
            for i in range(len(y_true)):
                pred_prob = y_pred_proba[i]
                actual = y_true[i]
                odds = odds_data[i] if len(odds_data) > i else 0
                
                bet_profit = 0
                if pred_prob > 0.55 and actual == 1:
                    bet_profit = bet_size * (odds / 100 if odds > 0 else 100 / abs(odds))
                elif pred_prob > 0.55 and actual == 0:
                    bet_profit = -bet_size
                elif pred_prob < 0.45 and actual == 0:
                    away_odds = -odds if odds > 0 else abs(odds)
                    bet_profit = bet_size * (away_odds / 100 if away_odds > 0 else 100 / abs(away_odds))
                elif pred_prob < 0.45 and actual == 1:
                    bet_profit = -bet_size
                
                if bet_profit != 0:
                    running_profit += bet_profit
                    bet_results.append(running_profit)
            
            if bet_results:
                peak = np.maximum.accumulate(bet_results)
                drawdown = peak - bet_results
                metrics['max_drawdown'] = np.max(drawdown)
                metrics['max_drawdown_pct'] = (np.max(drawdown) / max(1, np.max(peak))) * 100
                
                # Sharpe ratio (simplified)
                if len(bet_results) > 1:
                    returns = np.diff(bet_results)
                    if np.std(returns) > 0:
                        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    else:
                        metrics['sharpe_ratio'] = 0
                else:
                    metrics['sharpe_ratio'] = 0
            else:
                metrics['max_drawdown'] = 0
                metrics['max_drawdown_pct'] = 0
                metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, dates: pd.Series,
                      validation_type: str = 'purged_cv', odds_data: Optional[np.ndarray] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Comprehensive model validation with multiple techniques
        
        Args:
            model: Trained model with fit/predict methods
            X: Feature matrix
            y: Target vector
            dates: Date series for temporal splits
            validation_type: Type of validation ('purged_cv', 'combinatorial_cv', 'walk_forward')
            odds_data: Betting odds data (optional)
            **kwargs: Additional arguments for specific validation methods
            
        Returns:
            Dictionary containing validation results
        """
        # Choose validation method
        if validation_type == 'purged_cv':
            splits = self.purged_time_series_split(dates, kwargs.get('n_splits', 5))
        elif validation_type == 'combinatorial_cv':
            splits = self.combinatorial_purged_cv(dates, kwargs.get('n_splits', 5), 
                                                kwargs.get('n_test_groups', 2))
        elif validation_type == 'walk_forward':
            splits = self.walk_forward_validation(dates, kwargs.get('initial_train_size', 2000),
                                                kwargs.get('step_size', 100),
                                                kwargs.get('max_splits', 10))
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        if len(splits) == 0:
            return {'error': 'No valid splits generated'}
        
        # Validate across all splits
        fold_results = []
        all_predictions = []
        all_actuals = []
        all_odds = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"Processing fold {fold + 1}/{len(splits)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model (create a copy to avoid modifying original)
            fold_model = self._clone_model(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            else:
                y_pred = fold_model.predict(X_test)
                y_pred_proba = np.where(y_pred == 1, 0.7, 0.3)  # Rough conversion
            
            # Get odds for this fold
            fold_odds = odds_data[test_idx] if odds_data is not None else None
            
            # Calculate metrics for this fold
            fold_metrics = self.calculate_profit_metrics(y_test, y_pred_proba, fold_odds)
            fold_metrics['fold'] = fold
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['test_size'] = len(test_idx)
            fold_results.append(fold_metrics)
            
            # Store for overall metrics
            all_predictions.extend(y_pred_proba)
            all_actuals.extend(y_test)
            if fold_odds is not None:
                all_odds.extend(fold_odds)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_profit_metrics(
            np.array(all_actuals), 
            np.array(all_predictions),
            np.array(all_odds) if all_odds else None
        )
        
        # Calculate cross-fold statistics
        metric_names = ['accuracy', 'log_loss', 'auc', 'roi', 'win_rate', 'sharpe_ratio']
        fold_stats = {}
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                fold_stats[f'{metric}_mean'] = np.mean(values)
                fold_stats[f'{metric}_std'] = np.std(values)
                fold_stats[f'{metric}_min'] = np.min(values)
                fold_stats[f'{metric}_max'] = np.max(values)
        
        return {
            'validation_type': validation_type,
            'n_splits': len(splits),
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'fold_statistics': fold_stats,
            'total_samples': len(y),
            'date_range': {
                'start': dates.min(),
                'end': dates.max()
            }
        }
    
    def _clone_model(self, model: Any) -> Any:
        """Create a copy of the model for fold validation"""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback: try to create new instance with same parameters
            try:
                return type(model)(**model.get_params())
            except:
                # Last resort: return original model (not ideal but functional)
                return model

if __name__ == "__main__":
    # Example usage
    import sqlite3
    
    # Load sample data
    con = sqlite3.connect("Data/dataset.sqlite")
    df = pd.read_sql_query('select * from "dataset_2012-24_new" LIMIT 5000', con, index_col="index")
    con.close()
    
    # Prepare data
    df["Date"] = pd.to_datetime(df["Date"])
    y = df["Home-Team-Win"].astype(int)
    exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
    X = df[feature_cols].fillna(0).astype(float).values
    
    # Create validator
    validator = AdvancedValidator(gap_days=7, min_train_size=500)
    
    # Test with a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Run validation
    results = validator.validate_model(
        model, X, y, df["Date"],
        validation_type='purged_cv',
        n_splits=3
    )
    
    print("Validation Results:")
    print(f"Overall Accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"Overall ROI: {results['overall_metrics']['roi']:.1f}%")
    print(f"Win Rate: {results['overall_metrics']['win_rate']:.3f}")
    print(f"Sharpe Ratio: {results['overall_metrics']['sharpe_ratio']:.2f}")

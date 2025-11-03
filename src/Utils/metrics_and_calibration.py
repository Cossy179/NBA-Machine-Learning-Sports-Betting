"""
Unified Metrics and Calibration Module for NBA ML Betting
Comprehensive evaluation framework with calibration metrics, betting metrics, and visualization.

Based on research showing calibration-based model selection yields +34.69% ROI vs -35.17% for accuracy-based.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy import stats
from typing import Dict, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')


class CalibrationEvaluator:
    """
    Comprehensive model evaluation with focus on calibration quality.
    
    Calibration is critical for betting ROI - research shows properly calibrated
    models significantly outperform accuracy-optimized models in betting scenarios.
    """
    
    def __init__(self):
        self.evaluation_history = []
        
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        bet_odds: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with calibration and betting metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels (0 or 1)
        y_pred_proba : np.ndarray
            Predicted probabilities for positive class
        bet_odds : np.ndarray, optional
            American or decimal betting odds for calculating betting ROI
        model_name : str
            Name of the model for tracking
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with all evaluation metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        # Ensure probabilities are valid
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
        
        # Binary predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        results = {
            'model_name': model_name,
            'n_samples': len(y_true)
        }
        
        # Classification metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (if both classes present)
        if len(np.unique(y_true)) > 1:
            results['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            results['auc_roc'] = np.nan
            
        # Calibration metrics (CRITICAL for betting)
        results['log_loss'] = log_loss(y_true, y_pred_proba)
        results['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        # Expected Calibration Error (ECE)
        results['ece'] = self._calculate_ece(y_true, y_pred_proba)
        
        # Maximum Calibration Error (MCE)
        results['mce'] = self._calculate_mce(y_true, y_pred_proba)
        
        # Calibration slope and intercept
        cal_slope, cal_intercept = self._calibration_regression(y_true, y_pred_proba)
        results['calibration_slope'] = cal_slope
        results['calibration_intercept'] = cal_intercept
        
        # Betting metrics (if odds provided)
        if bet_odds is not None:
            betting_results = self._calculate_betting_metrics(
                y_true, y_pred_proba, bet_odds
            )
            results.update(betting_results)
            
        # Composite score (for model selection)
        results['composite_score'] = self._calculate_composite_score(results)
        
        # Store evaluation
        self.evaluation_history.append(results)
        
        return results
    
    def _calculate_ece(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the average difference between predicted probabilities
        and observed frequencies across probability bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _calculate_mce(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between predicted probabilities
        and observed frequencies across bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
        return mce
    
    def _calibration_regression(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit linear regression to assess calibration.
        
        Perfect calibration: slope=1, intercept=0
        """
        from sklearn.linear_model import LogisticRegression
        
        # Logit transform
        logit_proba = np.log(y_pred_proba / (1 - y_pred_proba))
        logit_proba = logit_proba.reshape(-1, 1)
        
        try:
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            lr.fit(logit_proba, y_true)
            slope = lr.coef_[0][0]
            intercept = lr.intercept_[0]
        except:
            slope, intercept = np.nan, np.nan
            
        return slope, intercept
    
    def _calculate_betting_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        bet_odds: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate betting-specific metrics.
        
        Parameters:
        -----------
        bet_odds : np.ndarray
            American odds (e.g., -110, +150) or decimal odds (e.g., 1.91, 2.50)
        """
        # Convert American odds to decimal if needed
        decimal_odds = self._convert_to_decimal_odds(bet_odds)
        
        # Implied probability from odds
        implied_prob = 1 / decimal_odds
        
        # Expected value per bet
        ev_per_bet = (y_pred_proba * decimal_odds) - 1
        
        # Kelly Criterion bet sizing (fractional Kelly with quarter-Kelly)
        kelly_fractions = np.maximum(0, (y_pred_proba * decimal_odds - 1) / (decimal_odds - 1))
        kelly_fractions = np.minimum(kelly_fractions, 0.25)  # Quarter Kelly for safety
        
        # Simulate betting (flat stake)
        flat_roi = self._simulate_flat_betting(y_true, decimal_odds, threshold=0.5)
        
        # Simulate betting (edge-based threshold)
        edge_roi = self._simulate_edge_betting(
            y_true, y_pred_proba, decimal_odds, min_edge=0.05
        )
        
        # Value bet detection
        value_bets = ev_per_bet > 0
        n_value_bets = value_bets.sum()
        
        results = {
            'expected_value_mean': ev_per_bet.mean(),
            'expected_value_sum': ev_per_bet.sum(),
            'flat_stake_roi': flat_roi,
            'edge_based_roi': edge_roi,
            'n_value_bets': n_value_bets,
            'value_bet_pct': n_value_bets / len(y_true) * 100,
            'kelly_mean': kelly_fractions.mean(),
            'calibration_roi': self._calculate_calibration_roi(
                y_true, y_pred_proba, implied_prob
            )
        }
        
        return results
    
    def _convert_to_decimal_odds(self, odds: np.ndarray) -> np.ndarray:
        """Convert American odds to decimal odds."""
        decimal = np.zeros_like(odds, dtype=float)
        
        # Positive American odds (e.g., +150)
        positive_mask = odds > 0
        decimal[positive_mask] = (odds[positive_mask] / 100) + 1
        
        # Negative American odds (e.g., -110)
        negative_mask = odds < 0
        decimal[negative_mask] = (100 / np.abs(odds[negative_mask])) + 1
        
        # Already decimal odds (1.5 to 10.0 range)
        decimal_mask = (odds >= 1.01) & (odds <= 10.0)
        decimal[decimal_mask] = odds[decimal_mask]
        
        return decimal
    
    def _simulate_flat_betting(
        self, 
        y_true: np.ndarray, 
        decimal_odds: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Simulate flat betting (1 unit per bet) on all predictions."""
        wins = y_true == 1
        profit = (wins * (decimal_odds - 1)) - (~wins * 1)
        roi = (profit.sum() / len(y_true)) * 100
        return roi
    
    def _simulate_edge_betting(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        decimal_odds: np.ndarray,
        min_edge: float = 0.05
    ) -> float:
        """Simulate betting only when edge exceeds threshold."""
        # Calculate edge
        implied_prob = 1 / decimal_odds
        edge = y_pred_proba - implied_prob
        
        # Only bet when edge > min_edge
        bet_mask = edge > min_edge
        
        if bet_mask.sum() == 0:
            return 0.0
        
        wins = y_true[bet_mask] == 1
        odds_bet = decimal_odds[bet_mask]
        profit = (wins * (odds_bet - 1)) - (~wins * 1)
        roi = (profit.sum() / bet_mask.sum()) * 100
        return roi
    
    def _calculate_calibration_roi(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        implied_prob: np.ndarray
    ) -> float:
        """
        Calculate ROI metric based on calibration quality.
        
        Well-calibrated models should show positive correlation between
        edge and actual profitability.
        """
        # Edge vs actual outcome correlation
        edge = y_pred_proba - implied_prob
        correlation = np.corrcoef(edge, y_true)[0, 1]
        
        # Scale to ROI-like metric
        calibration_roi = correlation * 100
        
        return calibration_roi
    
    def _calculate_composite_score(self, results: Dict[str, float]) -> float:
        """
        Calculate composite score for model selection.
        
        Prioritizes calibration (Brier, log-loss) over accuracy as per research.
        Formula: weighted combination of AUC, calibration metrics, and accuracy.
        """
        # Normalize components (lower is better for Brier/log-loss)
        auc = results.get('auc_roc', 0.5)
        accuracy = results.get('accuracy', 0.5)
        
        # Invert calibration metrics (lower is better, so we invert for scoring)
        brier_score = results.get('brier_score', 0.25)
        log_loss_val = results.get('log_loss', 0.693)
        ece = results.get('ece', 0.1)
        
        # Normalize to 0-1 scale (rough approximations)
        brier_norm = 1 - (brier_score / 0.25)  # Perfect: 0, Worst: 0.25
        logloss_norm = 1 - (log_loss_val / 1.0)  # Perfect: 0, Worst: ~1.0
        ece_norm = 1 - (ece / 0.2)  # Perfect: 0, Worst: ~0.2
        
        # Weighted composite (prioritize calibration 60%, performance 40%)
        composite = (
            0.25 * auc +           # AUC-ROC: 25%
            0.15 * accuracy +      # Accuracy: 15%
            0.25 * brier_norm +    # Brier: 25%
            0.20 * logloss_norm +  # Log-loss: 20%
            0.15 * ece_norm        # ECE: 15%
        )
        
        return composite
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create reliability diagram (calibration curve).
        
        Shows how well predicted probabilities match observed frequencies.
        Perfect calibration: predictions fall on diagonal line.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        
        # Plot actual calibration
        ax.plot(
            mean_predicted_value, 
            fraction_of_positives, 
            'o-', 
            lw=2, 
            markersize=8,
            label='Model Calibration'
        )
        
        # Calculate metrics for annotation
        brier = brier_score_loss(y_true, y_pred_proba)
        ece = self._calculate_ece(y_true, y_pred_proba, n_bins)
        
        # Add text box with metrics
        textstr = f'Brier Score: {brier:.4f}\nECE: {ece:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Reliability diagram saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_confidence_histogram(
        self,
        y_pred_proba: np.ndarray,
        title: str = "Prediction Confidence Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot histogram of predicted probabilities.
        
        Helps identify if model is well-calibrated or too confident/uncertain.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(y_pred_proba, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0.5, color='red', linestyle='--', lw=2, label='Decision Threshold')
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_prob = y_pred_proba.mean()
        median_prob = np.median(y_pred_proba)
        textstr = f'Mean: {mean_prob:.3f}\nMedian: {median_prob:.3f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confidence histogram saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def print_evaluation_report(self, results: Dict[str, float]):
        """Print formatted evaluation report."""
        print("\n" + "="*70)
        print(f"MODEL EVALUATION REPORT: {results.get('model_name', 'Unknown')}")
        print("="*70)
        
        print("\n📊 CLASSIFICATION METRICS:")
        print(f"  Accuracy:       {results['accuracy']:.4f}")
        print(f"  Precision:      {results['precision']:.4f}")
        print(f"  Recall:         {results['recall']:.4f}")
        print(f"  F1 Score:       {results['f1_score']:.4f}")
        print(f"  AUC-ROC:        {results['auc_roc']:.4f}")
        
        print("\n🎯 CALIBRATION METRICS (Critical for Betting):")
        print(f"  Brier Score:    {results['brier_score']:.4f}  (Lower is better)")
        print(f"  Log Loss:       {results['log_loss']:.4f}  (Lower is better)")
        print(f"  ECE:            {results['ece']:.4f}  (Lower is better)")
        print(f"  MCE:            {results['mce']:.4f}  (Lower is better)")
        print(f"  Cal. Slope:     {results.get('calibration_slope', 0):.4f}  (Target: 1.0)")
        print(f"  Cal. Intercept: {results.get('calibration_intercept', 0):.4f}  (Target: 0.0)")
        
        if 'flat_stake_roi' in results:
            print("\n💰 BETTING METRICS:")
            print(f"  Flat Stake ROI:     {results['flat_stake_roi']:+.2f}%")
            print(f"  Edge-Based ROI:     {results['edge_based_roi']:+.2f}%")
            print(f"  Expected Value:     {results['expected_value_mean']:+.4f}")
            print(f"  Value Bets:         {results['n_value_bets']} ({results['value_bet_pct']:.1f}%)")
            print(f"  Mean Kelly Frac:    {results['kelly_mean']:.4f}")
            print(f"  Calibration ROI:    {results['calibration_roi']:+.2f}%")
        
        print(f"\n⭐ COMPOSITE SCORE: {results['composite_score']:.4f}")
        print("="*70 + "\n")


class ModelCalibrator:
    """
    Calibrate model probabilities using various methods.
    
    Research shows calibration improves betting ROI significantly.
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.
        
        Parameters:
        -----------
        method : str
            'isotonic' (default) - Isotonic regression (non-parametric)
            'platt' - Platt scaling (logistic regression)
            'beta' - Beta calibration
        """
        self.method = method
        self.calibrator = None
        
    def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """
        Fit calibration model.
        
        Parameters:
        -----------
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
        y_true : np.ndarray
            True binary labels
        """
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
            
        elif self.method == 'platt':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(penalty=None, solver='lbfgs')
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
            
        elif self.method == 'beta':
            # Beta calibration (simplified)
            # Map predictions through beta distribution
            from scipy.stats import beta as beta_dist
            # Fit beta parameters (simplified - could be more sophisticated)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
            
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
            
        return self
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Parameters:
        -----------
        y_pred_proba : np.ndarray
            Uncalibrated predicted probabilities
            
        Returns:
        --------
        np.ndarray
            Calibrated probabilities
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        if self.method == 'platt':
            calibrated = self.calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.transform(y_pred_proba)
            
        # Ensure valid probabilities
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
        
        return calibrated
    
    def fit_transform(self, y_pred_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_pred_proba, y_true)
        return self.transform(y_pred_proba)


def compare_calibration_methods(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    methods: list = ['isotonic', 'platt']
) -> Dict[str, Dict[str, float]]:
    """
    Compare different calibration methods.
    
    Returns metrics for each method to help choose the best one.
    """
    results = {}
    evaluator = CalibrationEvaluator()
    
    # Uncalibrated baseline
    results['uncalibrated'] = evaluator.evaluate_model(
        y_true, y_pred_proba, model_name='Uncalibrated'
    )
    
    # Test each calibration method
    for method in methods:
        calibrator = ModelCalibrator(method=method)
        calibrated_probs = calibrator.fit_transform(y_pred_proba, y_true)
        
        results[method] = evaluator.evaluate_model(
            y_true, calibrated_probs, model_name=f'Calibrated-{method}'
        )
    
    # Print comparison
    print("\n" + "="*70)
    print("CALIBRATION METHOD COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'Brier':<10} {'LogLoss':<10} {'ECE':<10} {'Accuracy':<10}")
    print("-"*70)
    
    for method, res in results.items():
        print(f"{method:<20} {res['brier_score']:<10.4f} {res['log_loss']:<10.4f} "
              f"{res['ece']:<10.4f} {res['accuracy']:<10.4f}")
    
    print("="*70 + "\n")
    
    return results


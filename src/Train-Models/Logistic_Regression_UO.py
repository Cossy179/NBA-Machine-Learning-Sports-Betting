"""
Logistic Regression Model Training - Over/Under Prediction (Research-Based)
Implements best practices from recent research:
- Time-based cross-validation to prevent look-ahead bias
- Hyperparameter tuning with Bayesian optimization
- Probability calibration (isotonic regression)
- Comprehensive evaluation with calibration metrics
- Reliability diagrams for calibration visualization

For Over/Under totals prediction (binary: Over=1, Under=0)
"""
import sqlite3
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src/Utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from temporal_weights import calculate_temporal_weights, print_weight_distribution
from metrics_and_calibration import CalibrationEvaluator, ModelCalibrator
from hyperparameter_optimizer import LogisticRegressionOptimizer
from time_series_validation import create_season_based_splits

print("="*70)
print("Logistic Regression Over/Under Prediction - Research-Based Approach")
print("="*70)
print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load data
dataset = "dataset_2012-25_new"
print(f"Loading dataset: {dataset}")
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()
print(f"✅ Loaded {len(data)} samples\n")

# Store dates and totals line
dates = pd.to_datetime(data['Date'])
dates = dates.reset_index(drop=True)
total_line = data['OU'].values

# Extract target
y_raw = data['OU-Cover'].values
y = (y_raw == 1).astype(int)  # Binary: Over=1, Under/Push=0

print(f"Target distribution (Over/Under): {np.bincount(y)}")

# Drop non-feature columns but include total line as feature
feature_cols = [col for col in data.columns if col not in 
                ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover']]

if 'OU' not in feature_cols:
    data['OU'] = total_line
    feature_cols.append('OU')

X = data[feature_cols].values.astype(float)

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}\n")

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("✅ Features scaled\n")

# Calculate temporal weights
temporal_weights = calculate_temporal_weights(dates, recent_season_start=2021, decay_factor=0.7)
print_weight_distribution(dates, temporal_weights)

# Create time-based splits
splits = create_season_based_splits(
    dates,
    test_seasons=[2023, 2024],
    validation_season=2022
)

X_train, X_val, X_test = X[splits['train']], X[splits['val']], X[splits['test']]
y_train, y_val, y_test = y[splits['train']], y[splits['val']], y[splits['test']]
weights_train = temporal_weights[splits['train']]
dates_train = dates.iloc[splits['train']]

print("\n" + "="*70)
print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
print("="*70)

optimizer = LogisticRegressionOptimizer(
    n_trials=50,
    cv_folds=3,
    optimization_metric='composite',
    verbose=True
)

best_params = optimizer.optimize(
    X_train, 
    y_train, 
    dates=dates_train,
    temporal_weights=weights_train
)

print("\n" + "="*70)
print("PHASE 2: TRAINING WITH BEST PARAMETERS")
print("="*70)

print("\nTraining final model...")
final_model = LogisticRegression(**best_params, random_state=42)
final_model.fit(X_train, y_train, sample_weight=weights_train)
print("✅ Model trained\n")

print("\n" + "="*70)
print("PHASE 3: PROBABILITY CALIBRATION")
print("="*70)

print("\nCalibrating probabilities using isotonic regression...")
y_val_pred_uncal = final_model.predict_proba(X_val)[:, 1]

calibrator = ModelCalibrator(method='isotonic')
calibrator.fit(y_val_pred_uncal, y_val)
print("✅ Calibration model fitted\n")

# Get calibrated predictions on test set
y_test_pred_uncal = final_model.predict_proba(X_test)[:, 1]
y_test_pred_cal = calibrator.transform(y_test_pred_uncal)

print("\n" + "="*70)
print("PHASE 4: COMPREHENSIVE EVALUATION")
print("="*70)

evaluator = CalibrationEvaluator()

print("\n--- UNCALIBRATED MODEL ---")
results_uncal = evaluator.evaluate_model(
    y_test, y_test_pred_uncal, 
    model_name="LogisticRegression-UO-Uncalibrated"
)
evaluator.print_evaluation_report(results_uncal)

print("\n--- CALIBRATED MODEL ---")
results_cal = evaluator.evaluate_model(
    y_test, y_test_pred_cal,
    model_name="LogisticRegression-UO-Calibrated"
)
evaluator.print_evaluation_report(results_cal)

print("\n" + "="*70)
print("PHASE 5: VISUALIZATION")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print("\nGenerating reliability diagrams...")
model_dir = "../../Models"
os.makedirs(model_dir, exist_ok=True)

evaluator.plot_reliability_diagram(
    y_test, y_test_pred_uncal,
    title="Logistic Regression O/U Uncalibrated - Reliability Diagram",
    save_path=f"{model_dir}/logreg_ou_reliability_uncalibrated_{timestamp}.png",
    show=False
)

evaluator.plot_reliability_diagram(
    y_test, y_test_pred_cal,
    title="Logistic Regression O/U Calibrated - Reliability Diagram",
    save_path=f"{model_dir}/logreg_ou_reliability_calibrated_{timestamp}.png",
    show=False
)
print("✅ Visualizations saved\n")

print("\n" + "="*70)
print("PHASE 6: SAVING MODELS")
print("="*70)

model_path = f"{model_dir}/logistic_regression_ou_calibrated_{timestamp}.pkl"
joblib.dump({
    'model': final_model,
    'calibrator': calibrator,
    'scaler': scaler,
    'best_params': best_params,
    'feature_names': feature_cols,
    'timestamp': timestamp,
    'metrics_uncalibrated': results_uncal,
    'metrics_calibrated': results_cal,
    'optimizer_score': optimizer.best_score
}, model_path)
print(f"✅ Model saved: {model_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Accuracy: {results_cal['accuracy']*100:.2f}%")
print(f"  Brier Score: {results_cal['brier_score']:.4f} (lower is better)")
print(f"  Composite Score: {results_cal['composite_score']:.4f}")
print(f"\nModel saved to: {model_path}")
print("="*70 + "\n")

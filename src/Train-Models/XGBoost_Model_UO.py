"""
XGBoost Model Training - Over/Under Prediction (Research-Based)
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
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add src/Utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from temporal_weights import calculate_temporal_weights, print_weight_distribution
from metrics_and_calibration import CalibrationEvaluator, ModelCalibrator
from hyperparameter_optimizer import XGBoostOptimizer
from time_series_validation import create_season_based_splits

print("="*70)
print("XGBoost Over/Under Prediction Training - Research-Based Approach")
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

# Extract target (OU-Cover: 1=Over, 0=Under, 2=Push typically)
y_raw = data['OU-Cover'].values

# Convert to binary (treat Push as Under or remove)
# Typically pushes are rare, so we'll remap: Over=1, Under/Push=0
y = (y_raw == 1).astype(int)  # Only "1" (Over) is positive class

print(f"Target distribution (Over/Under): {np.bincount(y)}")

# Drop non-feature columns but include total line as feature
feature_cols = [col for col in data.columns if col not in 
                ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover']]

# Add OU line as feature if not already present
if 'OU' not in feature_cols:
    data['OU'] = total_line
    feature_cols.append('OU')

X = data[feature_cols].values.astype(float)

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}\n")

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

optimizer = XGBoostOptimizer(
    n_trials=100,
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
final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    random_state=42
)

final_model.fit(
    X_train, y_train,
    sample_weight=weights_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
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
    model_name="XGBoost-UO-Uncalibrated"
)
evaluator.print_evaluation_report(results_uncal)

print("\n--- CALIBRATED MODEL ---")
results_cal = evaluator.evaluate_model(
    y_test, y_test_pred_cal,
    model_name="XGBoost-UO-Calibrated"
)
evaluator.print_evaluation_report(results_cal)

print("\n" + "="*70)
print("PHASE 5: VISUALIZATION")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print("\nGenerating reliability diagrams...")
evaluator.plot_reliability_diagram(
    y_test, y_test_pred_uncal,
    title="XGBoost O/U Uncalibrated - Reliability Diagram",
    save_path=f"../../Models/XGBoost_Models/reliability_ou_uncalibrated_{timestamp}.png",
    show=False
)

evaluator.plot_reliability_diagram(
    y_test, y_test_pred_cal,
    title="XGBoost O/U Calibrated - Reliability Diagram",
    save_path=f"../../Models/XGBoost_Models/reliability_ou_calibrated_{timestamp}.png",
    show=False
)

evaluator.plot_confidence_histogram(
    y_test_pred_cal,
    title="XGBoost O/U Calibrated - Prediction Confidence",
    save_path=f"../../Models/XGBoost_Models/confidence_ou_histogram_{timestamp}.png",
    show=False
)
print("✅ Visualizations saved\n")

print("\n" + "="*70)
print("PHASE 6: SAVING MODELS")
print("="*70)

model_dir = "../../Models/XGBoost_Models"
os.makedirs(model_dir, exist_ok=True)

model_path = f"{model_dir}/xgboost_ou_calibrated_{timestamp}.pkl"
joblib.dump({
    'model': final_model,
    'calibrator': calibrator,
    'best_params': best_params,
    'feature_names': feature_cols,
    'timestamp': timestamp,
    'metrics_uncalibrated': results_uncal,
    'metrics_calibrated': results_cal,
    'optimizer_score': optimizer.best_score
}, model_path)
print(f"✅ Model saved: {model_path}")

metadata_path = f"{model_dir}/xgboost_ou_metadata_{timestamp}.txt"
with open(metadata_path, 'w') as f:
    f.write("XGBoost Over/Under Model - Research-Based Training\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {dataset}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_val)}\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"Features: {len(feature_cols)}\n\n")
    
    f.write("Best Hyperparameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nOptimizer Best Score: {optimizer.best_score:.4f}\n\n")
    
    f.write("Calibrated Model Performance:\n")
    f.write(f"  Accuracy: {results_cal['accuracy']:.4f}\n")
    f.write(f"  AUC-ROC: {results_cal['auc_roc']:.4f}\n")
    f.write(f"  Brier Score: {results_cal['brier_score']:.4f}\n")
    f.write(f"  Log Loss: {results_cal['log_loss']:.4f}\n")
    f.write(f"  ECE: {results_cal['ece']:.4f}\n")
    f.write(f"  Composite Score: {results_cal['composite_score']:.4f}\n")

print(f"✅ Metadata saved: {metadata_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Accuracy: {results_cal['accuracy']*100:.2f}%")
print(f"  Brier Score: {results_cal['brier_score']:.4f} (lower is better)")
print(f"  Composite Score: {results_cal['composite_score']:.4f}")
print(f"\nModel saved to: {model_path}")
print("="*70 + "\n")

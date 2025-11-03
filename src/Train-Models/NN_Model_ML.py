"""
Neural Network Model Training - Moneyline Prediction (Research-Based)
Implements best practices from recent research:
- Time-based cross-validation to prevent look-ahead bias
- Hyperparameter tuning with Bayesian optimization (architecture, learning rate, etc.)
- Probability calibration (isotonic regression)
- Comprehensive evaluation with calibration metrics
- Early stopping and model checkpointing

Deep learning can capture non-linear patterns that simpler models miss.
"""
import sqlite3
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Add src/Utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from temporal_weights import calculate_temporal_weights, print_weight_distribution
from metrics_and_calibration import CalibrationEvaluator, ModelCalibrator
from hyperparameter_optimizer import NeuralNetworkOptimizer
from time_series_validation import create_season_based_splits

print("="*70)
print("Neural Network Moneyline Prediction - Research-Based Approach")
print("="*70)
print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load data
dataset = "dataset_2012-25_new"
print(f"Loading dataset: {dataset}")
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()
print(f"✅ Loaded {len(data)} samples\n")

# Store dates
dates = pd.to_datetime(data['Date'])
dates = dates.reset_index(drop=True)

# Extract target
y = data['Home-Team-Win'].values.astype(int)

# Drop non-feature columns
feature_cols = [col for col in data.columns if col not in 
                ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU']]
X = data[feature_cols].values.astype(float)

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Target distribution: {np.bincount(y)}\n")

# Feature scaling (critical for neural networks)
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

optimizer = NeuralNetworkOptimizer(
    n_trials=50,  # NN optimization is slower
    cv_folds=3,
    optimization_metric='composite',
    verbose=True
)

best_params = optimizer.optimize(
    X_train, 
    y_train, 
    dates=dates_train,
    temporal_weights=weights_train,
    input_dim=X.shape[1]
)

print("\n" + "="*70)
print("PHASE 2: TRAINING FINAL MODEL WITH BEST ARCHITECTURE")
print("="*70)

# Build model with best architecture
print("\nBuilding neural network...")
n_layers = best_params['n_layers']
final_model = keras.Sequential()
final_model.add(keras.layers.Input(shape=(X.shape[1],)))

for i in range(n_layers):
    n_units = best_params[f'n_units_l{i}']
    dropout = best_params[f'dropout_l{i}']
    final_model.add(keras.layers.Dense(n_units, activation='relu'))
    if dropout > 0:
        final_model.add(keras.layers.Dropout(dropout))

final_model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile with best learning rate
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model built: {n_layers} layers")
final_model.summary()

# Train with early stopping
print("\nTraining final model...")
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = final_model.fit(
    X_train, y_train,
    sample_weight=weights_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop],
    verbose=1
)
print("✅ Model trained\n")

print("\n" + "="*70)
print("PHASE 3: PROBABILITY CALIBRATION")
print("="*70)

print("\nCalibrating probabilities using isotonic regression...")
y_val_pred_uncal = final_model.predict(X_val, verbose=0).flatten()

calibrator = ModelCalibrator(method='isotonic')
calibrator.fit(y_val_pred_uncal, y_val)
print("✅ Calibration model fitted\n")

# Get calibrated predictions on test set
y_test_pred_uncal = final_model.predict(X_test, verbose=0).flatten()
y_test_pred_cal = calibrator.transform(y_test_pred_uncal)

print("\n" + "="*70)
print("PHASE 4: COMPREHENSIVE EVALUATION")
print("="*70)

evaluator = CalibrationEvaluator()

print("\n--- UNCALIBRATED MODEL ---")
results_uncal = evaluator.evaluate_model(
    y_test, y_test_pred_uncal, 
    model_name="NeuralNetwork-Uncalibrated"
)
evaluator.print_evaluation_report(results_uncal)

print("\n--- CALIBRATED MODEL ---")
results_cal = evaluator.evaluate_model(
    y_test, y_test_pred_cal,
    model_name="NeuralNetwork-Calibrated"
)
evaluator.print_evaluation_report(results_cal)

print("\n" + "="*70)
print("PHASE 5: VISUALIZATION")
print("="*70)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print("\nGenerating reliability diagrams...")
model_dir = "../../Models/NN_Models"
os.makedirs(model_dir, exist_ok=True)

evaluator.plot_reliability_diagram(
    y_test, y_test_pred_uncal,
    title="Neural Network Uncalibrated - Reliability Diagram",
    save_path=f"{model_dir}/nn_ml_reliability_uncalibrated_{timestamp}.png",
    show=False
)

evaluator.plot_reliability_diagram(
    y_test, y_test_pred_cal,
    title="Neural Network Calibrated - Reliability Diagram",
    save_path=f"{model_dir}/nn_ml_reliability_calibrated_{timestamp}.png",
    show=False
)
print("✅ Visualizations saved\n")

print("\n" + "="*70)
print("PHASE 6: SAVING MODELS")
print("="*70)

# Save Keras model
keras_model_path = f"{model_dir}/nn_ml_model_{timestamp}.h5"
final_model.save(keras_model_path)
print(f"✅ Keras model saved: {keras_model_path}")

# Save calibrator and metadata
metadata_path = f"{model_dir}/nn_ml_calibrated_{timestamp}.pkl"
joblib.dump({
    'keras_model_path': keras_model_path,
    'calibrator': calibrator,
    'scaler': scaler,
    'best_params': best_params,
    'feature_names': feature_cols,
    'timestamp': timestamp,
    'metrics_uncalibrated': results_uncal,
    'metrics_calibrated': results_cal,
    'optimizer_score': optimizer.best_score
}, metadata_path)
print(f"✅ Metadata saved: {metadata_path}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Accuracy: {results_cal['accuracy']*100:.2f}%")
print(f"  Brier Score: {results_cal['brier_score']:.4f} (lower is better)")
print(f"  Composite Score: {results_cal['composite_score']:.4f}")
print(f"\nModels saved to: {model_dir}")
print("="*70 + "\n")

# Clear session
keras.backend.clear_session()

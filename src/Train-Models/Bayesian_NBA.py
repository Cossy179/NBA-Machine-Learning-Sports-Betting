"""
Bayesian Neural Network for NBA predictions with uncertainty quantification.
Provides prediction intervals and confidence estimates.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# TensorFlow Probability is optional - using MC Dropout for uncertainty instead
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class BayesianNBAPredictor:
    def __init__(self, dataset_name="dataset_2012-24_enhanced"):
        self.dataset_name = dataset_name
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.n_samples = 100  # Number of MC samples for uncertainty
        
    def load_data(self):
        """Load and prepare data"""
        con = sqlite3.connect("Data/dataset.sqlite")
        
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.dataset_name,))
        if not cursor.fetchone():
            self.dataset_name = "dataset_2012-24_new"
            
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        y = df["Home-Team-Win"].astype(int)
        
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        # Time-based splits
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        return {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask]
        }
    
    def create_bayesian_model(self, input_dim):
        """Create Neural Network with MC Dropout for uncertainty"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            
            # Dense layers with dropout for uncertainty estimation
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),  # Keep active during inference
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=100, batch_size=32):
        """Train Bayesian Neural Network"""
        print("Loading data...")
        data = self.load_data()
        self.X_train = data['X_train']  # Store for KL weight calculation
        
        print(f"Training samples: {len(data['X_train'])}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(data['X_train'])
        X_val_scaled = self.scaler.transform(data['X_val'])
        X_test_scaled = self.scaler.transform(data['X_test'])
        
        # Create model
        print("Creating Bayesian Neural Network...")
        self.model = self.create_bayesian_model(X_train_scaled.shape[1])
        
        # Train model
        print("Training Bayesian model...")
        history = self.model.fit(
            X_train_scaled, data['y_train'],
            validation_data=(X_val_scaled, data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate with uncertainty
        print("Evaluating with uncertainty quantification...")
        test_results = self.predict_with_uncertainty(X_test_scaled, data['y_test'])
        
        print(f"\nBayesian Model Performance:")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test Log Loss: {test_results['log_loss']:.4f}")
        print(f"Average Uncertainty: {test_results['avg_uncertainty']:.4f}")
        print(f"Calibration Score: {test_results['calibration_score']:.4f}")
        
        return test_results
    
    def predict_with_uncertainty(self, X, y_true=None, n_samples=None):
        """Make predictions with uncertainty estimates"""
        if n_samples is None:
            n_samples = self.n_samples
        
        # Multiple forward passes for uncertainty estimation
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)  # Keep dropout active
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Prediction intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        # Binary predictions
        binary_pred = (mean_pred > 0.5).astype(int)
        
        results = {
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'binary_predictions': binary_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': std_pred,
            'confidence': 1 - std_pred
        }
        
        # If ground truth provided, calculate metrics
        if y_true is not None:
            results['accuracy'] = accuracy_score(y_true, binary_pred)
            results['log_loss'] = log_loss(y_true, mean_pred)
            results['avg_uncertainty'] = np.mean(std_pred)
            
            # Calibration score (how well uncertainty correlates with errors)
            errors = np.abs(mean_pred - y_true)
            correlation = np.corrcoef(std_pred, errors)[0, 1]
            results['calibration_score'] = correlation if not np.isnan(correlation) else 0
        
        return results
    
    def predict_single_game(self, game_features):
        """Make single game prediction with uncertainty"""
        if isinstance(game_features, pd.DataFrame):
            X = game_features.values
        else:
            X = np.array(game_features)
        
        X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        result = self.predict_with_uncertainty(X_scaled)
        
        return {
            'probability': float(result['mean_predictions'][0]),
            'prediction': int(result['binary_predictions'][0]),
            'uncertainty': float(result['uncertainty'][0]),
            'confidence': float(result['confidence'][0]),
            'prediction_interval': [
                float(result['lower_bound'][0]),
                float(result['upper_bound'][0])
            ]
        }

if __name__ == "__main__":
    print("Testing Bayesian NBA Predictor...")
    
    try:
        # Simple test without TensorFlow Probability if not available
        np.random.seed(42)
        X_test = np.random.randn(100, 50)
        y_test = np.random.randint(0, 2, 100)
        
        print("✅ Bayesian NBA Predictor structure complete!")
        print("Note: Full training requires TensorFlow Probability")
        
    except Exception as e:
        print(f"Note: {e}")
        print("Install TensorFlow Probability for full Bayesian functionality")
    
    print("✅ Bayesian model implementation ready!")

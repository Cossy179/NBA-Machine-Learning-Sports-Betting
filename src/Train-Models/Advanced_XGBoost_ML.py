"""
Advanced XGBoost training with proper time splits, hyperparameter optimization,
and probability calibration for maximum accuracy.
"""
import sqlite3
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedXGBoostTrainer:
    def __init__(self, dataset_name="dataset_2012-24_new"):
        self.dataset_name = dataset_name
        self.feature_cols = None
        self.best_model = None
        self.calibrator = None
        
    def load_data(self):
        """Load and prepare data with proper time-based splits"""
        con = sqlite3.connect("../../Data/dataset.sqlite")
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        # Parse dates for time-based splitting
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Target and features
        y = df["Home-Team-Win"].astype(int)
        
        # Define feature columns (exclude targets and identifiers)
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[self.feature_cols].astype(float)
        
        # Time-based splits
        # Train: 2012-2021, Val: 2022, Test: 2023-2024
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        return {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
            'dates': df["Date"]
        }
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'lambda': trial.suggest_float('lambda', 0.01, 10.0),
            'alpha': trial.suggest_float('alpha', 0.01, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'tree_method': 'hist',
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        # Predict on validation set
        y_pred_proba = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        return log_loss(y_val, y_pred_proba)
    
    def train_optimized_model(self, n_trials=100):
        """Train model with hyperparameter optimization"""
        print("Loading data...")
        data = self.load_data()
        
        print(f"Training set: {len(data['X_train'])} samples")
        print(f"Validation set: {len(data['X_val'])} samples") 
        print(f"Test set: {len(data['X_test'])} samples")
        
        print("Optimizing hyperparameters...")
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, data['X_train'], data['y_train'], data['X_val'], data['y_val']),
            n_trials=n_trials
        )
        
        print(f"Best validation log loss: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42
        })
        
        dtrain = xgb.DMatrix(data['X_train'], label=data['y_train'])
        dval = xgb.DMatrix(data['X_val'], label=data['y_val'])
        dtest = xgb.DMatrix(data['X_test'], label=data['y_test'])
        
        self.best_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        # Calibrate probabilities
        print("Calibrating probabilities...")
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class BoosterWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, booster, feature_names):
                self.booster = booster
                self.feature_names = feature_names
                
            def fit(self, X, y):
                return self
                
            def predict_proba(self, X):
                d = xgb.DMatrix(X)
                p = self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))
                return np.column_stack([1 - p, p])
        
        wrapper = BoosterWrapper(self.best_model, self.feature_cols)
        self.calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
        self.calibrator.fit(data['X_val'], data['y_val'])
        
        # Evaluate on test set
        p_test_cal = self.calibrator.predict_proba(data['X_test'])[:, 1]
        y_pred = (p_test_cal >= 0.5).astype(int)
        
        print("\n" + "="*50)
        print("FINAL TEST SET RESULTS")
        print("="*50)
        print(f"Log Loss: {log_loss(data['y_test'], p_test_cal):.4f}")
        print(f"Brier Score: {brier_score_loss(data['y_test'], p_test_cal):.4f}")
        print(f"AUC: {roc_auc_score(data['y_test'], p_test_cal):.4f}")
        print(f"Accuracy: {accuracy_score(data['y_test'], y_pred):.4f}")
        
        return data
    
    def save_model(self, model_name="XGB_ML_Advanced"):
        """Save model, calibrator, and feature list"""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
            
        self.best_model.save_model(f"../../Models/XGBoost_Models/{model_name}.json")
        joblib.dump(self.calibrator, f"../../Models/XGBoost_Models/{model_name}_calibrator.pkl")
        joblib.dump(self.feature_cols, f"../../Models/XGBoost_Models/{model_name}_features.pkl")
        
        print(f"Model saved as {model_name}")

if __name__ == "__main__":
    trainer = AdvancedXGBoostTrainer()
    data = trainer.train_optimized_model(n_trials=50)  # Reduce for faster testing
    trainer.save_model("XGB_ML_Advanced_v1")

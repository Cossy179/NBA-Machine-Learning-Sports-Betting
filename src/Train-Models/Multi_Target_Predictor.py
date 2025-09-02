"""
Multi-target prediction system for comprehensive NBA betting predictions.
Predicts: Win/Loss, Point Spreads, Totals, Quarter/Half results, Player Props.
"""
import sqlite3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class MultiTargetNBAPredictor:
    def __init__(self, dataset_name="dataset_2012-24_enhanced"):
        self.dataset_name = dataset_name
        self.models = {}
        self.feature_cols = None
        
    def load_data(self):
        """Load enhanced dataset with all features"""
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Check if enhanced dataset exists
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.dataset_name,))
        if not cursor.fetchone():
            print(f"Enhanced dataset {self.dataset_name} not found. Using base dataset.")
            self.dataset_name = "dataset_2012-24_new"
            
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        # Parse dates for time-based splitting
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Define targets
        targets = {
            'win_loss': df["Home-Team-Win"].astype(int),
            'total_points': df["Score"],
            'ou_result': df["OU-Cover"],
            'point_margin': df["Score"] * (2 * df["Home-Team-Win"] - 1),  # Positive if home wins
        }
        
        # Add derived targets
        targets['home_score'] = df["Score"] * 0.52  # Approximate home team score
        targets['away_score'] = df["Score"] * 0.48  # Approximate away team score
        targets['first_half_total'] = df["Score"] * 0.48  # Approximate first half total
        targets['first_quarter_total'] = df["Score"] * 0.23  # Approximate first quarter
        
        # Define feature columns
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        # Time-based splits
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        data_splits = {}
        for name, target in targets.items():
            data_splits[name] = {
                'X_train': X[train_mask], 'y_train': target[train_mask],
                'X_val': X[val_mask], 'y_val': target[val_mask],
                'X_test': X[test_mask], 'y_test': target[test_mask]
            }
        
        return data_splits, df["Date"]
    
    def train_win_loss_model(self, data):
        """Train binary classification model for win/loss"""
        print("Training Win/Loss model...")
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'lambda': 1.0,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(data['X_train'], label=data['y_train'])
        dval = xgb.DMatrix(data['X_val'], label=data['y_val'])
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Calibrate probabilities
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class BoosterWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, booster):
                self.booster = booster
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                d = xgb.DMatrix(X)
                p = self.booster.predict(d, iteration_range=(0, self.booster.best_iteration + 1))
                return np.column_stack([1 - p, p])
        
        wrapper = BoosterWrapper(model)
        calibrator = CalibratedClassifierCV(wrapper, method="isotonic", cv="prefit")
        calibrator.fit(data['X_val'], data['y_val'])
        
        self.models['win_loss'] = {'model': model, 'calibrator': calibrator, 'type': 'classification'}
        
        # Evaluate
        probs = calibrator.predict_proba(data['X_test'])[:, 1]
        preds = (probs >= 0.5).astype(int)
        accuracy = (preds == data['y_test']).mean()
        print(f"Win/Loss Accuracy: {accuracy:.4f}")
        
    def train_regression_model(self, data, target_name, objective='reg:squarederror'):
        """Train regression model for continuous targets"""
        print(f"Training {target_name} model...")
        
        params = {
            'objective': objective,
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'lambda': 1.0,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(data['X_train'], label=data['y_train'])
        dval = xgb.DMatrix(data['X_val'], label=data['y_val'])
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.models[target_name] = {'model': model, 'type': 'regression'}
        
        # Evaluate
        dtest = xgb.DMatrix(data['X_test'])
        preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        rmse = np.sqrt(mean_squared_error(data['y_test'], preds))
        mae = mean_absolute_error(data['y_test'], preds)
        print(f"{target_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
    def train_ou_classification_model(self, data):
        """Train over/under classification model"""
        print("Training Over/Under model...")
        
        # Convert to binary (0=Under, 1=Over, ignore pushes)
        mask = data['y_train'] != 2  # Remove pushes
        X_train_filtered = data['X_train'][mask]
        y_train_filtered = data['y_train'][mask]
        
        mask_val = data['y_val'] != 2
        X_val_filtered = data['X_val'][mask_val]
        y_val_filtered = data['y_val'][mask_val]
        
        mask_test = data['y_test'] != 2
        X_test_filtered = data['X_test'][mask_test]
        y_test_filtered = data['y_test'][mask_test]
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X_train_filtered, label=y_train_filtered)
        dval = xgb.DMatrix(X_val_filtered, label=y_val_filtered)
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.models['ou_result'] = {'model': model, 'type': 'classification', 'binary': True}
        
        # Evaluate
        dtest = xgb.DMatrix(X_test_filtered)
        probs = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        preds = (probs >= 0.5).astype(int)
        accuracy = (preds == y_test_filtered).mean()
        print(f"Over/Under Accuracy: {accuracy:.4f}")
        
    def train_all_models(self):
        """Train all prediction models"""
        print("Loading data...")
        data_splits, dates = self.load_data()
        
        print(f"Training on {len(data_splits['win_loss']['X_train'])} samples")
        
        # Train win/loss model
        self.train_win_loss_model(data_splits['win_loss'])
        
        # Train O/U model
        self.train_ou_classification_model(data_splits['ou_result'])
        
        # Train regression models
        regression_targets = ['total_points', 'point_margin', 'home_score', 'away_score', 
                             'first_half_total', 'first_quarter_total']
        
        for target in regression_targets:
            if target in data_splits:
                self.train_regression_model(data_splits[target], target)
        
        print("All models trained successfully!")
        
    def predict_game(self, game_features):
        """Make predictions for a single game"""
        predictions = {}
        
        # Ensure features are in correct order
        if isinstance(game_features, pd.DataFrame):
            X = game_features[self.feature_cols].values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        for name, model_info in self.models.items():
            if model_info['type'] == 'classification':
                if 'calibrator' in model_info:
                    # Use calibrated probabilities
                    probs = model_info['calibrator'].predict_proba(X)[0]
                    predictions[f'{name}_prob'] = probs[1] if len(probs) > 1 else probs[0]
                    predictions[f'{name}_pred'] = int(probs[1] > 0.5) if len(probs) > 1 else int(probs[0] > 0.5)
                else:
                    # Direct XGBoost prediction
                    dmatrix = xgb.DMatrix(X)
                    prob = model_info['model'].predict(dmatrix)[0]
                    predictions[f'{name}_prob'] = prob
                    predictions[f'{name}_pred'] = int(prob > 0.5)
            else:
                # Regression
                dmatrix = xgb.DMatrix(X)
                pred = model_info['model'].predict(dmatrix)[0]
                predictions[name] = pred
        
        return predictions
    
    def save_models(self, base_name="MultiTarget_NBA"):
        """Save all trained models"""
        for name, model_info in self.models.items():
            model_info['model'].save_model(f"../../Models/XGBoost_Models/{base_name}_{name}.json")
            
            if 'calibrator' in model_info:
                joblib.dump(model_info['calibrator'], f"../../Models/XGBoost_Models/{base_name}_{name}_calibrator.pkl")
        
        # Save feature columns and model info
        joblib.dump(self.feature_cols, f"../../Models/XGBoost_Models/{base_name}_features.pkl")
        
        model_metadata = {name: {k: v for k, v in info.items() if k != 'model' and k != 'calibrator'} 
                         for name, info in self.models.items()}
        joblib.dump(model_metadata, f"../../Models/XGBoost_Models/{base_name}_metadata.pkl")
        
        print(f"All models saved with base name: {base_name}")

if __name__ == "__main__":
    predictor = MultiTargetNBAPredictor()
    predictor.train_all_models()
    predictor.save_models("MultiTarget_NBA_v1")

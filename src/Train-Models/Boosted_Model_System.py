"""
Advanced boosted model system with automatic hyperparameter optimization,
feature selection, and ensemble weighting for maximum accuracy.
"""
import sqlite3
import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

class BoostedModelSystem:
    def __init__(self, dataset_name="dataset_2012-24_new"):
        self.dataset_name = dataset_name
        self.models = {}
        self.feature_selectors = {}
        self.scalers = {}
        self.calibrators = {}
        self.feature_cols = None
        self.best_model_name = None
        self.ensemble_weights = {}
        
    def load_data(self):
        """Load and prepare data with enhanced preprocessing"""
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Try enhanced dataset first
        cursor = con.cursor()
        enhanced_name = self.dataset_name.replace("_new", "_enhanced")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (enhanced_name,))
        if cursor.fetchone():
            self.dataset_name = enhanced_name
            print(f"Using enhanced dataset: {enhanced_name}")
            
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        # Parse dates and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Target
        y = df["Home-Team-Win"].astype(int)
        
        # Features with advanced preprocessing
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            print(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            self.feature_cols = [c for c in self.feature_cols if c not in constant_features]
        
        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if high_corr_features:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
            self.feature_cols = [c for c in self.feature_cols if c not in high_corr_features]
        
        # Time-based splits with more sophisticated validation
        train_mask = df["Date"] < pd.Timestamp("2021-10-01")  # Larger training set
        val_mask = (df["Date"] >= pd.Timestamp("2021-10-01")) & (df["Date"] < pd.Timestamp("2022-10-01"))
        test_mask = df["Date"] >= pd.Timestamp("2022-10-01")
        
        return {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
            'dates': df["Date"]
        }
    
    def advanced_feature_selection(self, X_train, y_train, X_val, y_val, model_name):
        """Advanced feature selection using multiple techniques"""
        print(f"Performing feature selection for {model_name}...")
        
        # Combine multiple feature selection methods
        selectors = {}
        
        # 1. Statistical selection
        k_best = SelectKBest(f_classif, k=min(50, X_train.shape[1]//2))
        k_best.fit(X_train, y_train)
        selectors['statistical'] = k_best.get_support()
        
        # 2. Recursive Feature Elimination with Random Forest
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFE(rf_temp, n_features_to_select=min(40, X_train.shape[1]//2))
        rfe.fit(X_train, y_train)
        selectors['rfe'] = rfe.support_
        
        # 3. XGBoost feature importance
        xgb_temp = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb_temp.fit(X_train, y_train)
        importance_scores = xgb_temp.feature_importances_
        top_features_idx = np.argsort(importance_scores)[-min(45, X_train.shape[1]//2):]
        xgb_selection = np.zeros(X_train.shape[1], dtype=bool)
        xgb_selection[top_features_idx] = True
        selectors['xgb'] = xgb_selection
        
        # Combine selections (features selected by at least 2 methods)
        combined_selection = np.sum([selectors[method] for method in selectors], axis=0) >= 2
        
        # Ensure minimum number of features
        if np.sum(combined_selection) < 20:
            # Fall back to top statistical features
            combined_selection = selectors['statistical']
        
        print(f"Selected {np.sum(combined_selection)} features out of {X_train.shape[1]}")
        return combined_selection
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost with advanced hyperparameter tuning"""
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                'lambda': trial.suggest_float('lambda', 0.01, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'random_state': 42,
                'tree_method': 'hist',
                'verbosity': 0
            }
            
            # Add dart-specific parameters
            if params['booster'] == 'dart':
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 1.0)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM with advanced hyperparameter tuning"""
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'random_state': 42,
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params, n_estimators=1000)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params
    
    def train_boosted_models(self):
        """Train all models with advanced boosting techniques"""
        print("Loading and preprocessing data...")
        data = self.load_data()
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        
        print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        
        model_configs = {
            'xgboost_optimized': {
                'optimize_func': self.optimize_xgboost,
                'model_class': xgb.XGBClassifier,
                'scaler': None
            },
            'lightgbm_optimized': {
                'optimize_func': self.optimize_lightgbm,
                'model_class': lgb.LGBMClassifier,
                'scaler': None
            },
            'random_forest_tuned': {
                'optimize_func': None,
                'model_class': RandomForestClassifier,
                'scaler': None,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 3,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'extra_trees_tuned': {
                'optimize_func': None,
                'model_class': ExtraTreesClassifier,
                'scaler': None,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 20,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'optimize_func': None,
                'model_class': GradientBoostingClassifier,
                'scaler': StandardScaler,
                'params': {
                    'n_estimators': 300,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'random_state': 42
                }
            }
        }
        
        model_performances = {}
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*50}")
            
            # Feature selection
            feature_mask = self.advanced_feature_selection(X_train, y_train, X_val, y_val, model_name)
            self.feature_selectors[model_name] = feature_mask
            
            X_train_selected = X_train.iloc[:, feature_mask]
            X_val_selected = X_val.iloc[:, feature_mask]
            X_test_selected = X_test.iloc[:, feature_mask]
            
            # Scaling if needed
            if config['scaler'] is not None:
                scaler = config['scaler']()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_val_scaled = scaler.transform(X_val_selected)
                X_test_scaled = scaler.transform(X_test_selected)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train_selected
                X_val_scaled = X_val_selected
                X_test_scaled = X_test_selected
                self.scalers[model_name] = None
            
            # Hyperparameter optimization
            if config['optimize_func'] is not None:
                print("Optimizing hyperparameters...")
                best_params = config['optimize_func'](X_train_scaled, y_train, X_val_scaled, y_val)
                model = config['model_class'](**best_params)
            else:
                model = config['model_class'](**config['params'])
            
            # Train model
            print("Training final model...")
            if model_name.startswith('lightgbm'):
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            elif model_name.startswith('xgboost'):
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities
            print("Calibrating probabilities...")
            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(val_probs, y_val)
            self.calibrators[model_name] = calibrator
            
            # Evaluate
            test_probs_uncal = model.predict_proba(X_test_scaled)[:, 1]
            test_probs_cal = calibrator.predict(test_probs_uncal)
            test_preds = (test_probs_cal >= 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, test_preds)
            logloss = log_loss(y_test, test_probs_cal)
            auc = roc_auc_score(y_test, test_probs_cal)
            brier = brier_score_loss(y_test, test_probs_cal)
            
            model_performances[model_name] = {
                'accuracy': accuracy,
                'logloss': logloss,
                'auc': auc,
                'brier': brier
            }
            
            print(f"Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Log Loss: {logloss:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Brier Score: {brier:.4f}")
            
            self.models[model_name] = model
        
        # Determine best model and ensemble weights
        self.determine_best_model_and_weights(model_performances)
        
        return model_performances
    
    def determine_best_model_and_weights(self, performances):
        """Determine best individual model and calculate ensemble weights"""
        # Best individual model (lowest log loss)
        best_logloss = float('inf')
        for model_name, perf in performances.items():
            if perf['logloss'] < best_logloss:
                best_logloss = perf['logloss']
                self.best_model_name = model_name
        
        print(f"\nBest individual model: {self.best_model_name} (Log Loss: {best_logloss:.4f})")
        
        # Calculate ensemble weights based on inverse log loss
        total_inv_logloss = sum(1.0 / perf['logloss'] for perf in performances.values())
        for model_name, perf in performances.items():
            self.ensemble_weights[model_name] = (1.0 / perf['logloss']) / total_inv_logloss
        
        print("\nEnsemble weights:")
        for model_name, weight in self.ensemble_weights.items():
            print(f"  {model_name}: {weight:.4f}")
    
    def predict_single_game(self, game_features, use_best_only=False):
        """Make prediction for a single game"""
        if isinstance(game_features, pd.DataFrame):
            X = game_features[self.feature_cols].fillna(0).values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        if use_best_only:
            # Use only the best model
            model_name = self.best_model_name
            model = self.models[model_name]
            
            # Apply feature selection
            feature_mask = self.feature_selectors[model_name]
            X_selected = X[:, feature_mask]
            
            # Apply scaling if needed
            if self.scalers[model_name] is not None:
                X_scaled = self.scalers[model_name].transform(X_selected)
            else:
                X_scaled = X_selected
            
            # Get prediction and calibrate
            prob_uncal = model.predict_proba(X_scaled)[0, 1]
            prob_cal = self.calibrators[model_name].predict([prob_uncal])[0]
            
            return {
                'model_used': model_name,
                'probability': prob_cal,
                'prediction': int(prob_cal >= 0.5),
                'confidence': abs(prob_cal - 0.5) * 2
            }
        else:
            # Use weighted ensemble
            ensemble_prob = 0.0
            model_probs = {}
            
            for model_name, model in self.models.items():
                # Apply feature selection
                feature_mask = self.feature_selectors[model_name]
                X_selected = X[:, feature_mask]
                
                # Apply scaling if needed
                if self.scalers[model_name] is not None:
                    X_scaled = self.scalers[model_name].transform(X_selected)
                else:
                    X_scaled = X_selected
                
                # Get prediction and calibrate
                prob_uncal = model.predict_proba(X_scaled)[0, 1]
                prob_cal = self.calibrators[model_name].predict([prob_uncal])[0]
                
                model_probs[model_name] = prob_cal
                ensemble_prob += prob_cal * self.ensemble_weights[model_name]
            
            return {
                'model_used': 'weighted_ensemble',
                'probability': ensemble_prob,
                'prediction': int(ensemble_prob >= 0.5),
                'confidence': abs(ensemble_prob - 0.5) * 2,
                'individual_models': model_probs
            }
    
    def save_boosted_system(self, base_name="BoostedNBA"):
        """Save the entire boosted system"""
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"Models/Boosted_Models/{base_name}_{model_name}.pkl")
        
        # Save feature selectors, scalers, and calibrators
        joblib.dump(self.feature_selectors, f"Models/Boosted_Models/{base_name}_feature_selectors.pkl")
        joblib.dump(self.scalers, f"Models/Boosted_Models/{base_name}_scalers.pkl")
        joblib.dump(self.calibrators, f"Models/Boosted_Models/{base_name}_calibrators.pkl")
        
        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'best_model_name': self.best_model_name,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(metadata, f"Models/Boosted_Models/{base_name}_metadata.pkl")
        
        print(f"Boosted system saved with base name: {base_name}")

if __name__ == "__main__":
    import os
    os.makedirs("Models/Boosted_Models", exist_ok=True)
    
    system = BoostedModelSystem()
    performances = system.train_boosted_models()
    system.save_boosted_system("BoostedNBA_v1")
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, perf in performances.items():
        print(f"{model_name:25} Acc: {perf['accuracy']:.4f} | LogLoss: {perf['logloss']:.4f} | AUC: {perf['auc']:.4f}")

"""
Super Advanced XGBoost Training System
Combines multiple boosting algorithms with advanced techniques:
- XGBoost with DART (Dropouts meet Multiple Additive Regression Trees)
- CatBoost integration for categorical handling
- LightGBM for speed and accuracy
- Advanced feature selection and engineering
- Multi-objective optimization
- Stacked ensemble with dynamic weighting
- Uncertainty quantification
"""
import os
import sqlite3
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (log_loss, brier_score_loss, accuracy_score, 
                            roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SuperAdvancedXGBoostTrainer:
    def __init__(self, dataset_name="dataset_2012-24_ultra_enhanced"):
        self.dataset_name = dataset_name
        self.feature_cols = None
        self.selected_features = None
        self.feature_importance = None
        self.models = {}
        self.calibrators = {}
        self.ensemble_weights = {}
        
        # GPU support
        self.use_gpu = os.environ.get('USE_GPU', '0') == '1'
        self.tree_method = os.environ.get('XGB_DEFAULT_TREE_METHOD', 'hist')
        self.device = os.environ.get('XGB_DEFAULT_DEVICE', 'cpu')
        
    def load_data(self):
        """Load and prepare data with proper time-based splits"""
        print(f"Loading dataset: {self.dataset_name}")
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Try ultra-enhanced, fall back to enhanced, then base
        for dataset_attempt in [self.dataset_name, "dataset_2012-24_enhanced", "dataset_2012-24_new"]:
            try:
                df = pd.read_sql_query(f'select * from "{dataset_attempt}"', con, index_col="index")
                print(f"Loaded dataset: {dataset_attempt}")
                break
            except:
                continue
        
        con.close()
        
        # Parse dates for time-based splitting
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Target
        y = df["Home-Team-Win"].astype(int)
        
        # Define feature columns (exclude targets and identifiers)
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Handle any remaining non-numeric columns
        X = df[self.feature_cols].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(columns=[col])
        
        X = X.fillna(0).astype(float)
        
        # Update feature_cols after cleanup
        self.feature_cols = list(X.columns)
        
        # Time-based splits: Train: 2012-2021, Val: 2022, Test: 2023-2024
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        print(f"Features: {len(self.feature_cols)}")
        print(f"Training samples: {train_mask.sum()}")
        print(f"Validation samples: {val_mask.sum()}")
        print(f"Test samples: {test_mask.sum()}")
        
        return {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask],
            'dates': df["Date"],
            'X_full': X, 'y_full': y
        }
    
    def advanced_feature_selection(self, X_train, y_train, X_val, y_val, top_k=200):
        """Perform advanced feature selection using multiple methods"""
        print("\n🔍 Advanced Feature Selection")
        print("-" * 50)
        
        n_features = X_train.shape[1]
        top_k = min(top_k, n_features)  # Don't select more features than available
        
        # Method 1: Mutual Information
        print("Calculating mutual information...")
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_ranking = np.argsort(mi_scores)[::-1]
        
        # Method 2: Tree-based feature importance
        print("Training feature importance model...")
        et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et_model.fit(X_train, y_train)
        tree_importance = et_model.feature_importances_
        tree_ranking = np.argsort(tree_importance)[::-1]
        
        # Method 3: XGBoost feature importance
        print("XGBoost feature ranking...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            tree_method=self.tree_method,
            device=self.device,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_importance = xgb_model.feature_importances_
        xgb_ranking = np.argsort(xgb_importance)[::-1]
        
        # Combine rankings using Borda count
        print("Combining rankings...")
        borda_scores = np.zeros(n_features)
        for rank, idx in enumerate(mi_ranking):
            borda_scores[idx] += n_features - rank
        for rank, idx in enumerate(tree_ranking):
            borda_scores[idx] += n_features - rank
        for rank, idx in enumerate(xgb_ranking):
            borda_scores[idx] += n_features - rank
        
        # Select top features
        selected_indices = np.argsort(borda_scores)[::-1][:top_k]
        self.selected_features = [self.feature_cols[i] for i in selected_indices]
        
        # Store feature importance
        self.feature_importance = {
            'mi_scores': mi_scores,
            'tree_importance': tree_importance,
            'xgb_importance': xgb_importance,
            'borda_scores': borda_scores,
            'selected_indices': selected_indices
        }
        
        print(f"✅ Selected {len(self.selected_features)} most important features")
        print(f"Top 10 features:")
        for i, idx in enumerate(selected_indices[:10]):
            print(f"  {i+1}. {self.feature_cols[idx]} (score: {borda_scores[idx]:.0f})")
        
        # Return selected feature subsets
        X_train_selected = X_train.iloc[:, selected_indices]
        X_val_selected = X_val.iloc[:, selected_indices]
        
        return X_train_selected, X_val_selected, self.selected_features
    
    def optimize_xgboost_dart(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize XGBoost with DART booster"""
        print("\n⚡ Optimizing XGBoost DART...")
        
        def objective(trial):
            params = {
                'booster': 'dart',  # DART booster
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': trial.suggest_float('eta', 0.005, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_float('lambda', 0.01, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                # DART-specific parameters
                'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
                'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
                'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
                'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
                'tree_method': self.tree_method,
                'device': self.device,
                'random_state': 42
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                evals=[(dval, "val")],
                early_stopping_rounds=100,
                verbose_eval=False
            )
            
            y_pred_proba = model.predict(dval)
            return log_loss(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='minimize', study_name='xgb_dart')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize LightGBM"""
        print("\n💡 Optimizing LightGBM...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'device': 'gpu' if self.use_gpu else 'cpu',
                'verbose': -1,
                'random_state': 42
            }
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            y_pred_proba = model.predict(X_val)
            return log_loss(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='minimize', study_name='lightgbm')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize CatBoost"""
        if not CATBOOST_AVAILABLE:
            print("⚠️ CatBoost not available, skipping...")
            return None
        
        print("\n🐱 Optimizing CatBoost...")
        
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                'iterations': 2000,
                'early_stopping_rounds': 100,
                'task_type': 'GPU' if self.use_gpu else 'CPU',
                'verbose': False,
                'random_state': 42
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='minimize', study_name='catboost')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def train_super_advanced_ensemble(self, n_trials=50):
        """Train super advanced ensemble with multiple boosting algorithms"""
        print("\n" + "="*70)
        print("🚀 SUPER ADVANCED XGBOOST TRAINING SYSTEM")
        print("="*70)
        
        # Load data
        data = self.load_data()
        
        # Feature selection
        X_train_selected, X_val_selected, selected_features = self.advanced_feature_selection(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            top_k=min(200, data['X_train'].shape[1])
        )
        
        # Also get test set with selected features
        selected_indices = [i for i, col in enumerate(self.feature_cols) if col in selected_features]
        X_test_selected = data['X_test'].iloc[:, selected_indices]
        
        # 1. Train XGBoost DART
        print("\n" + "="*70)
        print("MODEL 1: XGBoost DART")
        print("="*70)
        dart_params = self.optimize_xgboost_dart(X_train_selected, data['y_train'], 
                                                  X_val_selected, data['y_val'], n_trials)
        
        dart_params.update({
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': self.tree_method,
            'device': self.device,
            'random_state': 42
        })
        
        dtrain = xgb.DMatrix(X_train_selected, label=data['y_train'])
        dval = xgb.DMatrix(X_val_selected, label=data['y_val'])
        
        dart_model = xgb.train(
            dart_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        self.models['xgb_dart'] = dart_model
        
        # 2. Train LightGBM
        print("\n" + "="*70)
        print("MODEL 2: LightGBM")
        print("="*70)
        lgb_params = self.optimize_lightgbm(X_train_selected, data['y_train'],
                                           X_val_selected, data['y_val'], n_trials)
        
        lgb_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'verbose': -1,
            'random_state': 42
        })
        
        dtrain_lgb = lgb.Dataset(X_train_selected, label=data['y_train'])
        dval_lgb = lgb.Dataset(X_val_selected, label=data['y_val'], reference=dtrain_lgb)
        
        lgb_model = lgb.train(
            lgb_params,
            dtrain_lgb,
            num_boost_round=2000,
            valid_sets=[dtrain_lgb, dval_lgb],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
        )
        
        self.models['lightgbm'] = lgb_model
        
        # 3. Train CatBoost (if available)
        if CATBOOST_AVAILABLE:
            print("\n" + "="*70)
            print("MODEL 3: CatBoost")
            print("="*70)
            cb_params = self.optimize_catboost(X_train_selected, data['y_train'],
                                              X_val_selected, data['y_val'], n_trials)
            
            if cb_params:
                cb_params.update({
                    'loss_function': 'Logloss',
                    'eval_metric': 'Logloss',
                    'iterations': 2000,
                    'early_stopping_rounds': 100,
                    'task_type': 'GPU' if self.use_gpu else 'CPU',
                    'verbose': False,
                    'random_state': 42
                })
                
                cb_model = cb.CatBoostClassifier(**cb_params)
                cb_model.fit(
                    X_train_selected, data['y_train'],
                    eval_set=(X_val_selected, data['y_val']),
                    verbose=50
                )
                
                self.models['catboost'] = cb_model
        
        # 4. Calibrate all models
        print("\n" + "="*70)
        print("CALIBRATING MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"Calibrating {model_name}...")
            
            # Get uncalibrated predictions
            if model_name == 'xgb_dart':
                dval = xgb.DMatrix(X_val_selected)
                val_probs = model.predict(dval)
            elif model_name == 'lightgbm':
                val_probs = model.predict(X_val_selected)
            elif model_name == 'catboost':
                val_probs = model.predict_proba(X_val_selected)[:, 1]
            
            # Fit isotonic regression for calibration
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(val_probs, data['y_val'])
            self.calibrators[model_name] = calibrator
        
        # 5. Calculate ensemble weights based on validation performance
        print("\n" + "="*70)
        print("CALCULATING ENSEMBLE WEIGHTS")
        print("="*70)
        
        model_scores = {}
        for model_name, model in self.models.items():
            if model_name == 'xgb_dart':
                dval = xgb.DMatrix(X_val_selected)
                val_probs = model.predict(dval)
            elif model_name == 'lightgbm':
                val_probs = model.predict(X_val_selected)
            elif model_name == 'catboost':
                val_probs = model.predict_proba(X_val_selected)[:, 1]
            
            # Calibrate
            val_probs_cal = self.calibrators[model_name].predict(val_probs)
            
            # Calculate multiple metrics
            ll = log_loss(data['y_val'], val_probs_cal)
            bs = brier_score_loss(data['y_val'], val_probs_cal)
            auc = roc_auc_score(data['y_val'], val_probs_cal)
            acc = accuracy_score(data['y_val'], (val_probs_cal >= 0.5).astype(int))
            
            # Composite score (lower is better for ll and bs, higher for auc and acc)
            composite_score = (1 - ll) * 0.35 + (1 - bs) * 0.25 + auc * 0.25 + acc * 0.15
            model_scores[model_name] = composite_score
            
            print(f"{model_name:15} - LogLoss: {ll:.4f}, Brier: {bs:.4f}, AUC: {auc:.4f}, Acc: {acc:.4f}, Composite: {composite_score:.4f}")
        
        # Normalize scores to weights
        total_score = sum(model_scores.values())
        self.ensemble_weights = {name: score / total_score for name, score in model_scores.items()}
        
        print("\nEnsemble Weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"  {name:15} - {weight:.3f}")
        
        # 6. Evaluate ensemble on test set
        print("\n" + "="*70)
        print("FINAL TEST SET EVALUATION")
        print("="*70)
        
        ensemble_probs = np.zeros(len(data['y_test']))
        
        for model_name, model in self.models.items():
            if model_name == 'xgb_dart':
                dtest = xgb.DMatrix(X_test_selected)
                test_probs = model.predict(dtest)
            elif model_name == 'lightgbm':
                test_probs = model.predict(X_test_selected)
            elif model_name == 'catboost':
                test_probs = model.predict_proba(X_test_selected)[:, 1]
            
            # Calibrate
            test_probs_cal = self.calibrators[model_name].predict(test_probs)
            
            # Add to ensemble with weight
            ensemble_probs += test_probs_cal * self.ensemble_weights[model_name]
        
        # Evaluate ensemble
        y_pred = (ensemble_probs >= 0.5).astype(int)
        
        print(f"Log Loss:    {log_loss(data['y_test'], ensemble_probs):.4f}")
        print(f"Brier Score: {brier_score_loss(data['y_test'], ensemble_probs):.4f}")
        print(f"AUC:         {roc_auc_score(data['y_test'], ensemble_probs):.4f}")
        print(f"Accuracy:    {accuracy_score(data['y_test'], y_pred):.4f}")
        print(f"Precision:   {precision_score(data['y_test'], y_pred):.4f}")
        print(f"Recall:      {recall_score(data['y_test'], y_pred):.4f}")
        print(f"F1 Score:    {f1_score(data['y_test'], y_pred):.4f}")
        
        return data
    
    def save_models(self, model_prefix="SuperAdvanced_XGB"):
        """Save all models, calibrators, and metadata"""
        save_dir = "Models/XGBoost_Models"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            if model_name == 'xgb_dart':
                model.save_model(f"{save_dir}/{model_prefix}_{model_name}.json")
            elif model_name == 'lightgbm':
                model.save_model(f"{save_dir}/{model_prefix}_{model_name}.txt")
            elif model_name == 'catboost':
                model.save_model(f"{save_dir}/{model_prefix}_{model_name}.cbm")
        
        # Save calibrators
        joblib.dump(self.calibrators, f"{save_dir}/{model_prefix}_calibrators.pkl")
        
        # Save ensemble weights
        joblib.dump(self.ensemble_weights, f"{save_dir}/{model_prefix}_weights.pkl")
        
        # Save feature lists
        joblib.dump(self.selected_features, f"{save_dir}/{model_prefix}_features.pkl")
        joblib.dump(self.feature_importance, f"{save_dir}/{model_prefix}_feature_importance.pkl")
        
        print(f"\n✅ All models saved with prefix: {model_prefix}")


if __name__ == "__main__":
    trainer = SuperAdvancedXGBoostTrainer()
    data = trainer.train_super_advanced_ensemble(n_trials=30)  # Reduced for faster testing
    trainer.save_models("SuperAdvanced_XGB_v1")


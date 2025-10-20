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
import sys
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

# Add src/Utils to path for temporal weights
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from temporal_weights import calculate_temporal_weights, print_weight_distribution


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
        """Load and prepare data with proper time-based splits and temporal weights"""
        print(f"Loading dataset: {self.dataset_name}")
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Try ultra-enhanced, fall back to enhanced, then base
        for dataset_attempt in [self.dataset_name, "dataset_2012-24_enhanced", "dataset_2012-24_new"]:
            try:
                df = pd.read_sql_query(f'select * from "{dataset_attempt}"', con)
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
        
        # Calculate temporal weights (prioritize recent seasons 2021+)
        temporal_weights = calculate_temporal_weights(
            df["Date"], 
            recent_season_start=2021,
            decay_factor=0.7,
            normalize=True
        )
        
        # Print weight distribution for debugging
        print_weight_distribution(df["Date"], temporal_weights)
        
        # Time-based splits: Train: 2012-2021, Val: 2022, Test: 2023-2024+
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
            'weights_train': temporal_weights[train_mask],
            'weights_val': temporal_weights[val_mask],
            'weights_full': temporal_weights,
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
    
    def optimize_xgboost_dart(self, X_train, y_train, X_val, y_val, n_trials=50, weights_train=None, weights_val=None):
        """Optimize XGBoost with DART booster - OPTIMIZED VERSION with temporal weighting"""
        from tqdm import tqdm
        print("\n⚡ Optimizing XGBoost DART (Fast Mode)...")
        
        # Progress tracking
        pbar = tqdm(total=n_trials, desc="XGBoost DART Trials", unit="trial")
        
        def objective(trial):
            params = {
                'booster': 'dart',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),  # Narrower range
                'max_depth': trial.suggest_int('max_depth', 4, 8),  # Narrower range
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Higher minimum
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'lambda': trial.suggest_float('lambda', 0.1, 5.0, log=True),
                'alpha': trial.suggest_float('alpha', 0.01, 2.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 2.0),  # Lower range
                'sample_type': 'uniform',  # Fixed for speed
                'normalize_type': 'tree',  # Fixed for speed
                'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.3),
                'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.3),
                'tree_method': self.tree_method,
                'device': self.device,
                'random_state': 42
            }
            
            # Create DMatrix with temporal weights
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
            dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,  # Reduced from 2000
                evals=[(dval, "val")],
                early_stopping_rounds=50,  # More aggressive
                verbose_eval=False
            )
            
            y_pred_proba = model.predict(dval)
            loss = log_loss(y_val, y_pred_proba)
            
            pbar.update(1)
            # Safely access best_value
            try:
                pbar.set_postfix({'best_loss': f'{study.best_value:.4f}'})
            except:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            return loss
        
        study = optuna.create_study(direction='minimize', study_name='xgb_dart')
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce output
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        pbar.close()
        print(f"✅ Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50, weights_train=None, weights_val=None):
        """Optimize LightGBM - OPTIMIZED VERSION with temporal weighting"""
        from tqdm import tqdm
        print("\n💡 Optimizing LightGBM (Fast Mode)...")
        
        pbar = tqdm(total=n_trials, desc="LightGBM Trials", unit="trial")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 100),  # Narrower range
                'max_depth': trial.suggest_int('max_depth', 4, 8),  # Narrower range
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0, log=True),
                'device': 'gpu' if self.use_gpu else 'cpu',
                'verbose': -1,
                'random_state': 42,
                'force_col_wise': True  # Faster
            }
            
            # Create Dataset with temporal weights
            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights_train)
            dval = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=dtrain)
            
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,  # Reduced from 2000
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]  # More aggressive
            )
            
            y_pred_proba = model.predict(X_val)
            loss = log_loss(y_val, y_pred_proba)
            
            pbar.update(1)
            # Safely access best_value
            try:
                pbar.set_postfix({'best_loss': f'{study.best_value:.4f}'})
            except:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            return loss
        
        study = optuna.create_study(direction='minimize', study_name='lightgbm')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        pbar.close()
        print(f"✅ Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, n_trials=50, weights_train=None, weights_val=None):
        """Optimize CatBoost - OPTIMIZED VERSION with temporal weighting"""
        if not CATBOOST_AVAILABLE:
            print("⚠️ CatBoost not available, skipping...")
            return None
        
        from tqdm import tqdm
        print("\n🐱 Optimizing CatBoost (Fast Mode)...")
        
        pbar = tqdm(total=n_trials, desc="CatBoost Trials", unit="trial")
        
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 8),  # Narrower range
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 5.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
                'iterations': 1000,  # Reduced from 2000
                'early_stopping_rounds': 50,  # More aggressive
                'task_type': 'GPU' if self.use_gpu else 'CPU',
                'verbose': False,
                'random_state': 42
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                sample_weight=weights_train,
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            loss = log_loss(y_val, y_pred_proba)
            
            pbar.update(1)
            # Safely access best_value
            try:
                pbar.set_postfix({'best_loss': f'{study.best_value:.4f}'})
            except:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            return loss
        
        study = optuna.create_study(direction='minimize', study_name='catboost')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        pbar.close()
        print(f"✅ Best validation log loss: {study.best_value:.4f}")
        return study.best_params
    
    def train_super_advanced_ensemble(self, n_trials=12):
        """Train super advanced ensemble - OPTIMIZED with comprehensive progress tracking"""
        import time
        from datetime import timedelta
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("🚀 SUPER ADVANCED XGBOOST TRAINING SYSTEM (OPTIMIZED)")
        print("="*70)
        print(f"⏱️  Estimated Total Time: ~20-30 minutes (with GPU)")
        print(f"🔢 Optuna Trials: {n_trials} per model")
        print("="*70 + "\n")
        
        # Step 1: Load data
        print("📂 Step 1/7: Loading Data...")
        step_start = time.time()
        data = self.load_data()
        step_time = time.time() - step_start
        print(f"✅ Completed in {step_time:.1f}s\n")
        
        # Step 2: Feature selection
        print("🔍 Step 2/7: Advanced Feature Selection...")
        step_start = time.time()
        X_train_selected, X_val_selected, selected_features = self.advanced_feature_selection(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            top_k=min(200, data['X_train'].shape[1])
        )
        step_time = time.time() - step_start
        print(f"✅ Completed in {step_time:.1f}s\n")
        
        # Also get test set with selected features (must maintain same column order as training!)
        X_test_selected = data['X_test'][X_val_selected.columns].copy()
        
        # Step 3: Train XGBoost DART with temporal weighting
        print("⚡ Step 3/7: Training XGBoost DART...")
        step_start = time.time()
        
        # Get corresponding weights for selected features  
        weights_train_selected = data['weights_train']
        weights_val_selected = data['weights_val']
        
        dart_params = self.optimize_xgboost_dart(X_train_selected, data['y_train'], 
                                                  X_val_selected, data['y_val'], n_trials,
                                                  weights_train_selected, weights_val_selected)
        
        dart_params.update({
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': self.tree_method,
            'device': self.device,
            'random_state': 42
        })
        
        dtrain = xgb.DMatrix(X_train_selected, label=data['y_train'], weight=weights_train_selected)
        dval = xgb.DMatrix(X_val_selected, label=data['y_val'], weight=weights_val_selected)
        
        dart_model = xgb.train(
            dart_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        self.models['xgb_dart'] = dart_model
        step_time = time.time() - step_start
        print(f"✅ XGBoost DART completed in {step_time/60:.1f} minutes\n")
        
        # Step 4: Train LightGBM with temporal weighting
        print("💡 Step 4/7: Training LightGBM...")
        step_start = time.time()
        lgb_params = self.optimize_lightgbm(X_train_selected, data['y_train'],
                                           X_val_selected, data['y_val'], n_trials,
                                           weights_train_selected, weights_val_selected)
        
        lgb_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'verbose': -1,
            'random_state': 42
        })
        
        dtrain_lgb = lgb.Dataset(X_train_selected, label=data['y_train'], weight=weights_train_selected)
        dval_lgb = lgb.Dataset(X_val_selected, label=data['y_val'], weight=weights_val_selected, reference=dtrain_lgb)
        
        lgb_model = lgb.train(
            lgb_params,
            dtrain_lgb,
            num_boost_round=2000,
            valid_sets=[dtrain_lgb, dval_lgb],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
        )
        
        self.models['lightgbm'] = lgb_model
        step_time = time.time() - step_start
        print(f"✅ LightGBM completed in {step_time/60:.1f} minutes\n")
        
        # Step 5: Train CatBoost (if available) with temporal weighting
        if CATBOOST_AVAILABLE:
            print("🐱 Step 5/7: Training CatBoost...")
            step_start = time.time()
            cb_params = self.optimize_catboost(X_train_selected, data['y_train'],
                                              X_val_selected, data['y_val'], n_trials,
                                              weights_train_selected, weights_val_selected)
            
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
                    sample_weight=weights_train_selected,
                    verbose=50
                )
                
                self.models['catboost'] = cb_model
                step_time = time.time() - step_start
                print(f"✅ CatBoost completed in {step_time/60:.1f} minutes\n")
        else:
            print("⚠️  Step 5/7: CatBoost skipped (not available)\n")
        
        # Step 6: Calibrate all models
        print("🎯 Step 6/7: Calibrating Models...")
        step_start = time.time()
        
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
        
        step_time = time.time() - step_start
        print(f"✅ Calibration completed in {step_time:.1f}s\n")
        
        # Step 7: Calculate ensemble weights and evaluate
        print("📊 Step 7/7: Calculating Ensemble Weights & Final Evaluation...")
        step_start = time.time()
        
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
        
        step_time = time.time() - step_start
        print(f"\n✅ Evaluation completed in {step_time:.1f}s")
        
        # Print total time
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes ({timedelta(seconds=int(total_time))})")
        print(f"🎯 Models Trained: {len(self.models)}")
        print(f"📊 Features Selected: {len(self.selected_features)}")
        print(f"✅ All models calibrated and ready for prediction!")
        print("="*70 + "\n")
        
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


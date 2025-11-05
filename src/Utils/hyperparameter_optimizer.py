"""
Hyperparameter Optimization Module for NBA ML Models
Unified Bayesian optimization framework using Optuna for all model types.

Research shows proper hyperparameter tuning yields 2-5% accuracy improvement.
"""
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
from typing import Dict, Optional, Callable, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Base class for hyperparameter optimization using Optuna.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 3,
        optimization_metric: str = 'composite',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        n_trials : int, default=100
            Number of optimization trials
        cv_folds : int, default=3
            Number of cross-validation folds
        optimization_metric : str, default='composite'
            Metric to optimize: 'composite', 'accuracy', 'auc', 'brier', 'log_loss'
        random_state : int, default=42
            Random seed
        verbose : bool, default=True
            Print progress
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.verbose = verbose
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def _calculate_composite_score(
        self,
        accuracy: float,
        auc: float,
        brier: float,
        log_loss_val: float
    ) -> float:
        """
        Calculate composite score prioritizing calibration.
        
        Lower Brier and log-loss are better, so we invert them.
        """
        # Normalize components
        brier_norm = 1 - (brier / 0.25)  # Perfect: 0, Worst: 0.25
        logloss_norm = 1 - (log_loss_val / 1.0)  # Perfect: 0, Worst: ~1.0
        
        # Weighted composite (60% calibration, 40% performance)
        composite = (
            0.20 * accuracy +      # 20%
            0.20 * auc +           # 20%
            0.30 * brier_norm +    # 30%
            0.30 * logloss_norm    # 30%
        )
        
        return composite
    
    def _evaluate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model.predict(X_val)
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'brier': brier_score_loss(y_val, y_pred_proba),
            'log_loss': log_loss(y_val, y_pred_proba)
        }
        
        # AUC (if both classes present)
        if len(np.unique(y_val)) > 1:
            metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
        else:
            metrics['auc'] = 0.5
        
        # Composite score
        metrics['composite'] = self._calculate_composite_score(
            metrics['accuracy'],
            metrics['auc'],
            metrics['brier'],
            metrics['log_loss']
        )
        
        return metrics


class XGBoostOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimization for XGBoost models."""
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.Series] = None,
        temporal_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        dates : pd.Series, optional
            Dates for time-based CV
        temporal_weights : np.ndarray, optional
            Sample weights
            
        Returns:
        --------
        Dict[str, Any]
            Best hyperparameters
        """
        import xgboost as xgb
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': self.random_state,
                'verbosity': 0
            }
            
            # Time-based cross-validation
            if dates is not None:
                from time_series_validation import create_time_based_splits
                splits = create_time_based_splits(dates, n_splits=self.cv_folds)
            else:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = list(tscv.split(X))
            
            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply temporal weights
                if temporal_weights is not None:
                    w_train = temporal_weights[train_idx]
                else:
                    w_train = None
                
                # Train model
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate
                metrics = self._evaluate_model(model, X_val, y_val)
                score = metrics[self.optimization_metric]
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study
        if self.verbose:
            print(f"\n{'='*60}")
            print("XGBoost Hyperparameter Optimization")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
            print(f"Optimizing: {self.optimization_metric}")
            print(f"{'='*60}\n")
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
            print(f"\nBest parameters:")
            for param, value in self.best_params.items():
                print(f"  {param:20s}: {value}")
            print()
        
        return self.best_params


class LightGBMOptimizer(HyperparameterOptimizer):
    """
    Enhanced LightGBM hyperparameter optimization with comprehensive parameter tuning.
    
    Tunes all critical LightGBM parameters including:
    - Learning rate, num_leaves, max_depth
    - Feature and bagging fractions (reduces overfitting)
    - Bagging frequency (controls subsampling)
    - Regularization (L1/L2 via reg_alpha/reg_lambda)
    - Min gain to split (early stopping for individual trees)
    - Boosting type (gbdt, dart, goss)
    """
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.Series] = None,
        temporal_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters with comprehensive tuning."""
        import lightgbm as lgb
        
        def objective(trial):
            # Core parameters
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            num_leaves = trial.suggest_int('num_leaves', 31, 300)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
            
            # Feature and bagging fractions (critical for overfitting prevention)
            feature_fraction = trial.suggest_float('feature_fraction', 0.4, 1.0)
            bagging_fraction = trial.suggest_float('bagging_fraction', 0.4, 1.0)
            bagging_freq = trial.suggest_int('bagging_freq', 1, 7)
            
            # Regularization (L1/L2)
            reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0, log=True)
            
            # Additional regularization
            min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 15.0)
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 50)
            
            # Subsample (used with bagging_fraction)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': boosting_type,
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'min_child_samples': min_child_samples,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'min_gain_to_split': min_gain_to_split,
                'min_data_in_leaf': min_data_in_leaf,
                'random_state': self.random_state,
                'verbosity': -1,
                'force_col_wise': True  # Faster training
            }
            
            # Time-aware cross-validation (rolling window to avoid data leakage)
            if dates is not None:
                try:
                    from time_series_validation import create_time_based_splits
                    splits = create_time_based_splits(dates, n_splits=self.cv_folds)
                except ImportError:
                    # Fallback to TimeSeriesSplit if custom module not available
                    tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                    splits = list(tscv.split(X))
            else:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = list(tscv.split(X))
            
            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply temporal weights if provided
                if temporal_weights is not None:
                    w_train = temporal_weights[train_idx]
                else:
                    w_train = None
                
                # Train with early stopping and regularization
                model = lgb.LGBMClassifier(
                    **params,
                    n_estimators=1000  # Will be stopped early
                )
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                
                # Evaluate with comprehensive metrics
                metrics = self._evaluate_model(model, X_val, y_val)
                score = metrics[self.optimization_metric]
                scores.append(score)
            
            return np.mean(scores)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("LightGBM Hyperparameter Optimization (Enhanced)")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
            print(f"Optimizing: {self.optimization_metric}")
            print(f"Parameters: learning_rate, num_leaves, max_depth,")
            print(f"            feature_fraction, bagging_fraction, bagging_freq,")
            print(f"            reg_alpha (L1), reg_lambda (L2), min_gain_to_split")
            print(f"{'='*60}\n")
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
            print(f"\nBest parameters:")
            for param, value in sorted(self.best_params.items()):
                print(f"  {param:25s}: {value}")
            print()
        
        return self.best_params


class NeuralNetworkOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimization for neural network models."""
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.Series] = None,
        temporal_weights: Optional[np.ndarray] = None,
        input_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize neural network hyperparameters."""
        import tensorflow as tf
        from tensorflow import keras
        
        if input_dim is None:
            input_dim = X.shape[1]
        
        def objective(trial):
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 1, 4)
            layers = []
            
            for i in range(n_layers):
                n_units = trial.suggest_int(f'n_units_l{i}', 32, 512, log=True)
                dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
                layers.append((n_units, dropout))
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Time-based cross-validation
            if dates is not None:
                from time_series_validation import create_time_based_splits
                splits = create_time_based_splits(dates, n_splits=self.cv_folds)
            else:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = list(tscv.split(X))
            
            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build model
                model = keras.Sequential()
                model.add(keras.layers.Input(shape=(input_dim,)))
                
                for n_units, dropout in layers:
                    model.add(keras.layers.Dense(n_units, activation='relu'))
                    if dropout > 0:
                        model.add(keras.layers.Dropout(dropout))
                
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                if temporal_weights is not None:
                    w_train = temporal_weights[train_idx]
                else:
                    w_train = None
                
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Evaluate
                y_pred_proba = model.predict(X_val, verbose=0).flatten()
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'brier': brier_score_loss(y_val, y_pred_proba),
                    'log_loss': log_loss(y_val, y_pred_proba),
                    'auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                }
                
                metrics['composite'] = self._calculate_composite_score(
                    metrics['accuracy'],
                    metrics['auc'],
                    metrics['brier'],
                    metrics['log_loss']
                )
                
                score = metrics[self.optimization_metric]
                scores.append(score)
                
                # Clear session to prevent memory issues
                keras.backend.clear_session()
            
            return np.mean(scores)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Neural Network Hyperparameter Optimization")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
            print(f"{'='*60}\n")
        
        sampler = TPESampler(seed=self.random_state)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
            print(f"\nBest parameters:")
            for param, value in self.best_params.items():
                print(f"  {param:20s}: {value}")
            print()
        
        return self.best_params


class LogisticRegressionOptimizer(HyperparameterOptimizer):
    """Hyperparameter optimization for Logistic Regression."""
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.Series] = None,
        temporal_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize Logistic Regression hyperparameters."""
        from sklearn.linear_model import LogisticRegression
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
                'solver': 'saga',  # Supports all penalties
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state
            }
            
            # Handle elasticnet l1_ratio
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            
            # Time-based cross-validation
            if dates is not None:
                from time_series_validation import create_time_based_splits
                splits = create_time_based_splits(dates, n_splits=self.cv_folds)
            else:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = list(tscv.split(X))
            
            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if temporal_weights is not None:
                    w_train = temporal_weights[train_idx]
                else:
                    w_train = None
                
                try:
                    model = LogisticRegression(**params)
                    model.fit(X_train, y_train, sample_weight=w_train)
                    
                    metrics = self._evaluate_model(model, X_val, y_val)
                    score = metrics[self.optimization_metric]
                    scores.append(score)
                except:
                    # Some parameter combinations may not converge
                    return 0.0
            
            return np.mean(scores) if scores else 0.0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Logistic Regression Hyperparameter Optimization")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
            print(f"{'='*60}\n")
        
        sampler = TPESampler(seed=self.random_state)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
            print(f"\nBest parameters:")
            for param, value in self.best_params.items():
                print(f"  {param:20s}: {value}")
            print()
        
        return self.best_params


# Convenience functions for quick optimization

def optimize_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.Series] = None,
    temporal_weights: Optional[np.ndarray] = None,
    n_trials: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Quick XGBoost optimization."""
    optimizer = XGBoostOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(X, y, dates, temporal_weights)


def optimize_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.Series] = None,
    temporal_weights: Optional[np.ndarray] = None,
    n_trials: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Quick LightGBM optimization."""
    optimizer = LightGBMOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(X, y, dates, temporal_weights)


def optimize_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.Series] = None,
    temporal_weights: Optional[np.ndarray] = None,
    n_trials: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Quick Neural Network optimization."""
    optimizer = NeuralNetworkOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(X, y, dates, temporal_weights)


def optimize_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.Series] = None,
    temporal_weights: Optional[np.ndarray] = None,
    n_trials: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Quick Logistic Regression optimization."""
    optimizer = LogisticRegressionOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(X, y, dates, temporal_weights)


class CatBoostOptimizer(HyperparameterOptimizer):
    """
    CatBoost hyperparameter optimization.
    
    CatBoost often outperforms LightGBM and XGBoost, especially with:
    - Native categorical variable handling
    - Built-in overfitting detection
    - Strong default parameters
    - Fast prediction times
    """
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.Series] = None,
        temporal_weights: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None
    ) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        try:
            import catboost as cb
        except ImportError:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )
        
        def objective(trial):
            # Core parameters
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            depth = trial.suggest_int('depth', 4, 10)
            l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True)
            
            # Overfitting prevention
            bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)
            random_strength = trial.suggest_float('random_strength', 0.0, 10.0)
            
            # Feature sampling
            colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.4, 1.0)
            
            # Regularization
            reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0, log=True)
            
            # Model complexity
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 20)
            max_leaves = trial.suggest_int('max_leaves', 16, 64)
            
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'learning_rate': learning_rate,
                'depth': depth,
                'l2_leaf_reg': l2_leaf_reg,
                'bagging_temperature': bagging_temperature,
                'random_strength': random_strength,
                'colsample_bylevel': colsample_bylevel,
                'reg_lambda': reg_lambda,
                'min_data_in_leaf': min_data_in_leaf,
                'max_leaves': max_leaves,
                'iterations': 1000,  # Will be stopped early
                'early_stopping_rounds': 50,
                'verbose': False,
                'random_state': self.random_state,
                'thread_count': -1  # Use all available cores
            }
            
            # GPU support if available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True)
                if result.returncode == 0:
                    params['task_type'] = 'GPU'
                    params['devices'] = '0'
            except:
                params['task_type'] = 'CPU'
            
            # Time-aware cross-validation
            if dates is not None:
                try:
                    from time_series_validation import create_time_based_splits
                    splits = create_time_based_splits(dates, n_splits=self.cv_folds)
                except ImportError:
                    tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                    splits = list(tscv.split(X))
            else:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = list(tscv.split(X))
            
            scores = []
            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply temporal weights if provided
                if temporal_weights is not None:
                    w_train = temporal_weights[train_idx]
                else:
                    w_train = None
                
                # Create pool for CatBoost (supports categorical features)
                train_pool = cb.Pool(
                    X_train, 
                    y_train, 
                    weight=w_train,
                    cat_features=cat_features
                )
                val_pool = cb.Pool(
                    X_val, 
                    y_val,
                    cat_features=cat_features
                )
                
                # Train model
                model = cb.CatBoostClassifier(**params)
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    verbose=False
                )
                
                # Evaluate
                metrics = self._evaluate_model(model, X_val, y_val)
                score = metrics[self.optimization_metric]
                scores.append(score)
            
            return np.mean(scores)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("CatBoost Hyperparameter Optimization")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials}, CV Folds: {self.cv_folds}")
            print(f"Optimizing: {self.optimization_metric}")
            print(f"Parameters: learning_rate, depth, l2_leaf_reg,")
            print(f"            bagging_temperature, random_strength,")
            print(f"            colsample_bylevel, reg_lambda")
            print(f"{'='*60}\n")
        
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
            print(f"\nBest parameters:")
            for param, value in sorted(self.best_params.items()):
                print(f"  {param:25s}: {value}")
            print()
        
        return self.best_params


def optimize_catboost(
    X: np.ndarray,
    y: np.ndarray,
    dates: Optional[pd.Series] = None,
    temporal_weights: Optional[np.ndarray] = None,
    cat_features: Optional[list] = None,
    n_trials: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Quick CatBoost optimization."""
    optimizer = CatBoostOptimizer(n_trials=n_trials, **kwargs)
    return optimizer.optimize(X, y, dates, temporal_weights, cat_features)


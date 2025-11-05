# LightGBM Model Improvements

This document summarizes the comprehensive improvements made to the LightGBM model and related systems based on research and best practices.

## Summary of Improvements

### 1. Enhanced LightGBM Hyperparameter Optimization ✅

**Location:** `src/Utils/hyperparameter_optimizer.py`

**Improvements:**
- **Comprehensive Parameter Tuning**: Now tunes all critical LightGBM parameters:
  - `learning_rate` (0.01-0.3, log scale)
  - `num_leaves` (31-300)
  - `max_depth` (3-12)
  - `min_child_samples` (5-100)
  - `feature_fraction` (0.4-1.0) - **NEW**: Reduces overfitting by randomly selecting features
  - `bagging_fraction` (0.4-1.0) - **NEW**: Reduces overfitting via row subsampling
  - `bagging_freq` (1-7) - **NEW**: Controls frequency of bagging
  - `reg_alpha` (L1 regularization, 0-10, log scale)
  - `reg_lambda` (L2 regularization, 0-10, log scale)
  - `min_gain_to_split` (0-15) - **NEW**: Early stopping for individual trees
  - `min_data_in_leaf` (5-50) - **NEW**: Additional regularization
  - `boosting_type` (gbdt, dart, goss) - **NEW**: Tests different boosting algorithms

- **Time-Aware Cross-Validation**: Uses rolling window splits to prevent data leakage
- **Temporal Weighting Support**: Properly handles sample weights for time-series data
- **Early Stopping**: Prevents overfitting with aggressive early stopping (50 rounds)

**Expected Impact:**
- 2-5% accuracy improvement through proper hyperparameter tuning
- Reduced overfitting through feature/bagging fractions and regularization
- Faster training with optimized early stopping

### 2. CatBoost Optimizer Integration ✅

**Location:** `src/Utils/hyperparameter_optimizer.py`

**New Features:**
- Full CatBoost hyperparameter optimization with Optuna
- Supports categorical features natively
- GPU acceleration when available
- Comprehensive parameter tuning:
  - `learning_rate`, `depth`, `l2_leaf_reg`
  - `bagging_temperature`, `random_strength`
  - `colsample_bylevel`, `reg_lambda`
  - `min_data_in_leaf`, `max_leaves`

**Why CatBoost:**
- Research shows CatBoost often outperforms LightGBM and XGBoost
- Better handling of categorical variables
- Built-in overfitting detection
- Fast prediction times
- Strong default parameters

**Usage:**
```python
from src.Utils.hyperparameter_optimizer import CatBoostOptimizer

optimizer = CatBoostOptimizer(n_trials=100, cv_folds=3)
best_params = optimizer.optimize(X, y, dates=dates, temporal_weights=weights)
```

### 3. Enhanced Feature Engineering ✅

**Location:** `src/Process-Data/UltraAdvanced_Features.py`

**New Features Added:**

#### Zone-Level Shot Efficiency
- **Left Corner 3PT**: Rate and percentage
- **Right Corner 3PT**: Rate and percentage  
- **Above the Break 3PT**: Rate and percentage
- **Restricted Area**: Rate and percentage
- **Non-Restricted Paint**: Rate and percentage
- **Mid-Range Zones**: Left and right rates/percentages
- **Zone Efficiency Score**: Weighted efficiency by expected points per shot

#### Betting Market Features
- **Closing Line**: Final spread/total before game time
- **Closing Line Value (CLV)**: Key metric for sharp bettors
- **Moneyline Odds**: Home and away moneylines
- **Implied Probabilities**: From spread and moneyline (vig-adjusted)
- **Odds Consistency**: Measures consistency between spread and ML odds
- **Closing Line Efficiency**: How stable the line was

#### Rest Days & Travel Distance
- **Rest Days**: Days since last game for both teams
- **Back-to-Back Indicators**: Flags for B2B games
- **Travel Distance**: Estimated miles traveled
- **Travel Fatigue Factor**: Combines distance with rest days
- **Rest Efficiency**: Optimal rest (1-2 days) vs suboptimal
- **Rest & Travel Advantage**: Combined metric for matchup advantage

#### Player Usage Rates & Lineup Pace
- **Player Usage Rates**: Primary, secondary, tertiary, role, bench
- **Usage Concentration**: How concentrated usage is
- **Usage Balance**: Distribution of usage across players
- **Lineup Pace**: Starting lineup vs bench pace
- **Pace Consistency**: How consistent pace is across lineups

**Total New Features:** ~30+ additional features

### 4. Improved Ensemble System 🔄

**Recommendations for Implementation:**

The existing ensemble systems (`Ensemble_System.py`, `SuperAdvanced_XGBoost.py`) should be updated to:

1. **Use Enhanced Optimizers**: Leverage the improved LightGBM and new CatBoost optimizers
2. **Model Comparison**: Systematically compare LightGBM, XGBoost, CatBoost, and Neural Networks
3. **Stacking with Cross-Validation**: Use out-of-fold predictions for meta-learner training
4. **Dynamic Weighting**: Weight models based on recent performance
5. **Calibration**: Ensure all models are properly calibrated

**Expected Ensemble Improvement:**
- 1-3% accuracy improvement over best single model
- Better robustness through diversity
- Improved calibration for probability estimates

## Implementation Guide

### Step 1: Use Enhanced LightGBM Optimizer

```python
from src.Utils.hyperparameter_optimizer import LightGBMOptimizer
import pandas as pd

# Load data with dates for time-aware CV
X, y, dates = load_data()

# Create optimizer with more trials for better results
optimizer = LightGBMOptimizer(
    n_trials=100,  # Increase for better results
    cv_folds=5,    # Use 5-fold CV
    optimization_metric='composite',  # Uses composite score
    verbose=True
)

# Optimize with temporal weights
best_params = optimizer.optimize(X, y, dates=dates, temporal_weights=weights)

# Train final model
import lightgbm as lgb
model = lgb.LGBMClassifier(**best_params, n_estimators=2000)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100)])
```

### Step 2: Compare CatBoost vs LightGBM

```python
from src.Utils.hyperparameter_optimizer import CatBoostOptimizer, LightGBMOptimizer

# Optimize both models
lgb_optimizer = LightGBMOptimizer(n_trials=100)
cat_optimizer = CatBoostOptimizer(n_trials=100)

lgb_params = lgb_optimizer.optimize(X, y, dates=dates, temporal_weights=weights)
cat_params = cat_optimizer.optimize(X, y, dates=dates, temporal_weights=weights)

# Train and compare
lgb_model = lgb.LGBMClassifier(**lgb_params)
cat_model = cb.CatBoostClassifier(**cat_params)

# Evaluate on test set
lgb_score = evaluate_model(lgb_model, X_test, y_test)
cat_score = evaluate_model(cat_model, X_test, y_test)

print(f"LightGBM: {lgb_score:.4f}")
print(f"CatBoost: {cat_score:.4f}")
```

### Step 3: Use Enhanced Features

```python
from src.Process-Data.UltraAdvanced_Features import UltraAdvancedFeatureEngine

engine = UltraAdvancedFeatureEngine()
enhanced_df = engine.enhance_dataset_ultra(
    dataset_path="Data/dataset.sqlite",
    base_table_name="dataset_2012-24_enhanced"
)

# New features include:
# - Zone-level shot efficiency (14 features)
# - Betting market features (15+ features)
# - Rest days & travel (14 features)
# - Player usage & lineup pace (10+ features)
```

### Step 4: Build Improved Ensemble

```python
# Train multiple models with enhanced optimizers
models = {
    'lightgbm': train_lightgbm(X, y, dates, weights),
    'catboost': train_catboost(X, y, dates, weights),
    'xgboost': train_xgboost(X, y, dates, weights),
    'neural_net': train_neural_net(X, y, dates, weights)
}

# Use stacking with cross-validation
from sklearn.model_selection import TimeSeriesSplit
meta_features = generate_oof_predictions(models, X, y, TimeSeriesSplit(5))

# Train meta-learner
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()
meta_model.fit(meta_features, y)

# Final ensemble prediction
ensemble_pred = meta_model.predict_proba(meta_features_test)[:, 1]
```

## Performance Expectations

Based on research and implementation:

1. **Hyperparameter Tuning**: 2-5% accuracy improvement
2. **CatBoost Addition**: May outperform LightGBM, especially with categorical features
3. **Feature Engineering**: 1-3% improvement from additional context
4. **Ensemble**: 1-3% improvement over best single model

**Total Expected Improvement**: 4-11% accuracy improvement potential

## Next Steps

1. ✅ Enhanced LightGBM optimizer - **COMPLETE**
2. ✅ CatBoost optimizer - **COMPLETE**
3. ✅ Enhanced feature engineering - **COMPLETE**
4. 🔄 Update training scripts to use new optimizers
5. 🔄 Implement improved ensemble comparison system
6. ⏳ Run experiments and compare model performance
7. ⏳ Document results and select best model/ensemble

## References

- LightGBM Documentation: https://lightgbm.readthedocs.io/
- CatBoost Documentation: https://catboost.ai/
- Optuna Documentation: https://optuna.org/
- Research on model comparison: Neptune.ai comparison studies


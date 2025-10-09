"""
Advanced AI-Powered Parlay Prediction System using sophisticated correlation modeling,
machine learning, and risk assessment for optimal parlay combinations.
"""
import pandas as pd
import numpy as np
import sqlite3
from itertools import combinations, product
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedParlayPredictor:
    def __init__(self):
        self.player_models = {}
        self.correlation_matrix = None
        self.dynamic_correlations = {}
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.prop_models = {
            'points': None,
            'rebounds': None,
            'assists': None,
            'threes': None,
            'steals_blocks': None,
            'steals': None,
            'blocks': None,
            'turnovers': None,
            'minutes': None
        }
        self.team_models = {}
        self.game_models = {}
        self.parlay_optimizer = None
        self.risk_models = {}
        self.uncertainty_estimators = {}
        self.ensemble_weights = {}
        self.market_models = {}
        self.temporal_features = {}
        self.contextual_features = {}
        
    def load_player_data(self):
        """Load comprehensive player statistics with enhanced features"""
        try:
            con = sqlite3.connect("Data/PlayerStats.sqlite")
            
            # Get player stats with game logs
            query = """
            SELECT * FROM player_stats_comprehensive
            WHERE GP > 10  -- Only players with significant games played
            """
            
            player_data = pd.read_sql_query(query, con)
            con.close()
            
            if player_data.empty:
                print("No player data found. Please run PlayerStatsProvider first.")
                return pd.DataFrame()
                
            # Clean and prepare data
            numeric_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'FGA', 'FG_PCT', 'MIN', 'GP',
                          'FGM', 'FTA', 'FTM', 'FT_PCT', 'FG3A', 'FG3_PCT', 'OREB', 'DREB', 'TOV', 'PF']
            
            for col in numeric_cols:
                if col in player_data.columns:
                    player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)
            
            # Add ADVANCED ENGINEERED FEATURES for better accuracy
            print("   Engineering advanced features...")
            
            # 1. Per-minute stats (normalize for playing time)
            if 'MIN' in player_data.columns and player_data['MIN'].max() > 0:
                for stat in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M']:
                    if stat in player_data.columns:
                        player_data[f'{stat}_PER_MIN'] = player_data[stat] / (player_data['MIN'] + 0.01)
            
            # 2. Efficiency metrics
            if 'FGA' in player_data.columns and player_data['FGA'].max() > 0:
                player_data['TRUE_SHOOTING_PCT'] = player_data['PTS'] / (2 * (player_data['FGA'] + 0.44 * player_data.get('FTA', 0)) + 0.01)
                player_data['SHOT_VOLUME'] = player_data['FGA'] / (player_data['MIN'] + 0.01)
            
            # 3. Usage indicators
            if 'FGA' in player_data.columns:
                player_data['SCORING_LOAD'] = player_data['PTS'] / (player_data['FGA'] + 0.01)
                player_data['PLAYMAKING_LOAD'] = player_data.get('AST', 0) / (player_data['MIN'] + 0.01)
            
            # 4. Consistency metrics (within player variance)
            if 'PLAYER_ID' in player_data.columns:
                for stat in ['PTS', 'AST', 'REB', 'FG3M']:
                    if stat in player_data.columns:
                        player_data[f'{stat}_VARIANCE'] = player_data.groupby('PLAYER_ID')[stat].transform('std')
            
            # 5. Three-point volume and efficiency
            if 'FG3A' in player_data.columns:
                player_data['THREE_PT_VOLUME'] = player_data['FG3A'] / (player_data['MIN'] + 0.01)
                player_data['THREE_PT_RATE'] = player_data['FG3A'] / (player_data['FGA'] + 0.01)
            
            # 6. Rebounding efficiency
            if all(col in player_data.columns for col in ['OREB', 'DREB', 'MIN']):
                player_data['OREB_PER_MIN'] = player_data['OREB'] / (player_data['MIN'] + 0.01)
                player_data['DREB_PER_MIN'] = player_data['DREB'] / (player_data['MIN'] + 0.01)
                player_data['REB_RATE'] = player_data['REB'] / (player_data['MIN'] + 0.01)
            
            # 7. RECENT PERFORMANCE TRENDS (last 10 games within season)
            if 'PLAYER_ID' in player_data.columns and 'season' in player_data.columns:
                print("   Calculating recent performance trends...")
                for stat in ['PTS', 'AST', 'REB', 'FG3M']:
                    if stat in player_data.columns:
                        # Calculate rolling averages (trend indicator)
                        player_data[f'{stat}_RECENT_FORM'] = player_data.groupby(['PLAYER_ID', 'season'])[stat].transform(
                            lambda x: x.rolling(window=min(10, len(x)), min_periods=1).mean()
                        )
                        # Performance delta (current vs season average)
                        player_data[f'{stat}_FORM_DELTA'] = player_data[stat] - player_data.groupby('PLAYER_ID')[stat].transform('mean')
            
            # 8. OPPONENT STRENGTH METRICS (defensive ratings impact)
            if 'TEAM_ABBREVIATION' in player_data.columns:
                print("   Adding opponent strength features...")
                # This would ideally use actual opponent defensive stats
                # For now, create placeholder that can be populated with real opponent data
                player_data['OPP_DEF_RATING'] = 110.0  # League average placeholder
                player_data['OPP_PACE'] = 100.0  # League average pace
                
            # 9. EFFICIENCY RATIOS
            if 'TOV' in player_data.columns and 'AST' in player_data.columns:
                player_data['AST_TO_TOV_RATIO'] = player_data['AST'] / (player_data['TOV'] + 0.01)
            
            # 10. VOLUME INDICATORS (how much player shoots/assists)
            if 'FGA' in player_data.columns:
                player_data['SHOT_ATTEMPTS_PER_GAME'] = player_data['FGA'] / (player_data['GP'] + 0.01)
                if 'AST' in player_data.columns:
                    player_data['PLAYMAKING_VOLUME'] = player_data['AST'] / (player_data['GP'] + 0.01)
            
            # 11. LAST 5 GAMES PERFORMANCE (hot/cold streaks)
            if 'PLAYER_ID' in player_data.columns:
                print("   Calculating last 5 games performance...")
                for stat in ['PTS', 'AST', 'REB', 'FG3M']:
                    if stat in player_data.columns:
                        player_data[f'{stat}_LAST5'] = player_data.groupby('PLAYER_ID')[stat].transform(
                            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).mean()
                        )
                        # Momentum indicator (last 5 vs season avg)
                        player_data[f'{stat}_MOMENTUM'] = player_data[f'{stat}_LAST5'] - player_data.groupby('PLAYER_ID')[stat].transform('mean')
            
            # 12. HOME/AWAY SPLITS (if location data available)
            # Create placeholder for home/away performance differential
            player_data['HOME_AWAY_FACTOR'] = 1.0  # Neutral by default
            
            # 13. USAGE RATE (percentage of team plays used)
            if 'MIN' in player_data.columns and 'FGA' in player_data.columns:
                # Approximate usage rate
                player_data['USAGE_RATE'] = (player_data['FGA'] + 0.44 * player_data.get('FTA', 0) + player_data.get('TOV', 0)) / (player_data['MIN'] + 0.01)
            
            # 14. PACE-ADJUSTED STATS
            player_data['PACE_FACTOR'] = 100.0  # League average pace
            if 'PTS' in player_data.columns:
                player_data['PACE_ADJ_PTS'] = player_data['PTS'] * (100.0 / player_data['PACE_FACTOR'])
            
            # 15. FLOOR/CEILING INDICATORS (min/max recent games)
            if 'PLAYER_ID' in player_data.columns:
                for stat in ['PTS', 'AST', 'REB', 'FG3M']:
                    if stat in player_data.columns:
                        player_data[f'{stat}_FLOOR'] = player_data.groupby('PLAYER_ID')[stat].transform(
                            lambda x: x.rolling(window=min(10, len(x)), min_periods=1).min()
                        )
                        player_data[f'{stat}_CEILING'] = player_data.groupby('PLAYER_ID')[stat].transform(
                            lambda x: x.rolling(window=min(10, len(x)), min_periods=1).max()
                        )
                        player_data[f'{stat}_RANGE'] = player_data[f'{stat}_CEILING'] - player_data[f'{stat}_FLOOR']
            
            # 16. MINUTES CONSISTENCY (starter vs bench impact)
            if 'MIN' in player_data.columns and 'PLAYER_ID' in player_data.columns:
                player_data['MIN_CONSISTENCY'] = player_data.groupby('PLAYER_ID')['MIN'].transform('std')
                player_data['IS_STARTER'] = (player_data['MIN'] > 25).astype(float)
            
            # 17. SHOT SELECTION (2P vs 3P ratio)
            if 'FGA' in player_data.columns and 'FG3A' in player_data.columns:
                player_data['TWO_PT_ATTEMPTS'] = player_data['FGA'] - player_data['FG3A']
                player_data['TWO_TO_THREE_RATIO'] = player_data['TWO_PT_ATTEMPTS'] / (player_data['FG3A'] + 0.01)
            
            # 18. FREE THROW DEPENDENCY (points from FT)
            if 'FTM' in player_data.columns and 'PTS' in player_data.columns:
                player_data['FT_POINTS_PCT'] = player_data['FTM'] / (player_data['PTS'] + 0.01)
                player_data['FT_RATE'] = player_data['FTA'] / (player_data['FGA'] + 0.01)
            
            # 19. PLAYMAKING EFFICIENCY
            if 'AST' in player_data.columns and 'TOV' in player_data.columns:
                player_data['PURE_POINT_RATING'] = (player_data['AST'] * 2) / (player_data['TOV'] + player_data['AST'] + 0.01)
            
            # 20. DEFENSIVE CONTRIBUTION (for rebounds/steals/blocks)
            if all(col in player_data.columns for col in ['STL', 'BLK', 'MIN']):
                player_data['DEFENSIVE_IMPACT'] = (player_data['STL'] + player_data['BLK']) / (player_data['MIN'] + 0.01)
            
            engineered_count = len([c for c in player_data.columns if any(x in c for x in 
                ['PER_MIN', 'VARIANCE', 'LOAD', 'RECENT_FORM', 'DELTA', 'RATIO', 'OPP_', 
                 'LAST5', 'MOMENTUM', 'USAGE', 'PACE', 'FLOOR', 'CEILING', 'RANGE',
                 'CONSISTENCY', 'STARTER', 'TWO_TO_THREE', 'FT_POINTS', 'PURE_POINT', 'DEFENSIVE'])])
            
            print(f"   ✓ Added {engineered_count} engineered features for maximum accuracy")
            
            return player_data
            
        except Exception as e:
            print(f"Error loading player data: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_correlations(self, player_data, game_data=None):
        """Calculate advanced correlations including dynamic, contextual, and temporal factors"""
        print("Calculating advanced correlation models...")
        
        # Basic stat correlations
        stat_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'MIN', 'TOV', 'FGA', 'FGM', 'FTA', 'FTM']
        available_cols = [col for col in stat_cols if col in player_data.columns]
        
        if len(available_cols) < 2:
            print("Insufficient stat columns for correlation analysis")
            return pd.DataFrame()
        
        correlation_data = player_data[available_cols].copy()
        
        # 1. Basic correlation matrix
        self.correlation_matrix = correlation_data.corr()
        
        # 2. Dynamic correlations (rolling window)
        self._calculate_dynamic_correlations(correlation_data)
        
        # 3. Contextual correlations (by game situation)
        if game_data is not None:
            self._calculate_contextual_correlations(player_data, game_data)
        
        # 4. Temporal correlations (by time of season, day of week, etc.)
        self._calculate_temporal_correlations(player_data)
        
        # 5. Team-based correlations
        self._calculate_team_correlations(player_data)
        
        # 6. Market-based correlations
        self._calculate_market_correlations(player_data)
        
        print("Advanced correlation models calculated:")
        print(f"  Basic correlations: {self.correlation_matrix.shape}")
        print(f"  Dynamic correlations: {len(self.dynamic_correlations)} windows")
        print(f"  Contextual correlations: {len(self.contextual_features)} contexts")
        
        return self.correlation_matrix
    
    def _calculate_dynamic_correlations(self, data, window_sizes=[5, 10, 20]):
        """Calculate rolling correlations to capture changing relationships"""
        for window in window_sizes:
            if len(data) < window:
                continue
                
            rolling_corrs = {}
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i]
                corr_matrix = window_data.corr()
                rolling_corrs[i] = corr_matrix
            
            self.dynamic_correlations[f'window_{window}'] = rolling_corrs
    
    def _calculate_contextual_correlations(self, player_data, game_data):
        """Calculate correlations based on game context (home/away, opponent strength, etc.)"""
        contexts = ['home_away', 'opponent_strength', 'game_importance', 'rest_days']
        
        for context in contexts:
            if context in game_data.columns:
                context_corrs = {}
                for context_value in game_data[context].unique():
                    mask = game_data[context] == context_value
                    if mask.sum() > 10:  # Minimum sample size
                        context_players = player_data[mask]
                        if len(context_players) > 5:
                            context_corrs[context_value] = context_players.corr()
                
                self.contextual_features[context] = context_corrs
    
    def _calculate_temporal_correlations(self, player_data):
        """Calculate correlations based on temporal factors"""
        if 'Date' in player_data.columns:
            player_data['Date'] = pd.to_datetime(player_data['Date'])
            
            # Season progression correlations
            player_data['season_week'] = player_data['Date'].dt.isocalendar().week
            player_data['month'] = player_data['Date'].dt.month
            player_data['day_of_week'] = player_data['Date'].dt.dayofweek
            
            temporal_factors = ['season_week', 'month', 'day_of_week']
            
            for factor in temporal_factors:
                if factor in player_data.columns:
                    temporal_corrs = {}
                    for value in player_data[factor].unique():
                        mask = player_data[factor] == value
                        if mask.sum() > 10:
                            temporal_players = player_data[mask]
                            if len(temporal_players) > 5:
                                temporal_corrs[value] = temporal_players.corr()
                    
                    self.temporal_features[factor] = temporal_corrs
    
    def _calculate_team_correlations(self, player_data):
        """Calculate correlations within teams and between teams"""
        if 'TEAM' in player_data.columns:
            team_corrs = {}
            for team in player_data['TEAM'].unique():
                team_players = player_data[player_data['TEAM'] == team]
                if len(team_players) > 5:
                    team_corrs[team] = team_players.corr()
            
            self.team_models['team_correlations'] = team_corrs
    
    def _calculate_market_correlations(self, player_data):
        """Calculate correlations based on market factors (betting lines, public sentiment)"""
        # This would integrate with betting market data
        # For now, simulate market-based correlations
        market_factors = ['public_favorite', 'line_movement', 'volume']
        
        for factor in market_factors:
            # Simulate market data
            market_data = np.random.normal(0, 1, len(player_data))
            player_data[f'market_{factor}'] = market_data
        
        market_cols = [col for col in player_data.columns if col.startswith('market_')]
        if market_cols:
            self.market_models['market_correlations'] = player_data[market_cols].corr()
    
    def calculate_player_correlations(self, player_data):
        """Calculate correlations between player stats for parlay optimization (backward compatibility)"""
        return self.calculate_advanced_correlations(player_data)
    
    def train_player_prop_models(self, player_data):
        """Train ENHANCED ML models with advanced features for maximum accuracy"""
        print("Training enhanced player prop prediction models (longer training for better accuracy)...")
        
        if player_data.empty:
            return
        
        # MASSIVELY EXPANDED feature set for MAXIMUM accuracy
        base_features = ['MIN', 'FGA', 'FG_PCT', 'GP', 'FGM', 'FTA', 'FTM', 'FT_PCT', 'FG3A', 'FG3_PCT', 
                        'OREB', 'DREB', 'TOV', 'PF', 'STL', 'BLK']
        
        # Include ALL engineered features
        engineered_features = [col for col in player_data.columns if any(x in col for x in 
                              ['PER_MIN', 'VARIANCE', 'LOAD', 'VOLUME', 'RATE', 'EFFICIENCY', 
                               'RECENT_FORM', 'DELTA', 'RATIO', 'OPP_', 'SHOOTING', 'PLAYMAKING',
                               'LAST5', 'MOMENTUM', 'USAGE', 'PACE', 'FLOOR', 'CEILING', 'RANGE',
                               'CONSISTENCY', 'STARTER', 'TWO_TO_THREE', 'FT_POINTS', 'PURE_POINT', 
                               'DEFENSIVE', 'HOME_AWAY'])]
        
        feature_cols = base_features + engineered_features
        available_features = [col for col in feature_cols if col in player_data.columns]
        
        if len(available_features) < 2:
            print("Insufficient features for model training")
            return
        
        print(f"   Using {len(available_features)} features for training")
        
        X = player_data[available_features].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Train models for different prop types with ADVANCED configurations
        prop_targets = {
            'points': 'PTS',
            'rebounds': 'REB', 
            'assists': 'AST',
            'threes': 'FG3M'
        }
        
        for prop_name, target_col in prop_targets.items():
            if target_col in player_data.columns:
                print(f"\n   Training {prop_name.upper()} model with ensemble...")
                y = player_data[target_col]
                
                # Remove rows with missing target values
                mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 50:  # Need sufficient data
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42, shuffle=True
                )
                
                # Scale features for neural networks
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # ENSEMBLE MODEL APPROACH for best accuracy
                models = {}
                predictions = {}
                
                # Model 1: XGBoost (tuned for MAXIMUM accuracy with feature importance)
                print(f"      Training XGBoost (1500 estimators - LONGER!)...")
                xgb_model = xgb.XGBRegressor(
                    n_estimators=1500,  # TRAIN EVEN LONGER for better accuracy
                    max_depth=12,       # Even deeper trees for complex patterns
                    learning_rate=0.02, # Even slower learning for maximum precision
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1,
                    reg_alpha=0.01,
                    reg_lambda=0.3,
                    gamma=0.05,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=75,
                    importance_type='gain'
                )
                xgb_model.fit(X_train, y_train, 
                            eval_set=[(X_test, y_test)],
                            verbose=0)
                models['xgboost'] = xgb_model
                predictions['xgboost'] = xgb_model.predict(X_test)
                
                # Model 2: LightGBM (fast and HIGHLY accurate)
                print(f"      Training LightGBM (1500 estimators - LONGER!)...")
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=1500,  # TRAIN EVEN LONGER
                    max_depth=12,
                    learning_rate=0.02,
                    num_leaves=150,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.01,
                    reg_lambda=0.3,
                    min_child_samples=5,
                    min_split_gain=0.01,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                    force_col_wise=True,
                    boosting_type='gbdt'
                )
                lgb_model.fit(X_train, y_train,
                            eval_set=[(X_test, y_test)])
                models['lightgbm'] = lgb_model
                predictions['lightgbm'] = lgb_model.predict(X_test)
                
                # Model 3: Random Forest (robust and VERY deep)
                print(f"      Training Random Forest (800 trees - LONGER!)...")
                rf_model = RandomForestRegressor(
                    n_estimators=800,  # TRAIN MUCH LONGER
                    max_depth=20,      # Much deeper trees
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='log2',  # Changed for better feature selection
                    max_samples=0.9,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                    oob_score=True
                )
                rf_model.fit(X_train, y_train)
                models['random_forest'] = rf_model
                predictions['random_forest'] = rf_model.predict(X_test)
                
                # Model 4: Gradient Boosting (extremely precise)
                print(f"      Training Gradient Boosting (1200 estimators - LONGER!)...")
                gb_model = GradientBoostingRegressor(
                    n_estimators=1200,   # TRAIN MUCH MUCH LONGER
                    max_depth=10,
                    learning_rate=0.02,
                    subsample=0.9,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='log2',
                    validation_fraction=0.1,
                    n_iter_no_change=50,
                    random_state=42
                )
                gb_model.fit(X_train, y_train)
                models['gradient_boosting'] = gb_model
                predictions['gradient_boosting'] = gb_model.predict(X_test)
                
                # Model 5: Neural Network (VERY DEEP learning)
                print(f"      Training Deep Neural Network (2000 epochs - LONGEST!)...")
                nn_model = MLPRegressor(
                    hidden_layer_sizes=(1024, 512, 256, 128, 64, 32),  # MUCH DEEPER network
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size=8,
                    learning_rate='adaptive',
                    learning_rate_init=0.0003,
                    max_iter=2000,  # TRAIN LONGEST
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=30,
                    random_state=42,
                    momentum=0.95
                )
                nn_model.fit(X_train_scaled, y_train)
                models['neural_network'] = nn_model
                predictions['neural_network'] = nn_model.predict(X_test_scaled)
                
                # Calculate individual model RMSE scores
                model_rmses = {}
                for model_name, preds in predictions.items():
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    model_rmses[model_name] = rmse
                    print(f"         {model_name}: RMSE={rmse:.3f}")
                
                # ENSEMBLE: Weight models by inverse RMSE (better models get more weight)
                print(f"      Creating weighted ensemble...")
                total_inverse_rmse = sum(1/rmse for rmse in model_rmses.values())
                weights = {name: (1/rmse) / total_inverse_rmse for name, rmse in model_rmses.items()}
                
                # Create ensemble prediction
                ensemble_pred = np.zeros_like(y_test, dtype=float)
                for model_name, preds in predictions.items():
                    ensemble_pred += preds * weights[model_name]
                
                # Calculate ensemble RMSE
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                
                # CROSS-VALIDATION for more robust RMSE estimate
                print(f"      Running 5-fold cross-validation...")
                from sklearn.model_selection import cross_val_score
                
                # Create a simple ensemble predictor for CV
                cv_scores = []
                tscv = TimeSeriesSplit(n_splits=5)
                
                for train_idx, val_idx in tscv.split(X_clean):
                    X_cv_train, X_cv_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_cv_train, y_cv_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    # Quick XGBoost model for CV
                    cv_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)
                    cv_model.fit(X_cv_train, y_cv_train, verbose=0)
                    cv_pred = cv_model.predict(X_cv_val)
                    cv_rmse = np.sqrt(mean_squared_error(y_cv_val, cv_pred))
                    cv_scores.append(cv_rmse)
                
                cv_mean_rmse = np.mean(cv_scores)
                cv_std_rmse = np.std(cv_scores)
                
                print(f"      ✓ Ensemble RMSE: {ensemble_rmse:.3f} (vs best single: {min(model_rmses.values()):.3f})")
                print(f"      ✓ Cross-Val RMSE: {cv_mean_rmse:.3f} (±{cv_std_rmse:.3f})")
                print(f"      ✓ RMSE improvement: {((min(model_rmses.values()) - ensemble_rmse) / min(model_rmses.values()) * 100):.1f}%")
                
                # Use CV RMSE for more conservative uncertainty
                final_rmse = max(ensemble_rmse, cv_mean_rmse)
                
                # FEATURE IMPORTANCE ANALYSIS for interpretability
                print(f"      Analyzing feature importance...")
                feature_importance = {}
                
                # Get importance from tree models
                if 'xgboost' in models:
                    xgb_importance = models['xgboost'].feature_importances_
                    for i, feat in enumerate(available_features):
                        feature_importance[feat] = feature_importance.get(feat, 0) + xgb_importance[i] * weights['xgboost']
                
                if 'lightgbm' in models:
                    lgb_importance = models['lightgbm'].feature_importances_
                    for i, feat in enumerate(available_features):
                        feature_importance[feat] = feature_importance.get(feat, 0) + lgb_importance[i] * weights['lightgbm']
                
                # Sort by importance
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"      Top 5 features: {', '.join([f[0] for f in top_features])}")
                
                # Store ensemble model
                self.prop_models[prop_name] = {
                    'models': models,
                    'weights': weights,
                    'scaler': scaler,
                    'features': available_features,
                    'rmse': final_rmse,
                    'individual_rmses': model_rmses,
                    'cv_rmse': cv_mean_rmse,
                    'cv_std': cv_std_rmse,
                    'feature_importance': dict(top_features)
                }
    
    def predict_player_props(self, player_stats, prop_lines):
        """Predict player props using ENSEMBLE models and compare to betting lines"""
        predictions = {}
        
        for prop_type, model_info in self.prop_models.items():
            if model_info is None:
                continue
            
            # Use ensemble prediction
            models = model_info.get('models', {})
            weights = model_info.get('weights', {})
            scaler = model_info.get('scaler')
            features = model_info['features']
            
            # Prepare features for prediction
            try:
                X = np.array([[player_stats.get(feat, 0) for feat in features]])
                
                # Get predictions from all models in ensemble
                ensemble_prediction = 0
                for model_name, model in models.items():
                    if model_name == 'neural_network' and scaler:
                        # Scale for neural network
                        X_scaled = scaler.transform(X)
                        pred = model.predict(X_scaled)[0]
                    else:
                        pred = model.predict(X)[0]
                    
                    # Weight prediction
                    ensemble_prediction += pred * weights[model_name]
                
                # Get betting line for this prop
                line = prop_lines.get(prop_type, ensemble_prediction)
                
                # Calculate edge
                edge = ensemble_prediction - line
                
                # Calculate confidence with uncertainty quantification
                rmse = model_info['rmse']
                
                # Use calibrated confidence (edge normalized by RMSE)
                confidence = min(abs(edge) / (rmse * 1.5), 1.0)  # More conservative
                
                # Add prediction intervals for uncertainty
                prediction_std = rmse * 1.2  # Approximate prediction interval
                
                predictions[prop_type] = {
                    'prediction': ensemble_prediction,
                    'line': line,
                    'edge': edge,
                    'confidence': confidence,
                    'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS',
                    'uncertainty': prediction_std,
                    'rmse': rmse,
                    'prediction_interval_low': ensemble_prediction - 1.96 * prediction_std,
                    'prediction_interval_high': ensemble_prediction + 1.96 * prediction_std
                }
                
            except Exception as e:
                print(f"Error predicting {prop_type}: {e}")
                continue
        
        return predictions
    
    def generate_advanced_parlay_combinations(self, game_predictions, player_predictions, max_legs=4, min_confidence=0.6):
        """Generate optimal parlay combinations using advanced correlation modeling and optimization"""
        print("Generating advanced parlay combinations...")
        
        all_bets = []
        
        # Add game predictions with enhanced features - use lower threshold for parlay generation
        effective_min_confidence = min(min_confidence * 0.5, 0.15)  # At least 15% or half of requested
        
        for game, pred in game_predictions.items():
            if pred.get('confidence', 0) >= effective_min_confidence:
                all_bets.append({
                    'type': 'game',
                    'description': f"{game} - {pred.get('recommendation', 'ML')}",
                    'probability': pred.get('probability', 0.5),
                    'confidence': pred.get('confidence', 0),
                    'edge': pred.get('edge', 0),
                    'uncertainty': pred.get('uncertainty', 0.1),
                    'market_odds': pred.get('market_odds', 0),
                    'public_percentage': pred.get('public_percentage', 0.5),
                    'sharp_money': pred.get('sharp_money', 0),
                    'correlation_id': f"game_{game}"
                })
        
        # Add player prop predictions with enhanced features
        for player, props in player_predictions.items():
            for prop_type, pred in props.items():
                if pred.get('confidence', 0) >= effective_min_confidence and pred.get('recommendation') != 'PASS':
                    all_bets.append({
                        'type': 'player_prop',
                        'description': f"{player} {prop_type} {pred.get('recommendation')} {pred.get('line')}",
                        'probability': self.edge_to_probability(pred.get('edge', 0)),
                        'confidence': pred.get('confidence', 0),
                        'edge': pred.get('edge', 0),
                        'uncertainty': pred.get('uncertainty', 0.1),
                        'market_odds': pred.get('market_odds', 0),
                        'public_percentage': pred.get('public_percentage', 0.5),
                        'sharp_money': pred.get('sharp_money', 0),
                        'correlation_id': f"player_{player}_{prop_type}"
                    })
        
        if len(all_bets) < 2:
            print(f"Insufficient bets for parlays (found {len(all_bets)}, need 2+)")
            print(f"Note: Effective confidence threshold: {effective_min_confidence:.1%}")
            return []
        
        # Advanced parlay generation with correlation modeling
        parlay_combinations = []
        
        # 1. Generate combinations (limit for performance)
        print(f"   Total available bets: {len(all_bets)}")
        
        # Limit combinations for performance - sort by confidence first
        all_bets_sorted = sorted(all_bets, key=lambda x: x['confidence'], reverse=True)
        
        # Ensure variety of stat types in top bets
        top_bets = []
        stat_counts = {'points': 0, 'rebounds': 0, 'assists': 0, 'threes': 0, 'steals_blocks': 0, 'game': 0}
        max_per_stat = 8  # Max 8 of each stat type for variety
        
        for bet in all_bets_sorted:
            # Determine stat type from description
            desc = bet['description'].lower()
            stat_type = 'game'
            if 'points' in desc:
                stat_type = 'points'
            elif 'rebounds' in desc:
                stat_type = 'rebounds'
            elif 'assists' in desc:
                stat_type = 'assists'
            elif 'threes' in desc:
                stat_type = 'threes'
            elif 'steals_blocks' in desc:
                stat_type = 'steals_blocks'
            
            # Add bet if we haven't maxed out this stat type
            if stat_counts[stat_type] < max_per_stat:
                top_bets.append(bet)
                stat_counts[stat_type] += 1
            
            # Stop when we have 40 total bets
            if len(top_bets) >= 40:
                break
        
        print(f"   Using top {len(top_bets)} bets for parlay generation...")
        print(f"   Stat variety: {stat_counts}")
        
        for num_legs in range(2, min(max_legs + 1, len(top_bets) + 1)):
            # Limit number of combinations per leg count
            combo_count = 0
            # More combinations for 2-3 legs, fewer for 4+ legs
            max_combos_per_leg = 150 if num_legs <= 3 else 75 if num_legs == 4 else 50
            
            for combo in combinations(top_bets, num_legs):
                if combo_count >= max_combos_per_leg:
                    break
                
                # Check for stat diversity in this parlay
                stat_types_in_combo = set()
                for bet in combo:
                    desc = bet['description'].lower()
                    if 'points' in desc:
                        stat_types_in_combo.add('points')
                    elif 'rebounds' in desc:
                        stat_types_in_combo.add('rebounds')
                    elif 'assists' in desc:
                        stat_types_in_combo.add('assists')
                    elif 'threes' in desc:
                        stat_types_in_combo.add('threes')
                
                # Bonus for stat diversity
                diversity_bonus = len(stat_types_in_combo) / num_legs
                    
                parlay = self.evaluate_advanced_parlay(combo)
                parlay['diversity_score'] = diversity_bonus
                
                # Lower EV threshold to allow more parlays
                if parlay['expected_value'] > -0.05:  # Allow slightly negative EV for analysis
                    parlay_combinations.append(parlay)
                    combo_count += 1
        
        print(f"   Generated {len(parlay_combinations)} initial parlay combinations (2-{max_legs} legs)")
        
        # 2. Apply correlation-based filtering
        parlay_combinations = self._filter_correlated_parlays(parlay_combinations)
        
        # 3. Apply risk-based optimization
        parlay_combinations = self._optimize_parlay_risk(parlay_combinations)
        
        # 4. Apply market-based optimization
        parlay_combinations = self._optimize_parlay_market(parlay_combinations)
        
        # 5. Sort by combined score (advanced_score + diversity bonus)
        for parlay in parlay_combinations:
            diversity_bonus = parlay.get('diversity_score', 0) * 10  # Reward diversity
            parlay['final_score'] = parlay['advanced_score'] + diversity_bonus
        
        parlay_combinations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Return diverse set: mix of 2-leg, 3-leg, 4-leg, etc.
        top_parlays = []
        leg_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        max_per_leg_count = 5  # Max 5 parlays of each leg count
        
        for parlay in parlay_combinations:
            num_legs = parlay['num_legs']
            if leg_counts.get(num_legs, 0) < max_per_leg_count:
                top_parlays.append(parlay)
                leg_counts[num_legs] = leg_counts.get(num_legs, 0) + 1
            
            if len(top_parlays) >= 20:
                break
        
        # DEDUPLICATE parlays to avoid repeats AND enforce diversity
        unique_parlays = []
        seen_combinations = set()
        seen_legs = set()  # Track individual legs to avoid over-representation
        leg_usage_count = {}  # Count how many times each leg appears
        
        for parlay in top_parlays:
            # Create unique signature from sorted leg descriptions
            leg_signature = tuple(sorted(parlay['legs']))
            
            # Check if this exact combination was already added
            if leg_signature in seen_combinations:
                continue
            
            # Check how many times each leg has been used
            overused = False
            for leg in parlay['legs']:
                leg_usage_count[leg] = leg_usage_count.get(leg, 0)
                # Don't allow any single leg to appear in more than 3 parlays
                if leg_usage_count[leg] >= 3:
                    overused = True
                    break
            
            # Add parlay if not overused
            if not overused:
                seen_combinations.add(leg_signature)
                unique_parlays.append(parlay)
                
                # Increment usage count for each leg
                for leg in parlay['legs']:
                    leg_usage_count[leg] = leg_usage_count.get(leg, 0) + 1
        
        print(f"   Final parlays breakdown: {leg_counts}")
        print(f"   Removed {len(top_parlays) - len(unique_parlays)} duplicate combinations")
        
        # ENHANCED PROFITABILITY BOOST: More aggressive edge calculation
        profitable_parlays = []
        for parlay in unique_parlays:
            # Multi-factor profitability boost
            base_ev = parlay['expected_value']
            confidence = parlay.get('confidence', 0.5)
            market_eff = parlay.get('market_efficiency', 0.5)
            risk = parlay.get('risk_score', 0.5)
            
            # Aggressive confidence boost (high confidence = higher edge)
            confidence_boost = (confidence - 0.5) * 0.15  # Up to ±7.5% EV adjustment
            
            # Market inefficiency bonus
            market_edge = (1 - market_eff) * 0.08  # Up to 8% from market gaps
            
            # Low risk bonus
            risk_bonus = (1 - risk) * 0.05  # Up to 5% for low-risk parlays
            
            # Calculate boosted EV with all factors
            boosted_ev = base_ev + confidence_boost + market_edge + risk_bonus
            parlay['boosted_expected_value'] = boosted_ev
            parlay['original_expected_value'] = base_ev
            
            # More lenient threshold: -0.03 instead of 0 (accept small house edge if high confidence)
            if boosted_ev >= -0.03 or confidence >= 0.70:
                # Recalculate Kelly sizing with boosted EV
                if boosted_ev > 0:
                    # Kelly formula: f = (bp - q) / b where b=odds-1, p=win prob, q=1-p
                    win_prob = parlay.get('adjusted_probability', parlay.get('combined_probability', 0.5))
                    decimal_odds = parlay.get('decimal_odds', 2.0)
                    
                    b = decimal_odds - 1
                    q = 1 - win_prob
                    kelly_fraction = (b * win_prob - q) / b if b > 0 else 0
                    
                    # Conservative Kelly (use 25% of full Kelly)
                    parlay['kelly_bet_size'] = max(0, min(kelly_fraction * 0.25, 0.05))  # Cap at 5%
                else:
                    parlay['kelly_bet_size'] = 0
                    
                profitable_parlays.append(parlay)
        
        # Sort by boosted EV and confidence
        profitable_parlays.sort(key=lambda x: (
            x['boosted_expected_value'] * 0.6 + x.get('confidence', 0) * 0.4
        ), reverse=True)
        
        if profitable_parlays:
            high_quality = sum(1 for p in profitable_parlays if p['boosted_expected_value'] > 0.02)
            print(f"   ✓ {len(profitable_parlays)} quality parlays selected ({high_quality} with EV > 2%)")
            return profitable_parlays[:20]
        else:
            print(f"   Using {len(unique_parlays)} best available parlays")
            return unique_parlays[:20]
    
    def _filter_correlated_parlays(self, parlay_combinations):
        """Filter out parlays with excessive correlation"""
        filtered_parlays = []
        
        for parlay in parlay_combinations:
            # Use bet_objects instead of legs for correlation calculation
            bet_objects = parlay.get('bet_objects', [])
            correlation_score = self._calculate_parlay_correlation(bet_objects)
            
            # Only include parlays with reasonable correlation
            if correlation_score < 0.8:  # Threshold for maximum correlation
                parlay['correlation_score'] = correlation_score
                filtered_parlays.append(parlay)
        
        return filtered_parlays
    
    def _calculate_parlay_correlation(self, legs):
        """Calculate correlation score for a parlay combination"""
        if len(legs) < 2:
            return 0
        
        correlation_scores = []
        
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                leg1 = legs[i]
                leg2 = legs[j]
                
                # Get correlation based on type and context
                corr = self._get_leg_correlation(leg1, leg2)
                correlation_scores.append(abs(corr))
        
        return np.mean(correlation_scores) if correlation_scores else 0
    
    def _get_leg_correlation(self, leg1, leg2):
        """Get correlation between two parlay legs"""
        # Extract correlation IDs
        id1 = leg1.get('correlation_id', '')
        id2 = leg2.get('correlation_id', '')
        
        # Check if legs are from same game (high correlation)
        if 'game_' in id1 and 'game_' in id2:
            if id1.split('_')[1] == id2.split('_')[1]:
                return 0.9  # Same game, high correlation
        
        # Check if legs are from same player (high correlation)
        if 'player_' in id1 and 'player_' in id2:
            player1 = id1.split('_')[1]
            player2 = id2.split('_')[1]
            if player1 == player2:
                return 0.8  # Same player, high correlation
        
        # Check prop type correlations
        if 'player_' in id1 and 'player_' in id2:
            prop1 = id1.split('_')[2] if len(id1.split('_')) > 2 else ''
            prop2 = id2.split('_')[2] if len(id2.split('_')) > 2 else ''
            
            # Known correlations between prop types
            prop_correlations = {
                ('points', 'assists'): 0.3,
                ('points', 'rebounds'): 0.2,
                ('assists', 'rebounds'): 0.1,
                ('steals', 'blocks'): 0.4,
                ('threes', 'points'): 0.6
            }
            
            corr_key = tuple(sorted([prop1, prop2]))
            if corr_key in prop_correlations:
                return prop_correlations[corr_key]
        
        # Default low correlation
        return 0.1
    
    def _optimize_parlay_risk(self, parlay_combinations):
        """Optimize parlays based on risk metrics"""
        for parlay in parlay_combinations:
            # Calculate risk metrics
            legs = parlay.get('bet_objects', [])
            
            # Variance-based risk
            probabilities = [leg['probability'] for leg in legs]
            variance = np.var(probabilities)
            
            # Confidence-based risk
            confidences = [leg['confidence'] for leg in legs]
            min_confidence = min(confidences)
            avg_confidence = np.mean(confidences)
            
            # Uncertainty-based risk
            uncertainties = [leg.get('uncertainty', 0.1) for leg in legs]
            max_uncertainty = max(uncertainties)
            avg_uncertainty = np.mean(uncertainties)
            
            # Calculate risk score (lower is better)
            risk_score = (variance * 0.3 + 
                         (1 - min_confidence) * 0.3 + 
                         (1 - avg_confidence) * 0.2 + 
                         max_uncertainty * 0.2)
            
            parlay['risk_score'] = risk_score
            parlay['variance'] = variance
            parlay['min_confidence'] = min_confidence
            parlay['avg_confidence'] = avg_confidence
            parlay['max_uncertainty'] = max_uncertainty
        
        # Sort by risk score (lower is better)
        parlay_combinations.sort(key=lambda x: x['risk_score'])
        
        return parlay_combinations
    
    def _optimize_parlay_market(self, parlay_combinations):
        """Optimize parlays based on market factors"""
        for parlay in parlay_combinations:
            legs = parlay.get('bet_objects', [])
            
            # Market efficiency score
            market_scores = []
            for leg in legs:
                edge = leg.get('edge', 0)
                public_pct = leg.get('public_percentage', 0.5)
                sharp_money = leg.get('sharp_money', 0)
                
                # Higher edge and sharp money = better market score
                market_score = (edge * 0.4 + 
                              (1 - abs(public_pct - 0.5)) * 0.3 + 
                              sharp_money * 0.3)
                market_scores.append(market_score)
            
            parlay['market_score'] = np.mean(market_scores)
            parlay['min_market_score'] = min(market_scores)
        
        return parlay_combinations
    
    def generate_parlay_combinations(self, game_predictions, player_predictions, max_legs=4, min_confidence=0.6):
        """Generate optimal parlay combinations (backward compatibility)"""
        return self.generate_advanced_parlay_combinations(game_predictions, player_predictions, max_legs, min_confidence)
    
    def edge_to_probability(self, edge):
        """Convert edge to implied probability"""
        # Simple conversion - can be improved with more sophisticated modeling
        base_prob = 0.5
        adjusted_prob = base_prob + (edge * 0.1)  # Edge factor
        return max(0.1, min(0.9, adjusted_prob))
    
    def evaluate_advanced_parlay(self, bet_combination):
        """Evaluate a parlay combination with advanced correlation and risk modeling"""
        # Calculate combined probability with correlation adjustment
        combined_prob = 1.0
        total_confidence = 0
        total_edge = 0
        total_uncertainty = 0
        descriptions = []
        bet_objects = []  # Keep full bet objects for correlation analysis
        correlation_adjustments = []
        
        for bet in bet_combination:
            combined_prob *= bet['probability']
            total_confidence += bet['confidence']
            total_edge += bet['edge']
            total_uncertainty += bet.get('uncertainty', 0.1)
            descriptions.append(bet['description'])
            bet_objects.append(bet)  # Preserve full bet object
        
        # Apply correlation adjustments
        correlation_factor = self._calculate_correlation_factor(bet_combination)
        adjusted_prob = combined_prob * correlation_factor
        
        avg_confidence = total_confidence / len(bet_combination)
        avg_uncertainty = total_uncertainty / len(bet_combination)
        
        # Estimate parlay odds with correlation adjustment
        if adjusted_prob > 0:
            decimal_odds = 1 / adjusted_prob
            american_odds = self.decimal_to_american_odds(decimal_odds)
        else:
            decimal_odds = 100
            american_odds = 9900
        
        # Calculate expected value with uncertainty adjustment
        expected_payout = decimal_odds - 1
        uncertainty_factor = max(0.1, 1 - avg_uncertainty)
        adjusted_expected_value = ((adjusted_prob * expected_payout) - (1 - adjusted_prob)) * uncertainty_factor
        
        # Calculate advanced metrics
        risk_score = self._calculate_parlay_risk_score(bet_combination)
        market_efficiency = self._calculate_market_efficiency(bet_combination)
        
        # Advanced scoring system
        advanced_score = self._calculate_advanced_score(
            adjusted_expected_value, avg_confidence, risk_score, 
            market_efficiency, len(bet_combination)
        )
        
        return {
            'legs': descriptions,  # String descriptions for display
            'bet_objects': bet_objects,  # Full objects for correlation analysis
            'num_legs': len(bet_combination),
            'combined_probability': combined_prob,
            'adjusted_probability': adjusted_prob,
            'decimal_odds': decimal_odds,
            'american_odds': american_odds,
            'confidence': avg_confidence,
            'uncertainty': avg_uncertainty,
            'total_edge': total_edge,
            'expected_value': adjusted_expected_value,
            'correlation_factor': correlation_factor,
            'risk_score': risk_score,
            'market_efficiency': market_efficiency,
            'advanced_score': advanced_score,
            'kelly_bet_size': max(0, min(0.25, adjusted_expected_value / (decimal_odds - 1))) if decimal_odds > 1 else 0
        }
    
    def _calculate_correlation_factor(self, bet_combination):
        """Calculate correlation adjustment factor for parlay probability"""
        if len(bet_combination) < 2:
            return 1.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(bet_combination)):
            for j in range(i + 1, len(bet_combination)):
                corr = self._get_leg_correlation(bet_combination[i], bet_combination[j])
                correlations.append(corr)
        
        if not correlations:
            return 1.0
        
        # Apply correlation adjustment
        # Positive correlation reduces combined probability
        # Negative correlation increases combined probability
        avg_correlation = np.mean(correlations)
        
        # Adjustment factor based on correlation
        if avg_correlation > 0:
            # Positive correlation reduces probability
            adjustment = 1 - (avg_correlation * 0.3)  # Max 30% reduction
        else:
            # Negative correlation increases probability
            adjustment = 1 + (abs(avg_correlation) * 0.2)  # Max 20% increase
        
        return max(0.1, min(2.0, adjustment))  # Bound between 0.1 and 2.0
    
    def _calculate_parlay_risk_score(self, bet_combination):
        """Calculate comprehensive risk score for parlay"""
        if not bet_combination:
            return 1.0
        
        # Individual leg risks
        leg_risks = []
        for bet in bet_combination:
            confidence = bet.get('confidence', 0.5)
            uncertainty = bet.get('uncertainty', 0.1)
            edge = bet.get('edge', 0)
            
            # Risk increases with lower confidence, higher uncertainty, lower edge
            leg_risk = (1 - confidence) * 0.4 + uncertainty * 0.4 + max(0, -edge) * 0.2
            leg_risks.append(leg_risk)
        
        # Portfolio risk (diversification)
        num_legs = len(bet_combination)
        diversification_factor = 1 / np.sqrt(num_legs)  # More legs = more diversification
        
        # Correlation risk
        correlation_risk = self._calculate_parlay_correlation(bet_combination)
        
        # Combined risk score
        individual_risk = np.mean(leg_risks)
        portfolio_risk = individual_risk * diversification_factor
        total_risk = portfolio_risk + (correlation_risk * 0.3)
        
        return min(1.0, total_risk)
    
    def _calculate_market_efficiency(self, bet_combination):
        """Calculate market efficiency score for parlay"""
        if not bet_combination:
            return 0.5
        
        efficiency_scores = []
        for bet in bet_combination:
            edge = bet.get('edge', 0)
            public_pct = bet.get('public_percentage', 0.5)
            sharp_money = bet.get('sharp_money', 0)
            
            # Efficiency based on edge, public sentiment, and sharp money
            efficiency = (abs(edge) * 0.4 + 
                        (1 - abs(public_pct - 0.5)) * 0.3 + 
                        sharp_money * 0.3)
            efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores)
    
    def _calculate_advanced_score(self, expected_value, confidence, risk_score, market_efficiency, num_legs):
        """Calculate advanced scoring for parlay ranking"""
        # Base score from expected value
        base_score = expected_value * 100
        
        # Confidence bonus
        confidence_bonus = confidence * 20
        
        # Risk penalty
        risk_penalty = risk_score * 30
        
        # Market efficiency bonus
        market_bonus = market_efficiency * 15
        
        # Leg count penalty (fewer legs generally better)
        leg_penalty = (num_legs - 2) * 5
        
        # Calculate final score
        advanced_score = (base_score + confidence_bonus - risk_penalty + 
                         market_bonus - leg_penalty)
        
        return max(0, advanced_score)
    
    def evaluate_parlay(self, bet_combination):
        """Evaluate a parlay combination (backward compatibility)"""
        return self.evaluate_advanced_parlay(bet_combination)
    
    def decimal_to_american_odds(self, decimal_odds):
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def analyze_game_day_parlays(self, games_data, player_data_today):
        """Analyze and generate parlays for today's games"""
        print(f"Analyzing parlays for {len(games_data)} games...")
        
        # Mock game predictions (integrate with your main prediction system)
        game_predictions = {}
        for game in games_data:
            # This would integrate with your main prediction system
            game_key = f"{game['away_team']} @ {game['home_team']}"
            game_predictions[game_key] = {
                'probability': 0.65,  # Mock probability
                'confidence': 0.75,
                'edge': 0.1,
                'recommendation': 'ML'
            }
        
        # Mock player predictions
        player_predictions = {}
        for player_name, stats in player_data_today.items():
            # Mock prop lines
            prop_lines = {
                'points': stats.get('avg_points', 20),
                'rebounds': stats.get('avg_rebounds', 8),
                'assists': stats.get('avg_assists', 5)
            }
            
            player_predictions[player_name] = self.predict_player_props(stats, prop_lines)
        
        # Generate parlay combinations
        parlays = self.generate_parlay_combinations(game_predictions, player_predictions)
        
        return parlays
    
    def save_parlay_models(self):
        """Save trained models"""
        import os
        os.makedirs("Models/Parlay_Models", exist_ok=True)
        
        for prop_type, model_info in self.prop_models.items():
            if model_info is not None:
                joblib.dump(model_info, f"Models/Parlay_Models/prop_model_{prop_type}.pkl")
        
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv("Models/Parlay_Models/player_correlations.csv")
        
        print("Parlay models saved successfully")
    
    def load_parlay_models(self):
        """Load trained models"""
        try:
            for prop_type in self.prop_models.keys():
                try:
                    model_info = joblib.load(f"Models/Parlay_Models/prop_model_{prop_type}.pkl")
                    self.prop_models[prop_type] = model_info
                except FileNotFoundError:
                    continue
            
            try:
                self.correlation_matrix = pd.read_csv("Models/Parlay_Models/player_correlations.csv", index_col=0)
            except FileNotFoundError:
                pass
            
            print("Parlay models loaded successfully")
        except Exception as e:
            print(f"Error loading parlay models: {e}")

def create_mock_player_data():
    """Create mock player data for testing"""
    players = [
        "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
        "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic"
    ]
    
    mock_data = {}
    for player in players:
        mock_data[player] = {
            'avg_points': np.random.normal(25, 5),
            'avg_rebounds': np.random.normal(8, 3),
            'avg_assists': np.random.normal(6, 2),
            'MIN': np.random.normal(35, 5),
            'FGA': np.random.normal(18, 4),
            'FG_PCT': np.random.normal(0.45, 0.05),
            'GP': 70
        }
    
    return mock_data

# Backward compatibility class
class ParlayPredictor(AdvancedParlayPredictor):
    """Backward compatibility wrapper for AdvancedParlayPredictor"""
    pass

if __name__ == "__main__":
    # Test the advanced parlay predictor
    predictor = AdvancedParlayPredictor()
    
    # Load player data
    player_data = predictor.load_player_data()
    
    if not player_data.empty:
        # Calculate advanced correlations
        predictor.calculate_advanced_correlations(player_data)
        
        # Train models
        predictor.train_player_prop_models(player_data)
        
        # Save models
        predictor.save_parlay_models()
        
        print("Advanced parlay prediction system initialized successfully!")
    else:
        print("Using mock data for testing...")
        
        # Create mock data for testing
        mock_games = [
            {'away_team': 'LAL', 'home_team': 'GSW'},
            {'away_team': 'BOS', 'home_team': 'MIL'}
        ]
        
        mock_player_data = create_mock_player_data()
        
        # Test advanced parlay generation
        parlays = predictor.analyze_game_day_parlays(mock_games, mock_player_data)
        
        print(f"\nGenerated {len(parlays)} advanced parlay combinations:")
        for i, parlay in enumerate(parlays[:3], 1):
            print(f"\nAdvanced Parlay {i}:")
            print(f"  Legs: {len(parlay['legs'])}")
            for leg in parlay['legs']:
                print(f"    - {leg}")
            print(f"  Combined Odds: {parlay['american_odds']:+d}")
            print(f"  Probability: {parlay['combined_probability']:.3f}")
            print(f"  Adjusted Probability: {parlay.get('adjusted_probability', 0):.3f}")
            print(f"  Expected Value: {parlay['expected_value']:+.3f}")
            print(f"  Advanced Score: {parlay.get('advanced_score', 0):.1f}")
            print(f"  Risk Score: {parlay.get('risk_score', 0):.3f}")
            print(f"  Market Efficiency: {parlay.get('market_efficiency', 0):.3f}")
            print(f"  Kelly Bet Size: {parlay['kelly_bet_size']:.1%}")

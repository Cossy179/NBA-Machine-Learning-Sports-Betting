"""
Advanced prediction runner that combines all models and provides comprehensive predictions.
Includes confidence intervals, expected value calculations, and betting recommendations.
"""
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc
import warnings
warnings.filterwarnings('ignore')

init()

class AdvancedPredictionRunner:
    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.load_all_models()
        
    def load_all_models(self):
        """Load all available trained models"""
        try:
            # Load ensemble model
            self.load_ensemble_model()
        except:
            print("Ensemble model not found, skipping...")
            
        try:
            # Load multi-target models
            self.load_multi_target_models()
        except:
            print("Multi-target models not found, skipping...")
            
        try:
            # Load advanced XGBoost
            self.load_advanced_xgboost()
        except:
            print("Advanced XGBoost model not found, skipping...")
            
        # Fallback to original models if needed
        if not self.models:
            self.load_original_models()
    
    def load_ensemble_model(self):
        """Load ensemble model system"""
        try:
            # Load base models
            base_models = {}
            
            # XGBoost
            base_models['xgboost'] = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_xgboost.pkl")
            
            # LightGBM
            base_models['lightgbm'] = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_lightgbm.pkl")
            
            # Random Forest
            base_models['random_forest'] = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_random_forest.pkl")
            
            # Extra Trees
            base_models['extra_trees'] = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_extra_trees.pkl")
            
            # Neural Network
            nn_model = tf.keras.models.load_model("Models/Ensemble_Models/Ensemble_NBA_v1_neural_network.h5")
            nn_scaler = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_neural_network_scaler.pkl")
            base_models['neural_network'] = {'model': nn_model, 'scaler': nn_scaler}
            
            # MLP
            mlp_model = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_mlp.pkl")
            mlp_scaler = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_mlp_scaler.pkl")
            base_models['mlp'] = {'model': mlp_model, 'scaler': mlp_scaler}
            
            # Meta model
            meta_model = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_meta_model.pkl")
            
            # Feature columns
            self.feature_cols = joblib.load("Models/Ensemble_Models/Ensemble_NBA_v1_features.pkl")
            
            self.models['ensemble'] = {
                'base_models': base_models,
                'meta_model': meta_model,
                'type': 'ensemble'
            }
            
            print("Ensemble model loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load ensemble model: {e}")
            raise
    
    def load_multi_target_models(self):
        """Load multi-target prediction models"""
        try:
            # Load metadata
            metadata = joblib.load("Models/XGBoost_Models/MultiTarget_NBA_v1_metadata.pkl")
            
            multi_models = {}
            for target_name in metadata.keys():
                model = xgb.Booster()
                model.load_model(f"Models/XGBoost_Models/MultiTarget_NBA_v1_{target_name}.json")
                multi_models[target_name] = model
                
                # Load calibrator if exists
                try:
                    calibrator = joblib.load(f"Models/XGBoost_Models/MultiTarget_NBA_v1_{target_name}_calibrator.pkl")
                    multi_models[f'{target_name}_calibrator'] = calibrator
                except:
                    pass
            
            if not self.feature_cols:
                self.feature_cols = joblib.load("Models/XGBoost_Models/MultiTarget_NBA_v1_features.pkl")
            
            self.models['multi_target'] = {
                'models': multi_models,
                'metadata': metadata,
                'type': 'multi_target'
            }
            
            print("Multi-target models loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load multi-target models: {e}")
            raise
    
    def load_advanced_xgboost(self):
        """Load advanced XGBoost model"""
        try:
            model = xgb.Booster()
            model.load_model("Models/XGBoost_Models/XGB_ML_Advanced_v1.json")
            calibrator = joblib.load("Models/XGBoost_Models/XGB_ML_Advanced_v1_calibrator.pkl")
            
            if not self.feature_cols:
                self.feature_cols = joblib.load("Models/XGBoost_Models/XGB_ML_Advanced_v1_features.pkl")
            
            self.models['advanced_xgb'] = {
                'model': model,
                'calibrator': calibrator,
                'type': 'classification'
            }
            
            print("Advanced XGBoost model loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load advanced XGBoost model: {e}")
            raise
    
    def load_original_models(self):
        """Load original models as fallback"""
        try:
            # Load original XGBoost
            xgb_ml = xgb.Booster()
            xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.7%_ML-4.json')
            
            xgb_uo = xgb.Booster()
            xgb_uo.load_model('Models/XGBoost_Models/XGBoost_53.7%_UO-9.json')
            
            self.models['original'] = {
                'ml_model': xgb_ml,
                'uo_model': xgb_uo,
                'type': 'original'
            }
            
            print("Original models loaded as fallback!")
            
        except Exception as e:
            print(f"Failed to load original models: {e}")
    
    def make_ensemble_prediction(self, game_features):
        """Make prediction using ensemble model"""
        if 'ensemble' not in self.models:
            return None
            
        ensemble_info = self.models['ensemble']
        base_models = ensemble_info['base_models']
        meta_model = ensemble_info['meta_model']
        
        # Prepare features
        if isinstance(game_features, pd.DataFrame):
            X = game_features[self.feature_cols].values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        # Get base model predictions
        base_predictions = np.zeros((1, len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            if name in ['neural_network', 'mlp']:
                scaler = model['scaler']
                actual_model = model['model']
                X_scaled = scaler.transform(X)
                
                if name == 'neural_network':
                    base_predictions[0, i] = actual_model.predict(X_scaled)[0, 0]
                else:  # MLP
                    base_predictions[0, i] = actual_model.predict_proba(X_scaled)[0, 1]
            else:
                base_predictions[0, i] = model.predict_proba(X)[0, 1]
        
        # Get ensemble prediction
        ensemble_prob = meta_model.predict_proba(base_predictions)[0, 1]
        
        return {
            'probability': ensemble_prob,
            'prediction': int(ensemble_prob >= 0.5),
            'confidence': abs(ensemble_prob - 0.5) * 2,
            'base_predictions': dict(zip(base_models.keys(), base_predictions[0]))
        }
    
    def make_multi_target_predictions(self, game_features, ou_line=None):
        """Make multi-target predictions"""
        if 'multi_target' not in self.models:
            return None
            
        multi_info = self.models['multi_target']
        models = multi_info['models']
        
        # Prepare features
        if isinstance(game_features, pd.DataFrame):
            if 'OU' in game_features.columns and ou_line is not None:
                game_features = game_features.copy()
                game_features['OU'] = ou_line
            X = game_features[self.feature_cols].values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        dmatrix = xgb.DMatrix(X)
        predictions = {}
        
        for name, model in models.items():
            if '_calibrator' in name:
                continue
                
            try:
                pred = model.predict(dmatrix)[0]
                predictions[name] = pred
                
                # Use calibrator if available
                calibrator_name = f'{name}_calibrator'
                if calibrator_name in models:
                    if name in ['win_loss', 'ou_result']:
                        # Convert to probability for classification tasks
                        if isinstance(pred, (list, np.ndarray)) and len(pred) > 1:
                            predictions[f'{name}_calibrated'] = pred[1] if len(pred) > 1 else pred[0]
                        else:
                            predictions[f'{name}_calibrated'] = pred
                            
            except Exception as e:
                print(f"Error predicting {name}: {e}")
        
        return predictions
    
    def calculate_betting_edge(self, model_prob, odds):
        """Calculate betting edge and Kelly Criterion"""
        if not odds or odds == 0:
            return {'edge': 0, 'kelly': 0, 'expected_value': 0}
            
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Calculate implied probability (with vig)
        implied_prob = 1 / decimal_odds
        
        # Calculate edge
        edge = model_prob - implied_prob
        
        # Expected value
        if model_prob > implied_prob:
            expected_value = (model_prob * (decimal_odds - 1)) - ((1 - model_prob) * 1)
        else:
            expected_value = 0
        
        # Kelly Criterion
        if edge > 0:
            kelly_fraction = edge / (decimal_odds - 1)
            kelly_percentage = max(0, min(25, kelly_fraction * 100))  # Cap at 25%
        else:
            kelly_percentage = 0
        
        return {
            'edge': edge,
            'kelly': kelly_percentage,
            'expected_value': expected_value,
            'implied_probability': implied_prob
        }
    
    def run_comprehensive_prediction(self, data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion=True):
        """Run comprehensive predictions using all available models"""
        print("=" * 60)
        print("COMPREHENSIVE NBA PREDICTIONS")
        print("=" * 60)
        
        for i, game in enumerate(games):
            home_team = game[0]
            away_team = game[1]
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{home_team}{Style.RESET_ALL} vs {Fore.RED}{away_team}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # Prepare game features
            if isinstance(data, np.ndarray):
                game_features = data[i] if len(data) > i else data[0]
            else:
                game_features = frame_ml.iloc[i] if len(frame_ml) > i else frame_ml.iloc[0]
            
            # Ensemble prediction
            ensemble_pred = self.make_ensemble_prediction(game_features)
            if ensemble_pred:
                prob = ensemble_pred['probability']
                confidence = ensemble_pred['confidence']
                
                winner = home_team if prob > 0.5 else away_team
                winner_prob = prob if prob > 0.5 else (1 - prob)
                
                print(f"\n{Fore.MAGENTA}🏆 ENSEMBLE PREDICTION:{Style.RESET_ALL}")
                print(f"   Winner: {Fore.GREEN if prob > 0.5 else Fore.RED}{winner}{Style.RESET_ALL} ({winner_prob:.1%})")
                print(f"   Confidence: {Fore.YELLOW}{confidence:.1%}{Style.RESET_ALL}")
                
                # Show base model agreement
                base_preds = ensemble_pred['base_predictions']
                agreement = sum(1 for p in base_preds.values() if (p > 0.5) == (prob > 0.5))
                print(f"   Model Agreement: {agreement}/{len(base_preds)} models")
            
            # Multi-target predictions
            ou_line = todays_games_uo[i] if i < len(todays_games_uo) else None
            multi_preds = self.make_multi_target_predictions(game_features, ou_line)
            if multi_preds:
                print(f"\n{Fore.BLUE}📊 MULTI-TARGET PREDICTIONS:{Style.RESET_ALL}")
                
                # Total points prediction
                if 'total_points' in multi_preds:
                    total_pred = multi_preds['total_points']
                    print(f"   Total Points: {total_pred:.1f}")
                    
                    if ou_line:
                        ou_recommendation = "OVER" if total_pred > ou_line else "UNDER"
                        ou_edge = abs(total_pred - ou_line)
                        print(f"   O/U Recommendation: {Fore.BLUE if ou_recommendation == 'OVER' else Fore.MAGENTA}{ou_recommendation} {ou_line}{Style.RESET_ALL} (Edge: {ou_edge:.1f})")
                
                # Point margin
                if 'point_margin' in multi_preds:
                    margin = multi_preds['point_margin']
                    print(f"   Predicted Margin: {margin:+.1f} points")
                
                # Individual team scores
                if 'home_score' in multi_preds and 'away_score' in multi_preds:
                    home_score = multi_preds['home_score']
                    away_score = multi_preds['away_score']
                    print(f"   Score Prediction: {home_team} {home_score:.0f} - {away_team} {away_score:.0f}")
                
                # Quarter/Half predictions
                if 'first_half_total' in multi_preds:
                    fh_total = multi_preds['first_half_total']
                    print(f"   First Half Total: {fh_total:.1f}")
                
                if 'first_quarter_total' in multi_preds:
                    q1_total = multi_preds['first_quarter_total']
                    print(f"   First Quarter Total: {q1_total:.1f}")
            
            # Betting analysis
            if kelly_criterion and i < len(home_team_odds) and i < len(away_team_odds):
                print(f"\n{Fore.YELLOW}💰 BETTING ANALYSIS:{Style.RESET_ALL}")
                
                home_odds = home_team_odds[i]
                away_odds = away_team_odds[i]
                
                if ensemble_pred:
                    home_prob = ensemble_pred['probability']
                    away_prob = 1 - home_prob
                    
                    # Home team analysis
                    if home_odds:
                        home_analysis = self.calculate_betting_edge(home_prob, int(home_odds))
                        edge_color = Fore.GREEN if home_analysis['edge'] > 0 else Fore.RED
                        print(f"   {home_team}:")
                        print(f"     Model Probability: {home_prob:.1%}")
                        print(f"     Betting Edge: {edge_color}{home_analysis['edge']:+.1%}{Style.RESET_ALL}")
                        print(f"     Expected Value: {edge_color}{home_analysis['expected_value']:+.3f}{Style.RESET_ALL}")
                        if home_analysis['kelly'] > 0:
                            print(f"     Kelly Bet: {Fore.GREEN}{home_analysis['kelly']:.1f}% of bankroll{Style.RESET_ALL}")
                    
                    # Away team analysis  
                    if away_odds:
                        away_analysis = self.calculate_betting_edge(away_prob, int(away_odds))
                        edge_color = Fore.GREEN if away_analysis['edge'] > 0 else Fore.RED
                        print(f"   {away_team}:")
                        print(f"     Model Probability: {away_prob:.1%}")
                        print(f"     Betting Edge: {edge_color}{away_analysis['edge']:+.1%}{Style.RESET_ALL}")
                        print(f"     Expected Value: {edge_color}{away_analysis['expected_value']:+.3f}{Style.RESET_ALL}")
                        if away_analysis['kelly'] > 0:
                            print(f"     Kelly Bet: {Fore.GREEN}{away_analysis['kelly']:.1f}% of bankroll{Style.RESET_ALL}")
                    
                    # Best bet recommendation
                    best_bets = []
                    if home_odds and home_analysis['edge'] > 0.02:  # 2% edge threshold
                        best_bets.append((home_team, home_analysis['edge'], home_analysis['kelly']))
                    if away_odds and away_analysis['edge'] > 0.02:
                        best_bets.append((away_team, away_analysis['edge'], away_analysis['kelly']))
                    
                    if best_bets:
                        best_bet = max(best_bets, key=lambda x: x[1])
                        print(f"\n   {Fore.GREEN}⭐ RECOMMENDED BET: {best_bet[0]} ({best_bet[1]:+.1%} edge, {best_bet[2]:.1f}% Kelly){Style.RESET_ALL}")
                    else:
                        print(f"\n   {Fore.YELLOW}⚠️  NO STRONG BETTING OPPORTUNITIES{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        deinit()

# Global instance for backward compatibility
advanced_runner = AdvancedPredictionRunner()

def advanced_prediction_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion=True):
    """Main function for advanced predictions"""
    advanced_runner.run_comprehensive_prediction(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion)

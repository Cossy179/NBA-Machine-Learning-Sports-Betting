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
    
    def make_advanced_ensemble_prediction(self, game_features):
        """Make prediction using advanced ensemble model with sophisticated stacking"""
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
        
        # Get base model predictions with confidence scores
        base_predictions = np.zeros((1, len(base_models)))
        base_confidences = np.zeros((1, len(base_models)))
        base_uncertainties = np.zeros((1, len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            try:
                if name in ['neural_network', 'mlp']:
                    scaler = model['scaler']
                    actual_model = model['model']
                    X_scaled = scaler.transform(X)
                    
                    if name == 'neural_network':
                        pred = actual_model.predict(X_scaled)[0, 0]
                        # Estimate uncertainty from model variance
                        uncertainty = 0.1  # Simplified uncertainty estimation
                    else:  # MLP
                        pred_proba = actual_model.predict_proba(X_scaled)[0]
                        pred = pred_proba[1]
                        uncertainty = np.std(pred_proba)  # Use std as uncertainty measure
                else:
                    pred_proba = model.predict_proba(X)[0]
                    pred = pred_proba[1]
                    uncertainty = np.std(pred_proba)
                
                base_predictions[0, i] = pred
                base_confidences[0, i] = abs(pred - 0.5) * 2  # Confidence from distance to 0.5
                base_uncertainties[0, i] = uncertainty
                
            except Exception as e:
                print(f"Error in {name} prediction: {e}")
                base_predictions[0, i] = 0.5
                base_confidences[0, i] = 0
                base_uncertainties[0, i] = 0.5
        
        # Advanced ensemble methods
        # 1. Weighted average based on confidence
        confidence_weights = base_confidences[0] / (np.sum(base_confidences[0]) + 1e-8)
        weighted_avg = np.sum(base_predictions[0] * confidence_weights)
        
        # 2. Meta-model prediction
        meta_features = np.concatenate([base_predictions[0], base_confidences[0], base_uncertainties[0]])
        meta_features = meta_features.reshape(1, -1)
        meta_prob = meta_model.predict_proba(meta_features)[0, 1]
        
        # 3. Uncertainty-weighted ensemble
        uncertainty_weights = 1 / (base_uncertainties[0] + 1e-8)
        uncertainty_weights = uncertainty_weights / np.sum(uncertainty_weights)
        uncertainty_weighted = np.sum(base_predictions[0] * uncertainty_weights)
        
        # 4. Bayesian model averaging (simplified)
        # Use model performance as prior weights
        model_weights = np.array([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])[:len(base_models)]  # Example weights
        model_weights = model_weights / np.sum(model_weights)
        bayesian_avg = np.sum(base_predictions[0] * model_weights)
        
        # 5. Dynamic ensemble selection
        # Select best performing models based on recent accuracy
        recent_performance = np.array([0.8, 0.75, 0.82, 0.78, 0.85, 0.77])[:len(base_models)]  # Example performance
        top_models = np.argsort(recent_performance)[-3:]  # Top 3 models
        dynamic_weights = np.zeros(len(base_models))
        dynamic_weights[top_models] = recent_performance[top_models]
        dynamic_weights = dynamic_weights / np.sum(dynamic_weights)
        dynamic_avg = np.sum(base_predictions[0] * dynamic_weights)
        
        # Combine all ensemble methods
        ensemble_methods = [weighted_avg, meta_prob, uncertainty_weighted, bayesian_avg, dynamic_avg]
        ensemble_weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # Weights for different methods
        
        final_prob = np.sum([method * weight for method, weight in zip(ensemble_methods, ensemble_weights)])
        
        # Calculate ensemble confidence and uncertainty
        ensemble_confidence = abs(final_prob - 0.5) * 2
        ensemble_uncertainty = np.std(ensemble_methods)
        
        # Model agreement analysis
        agreement_threshold = 0.1
        agreeing_models = np.sum(np.abs(base_predictions[0] - final_prob) < agreement_threshold)
        agreement_ratio = agreeing_models / len(base_models)
        
        # Prediction reliability score
        reliability_score = (ensemble_confidence + (1 - ensemble_uncertainty) + agreement_ratio) / 3
        
        return {
            'probability': final_prob,
            'prediction': int(final_prob >= 0.5),
            'confidence': ensemble_confidence,
            'uncertainty': ensemble_uncertainty,
            'reliability_score': reliability_score,
            'agreement_ratio': agreement_ratio,
            'base_predictions': dict(zip(base_models.keys(), base_predictions[0])),
            'base_confidences': dict(zip(base_models.keys(), base_confidences[0])),
            'ensemble_methods': {
                'weighted_avg': weighted_avg,
                'meta_model': meta_prob,
                'uncertainty_weighted': uncertainty_weighted,
                'bayesian_avg': bayesian_avg,
                'dynamic_avg': dynamic_avg
            }
        }
    
    def make_ensemble_prediction(self, game_features):
        """Make prediction using ensemble model (backward compatibility)"""
        return self.make_advanced_ensemble_prediction(game_features)
    
    def make_advanced_multi_target_predictions(self, game_features, ou_line=None):
        """Make advanced multi-target predictions with uncertainty quantification"""
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
        uncertainties = {}
        confidences = {}
        
        for name, model in models.items():
            if '_calibrator' in name:
                continue
                
            try:
                # Get prediction
                pred = model.predict(dmatrix)[0]
                predictions[name] = pred
                
                # Calculate uncertainty using prediction variance
                # For XGBoost, we can use leaf indices to estimate uncertainty
                leaf_indices = model.predict(dmatrix, pred_leaf=True)[0]
                uncertainty = np.std(leaf_indices) / len(leaf_indices)  # Normalized uncertainty
                uncertainties[name] = uncertainty
                
                # Calculate confidence based on prediction strength
                if name in ['win_loss', 'ou_result']:
                    # Classification confidence
                    if isinstance(pred, (list, np.ndarray)) and len(pred) > 1:
                        prob = pred[1] if len(pred) > 1 else pred[0]
                        confidence = abs(prob - 0.5) * 2
                    else:
                        confidence = abs(pred - 0.5) * 2
                else:
                    # Regression confidence (based on prediction magnitude)
                    confidence = min(1.0, abs(pred) / 50)  # Normalize by expected range
                
                confidences[name] = confidence
                
                # Use calibrator if available
                calibrator_name = f'{name}_calibrator'
                if calibrator_name in models:
                    calibrator = models[calibrator_name]
                    try:
                        if name in ['win_loss', 'ou_result']:
                            # Classification calibration
                            if isinstance(pred, (list, np.ndarray)) and len(pred) > 1:
                                calibrated_prob = calibrator.predict_proba([[pred[1]]])[0, 1]
                            else:
                                calibrated_prob = calibrator.predict_proba([[pred]])[0, 1]
                            predictions[f'{name}_calibrated'] = calibrated_prob
                            confidences[f'{name}_calibrated'] = abs(calibrated_prob - 0.5) * 2
                        else:
                            # Regression calibration
                            calibrated_pred = calibrator.predict([[pred]])[0]
                            predictions[f'{name}_calibrated'] = calibrated_pred
                    except Exception as e:
                        print(f"Error calibrating {name}: {e}")
                        predictions[f'{name}_calibrated'] = pred
                        confidences[f'{name}_calibrated'] = confidence
                            
            except Exception as e:
                print(f"Error predicting {name}: {e}")
                predictions[name] = 0
                uncertainties[name] = 1.0
                confidences[name] = 0
        
        # Add ensemble predictions for key targets
        key_targets = ['win_loss', 'ou_result', 'total_points', 'point_margin']
        for target in key_targets:
            if target in predictions:
                # Create ensemble prediction using multiple models
                ensemble_pred = self._create_multi_target_ensemble(predictions, target, confidences)
                if ensemble_pred is not None:
                    predictions[f'{target}_ensemble'] = ensemble_pred
        
        # Add prediction quality metrics
        predictions['_uncertainties'] = uncertainties
        predictions['_confidences'] = confidences
        predictions['_overall_confidence'] = np.mean(list(confidences.values()))
        predictions['_overall_uncertainty'] = np.mean(list(uncertainties.values()))
        
        return predictions
    
    def _create_multi_target_ensemble(self, predictions, target, confidences):
        """Create ensemble prediction for multi-target models"""
        target_variations = [f'{target}_calibrated', f'{target}_ensemble']
        available_predictions = [predictions[target]]
        
        for variation in target_variations:
            if variation in predictions:
                available_predictions.append(predictions[variation])
        
        if len(available_predictions) > 1:
            # Weight by confidence
            weights = [confidences.get(target, 0.5)]
            for variation in target_variations:
                if variation in predictions:
                    weights.append(confidences.get(variation, 0.5))
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            return np.sum([pred * weight for pred, weight in zip(available_predictions, weights)])
        
        return None
    
    def make_multi_target_predictions(self, game_features, ou_line=None):
        """Make multi-target predictions (backward compatibility)"""
        return self.make_advanced_multi_target_predictions(game_features, ou_line)
    
    def calculate_advanced_betting_edge(self, model_prob, odds, confidence=0.5, uncertainty=0.1):
        """Calculate advanced betting edge with confidence and uncertainty adjustments"""
        if not odds or odds == 0:
            return {'edge': 0, 'kelly': 0, 'expected_value': 0, 'confidence_adjusted_edge': 0}
            
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Calculate implied probability (with vig)
        implied_prob = 1 / decimal_odds
        
        # Basic edge calculation
        basic_edge = model_prob - implied_prob
        
        # Confidence-adjusted probability
        # Higher confidence increases the effective probability
        confidence_factor = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        confidence_adjusted_prob = model_prob * confidence_factor + implied_prob * (1 - confidence_factor)
        
        # Uncertainty adjustment
        # Higher uncertainty reduces the effective edge
        uncertainty_factor = max(0.1, 1 - uncertainty)  # Range: 0.1 to 1.0
        uncertainty_adjusted_prob = confidence_adjusted_prob * uncertainty_factor + implied_prob * (1 - uncertainty_factor)
        
        # Final edge calculation
        edge = uncertainty_adjusted_prob - implied_prob
        confidence_adjusted_edge = confidence_adjusted_prob - implied_prob
        
        # Expected value calculations
        basic_ev = (model_prob * (decimal_odds - 1)) - ((1 - model_prob) * 1) if model_prob > implied_prob else 0
        adjusted_ev = (uncertainty_adjusted_prob * (decimal_odds - 1)) - ((1 - uncertainty_adjusted_prob) * 1) if uncertainty_adjusted_prob > implied_prob else 0
        
        # Advanced Kelly Criterion with confidence and uncertainty
        if edge > 0:
            # Base Kelly fraction
            base_kelly = edge / (decimal_odds - 1)
            
            # Adjust for confidence (higher confidence = higher bet)
            confidence_kelly = base_kelly * confidence_factor
            
            # Adjust for uncertainty (higher uncertainty = lower bet)
            uncertainty_kelly = confidence_kelly * uncertainty_factor
            
            # Risk management: cap based on confidence and uncertainty
            max_kelly = min(0.25, 0.1 + (confidence * 0.15))  # Max 10-25% based on confidence
            kelly_percentage = max(0, min(max_kelly, uncertainty_kelly)) * 100
        else:
            kelly_percentage = 0
        
        # Value rating (0-100 scale)
        value_rating = max(0, min(100, (edge / implied_prob) * 100)) if implied_prob > 0 else 0
        
        # Risk-adjusted return
        risk_adjusted_return = adjusted_ev * confidence_factor * uncertainty_factor
        
        return {
            'edge': edge,
            'confidence_adjusted_edge': confidence_adjusted_edge,
            'kelly': kelly_percentage,
            'expected_value': adjusted_ev,
            'basic_expected_value': basic_ev,
            'implied_probability': implied_prob,
            'confidence_adjusted_probability': confidence_adjusted_prob,
            'uncertainty_adjusted_probability': uncertainty_adjusted_prob,
            'value_rating': value_rating,
            'risk_adjusted_return': risk_adjusted_return,
            'confidence_factor': confidence_factor,
            'uncertainty_factor': uncertainty_factor
        }
    
    def calculate_betting_edge(self, model_prob, odds):
        """Calculate betting edge and Kelly Criterion (backward compatibility)"""
        return self.calculate_advanced_betting_edge(model_prob, odds)
    
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
            
            # Advanced ensemble prediction
            ensemble_pred = self.make_advanced_ensemble_prediction(game_features)
            if ensemble_pred:
                prob = ensemble_pred['probability']
                confidence = ensemble_pred['confidence']
                uncertainty = ensemble_pred.get('uncertainty', 0.1)
                reliability = ensemble_pred.get('reliability_score', 0.5)
                agreement = ensemble_pred.get('agreement_ratio', 0.5)
                
                winner = home_team if prob > 0.5 else away_team
                winner_prob = prob if prob > 0.5 else (1 - prob)
                
                print(f"\n{Fore.MAGENTA}🏆 ADVANCED ENSEMBLE PREDICTION:{Style.RESET_ALL}")
                print(f"   Winner: {Fore.GREEN if prob > 0.5 else Fore.RED}{winner}{Style.RESET_ALL} ({winner_prob:.1%})")
                print(f"   Confidence: {Fore.YELLOW}{confidence:.1%}{Style.RESET_ALL}")
                print(f"   Uncertainty: {Fore.CYAN}{uncertainty:.3f}{Style.RESET_ALL}")
                print(f"   Reliability: {Fore.BLUE}{reliability:.1%}{Style.RESET_ALL}")
                print(f"   Model Agreement: {Fore.GREEN}{agreement:.1%}{Style.RESET_ALL}")
                
                # Show base model predictions with confidence
                base_preds = ensemble_pred['base_predictions']
                base_confidences = ensemble_pred.get('base_confidences', {})
                print(f"   Base Model Predictions:")
                for name, pred in base_preds.items():
                    conf = base_confidences.get(name, 0)
                    print(f"     {name}: {pred:.3f} (conf: {conf:.1%})")
                
                # Show ensemble methods
                methods = ensemble_pred.get('ensemble_methods', {})
                if methods:
                    print(f"   Ensemble Methods:")
                    for method, value in methods.items():
                        print(f"     {method}: {value:.3f}")
            
            # Advanced multi-target predictions
            ou_line = todays_games_uo[i] if i < len(todays_games_uo) else None
            multi_preds = self.make_advanced_multi_target_predictions(game_features, ou_line)
            if multi_preds:
                print(f"\n{Fore.BLUE}📊 ADVANCED MULTI-TARGET PREDICTIONS:{Style.RESET_ALL}")
                
                # Overall confidence and uncertainty
                overall_conf = multi_preds.get('_overall_confidence', 0.5)
                overall_uncertainty = multi_preds.get('_overall_uncertainty', 0.1)
                print(f"   Overall Confidence: {Fore.YELLOW}{overall_conf:.1%}{Style.RESET_ALL}")
                print(f"   Overall Uncertainty: {Fore.CYAN}{overall_uncertainty:.3f}{Style.RESET_ALL}")
                
                # Total points prediction with uncertainty
                if 'total_points' in multi_preds:
                    total_pred = multi_preds['total_points']
                    total_uncertainty = multi_preds.get('_uncertainties', {}).get('total_points', 0.1)
                    total_confidence = multi_preds.get('_confidences', {}).get('total_points', 0.5)
                    
                    print(f"   Total Points: {total_pred:.1f} ± {total_uncertainty:.1f} (conf: {total_confidence:.1%})")
                    
                    if ou_line:
                        ou_recommendation = "OVER" if total_pred > ou_line else "UNDER"
                        ou_edge = abs(total_pred - ou_line)
                        confidence_level = "HIGH" if total_confidence > 0.7 else "MEDIUM" if total_confidence > 0.5 else "LOW"
                        print(f"   O/U Recommendation: {Fore.BLUE if ou_recommendation == 'OVER' else Fore.MAGENTA}{ou_recommendation} {ou_line}{Style.RESET_ALL} (Edge: {ou_edge:.1f}, {confidence_level})")
                
                # Point margin with confidence
                if 'point_margin' in multi_preds:
                    margin = multi_preds['point_margin']
                    margin_confidence = multi_preds.get('_confidences', {}).get('point_margin', 0.5)
                    print(f"   Predicted Margin: {margin:+.1f} points (conf: {margin_confidence:.1%})")
                
                # Individual team scores
                if 'home_score' in multi_preds and 'away_score' in multi_preds:
                    home_score = multi_preds['home_score']
                    away_score = multi_preds['away_score']
                    home_conf = multi_preds.get('_confidences', {}).get('home_score', 0.5)
                    away_conf = multi_preds.get('_confidences', {}).get('away_score', 0.5)
                    print(f"   Score Prediction: {home_team} {home_score:.0f} (conf: {home_conf:.1%}) - {away_team} {away_score:.0f} (conf: {away_conf:.1%})")
                
                # Quarter/Half predictions
                if 'first_half_total' in multi_preds:
                    fh_total = multi_preds['first_half_total']
                    fh_conf = multi_preds.get('_confidences', {}).get('first_half_total', 0.5)
                    print(f"   First Half Total: {fh_total:.1f} (conf: {fh_conf:.1%})")
                
                if 'first_quarter_total' in multi_preds:
                    q1_total = multi_preds['first_quarter_total']
                    q1_conf = multi_preds.get('_confidences', {}).get('first_quarter_total', 0.5)
                    print(f"   First Quarter Total: {q1_total:.1f} (conf: {q1_conf:.1%})")
                
                # Show ensemble predictions if available
                ensemble_targets = [key for key in multi_preds.keys() if key.endswith('_ensemble')]
                if ensemble_targets:
                    print(f"   Ensemble Predictions:")
                    for target in ensemble_targets:
                        base_target = target.replace('_ensemble', '')
                        ensemble_pred = multi_preds[target]
                        base_pred = multi_preds.get(base_target, 0)
                        improvement = abs(ensemble_pred - base_pred)
                        print(f"     {base_target}: {ensemble_pred:.3f} (vs base: {base_pred:.3f}, Δ: {improvement:.3f})")
            
            # Advanced betting analysis
            if kelly_criterion and i < len(home_team_odds) and i < len(away_team_odds):
                print(f"\n{Fore.YELLOW}💰 ADVANCED BETTING ANALYSIS:{Style.RESET_ALL}")
                
                home_odds = home_team_odds[i]
                away_odds = away_team_odds[i]
                
                if ensemble_pred:
                    home_prob = ensemble_pred['probability']
                    away_prob = 1 - home_prob
                    confidence = ensemble_pred.get('confidence', 0.5)
                    uncertainty = ensemble_pred.get('uncertainty', 0.1)
                    reliability = ensemble_pred.get('reliability_score', 0.5)
                    
                    # Home team analysis
                    if home_odds:
                        home_analysis = self.calculate_advanced_betting_edge(home_prob, int(home_odds), confidence, uncertainty)
                        edge_color = Fore.GREEN if home_analysis['edge'] > 0 else Fore.RED
                        print(f"   {home_team}:")
                        print(f"     Model Probability: {home_prob:.1%}")
                        print(f"     Confidence-Adjusted Prob: {home_analysis['confidence_adjusted_probability']:.1%}")
                        print(f"     Uncertainty-Adjusted Prob: {home_analysis['uncertainty_adjusted_probability']:.1%}")
                        print(f"     Basic Edge: {edge_color}{home_analysis['basic_expected_value']:+.3f}{Style.RESET_ALL}")
                        print(f"     Adjusted Edge: {edge_color}{home_analysis['edge']:+.1%}{Style.RESET_ALL}")
                        print(f"     Expected Value: {edge_color}{home_analysis['expected_value']:+.3f}{Style.RESET_ALL}")
                        print(f"     Value Rating: {Fore.CYAN}{home_analysis['value_rating']:.0f}/100{Style.RESET_ALL}")
                        print(f"     Risk-Adjusted Return: {Fore.BLUE}{home_analysis['risk_adjusted_return']:+.3f}{Style.RESET_ALL}")
                        if home_analysis['kelly'] > 0:
                            print(f"     Kelly Bet: {Fore.GREEN}{home_analysis['kelly']:.1f}% of bankroll{Style.RESET_ALL}")
                    
                    # Away team analysis  
                    if away_odds:
                        away_analysis = self.calculate_advanced_betting_edge(away_prob, int(away_odds), confidence, uncertainty)
                        edge_color = Fore.GREEN if away_analysis['edge'] > 0 else Fore.RED
                        print(f"   {away_team}:")
                        print(f"     Model Probability: {away_prob:.1%}")
                        print(f"     Confidence-Adjusted Prob: {away_analysis['confidence_adjusted_probability']:.1%}")
                        print(f"     Uncertainty-Adjusted Prob: {away_analysis['uncertainty_adjusted_probability']:.1%}")
                        print(f"     Basic Edge: {edge_color}{away_analysis['basic_expected_value']:+.3f}{Style.RESET_ALL}")
                        print(f"     Adjusted Edge: {edge_color}{away_analysis['edge']:+.1%}{Style.RESET_ALL}")
                        print(f"     Expected Value: {edge_color}{away_analysis['expected_value']:+.3f}{Style.RESET_ALL}")
                        print(f"     Value Rating: {Fore.CYAN}{away_analysis['value_rating']:.0f}/100{Style.RESET_ALL}")
                        print(f"     Risk-Adjusted Return: {Fore.BLUE}{away_analysis['risk_adjusted_return']:+.3f}{Style.RESET_ALL}")
                        if away_analysis['kelly'] > 0:
                            print(f"     Kelly Bet: {Fore.GREEN}{away_analysis['kelly']:.1f}% of bankroll{Style.RESET_ALL}")
                    
                    # Advanced bet recommendations
                    best_bets = []
                    if home_odds and home_analysis['edge'] > 0.01:  # Lower threshold for advanced analysis
                        bet_quality = (home_analysis['value_rating'] + reliability * 100) / 2
                        best_bets.append((home_team, home_analysis['edge'], home_analysis['kelly'], 
                                        home_analysis['value_rating'], bet_quality))
                    if away_odds and away_analysis['edge'] > 0.01:
                        bet_quality = (away_analysis['value_rating'] + reliability * 100) / 2
                        best_bets.append((away_team, away_analysis['edge'], away_analysis['kelly'], 
                                        away_analysis['value_rating'], bet_quality))
                    
                    if best_bets:
                        # Sort by bet quality (combination of value rating and reliability)
                        best_bet = max(best_bets, key=lambda x: x[4])
                        print(f"\n   {Fore.GREEN}⭐ RECOMMENDED BET: {best_bet[0]}{Style.RESET_ALL}")
                        print(f"     Edge: {best_bet[1]:+.1%}")
                        print(f"     Kelly: {best_bet[2]:.1f}% of bankroll")
                        print(f"     Value Rating: {best_bet[3]:.0f}/100")
                        print(f"     Bet Quality: {best_bet[4]:.0f}/100")
                        
                        # Show all viable bets
                        if len(best_bets) > 1:
                            print(f"   {Fore.CYAN}Other Viable Bets:{Style.RESET_ALL}")
                            for bet in sorted(best_bets[1:], key=lambda x: x[4], reverse=True):
                                print(f"     {bet[0]}: Edge {bet[1]:+.1%}, Kelly {bet[2]:.1f}%, Quality {bet[4]:.0f}/100")
                    else:
                        print(f"\n   {Fore.YELLOW}⚠️  NO STRONG BETTING OPPORTUNITIES{Style.RESET_ALL}")
                        print(f"   (Minimum edge threshold: 1%, Reliability: {reliability:.1%})")
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        deinit()

# Global instance for backward compatibility
advanced_runner = AdvancedPredictionRunner()

def advanced_prediction_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion=True):
    """Main function for advanced predictions"""
    advanced_runner.run_comprehensive_prediction(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion)

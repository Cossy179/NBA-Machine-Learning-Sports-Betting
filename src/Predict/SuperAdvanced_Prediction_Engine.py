"""
Super Advanced Prediction Engine
Integrates all advanced models with sentiment analysis for maximum accuracy.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

# Add Utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from SentimentAnalysis import NBASentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')


class SuperAdvancedPredictionEngine:
    def __init__(self):
        self.models = {}
        self.calibrators = {}
        self.ensemble_weights = {}
        self.selected_features = None
        self.sentiment_analyzer = NBASentimentAnalyzer()
        self.load_all_models()
        
    def load_all_models(self):
        """Load all super advanced models"""
        model_dir = "Models/XGBoost_Models"
        model_prefix = "SuperAdvanced_XGB_v1"
        
        try:
            # Load XGBoost DART
            xgb_model = xgb.Booster()
            xgb_model.load_model(f"{model_dir}/{model_prefix}_xgb_dart.json")
            self.models['xgb_dart'] = xgb_model
            print("✅ Loaded XGBoost DART model")
            
        except Exception as e:
            print(f"⚠️  Could not load XGBoost DART: {e}")
        
        try:
            # Load LightGBM
            lgb_model = lgb.Booster(model_file=f"{model_dir}/{model_prefix}_lightgbm.txt")
            self.models['lightgbm'] = lgb_model
            print("✅ Loaded LightGBM model")
            
        except Exception as e:
            print(f"⚠️ Could not load LightGBM: {e}")
        
        if CATBOOST_AVAILABLE:
            try:
                # Load CatBoost
                cb_model = cb.CatBoostClassifier()
                cb_model.load_model(f"{model_dir}/{model_prefix}_catboost.cbm")
                self.models['catboost'] = cb_model
                print("✅ Loaded CatBoost model")
                
            except Exception as e:
                print(f"⚠️ Could not load CatBoost: {e}")
        
        try:
            # Load calibrators
            self.calibrators = joblib.load(f"{model_dir}/{model_prefix}_calibrators.pkl")
            print("✅ Loaded calibrators")
            
        except Exception as e:
            print(f"⚠️ Could not load calibrators: {e}")
        
        try:
            # Load ensemble weights
            self.ensemble_weights = joblib.load(f"{model_dir}/{model_prefix}_weights.pkl")
            print("✅ Loaded ensemble weights")
            
        except Exception as e:
            print(f"⚠️ Could not load ensemble weights: {e}")
            # Default weights
            n_models = len(self.models)
            if n_models > 0:
                self.ensemble_weights = {name: 1.0/n_models for name in self.models.keys()}
        
        try:
            # Load selected features
            self.selected_features = joblib.load(f"{model_dir}/{model_prefix}_features.pkl")
            print(f"✅ Loaded {len(self.selected_features)} selected features")
            
        except Exception as e:
            print(f"⚠️ Could not load selected features: {e}")
    
    def prepare_features(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Select only the features the model was trained on
        if self.selected_features:
            # Get available features
            available_features = [f for f in self.selected_features if f in game_data.columns]
            missing_features = [f for f in self.selected_features if f not in game_data.columns]
            
            if missing_features:
                print(f"⚠️ Missing {len(missing_features)} features, filling with zeros")
                # Add missing features with zeros
                for feat in missing_features:
                    game_data[feat] = 0
            
            # Select features in the correct order
            game_data = game_data[self.selected_features]
        
        # Fill any NaN values
        game_data = game_data.fillna(0)
        
        return game_data
    
    def predict_with_sentiment(self, game_data: pd.DataFrame, home_team: str, 
                              away_team: str, use_sentiment: bool = True) -> Dict:
        """Make prediction with optional sentiment analysis"""
        
        if len(self.models) == 0:
            raise ValueError("No models loaded!")
        
        # Prepare features
        X = self.prepare_features(game_data)
        
        # Get base predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'xgb_dart':
                    dtest = xgb.DMatrix(X)
                    prob = model.predict(dtest)[0]
                elif model_name == 'lightgbm':
                    prob = model.predict(X)[0]
                elif model_name == 'catboost':
                    prob = model.predict_proba(X)[0, 1]
                
                # Apply calibration if available
                if model_name in self.calibrators:
                    prob = self.calibrators[model_name].predict([prob])[0]
                
                predictions[model_name] = prob
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions!")
        
        # Calculate weighted ensemble prediction
        weighted_prob = 0
        total_weight = 0
        
        for model_name, prob in predictions.items():
            weight = self.ensemble_weights.get(model_name, 1.0 / len(predictions))
            weighted_prob += prob * weight
            total_weight += weight
        
        base_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        
        # Base prediction result
        result = {
            'home_win_probability': float(base_prob),
            'away_win_probability': float(1 - base_prob),
            'prediction': 'HOME' if base_prob > 0.5 else 'AWAY',
            'confidence': abs(base_prob - 0.5) * 2,
            'base_probability': float(base_prob),
            'model_predictions': predictions,
            'ensemble_weights': self.ensemble_weights
        }
        
        # Add sentiment analysis if requested
        if use_sentiment:
            try:
                print(f"\n🎭 Getting sentiment analysis for {home_team} vs {away_team}...")
                sentiment = self.sentiment_analyzer.get_game_sentiment(home_team, away_team)
                
                # Adjust prediction with sentiment
                adjusted = self.sentiment_analyzer.adjust_prediction_with_sentiment(result, sentiment)
                
                result.update(adjusted)
                result['sentiment_data'] = sentiment
                result['sentiment_enabled'] = True
                
                # Calculate final confidence incorporating sentiment
                sentiment_confidence_boost = abs(sentiment['sentiment_differential']) * 0.05
                result['final_confidence'] = min(1.0, result['confidence'] + sentiment_confidence_boost)
                
            except Exception as e:
                print(f"⚠️ Sentiment analysis failed: {e}")
                result['sentiment_enabled'] = False
                result['final_confidence'] = result['confidence']
        else:
            result['sentiment_enabled'] = False
            result['final_confidence'] = result['confidence']
        
        return result
    
    def predict_game(self, game_features: pd.DataFrame, home_team: str, 
                    away_team: str, include_sentiment: bool = True) -> Dict:
        """Predict a single game with all advanced features"""
        
        print(f"\n{'='*70}")
        print(f"🏀 SUPER ADVANCED PREDICTION")
        print(f"   {home_team} vs {away_team}")
        print(f"{'='*70}")
        
        # Make prediction with sentiment
        prediction = self.predict_with_sentiment(
            game_features, home_team, away_team, use_sentiment=include_sentiment
        )
        
        # Print results
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"   Away Win Probability: {prediction['away_win_probability']:.1%}")
        print(f"   Prediction: {prediction['prediction']}")
        print(f"   Base Confidence: {prediction['confidence']:.1%}")
        
        if prediction.get('sentiment_enabled'):
            print(f"\n🎭 SENTIMENT ANALYSIS:")
            print(f"   Sentiment Differential: {prediction.get('sentiment_score', 0):.3f}")
            print(f"   Narrative: {prediction.get('narrative', 'N/A')}")
            print(f"   Final Confidence: {prediction['final_confidence']:.1%}")
            
            if prediction.get('contrarian_opportunity'):
                print(f"   💡 CONTRARIAN VALUE DETECTED!")
        
        print(f"\n🤖 MODEL BREAKDOWN:")
        for model_name, prob in prediction['model_predictions'].items():
            weight = prediction['ensemble_weights'].get(model_name, 0)
            print(f"   {model_name:15} - {prob:.3f} (weight: {weight:.3f})")
        
        print(f"{'='*70}\n")
        
        return prediction
    
    def predict_multiple_games(self, games: List[Dict], include_sentiment: bool = True) -> List[Dict]:
        """Predict multiple games"""
        results = []
        
        for game in games:
            try:
                prediction = self.predict_game(
                    game['features'],
                    game['home_team'],
                    game['away_team'],
                    include_sentiment=include_sentiment
                )
                
                prediction['game_info'] = game
                results.append(prediction)
                
            except Exception as e:
                print(f"Error predicting {game.get('home_team')} vs {game.get('away_team')}: {e}")
                continue
        
        return results


def compare_with_without_sentiment(engine: SuperAdvancedPredictionEngine, 
                                   game_features: pd.DataFrame,
                                   home_team: str, away_team: str):
    """Compare predictions with and without sentiment"""
    
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT SENTIMENT ANALYSIS")
    print("="*70)
    
    # Without sentiment
    print("\n1️⃣  WITHOUT SENTIMENT:")
    print("-" * 70)
    pred_no_sentiment = engine.predict_with_sentiment(
        game_features, home_team, away_team, use_sentiment=False
    )
    print(f"Probability: {pred_no_sentiment['home_win_probability']:.3f}")
    print(f"Confidence: {pred_no_sentiment['confidence']:.3f}")
    
    # With sentiment
    print("\n2️⃣  WITH SENTIMENT:")
    print("-" * 70)
    pred_with_sentiment = engine.predict_with_sentiment(
        game_features, home_team, away_team, use_sentiment=True
    )
    print(f"Probability: {pred_with_sentiment['home_win_probability']:.3f}")
    print(f"Confidence: {pred_with_sentiment['final_confidence']:.3f}")
    print(f"Sentiment Score: {pred_with_sentiment.get('sentiment_score', 0):.3f}")
    print(f"Narrative: {pred_with_sentiment.get('narrative', 'N/A')}")
    
    # Calculate differences
    prob_diff = pred_with_sentiment['home_win_probability'] - pred_no_sentiment['home_win_probability']
    conf_diff = pred_with_sentiment['final_confidence'] - pred_no_sentiment['confidence']
    
    print("\n📊 DIFFERENCES:")
    print(f"Probability Change: {prob_diff:+.3f}")
    print(f"Confidence Change: {conf_diff:+.3f}")
    
    if abs(prob_diff) > 0.05:
        print(f"\n⚠️  SIGNIFICANT SENTIMENT IMPACT!")
    
    print("="*70)


if __name__ == "__main__":
    # Test the prediction engine
    print("Testing Super Advanced Prediction Engine...")
    print("="*70)
    
    try:
        engine = SuperAdvancedPredictionEngine()
        
        if len(engine.models) == 0:
            print("\n⚠️  No models loaded. Please train models first using:")
            print("   py train.py --xgboost")
        else:
            print(f"\n✅ Loaded {len(engine.models)} models successfully!")
            print("Ready for predictions with sentiment analysis!")
            
    except Exception as e:
        print(f"\n❌ Error initializing engine: {e}")
        print("\nMake sure you have trained the models first:")
        print("   py train.py --xgboost")


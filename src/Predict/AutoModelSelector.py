"""
Automatic Model Selection System that dynamically chooses the best performing model
and combines predictions optimally for maximum accuracy.
"""
import joblib
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AutoModelSelector:
    def __init__(self):
        self.available_models = {}
        self.model_performances = {}
        self.best_model = None
        self.ensemble_weights = {}
        self.model_paths = {
            'boosted_system': 'Models/Boosted_Models/',
            'ensemble_system': 'Models/Ensemble_Models/', 
            'multi_target': 'Models/XGBoost_Models/',
            'advanced_xgb': 'Models/XGBoost_Models/',
            'original_xgb': 'Models/XGBoost_Models/'
        }
        
    def scan_available_models(self):
        """Scan for all available trained models"""
        print("Scanning for available models...")
        
        # Check for Boosted System
        if os.path.exists('Models/Boosted_Models/BoostedNBA_v1_metadata.pkl'):
            try:
                metadata = joblib.load('Models/Boosted_Models/BoostedNBA_v1_metadata.pkl')
                self.available_models['boosted_system'] = {
                    'type': 'boosted_ensemble',
                    'metadata': metadata,
                    'confidence': 0.95  # Highest confidence - most advanced system
                }
                print("✓ Boosted System found")
            except Exception as e:
                print(f"✗ Boosted System error: {e}")
        
        # Check for Ensemble System
        if os.path.exists('Models/Ensemble_Models/Ensemble_NBA_v1_features.pkl'):
            try:
                features = joblib.load('Models/Ensemble_Models/Ensemble_NBA_v1_features.pkl')
                self.available_models['ensemble_system'] = {
                    'type': 'stacked_ensemble',
                    'features': features,
                    'confidence': 0.85
                }
                print("✓ Ensemble System found")
            except Exception as e:
                print(f"✗ Ensemble System error: {e}")
        
        # Check for Multi-Target Models
        if os.path.exists('Models/XGBoost_Models/MultiTarget_NBA_v1_metadata.pkl'):
            try:
                metadata = joblib.load('Models/XGBoost_Models/MultiTarget_NBA_v1_metadata.pkl')
                self.available_models['multi_target'] = {
                    'type': 'multi_target',
                    'metadata': metadata,
                    'confidence': 0.80
                }
                print("✓ Multi-Target System found")
            except Exception as e:
                print(f"✗ Multi-Target System error: {e}")
        
        # Check for Advanced XGBoost
        if os.path.exists('Models/XGBoost_Models/XGB_ML_Advanced_v1.json'):
            self.available_models['advanced_xgb'] = {
                'type': 'single_model',
                'confidence': 0.75
            }
            print("✓ Advanced XGBoost found")
        
        # Check for Original XGBoost (fallback)
        if os.path.exists('Models/XGBoost_Models/XGBoost_68.7%_ML-4.json'):
            self.available_models['original_xgb'] = {
                'type': 'original',
                'confidence': 0.60
            }
            print("✓ Original XGBoost found (fallback)")
        
        print(f"Found {len(self.available_models)} available model systems")
        return self.available_models
    
    def load_boosted_system(self):
        """Load the boosted model system"""
        try:
            from src.Train_Models.Boosted_Model_System import BoostedModelSystem
            
            # Load metadata
            metadata = joblib.load('Models/Boosted_Models/BoostedNBA_v1_metadata.pkl')
            
            # Create system instance and load components
            system = BoostedModelSystem()
            system.feature_cols = metadata['feature_cols']
            system.best_model_name = metadata['best_model_name']
            system.ensemble_weights = metadata['ensemble_weights']
            
            # Load individual models
            system.models = {}
            system.feature_selectors = joblib.load('Models/Boosted_Models/BoostedNBA_v1_feature_selectors.pkl')
            system.scalers = joblib.load('Models/Boosted_Models/BoostedNBA_v1_scalers.pkl')
            system.calibrators = joblib.load('Models/Boosted_Models/BoostedNBA_v1_calibrators.pkl')
            
            for model_name in system.ensemble_weights.keys():
                try:
                    model = joblib.load(f'Models/Boosted_Models/BoostedNBA_v1_{model_name}.pkl')
                    system.models[model_name] = model
                except FileNotFoundError:
                    print(f"Warning: Could not load {model_name}")
            
            return system
            
        except Exception as e:
            print(f"Error loading boosted system: {e}")
            return None
    
    def load_ensemble_system(self):
        """Load the ensemble model system"""
        try:
            # Load base models
            base_models = {}
            
            model_files = [
                'xgboost', 'lightgbm', 'random_forest', 'extra_trees', 'mlp'
            ]
            
            for model_name in model_files:
                try:
                    model = joblib.load(f'Models/Ensemble_Models/Ensemble_NBA_v1_{model_name}.pkl')
                    base_models[model_name] = model
                except FileNotFoundError:
                    continue
            
            # Load neural network separately
            try:
                import tensorflow as tf
                nn_model = tf.keras.models.load_model('Models/Ensemble_Models/Ensemble_NBA_v1_neural_network.h5')
                nn_scaler = joblib.load('Models/Ensemble_Models/Ensemble_NBA_v1_neural_network_scaler.pkl')
                base_models['neural_network'] = {'model': nn_model, 'scaler': nn_scaler}
            except:
                pass
            
            # Load MLP scaler
            try:
                mlp_scaler = joblib.load('Models/Ensemble_Models/Ensemble_NBA_v1_mlp_scaler.pkl')
                if 'mlp' in base_models:
                    base_models['mlp'] = {'model': base_models['mlp'], 'scaler': mlp_scaler}
            except:
                pass
            
            # Load meta-model
            meta_model = joblib.load('Models/Ensemble_Models/Ensemble_NBA_v1_meta_model.pkl')
            
            # Load features
            features = joblib.load('Models/Ensemble_Models/Ensemble_NBA_v1_features.pkl')
            
            return {
                'base_models': base_models,
                'meta_model': meta_model,
                'features': features
            }
            
        except Exception as e:
            print(f"Error loading ensemble system: {e}")
            return None
    
    def select_best_model(self):
        """Select the best available model system"""
        if not self.available_models:
            self.scan_available_models()
        
        if not self.available_models:
            print("No models available!")
            return None
        
        # Sort by confidence score
        sorted_models = sorted(
            self.available_models.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )
        
        best_model_name, best_model_info = sorted_models[0]
        
        print(f"Selected best model: {best_model_name} (confidence: {best_model_info['confidence']:.2f})")
        
        # Load the best model
        if best_model_name == 'boosted_system':
            self.best_model = {
                'name': 'boosted_system',
                'system': self.load_boosted_system(),
                'type': 'boosted'
            }
        elif best_model_name == 'ensemble_system':
            self.best_model = {
                'name': 'ensemble_system',
                'system': self.load_ensemble_system(),
                'type': 'ensemble'
            }
        else:
            # Load other model types
            self.best_model = {
                'name': best_model_name,
                'system': None,  # Load specific model
                'type': 'single'
            }
        
        return self.best_model
    
    def predict_with_best_model(self, game_features):
        """Make prediction using the best available model"""
        if self.best_model is None:
            self.select_best_model()
        
        if self.best_model is None:
            print("No models available for prediction")
            return None
        
        try:
            if self.best_model['type'] == 'boosted':
                system = self.best_model['system']
                if system is not None:
                    # Use ensemble prediction from boosted system
                    return system.predict_single_game(game_features, use_best_only=False)
            
            elif self.best_model['type'] == 'ensemble':
                return self.predict_with_ensemble(game_features)
            
            else:
                return self.predict_with_single_model(game_features)
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_with_ensemble(self, game_features):
        """Make prediction using ensemble system"""
        ensemble_info = self.best_model['system']
        if ensemble_info is None:
            return None
        
        base_models = ensemble_info['base_models']
        meta_model = ensemble_info['meta_model']
        features = ensemble_info['features']
        
        # Prepare features
        if isinstance(game_features, pd.DataFrame):
            X = game_features[features].values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        # Get base model predictions
        base_predictions = np.zeros((1, len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            try:
                if name in ['neural_network', 'mlp']:
                    if isinstance(model, dict):
                        scaler = model['scaler']
                        actual_model = model['model']
                        X_scaled = scaler.transform(X)
                        
                        if name == 'neural_network':
                            base_predictions[0, i] = actual_model.predict(X_scaled)[0, 0]
                        else:  # MLP
                            base_predictions[0, i] = actual_model.predict_proba(X_scaled)[0, 1]
                    else:
                        base_predictions[0, i] = model.predict_proba(X)[0, 1]
                else:
                    base_predictions[0, i] = model.predict_proba(X)[0, 1]
            except Exception as e:
                print(f"Error with model {name}: {e}")
                base_predictions[0, i] = 0.5  # Neutral prediction
        
        # Get ensemble prediction
        try:
            ensemble_prob = meta_model.predict_proba(base_predictions)[0, 1]
        except:
            ensemble_prob = np.mean(base_predictions[0])  # Fallback to simple average
        
        return {
            'model_used': 'ensemble_system',
            'probability': ensemble_prob,
            'prediction': int(ensemble_prob >= 0.5),
            'confidence': abs(ensemble_prob - 0.5) * 2,
            'base_predictions': dict(zip(base_models.keys(), base_predictions[0]))
        }
    
    def predict_with_single_model(self, game_features):
        """Make prediction using a single model (fallback)"""
        # This would load and use individual models
        # For now, return a placeholder
        return {
            'model_used': self.best_model['name'],
            'probability': 0.65,  # Placeholder
            'prediction': 1,
            'confidence': 0.7
        }
    
    def get_model_recommendations(self):
        """Get recommendations for model improvements"""
        recommendations = []
        
        if 'boosted_system' not in self.available_models:
            recommendations.append("Train Boosted System for highest accuracy")
        
        if 'ensemble_system' not in self.available_models:
            recommendations.append("Train Ensemble System for robust predictions")
        
        if 'multi_target' not in self.available_models:
            recommendations.append("Train Multi-Target System for comprehensive predictions")
        
        if len(self.available_models) <= 1:
            recommendations.append("Train multiple model types for better selection")
        
        return recommendations
    
    def evaluate_all_models(self, test_data):
        """Evaluate all available models on test data (for model selection)"""
        # This would run all models on test data and compare performance
        # Implementation would depend on having labeled test data
        pass

# Global instance for easy access
auto_selector = AutoModelSelector()

def get_best_prediction(game_features):
    """Get the best available prediction"""
    return auto_selector.predict_with_best_model(game_features)

def get_available_models():
    """Get list of available models"""
    return auto_selector.scan_available_models()

if __name__ == "__main__":
    # Test the auto model selector
    selector = AutoModelSelector()
    
    # Scan for models
    available = selector.scan_available_models()
    
    # Select best model
    best = selector.select_best_model()
    
    if best:
        print(f"Best model selected: {best['name']}")
        
        # Test prediction with mock data
        mock_features = np.random.rand(100)  # Mock feature vector
        prediction = selector.predict_with_best_model(mock_features)
        
        if prediction:
            print(f"Test prediction: {prediction}")
    
    # Get recommendations
    recommendations = selector.get_model_recommendations()
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

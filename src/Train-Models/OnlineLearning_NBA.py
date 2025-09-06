"""
Online Learning system for NBA predictions that adapts in real-time.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class OnlineNBAPredictor:
    def __init__(self, adaptation_rate=0.1):
        self.adaptation_rate = adaptation_rate
        self.online_model = SGDClassifier(
            loss='log_loss',
            learning_rate='adaptive',
            eta0=0.01,
            alpha=0.0001,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.prediction_history = []
        self.recent_performance = {'accuracy': 0.5, 'confidence': 0.5}
        
    def partial_fit(self, X, y):
        """Update model with new game results"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.reshape(1, -1) if X.ndim == 1 else X
        y = np.array([y]) if np.isscalar(y) else y
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Update model
        self.online_model.partial_fit(X_scaled, y, classes=[0, 1])
        
        # Track performance
        self.update_performance_metrics()
    
    def predict_with_adaptation(self, X):
        """Make prediction with confidence scoring"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.reshape(1, -1) if X.ndim == 1 else X
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        if hasattr(self.online_model, 'predict_proba'):
            prob = self.online_model.predict_proba(X_scaled)[0, 1]
        else:
            decision = self.online_model.decision_function(X_scaled)[0]
            prob = 1 / (1 + np.exp(-decision))
        
        pred = int(prob > 0.5)
        conf = abs(prob - 0.5) * 2
        
        result = {
            'probability': prob,
            'prediction': pred,
            'confidence': conf,
            'recent_accuracy': self.recent_performance['accuracy']
        }
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': result,
            'features': X.flatten()
        })
        
        return result
    
    def update_performance_metrics(self, window_size=50):
        """Update recent performance metrics"""
        if len(self.prediction_history) < 2:
            return
        
        recent_predictions = self.prediction_history[-window_size:]
        correct = sum(1 for p in recent_predictions if p.get('actual_outcome') == p['prediction']['prediction'])
        total = sum(1 for p in recent_predictions if 'actual_outcome' in p)
        
        if total > 0:
            self.recent_performance['accuracy'] = correct / total
    
    def save_model(self, filepath=None):
        """Save online learning model"""
        if filepath is None:
            filepath = f"Models/Online_Models/NBA_Online_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        import os
        os.makedirs("Models/Online_Models", exist_ok=True)
        
        joblib.dump({
            'model': self.online_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'recent_performance': self.recent_performance
        }, f"{filepath}.pkl")
        
        print(f"Online model saved to {filepath}")

if __name__ == "__main__":
    print("Testing Online Learning NBA Predictor...")
    
    # Test with random data
    np.random.seed(42)
    X_test = np.random.randn(1000, 50)
    y_test = np.random.randint(0, 2, 1000)
    
    online_predictor = OnlineNBAPredictor()
    online_predictor.scaler.fit(X_test[:100])
    
    # Simulate online learning
    accuracies = []
    for i in range(100, 1000, 10):
        X_batch = X_test[i:i+10]
        y_batch = y_test[i:i+10]
        
        predictions = []
        for j in range(len(X_batch)):
            pred_result = online_predictor.predict_with_adaptation(X_batch[j:j+1])
            predictions.append(pred_result['prediction'])
            online_predictor.partial_fit(X_batch[j:j+1], y_batch[j])
        
        batch_accuracy = accuracy_score(y_batch, predictions)
        accuracies.append(batch_accuracy)
    
    print(f"Average accuracy: {np.mean(accuracies):.3f}")
    print("✅ Online Learning test complete!")

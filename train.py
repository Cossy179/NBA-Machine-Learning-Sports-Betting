#!/usr/bin/env python3
"""
🏀 NBA Machine Learning Sports Betting - Unified Training Script
Trains all advanced models with enhanced features and validation.
"""
import sys
import os
import argparse
import warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def print_header():
    """Print training script header"""
    print("🏀" + "="*70 + "🏀")
    print("🤖 NBA Machine Learning Sports Betting - Advanced Training System 🤖")
    print("🏀" + "="*70 + "🏀")
    print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def train_enhanced_features():
    """Train enhanced feature engineering"""
    print("📊 Step 1: Enhanced Feature Engineering")
    print("-" * 50)
    
    try:
        sys.path.append('src/Process-Data')
        from Enhanced_Features import EnhancedFeatureEngine
        
        enhancer = EnhancedFeatureEngine()
        print("🔧 Creating enhanced dataset with advanced features...")
        
        enhanced_df = enhancer.enhance_dataset()
        print(f"✅ Enhanced dataset created with {len(enhanced_df.columns)} total features")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced feature engineering failed: {e}")
        return False

def train_ensemble_models():
    """Train advanced ensemble models"""
    print("\n🤖 Step 2: Advanced Ensemble Models")
    print("-" * 50)
    
    try:
        sys.path.append('src/Train-Models')
        from Ensemble_System import EnsembleNBAPredictor
        
        print("🔧 Training advanced ensemble system...")
        ensemble = EnsembleNBAPredictor()
        ensemble.train_ensemble()
        ensemble.save_ensemble("Ensemble_NBA_v2")
        
        print("✅ Advanced ensemble models trained successfully")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        return False

def train_multi_target_models():
    """Train multi-target prediction models"""
    print("\n🎯 Step 3: Multi-Target Prediction Models")
    print("-" * 50)
    
    try:
        sys.path.append('src/Train-Models')
        from Multi_Target_Predictor import MultiTargetNBAPredictor
        
        print("🔧 Training multi-target prediction system...")
        multi_target = MultiTargetNBAPredictor()
        multi_target.train_all_models()
        
        print("✅ Multi-target models trained successfully")
        return True
        
    except Exception as e:
        print(f"❌ Multi-target training failed: {e}")
        return False

def train_advanced_xgboost():
    """Train optimized XGBoost models"""
    print("\n⚡ Step 4: Advanced XGBoost with Hyperparameter Optimization")
    print("-" * 50)
    
    try:
        sys.path.append('src/Train-Models')
        from Advanced_XGBoost_ML import AdvancedXGBoostTrainer
        
        print("🔧 Training advanced XGBoost with Optuna optimization...")
        trainer = AdvancedXGBoostTrainer()
        trainer.train_optimized_model(n_trials=50)  # Reduced for faster training
        trainer.save_model()
        
        print("✅ Advanced XGBoost trained successfully")
        return True
        
    except Exception as e:
        print(f"❌ Advanced XGBoost training failed: {e}")
        return False

def train_neural_networks():
    """Train advanced neural network models"""
    print("\n🧠 Step 5: Advanced Neural Networks")
    print("-" * 50)
    
    success_count = 0
    
    # Train Transformer model
    try:
        sys.path.append('src/Train-Models')
        from Transformer_NBA import NBATransformerPredictor
        
        print("🔧 Training Transformer model...")
        transformer = NBATransformerPredictor()
        results = transformer.train_model(epochs=30)  # Reduced for faster training
        transformer.save_model()
        
        print(f"✅ Transformer model trained (Accuracy: {results['test_accuracy']:.3f})")
        success_count += 1
        
    except Exception as e:
        print(f"⚠️ Transformer training failed: {e}")
    
    # Train Graph Neural Network
    try:
        from GraphNN_NBA import NBAGraphNeuralNetwork
        
        print("🔧 Training Graph Neural Network...")
        gnn = NBAGraphNeuralNetwork()
        results = gnn.train_model(epochs=30)
        gnn.save_model()
        
        print(f"✅ Graph NN trained (Accuracy: {results['test_accuracy']:.3f})")
        success_count += 1
        
    except Exception as e:
        print(f"⚠️ Graph NN training failed: {e}")
    
    # Train Bayesian model
    try:
        from Bayesian_NBA import BayesianNBAPredictor
        
        print("🔧 Training Bayesian Neural Network...")
        bayesian = BayesianNBAPredictor()
        results = bayesian.train_model(epochs=30)
        
        print(f"✅ Bayesian NN trained (Accuracy: {results['accuracy']:.3f})")
        success_count += 1
        
    except Exception as e:
        print(f"⚠️ Bayesian NN training failed: {e}")
    
    return success_count > 0

def train_online_learning():
    """Initialize online learning system"""
    print("\n🔄 Step 6: Online Learning System")
    print("-" * 50)
    
    try:
        from OnlineLearning_NBA import OnlineNBAPredictor
        
        print("🔧 Initializing online learning system...")
        online_predictor = OnlineNBAPredictor()
        
        # Create dummy data to initialize
        import numpy as np
        X_dummy = np.random.randn(100, 50)
        online_predictor.scaler.fit(X_dummy)
        online_predictor.initialize_online_models(50)
        
        # Save initial state
        online_predictor.save_online_state()
        
        print("✅ Online learning system initialized")
        return True
        
    except Exception as e:
        print(f"❌ Online learning initialization failed: {e}")
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='NBA ML Training Script')
    parser.add_argument('--features', action='store_true', help='Train enhanced features')
    parser.add_argument('--ensemble', action='store_true', help='Train ensemble models')
    parser.add_argument('--multi-target', action='store_true', help='Train multi-target models')
    parser.add_argument('--xgboost', action='store_true', help='Train advanced XGBoost')
    parser.add_argument('--neural', action='store_true', help='Train neural networks')
    parser.add_argument('--online', action='store_true', help='Initialize online learning')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--quick', action='store_true', help='Quick training (reduced epochs)')
    
    args = parser.parse_args()
    
    print_header()
    
    # Create model directories
    os.makedirs("Models/Ensemble_Models", exist_ok=True)
    os.makedirs("Models/XGBoost_Models", exist_ok=True)
    os.makedirs("Models/NN_Models", exist_ok=True)
    os.makedirs("Models/Online_Models", exist_ok=True)
    
    training_results = {}
    
    # Train components based on arguments
    if args.all or args.features:
        training_results['features'] = train_enhanced_features()
    
    if args.all or args.ensemble:
        training_results['ensemble'] = train_ensemble_models()
    
    if args.all or args.multi_target:
        training_results['multi_target'] = train_multi_target_models()
    
    if args.all or args.xgboost:
        training_results['xgboost'] = train_advanced_xgboost()
    
    if args.all or args.neural:
        training_results['neural'] = train_neural_networks()
    
    if args.all or args.online:
        training_results['online'] = train_online_learning()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    total_components = len(training_results)
    successful_components = sum(training_results.values())
    
    for component, success in training_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{component.upper():20} {status}")
    
    print(f"\nOverall: {successful_components}/{total_components} components trained successfully")
    
    if successful_components == total_components:
        print("\n🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("🎯 Your NBA prediction system is ready for use!")
        print("\nNext steps:")
        print("  • Run backtesting: python backtest.py")
        print("  • Make predictions: python predict.py")
    else:
        print(f"\n⚠️ {total_components - successful_components} components failed to train")
        print("Check the error messages above and retry failed components")
    
    print(f"\n⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return successful_components == total_components

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

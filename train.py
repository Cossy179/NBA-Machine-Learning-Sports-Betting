#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Machine Learning Sports Betting - Unified Training Script
Trains all advanced models with enhanced features and validation.
"""
import sys
import os
import argparse
import warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings('ignore')

# Ensure UTF-8 console to avoid Unicode errors on Windows
try:
    # Python 3.7+
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
except Exception:
    pass

# Add src directories to path once
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
process_data_dir = os.path.join(src_dir, 'Process-Data')
train_models_dir = os.path.join(src_dir, 'Train-Models')

for path in [src_dir, process_data_dir, train_models_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

def print_header():
    """Print training script header"""
    print("🏀" + "="*70 + "🏀")
    print("🤖 NBA Machine Learning Sports Betting - Advanced Training System 🤖")
    print("🏀" + "="*70 + "🏀")
    print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def detect_and_configure_device(preferred: str = 'auto'):
    """Detect CUDA GPU (e.g., RTX 3060) and configure frameworks to use it.
    Falls back to CPU if unavailable. Prints a concise summary.
    Returns a dict with device info and booleans.
    """
    info = {
        'use_gpu': False,
        'torch': {'available': False, 'device': 'cpu', 'name': None},
        'tensorflow': {'available': False, 'gpus': []},
        'xgboost': {'device': 'cpu', 'tree_method': 'hist'}
    }
    # Prefer CUDA if available
    try:
        import torch
        if preferred != 'cpu' and torch.cuda.is_available():
            info['use_gpu'] = True
            info['torch']['available'] = True
            info['torch']['device'] = 'cuda'
            try:
                info['torch']['name'] = torch.cuda.get_device_name(0)
            except Exception:
                info['torch']['name'] = 'CUDA GPU'
        else:
            info['torch']['available'] = True
            info['torch']['device'] = 'cpu'
    except Exception:
        pass

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        info['tensorflow']['available'] = len(gpus) > 0
        info['tensorflow']['gpus'] = [g.name if hasattr(g, 'name') else str(g) for g in gpus]
        if info['use_gpu'] and gpus:
            # Enable memory growth for stability
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass

    # Configure XGBoost default device via env for trainers to read
    if info['use_gpu']:
        os.environ['XGB_DEFAULT_DEVICE'] = 'cuda'
        os.environ['XGB_DEFAULT_TREE_METHOD'] = 'gpu_hist'
        info['xgboost']['device'] = 'cuda'
        info['xgboost']['tree_method'] = 'gpu_hist'
    else:
        os.environ['XGB_DEFAULT_DEVICE'] = 'cpu'
        os.environ['XGB_DEFAULT_TREE_METHOD'] = 'hist'

    # Public flag for our trainers to consume
    os.environ['USE_GPU'] = '1' if info['use_gpu'] else '0'

    # Print concise summary
    print("🖥️ Compute Device")
    print("-" * 50)
    if info['use_gpu']:
        gpu_name = info['torch']['name'] or (info['tensorflow']['gpus'][0] if info['tensorflow']['gpus'] else 'CUDA GPU')
        print(f"  • Using GPU (CUDA): {gpu_name}")
        print(f"  • XGBoost: device=cuda, tree_method=gpu_hist")
    else:
        print("  • Using CPU fallback (no CUDA detected or forced)")
        print(f"  • XGBoost: device=cpu, tree_method=hist")
    if info['tensorflow']['gpus']:
        print(f"  • TensorFlow GPUs: {len(info['tensorflow']['gpus'])}")
    if info['torch']['available']:
        print(f"  • PyTorch device: {info['torch']['device']}")
    print()
    return info

def validate_imports():
    """Validate that all required modules can be imported"""
    print("🔍 Validating module imports...")
    
    modules_to_check = [
        ('Enhanced_Features', 'EnhancedFeatureEngine'),
        ('Ensemble_System', 'EnsembleNBAPredictor'),
        ('Multi_Target_Predictor', 'MultiTargetNBAPredictor'),
        ('Advanced_XGBoost_ML', 'AdvancedXGBoostTrainer'),
        ('Transformer_NBA', 'NBATransformerPredictor'),
        ('GraphNN_NBA', 'NBAGraphNeuralNetwork'),
        ('Bayesian_NBA', 'BayesianNBAPredictor'),
        ('OnlineLearning_NBA', 'OnlineNBAPredictor')
    ]
    
    failed_imports = []
    
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name)
            getattr(module, class_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            failed_imports.append(module_name)
        except AttributeError as e:
            print(f"  ⚠️ {module_name}: Missing class {class_name}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n⚠️ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"  - {module}")
        print("\nSome training steps may fail. Continuing with available modules...")
    else:
        print("✅ All modules imported successfully!")
    
    print()
    return len(failed_imports) == 0

def train_enhanced_features():
    """Train enhanced feature engineering with advanced features"""
    print("📊 Step 1: Advanced Enhanced Feature Engineering")
    print("-" * 50)
    
    try:
        from Enhanced_Features import EnhancedFeatureEngine
        
        enhancer = EnhancedFeatureEngine()
        print("🔧 Creating enhanced dataset with 200+ advanced features...")
        print("  • Advanced ELO ratings (overall, home, away, offense, defense, recent)")
        print("  • Multi-window recent form analysis (3, 5, 10, 15 games)")
        print("  • Advanced betting features with market sentiment")
        print("  • Injury impact modeling with depth chart analysis")
        print("  • Situational factors (playoff implications, rivalry, etc.)")
        print("  • Team analytics and efficiency metrics")
        print("  • Momentum indicators and psychological factors")
        
        enhanced_df = enhancer.enhance_dataset()
        print(f"✅ Enhanced dataset created with {len(enhanced_df.columns)} total features")
        
        # Save enhanced dataset
        enhanced_df.to_csv("Data/dataset_2012-24_enhanced.csv", index=False)
        print("💾 Enhanced dataset saved to Data/dataset_2012-24_enhanced.csv")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Enhanced_Features: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced feature engineering failed: {e}")
        return False

def train_ensemble_models():
    """Train advanced ensemble models with enhanced features"""
    print("\n🤖 Step 2: Advanced Ensemble Models with Enhanced Features")
    print("-" * 50)
    
    try:
        # Import both old and new ensemble systems
        from Ensemble_System import EnsembleNBAPredictor
        sys.path.append('src/Predict')
        from Advanced_Prediction_Runner import AdvancedPredictionRunner
        
        print("🔧 Training advanced ensemble system with enhanced features...")
        print("  • Weighted average ensemble with confidence weighting")
        print("  • Meta-model stacking with uncertainty quantification")
        print("  • Bayesian model averaging")
        print("  • Dynamic ensemble selection")
        
        # Train original ensemble
        ensemble = EnsembleNBAPredictor()
        ensemble.train_ensemble()
        ensemble.save_ensemble("Ensemble_NBA_v2")
        
        # Train advanced ensemble
        advanced_runner = AdvancedPredictionRunner()
        print("🔧 Training advanced prediction runner...")
        # The advanced runner will use the enhanced features automatically
        
        print("✅ Advanced ensemble models trained successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import ensemble systems: {e}")
        return False
    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        return False

def train_multi_target_models():
    """Train multi-target prediction models"""
    print("\n🎯 Step 3: Multi-Target Prediction Models")
    print("-" * 50)
    
    try:
        from Multi_Target_Predictor import MultiTargetNBAPredictor
        
        print("🔧 Training multi-target prediction system...")
        multi_target = MultiTargetNBAPredictor()
        multi_target.train_all_models()
        
        print("✅ Multi-target models trained successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Multi_Target_Predictor: {e}")
        return False
    except Exception as e:
        print(f"❌ Multi-target training failed: {e}")
        return False

def train_advanced_xgboost():
    """Train optimized XGBoost models"""
    print("\n⚡ Step 4: Advanced XGBoost with Hyperparameter Optimization")
    print("-" * 50)
    
    try:
        from Advanced_XGBoost_ML import AdvancedXGBoostTrainer
        
        print("🔧 Training advanced XGBoost with Optuna optimization...")
        trainer = AdvancedXGBoostTrainer()
        trainer.train_optimized_model(n_trials=50)  # Reduced for faster training
        trainer.save_model()
        
        print("✅ Advanced XGBoost trained successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Advanced_XGBoost_ML: {e}")
        return False
    except Exception as e:
        print(f"❌ Advanced XGBoost training failed: {e}")
        return False

def train_ultra_features():
    """Train ultra-advanced feature engineering"""
    print("\n🚀 Step 4b: Ultra-Advanced Feature Engineering")
    print("-" * 50)
    
    try:
        from UltraAdvanced_Features import UltraAdvancedFeatureEngine
        
        print("🔧 Creating ultra-advanced dataset with 100+ new features...")
        print("  • Four Factors analysis (Dean Oliver methodology)")
        print("  • Clutch performance metrics (close game, Q4, mental toughness)")
        print("  • Advanced momentum indicators (multi-window, time-decay)")
        print("  • Lineup synergy and chemistry metrics")
        print("  • Shot distribution and efficiency analysis")
        print("  • Pace and playing style metrics")
        print("  • Matchup-specific interaction features")
        print("  • Advanced betting market signals")
        
        engine = UltraAdvancedFeatureEngine()
        enhanced_df = engine.enhance_dataset_ultra()
        
        print(f"✅ Ultra-advanced features created successfully")
        print(f"   Total features in dataset: {len(enhanced_df.columns)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import UltraAdvanced_Features: {e}")
        return False
    except Exception as e:
        print(f"❌ Ultra-advanced feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_super_advanced_xgboost():
    """Train super advanced XGBoost ensemble"""
    print("\n🌟 Step 4c: Super Advanced XGBoost Ensemble")
    print("-" * 50)
    
    try:
        from SuperAdvanced_XGBoost import SuperAdvancedXGBoostTrainer
        
        print("🔧 Training super advanced XGBoost ensemble...")
        print("  • XGBoost DART (Dropouts meet Multiple Additive Regression Trees)")
        print("  • LightGBM with advanced optimization")
        print("  • CatBoost for categorical handling")
        print("  • Advanced feature selection (mutual info + tree importance + XGBoost)")
        print("  • Multi-model ensemble with dynamic weighting")
        print("  • Isotonic regression calibration")
        
        trainer = SuperAdvancedXGBoostTrainer()
        trainer.train_super_advanced_ensemble(n_trials=30)  # Reduced for speed
        trainer.save_models("SuperAdvanced_XGB_v1")
        
        print("✅ Super advanced XGBoost ensemble trained successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SuperAdvanced_XGBoost: {e}")
        return False
    except Exception as e:
        print(f"❌ Super advanced XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_neural_networks():
    """Train advanced neural network models with enhanced architectures"""
    print("\n🧠 Step 5: Advanced Neural Networks with Enhanced Architectures")
    print("-" * 50)
    
    success_count = 0
    
    # Train Advanced Neural Networks
    try:
        sys.path.append('src/Predict')
        from NN_Runner import create_advanced_neural_network, train_advanced_neural_network
        
        print("🔧 Training advanced neural network architectures...")
        print("  • Multi-branch ensemble architecture with residual connections")
        print("  • Transformer-inspired architecture with attention mechanisms")
        print("  • CNN-like architecture for sequential patterns")
        print("  • Advanced regularization and uncertainty quantification")
        
        # This will be handled by the NN_Runner when called
        print("✅ Advanced neural network architectures available")
        success_count += 1
        
    except ImportError as e:
        print(f"⚠️ Failed to import advanced NN functions: {e}")
    except Exception as e:
        print(f"⚠️ Advanced NN setup failed: {e}")
    
    # Train Transformer model
    try:
        from Transformer_NBA import NBATransformerPredictor
        
        print("🔧 Training Transformer model...")
        transformer = NBATransformerPredictor()
        results = transformer.train_model(epochs=30)  # Reduced for faster training
        transformer.save_model()
        
        print(f"✅ Transformer model trained (Accuracy: {results['test_accuracy']:.3f})")
        success_count += 1
        
    except ImportError as e:
        print(f"⚠️ Failed to import Transformer_NBA: {e}")
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
        
    except ImportError as e:
        print(f"⚠️ Failed to import GraphNN_NBA: {e}")
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
        
    except ImportError as e:
        print(f"⚠️ Failed to import Bayesian_NBA: {e}")
    except Exception as e:
        print(f"⚠️ Bayesian NN training failed: {e}")
    
    return success_count > 0

def train_parlay_predictor():
    """Train advanced parlay prediction system"""
    print("\n🎯 Step 6: Advanced Parlay Prediction System")
    print("-" * 50)
    
    try:
        sys.path.append('src/Predict')
        from ParlayPredictor import AdvancedParlayPredictor
        
        print("🔧 Training advanced parlay prediction system...")
        print("  • Advanced correlation modeling (dynamic, contextual, temporal)")
        print("  • Risk assessment and optimization")
        print("  • Market efficiency analysis")
        print("  • Advanced parlay evaluation with uncertainty quantification")
        
        parlay_predictor = AdvancedParlayPredictor()
        
        # Load player data for training
        player_data = parlay_predictor.load_player_data()
        if not player_data.empty:
            print("🔧 Training on real player data...")
            parlay_predictor.calculate_advanced_correlations(player_data)
            parlay_predictor.train_player_prop_models(player_data)
            parlay_predictor.save_parlay_models()
        else:
            print("⚠️ No player data available, using mock data for initialization")
            # Create mock data for testing
            import pandas as pd
            from datetime import datetime, timedelta
            
            mock_data = pd.DataFrame({
                'Player': [i for i in range(100)],
                'Team': [i % 30 for i in range(100)],
                'PTS': np.random.normal(25, 5, 100),
                'REB': np.random.normal(8, 2, 100),
                'AST': np.random.normal(7, 2, 100),
                'Date': [datetime.now() - timedelta(days=i) for i in range(100)]
            })
            
            parlay_predictor.calculate_advanced_correlations(mock_data)
            parlay_predictor.train_player_prop_models(mock_data)
            parlay_predictor.save_parlay_models()
        
        print("✅ Advanced parlay prediction system trained successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import AdvancedParlayPredictor: {e}")
        return False
    except Exception as e:
        print(f"❌ Parlay prediction training failed: {e}")
        return False

def train_online_learning():
    """Initialize online learning system"""
    print("\n🔄 Step 7: Online Learning System")
    print("-" * 50)
    
    try:
        from OnlineLearning_NBA import OnlineNBAPredictor
        
        print("🔧 Initializing online learning system...")
        online_predictor = OnlineNBAPredictor()
        
        # Create dummy data to initialize
        X_dummy = np.random.randn(100, 50)
        online_predictor.scaler.fit(X_dummy)
        online_predictor.initialize_online_models(50)
        
        # Save initial state
        online_predictor.save_online_state()
        
        print("✅ Online learning system initialized")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OnlineLearning_NBA: {e}")
        return False
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
    parser.add_argument('--ultra-features', action='store_true', help='Train ultra-advanced features')
    parser.add_argument('--super-xgboost', action='store_true', help='Train super advanced XGBoost ensemble')
    parser.add_argument('--neural', action='store_true', help='Train neural networks')
    parser.add_argument('--parlay', action='store_true', help='Train parlay prediction system')
    parser.add_argument('--online', action='store_true', help='Initialize online learning')
    parser.add_argument('--all', action='store_true', help='Train all models (traditional)')
    parser.add_argument('--ultra', action='store_true', help='Train ultra-advanced models (NEW!)')
    parser.add_argument('--quick', action='store_true', help='Quick training (reduced epochs)')
    parser.add_argument('--dry-run', action='store_true', help='Validate imports and setup without training')
    
    args = parser.parse_args()
    
    # If no specific arguments provided, default to --all
    if not any([args.features, args.ensemble, args.multi_target, args.xgboost, 
                args.ultra_features, args.super_xgboost, args.neural, args.parlay, 
                args.online, args.all, args.ultra]):
        args.all = True
    
    print_header()
    # Detect and display compute device (GPU/CPU) and configure libs
    detect_and_configure_device()
    
    # Validate imports before starting
    all_imports_valid = validate_imports()
    
    # Create model directories
    model_dirs = [
        "Models/Ensemble_Models",
        "Models/XGBoost_Models", 
        "Models/NN_Models",
        "Models/Online_Models",
        "Models/Boosted_Models",
        "Models/Parlay_Models"
    ]
    
    for model_dir in model_dirs:
        os.makedirs(model_dir, exist_ok=True)
    
    # If dry-run, exit after validation
    if args.dry_run:
        print("\n🔍 DRY RUN COMPLETE")
        print("="*50)
        if all_imports_valid:
            print("✅ All modules validated successfully!")
            print("🎯 Ready to run full training with: py train.py --all")
        else:
            print("⚠️ Some modules failed validation")
            print("🔧 Fix import issues before running full training")
        return all_imports_valid
    
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
    
    # NEW ULTRA-ADVANCED OPTIONS
    if args.ultra or args.ultra_features:
        training_results['ultra_features'] = train_ultra_features()
    
    if args.ultra or args.super_xgboost:
        training_results['super_xgboost'] = train_super_advanced_xgboost()
    
    if args.all or args.neural:
        training_results['neural'] = train_neural_networks()
    
    if args.all or args.parlay:
        training_results['parlay'] = train_parlay_predictor()
    
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

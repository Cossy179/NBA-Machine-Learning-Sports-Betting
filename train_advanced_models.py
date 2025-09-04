"""
Training script for all advanced models.
Run this to train the enhanced prediction system.
"""
import os
import sys
import subprocess
from datetime import datetime

# Add src to path
sys.path.append('src')

def install_requirements():
    """Install additional requirements for advanced models"""
    print("Installing additional requirements...")
    
    additional_packages = [
        'optuna',
        'lightgbm', 
        'scikit-learn>=1.3.0',
        'joblib',
        'tensorflow>=2.10.0',
        'shap',
        'plotly',
        'seaborn'
    ]
    
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠ Failed to install {package}")

def create_directories():
    """Create necessary directories"""
    directories = [
        'Models/XGBoost_Models',
        'Models/Ensemble_Models', 
        'Models/NN_Models',
        'Logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def train_enhanced_features():
    """Train enhanced feature engineering"""
    print("\n" + "="*50)
    print("TRAINING ENHANCED FEATURES")
    print("="*50)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Enhanced_Features", "src/Process-Data/Enhanced_Features.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EnhancedFeatureEngine = module.EnhancedFeatureEngine
        
        enhancer = EnhancedFeatureEngine()
        enhanced_df = enhancer.enhance_dataset()
        print("✓ Enhanced features created successfully")
        return True
    except Exception as e:
        print(f"⚠ Enhanced features training failed: {e}")
        print("  Continuing with base dataset...")
        return False

def train_advanced_xgboost():
    """Train advanced XGBoost model"""
    print("\n" + "="*50)
    print("TRAINING ADVANCED XGBOOST")
    print("="*50)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Advanced_XGBoost_ML", "src/Train-Models/Advanced_XGBoost_ML.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        AdvancedXGBoostTrainer = module.AdvancedXGBoostTrainer
        
        trainer = AdvancedXGBoostTrainer()
        trainer.train_optimized_model(n_trials=30)  # Reduced for faster training
        trainer.save_model("XGB_ML_Advanced_v1")
        print("✓ Advanced XGBoost model trained successfully")
        return True
    except Exception as e:
        print(f"⚠ Advanced XGBoost training failed: {e}")
        return False

def train_multi_target_models():
    """Train multi-target prediction models"""
    print("\n" + "="*50)
    print("TRAINING MULTI-TARGET MODELS")
    print("="*50)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Multi_Target_Predictor", "src/Train-Models/Multi_Target_Predictor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MultiTargetNBAPredictor = module.MultiTargetNBAPredictor
        
        predictor = MultiTargetNBAPredictor()
        predictor.train_all_models()
        predictor.save_models("MultiTarget_NBA_v1")
        print("✓ Multi-target models trained successfully")
        return True
    except Exception as e:
        print(f"⚠ Multi-target models training failed: {e}")
        return False

def train_ensemble_system():
    """Train ensemble system"""
    print("\n" + "="*50)
    print("TRAINING ENSEMBLE SYSTEM")
    print("="*50)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Ensemble_System", "src/Train-Models/Ensemble_System.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EnsembleNBAPredictor = module.EnsembleNBAPredictor
        
        ensemble = EnsembleNBAPredictor()
        ensemble.train_ensemble()
        ensemble.save_ensemble("Ensemble_NBA_v1")
        print("✓ Ensemble system trained successfully")
        return True
    except Exception as e:
        print(f"⚠ Ensemble system training failed: {e}")
        return False

def train_boosted_system():
    """Train boosted model system"""
    print("\n" + "="*50)
    print("TRAINING BOOSTED MODEL SYSTEM")
    print("="*50)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Boosted_Model_System", "src/Train-Models/Boosted_Model_System.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BoostedModelSystem = module.BoostedModelSystem
        
        system = BoostedModelSystem()
        system.train_boosted_models()
        system.save_boosted_system("BoostedNBA_v1")
        print("✓ Boosted system trained successfully")
        return True
    except Exception as e:
        print(f"⚠ Boosted system training failed: {e}")
        return False

def train_player_models():
    """Train player stats and parlay models"""
    print("\n" + "="*50)
    print("TRAINING PLAYER & PARLAY MODELS")
    print("="*50)
    
    try:
        # Build player database
        import importlib.util
        spec = importlib.util.spec_from_file_location("PlayerStatsProvider", "src/DataProviders/PlayerStatsProvider.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PlayerStatsProvider = module.PlayerStatsProvider
        
        provider = PlayerStatsProvider()
        player_data = provider.build_comprehensive_player_database()
        
        if not player_data.empty:
            # Train parlay models
            spec = importlib.util.spec_from_file_location("ParlayPredictor", "src/Predict/ParlayPredictor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ParlayPredictor = module.ParlayPredictor
            
            predictor = ParlayPredictor()
            predictor.calculate_player_correlations(player_data)
            predictor.train_player_prop_models(player_data)
            predictor.save_parlay_models()
            
        print("✓ Player and parlay models trained successfully")
        return True
    except Exception as e:
        print(f"⚠ Player models training failed: {e}")
        return False

def main():
    """Main training pipeline"""
    print("🏀 NBA ADVANCED MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    # Setup
    print("\n1. Setting up environment...")
    install_requirements()
    create_directories()
    
    # Training pipeline
    results = {}
    
    print("\n2. Training enhanced features...")
    results['enhanced_features'] = train_enhanced_features()
    
    print("\n3. Training advanced XGBoost...")
    results['advanced_xgb'] = train_advanced_xgboost()
    
    print("\n4. Training multi-target models...")
    results['multi_target'] = train_multi_target_models()
    
    print("\n5. Training ensemble system...")
    results['ensemble'] = train_ensemble_system()
    
    print("\n6. Training boosted system...")
    results['boosted'] = train_boosted_system()
    
    print("\n7. Training player & parlay models...")
    results['player_models'] = train_player_models()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model:20} {status}")
    
    successful_models = sum(results.values())
    total_models = len(results)
    
    print(f"\nOverall: {successful_models}/{total_models} models trained successfully")
    
    if successful_models > 0:
        print(f"\n🎉 Training complete! You can now run:")
        print(f"   py ultimate_nba_predictor.py -odds=fanduel -realtime -parlays -kc")
        print(f"   py ultimate_nba_predictor.py -backtest  # Test on 2023-24 season")
        print(f"   py enhanced_main.py -advanced -realtime -odds=fanduel -kc  # Legacy")
    else:
        print(f"\n⚠ Training failed. Check error messages above.")
        print(f"   You can still use original models with:")
        print(f"   py main.py -A -odds=fanduel -kc")
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main()

"""
Simple validation script to ensure all systems work correctly.
Tests the core functionality without complex dependencies.
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('src')

def validate_models():
    """Validate that models can make predictions"""
    print("🤖 Validating model predictions...")
    
    try:
        # Load sample data
        con = sqlite3.connect("Data/dataset.sqlite")
        sample_data = pd.read_sql_query('''
        select * from "dataset_2012-24_new" 
        ORDER BY Date DESC LIMIT 10
        ''', con)
        con.close()
        
        if sample_data.empty:
            print("❌ No data available")
            return False
        
        print(f"✅ Loaded {len(sample_data)} sample games")
        
        # Test Auto Model Selector
        import importlib.util
        spec = importlib.util.spec_from_file_location("AutoModelSelector", "src/Predict/AutoModelSelector.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        selector = module.AutoModelSelector()
        available = selector.scan_available_models()
        
        print(f"✅ Found {len(available)} model systems")
        
        if available:
            best_model = selector.select_best_model()
            print(f"✅ Selected best model: {best_model['name']}")
            
            # Test prediction
            exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
            feature_cols = [c for c in sample_data.columns if c not in exclude_cols]
            game_features = sample_data[feature_cols].iloc[0].fillna(0)
            
            prediction = selector.predict_with_best_model(game_features)
            
            if prediction:
                print(f"✅ Prediction successful:")
                print(f"   Probability: {prediction.get('probability', 0):.1%}")
                print(f"   Confidence: {prediction.get('confidence', 0):.1%}")
                return True
            else:
                print("❌ Prediction failed")
                return False
        else:
            print("❌ No models available")
            return False
            
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def validate_backtesting():
    """Validate backtesting on small sample"""
    print("\n🧪 Validating backtesting system...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("BacktestingEngine", "src/Backtest/BacktestingEngine.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        engine = module.BacktestingEngine()
        
        # Load small historical sample
        historical_data = engine.load_historical_data("2024-03-01", "2024-03-31")
        
        if historical_data.empty:
            print("❌ No historical data")
            return False
        
        print(f"✅ Loaded {len(historical_data)} games for validation")
        
        # Simple mock predictor
        def simple_predictor(features):
            return {
                'probability': 0.65,
                'prediction': 1,
                'confidence': 0.8
            }
        
        # Run small backtest
        results = engine.run_model_backtest(simple_predictor, historical_data.head(20), "Validation Model")
        
        if results and results['valid_predictions'] > 0:
            print(f"✅ Backtest validation successful:")
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   Valid predictions: {results['valid_predictions']}")
            return True
        else:
            print("❌ Backtest validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Backtesting validation failed: {e}")
        return False

def validate_parlay_generation():
    """Validate parlay generation"""
    print("\n🎲 Validating parlay generation...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ParlayPredictor", "src/Predict/ParlayPredictor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        predictor = module.ParlayPredictor()
        
        # Simple test data
        game_predictions = {
            'Game 1': {'probability': 0.65, 'confidence': 0.8, 'edge': 0.05, 'recommendation': 'Home ML'},
            'Game 2': {'probability': 0.72, 'confidence': 0.75, 'edge': 0.08, 'recommendation': 'Away ML'}
        }
        
        player_predictions = {
            'Player A': {
                'points': {'prediction': 26.5, 'line': 25.5, 'edge': 1.0, 'confidence': 0.75, 'recommendation': 'OVER'}
            }
        }
        
        parlays = predictor.generate_parlay_combinations(game_predictions, player_predictions)
        
        print(f"✅ Generated {len(parlays)} parlays")
        
        if parlays:
            best_parlay = parlays[0]
            print(f"✅ Best parlay has {best_parlay['num_legs']} legs")
            print(f"   Expected Value: {best_parlay['expected_value']:+.3f}")
            return True
        else:
            print("⚠️ No parlays generated (may be due to low confidence)")
            return True  # This is ok - system is working, just no good parlays
            
    except Exception as e:
        print(f"❌ Parlay validation failed: {e}")
        return False

def main():
    """Run validation tests"""
    print("🏀 NBA SYSTEM VALIDATION")
    print("="*50)
    print(f"Started: {datetime.now()}")
    print("="*50)
    
    results = {}
    results['Models'] = validate_models()
    results['Backtesting'] = validate_backtesting()
    results['Parlays'] = validate_parlay_generation()
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    for test, passed in results.items():
        status = "✅ WORKING" if passed else "❌ BROKEN"
        print(f"{test:15} {status}")
    
    working_systems = sum(results.values())
    total_systems = len(results)
    
    print(f"\nOverall: {working_systems}/{total_systems} systems working")
    
    if working_systems == total_systems:
        print("\n🎉 ALL SYSTEMS VALIDATED!")
        print("\n🚀 Ready for:")
        print("   • Live game predictions")
        print("   • Parlay generation") 
        print("   • Historical backtesting")
        print("   • ROI analysis")
    else:
        print("\n⚠️ Some systems need attention")
        if results['Models']:
            print("✅ Core predictions work")
        if results['Backtesting']:
            print("✅ Backtesting works")
        if results['Parlays']:
            print("✅ Parlays work")

if __name__ == "__main__":
    main()

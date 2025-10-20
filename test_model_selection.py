#!/usr/bin/env python3
"""
Test script to demonstrate model selection functionality
"""
import os
import sys

def test_model_selection():
    """Test the model selection functionality"""
    print("=" * 70)
    print("NBA MODEL SELECTION TEST")
    print("=" * 70)
    
    # Check for available model files
    model_dirs = [
        "Models/XGBoost_Models/",
        "Models/Neural_Networks/",
        "Models/Ensemble_Models/",
        "Models/Advanced_Models/"
    ]
    
    available_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"\nChecking {model_dir}:")
            for file in os.listdir(model_dir):
                if file.endswith(('.json', '.pkl', '.joblib', '.h5')):
                    model_name = file.replace('.json', '').replace('.pkl', '').replace('.joblib', '').replace('.h5', '')
                    available_models.append({
                        'name': model_name,
                        'path': os.path.join(model_dir, file),
                        'type': 'XGBoost' if 'XGBoost' in model_dir else 'Neural Network' if 'Neural' in model_dir else 'Ensemble' if 'Ensemble' in model_dir else 'Advanced'
                    })
                    print(f"  - {model_name} ({file})")
    
    if available_models:
        print(f"\nFound {len(available_models)} trained models:")
        print("-" * 50)
        
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Path: {model['path']}")
            print()
        
        print("Usage Examples:")
        print("  python predict.py --model xgb")
        print("  python predict.py --model advanced")
        print("  python predict.py --model super")
        print("  python predict.py --model ensemble")
        print()
        print("Model Selection Logic:")
        print("  - Use partial matching (e.g., 'xgb' matches 'XGB_ML_Advanced')")
        print("  - Case insensitive")
        print("  - Falls back to best available model if not found")
        
    else:
        print("\nNo trained models found!")
        print("Train models first: python train.py --all")
    
    print("\n" + "=" * 70)
    print("MODEL SELECTION FEATURES ADDED:")
    print("=" * 70)
    print("1. --model argument: Specify which model to use")
    print("2. --list-models argument: List all available models")
    print("3. Partial matching: 'xgb' matches 'XGB_ML_Advanced'")
    print("4. Fallback: Uses best available if specified model not found")
    print("5. Model display: Shows which model is being used")
    print("=" * 70)

if __name__ == "__main__":
    test_model_selection()

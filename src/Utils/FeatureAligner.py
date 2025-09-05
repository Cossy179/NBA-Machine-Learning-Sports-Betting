"""
Feature Alignment Utility to handle feature mismatches between training and prediction.
Ensures consistent feature sets across different model versions.
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Optional

class FeatureAligner:
    def __init__(self):
        self.feature_mappings = {}
        self.default_values = {}
        
    def align_features(self, input_features, target_feature_list, model_name="Unknown"):
        """Align input features to match target feature list"""
        try:
            # Convert input to DataFrame if it's not already
            if isinstance(input_features, np.ndarray):
                # If we don't have column names, we can't align properly
                print(f"⚠️ Warning: Cannot align numpy array features for {model_name}")
                return input_features
            
            if isinstance(input_features, pd.Series):
                input_df = input_features.to_frame().T
            elif isinstance(input_features, pd.DataFrame):
                input_df = input_features.copy()
            else:
                print(f"⚠️ Warning: Unknown feature format for {model_name}")
                return input_features
            
            # Create aligned feature DataFrame
            aligned_features = pd.DataFrame(index=input_df.index)
            
            missing_features = []
            extra_features = []
            
            for feature in target_feature_list:
                if feature in input_df.columns:
                    aligned_features[feature] = input_df[feature]
                else:
                    # Feature missing - use default value
                    default_val = self.get_default_value(feature)
                    aligned_features[feature] = default_val
                    missing_features.append(feature)
            
            # Track extra features that weren't needed
            for feature in input_df.columns:
                if feature not in target_feature_list:
                    extra_features.append(feature)
            
            if missing_features or extra_features:
                print(f"🔧 Feature alignment for {model_name}:")
                if missing_features:
                    print(f"  Added {len(missing_features)} missing features with defaults")
                if extra_features:
                    print(f"  Removed {len(extra_features)} extra features")
            
            return aligned_features
            
        except Exception as e:
            print(f"❌ Feature alignment failed for {model_name}: {e}")
            return input_features
    
    def get_default_value(self, feature_name):
        """Get appropriate default value for a feature"""
        # Define default values based on feature patterns
        if any(keyword in feature_name.lower() for keyword in ['pct', 'percentage', 'rate']):
            return 0.5  # Percentage features default to 50%
        elif any(keyword in feature_name.lower() for keyword in ['rank', 'rating']):
            return 15  # Rank features default to middle
        elif any(keyword in feature_name.lower() for keyword in ['elo']):
            return 1500  # ELO default
        elif any(keyword in feature_name.lower() for keyword in ['rest', 'days']):
            return 2  # Rest days default
        elif any(keyword in feature_name.lower() for keyword in ['confidence', 'prob']):
            return 0.5  # Probability features
        else:
            return 0  # Most other features default to 0
    
    def save_feature_mapping(self, model_name, feature_list):
        """Save feature list for a specific model"""
        self.feature_mappings[model_name] = feature_list
        
        # Save to disk
        os.makedirs("Models/Feature_Mappings", exist_ok=True)
        joblib.dump(feature_list, f"Models/Feature_Mappings/{model_name}_features.pkl")
    
    def load_feature_mapping(self, model_name):
        """Load feature list for a specific model"""
        try:
            feature_list = joblib.load(f"Models/Feature_Mappings/{model_name}_features.pkl")
            self.feature_mappings[model_name] = feature_list
            return feature_list
        except FileNotFoundError:
            return None
    
    def get_common_features(self, feature_lists):
        """Get common features across multiple models"""
        if not feature_lists:
            return []
        
        common_features = set(feature_lists[0])
        for feature_list in feature_lists[1:]:
            common_features = common_features.intersection(set(feature_list))
        
        return list(common_features)

# Global feature aligner instance
feature_aligner = FeatureAligner()

def align_features_for_model(input_features, model_name, expected_features=None):
    """Convenience function to align features for a specific model"""
    if expected_features is None:
        expected_features = feature_aligner.load_feature_mapping(model_name)
        
    if expected_features is None:
        print(f"⚠️ No feature mapping found for {model_name}")
        return input_features
    
    return feature_aligner.align_features(input_features, expected_features, model_name)

if __name__ == "__main__":
    # Test the feature aligner
    aligner = FeatureAligner()
    
    # Mock test data
    input_data = pd.DataFrame({
        'PTS': [25.5],
        'AST': [7.2], 
        'REB': [8.1],
        'NEW_FEATURE': [1.0]  # Extra feature
    })
    
    target_features = ['PTS', 'AST', 'REB', 'MISSING_FEATURE']
    
    aligned = aligner.align_features(input_data, target_features, "Test Model")
    
    print("Original features:", list(input_data.columns))
    print("Target features:", target_features)
    print("Aligned features:", list(aligned.columns))
    print("Aligned data:", aligned.values)

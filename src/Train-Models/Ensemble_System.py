"""
Advanced ensemble system combining multiple model types with stacking.
Combines XGBoost, LightGBM, Neural Networks, and traditional ML models.
"""
import sqlite3
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class EnsembleNBAPredictor:
    def __init__(self, dataset_name="dataset_2012-24_new"):
        self.dataset_name = dataset_name
        self.base_models = {}
        self.meta_model = None
        self.feature_cols = None
        
    def load_data(self):
        """Load and prepare data"""
        con = sqlite3.connect("../../Data/dataset.sqlite")
        
        # Try enhanced dataset first
        cursor = con.cursor()
        enhanced_name = self.dataset_name.replace("_new", "_enhanced")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (enhanced_name,))
        if cursor.fetchone():
            self.dataset_name = enhanced_name
            print(f"Using enhanced dataset: {enhanced_name}")
            
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        # Parse dates and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Target
        y = df["Home-Team-Win"].astype(int)
        
        # Features
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        # Time-based splits
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        return {
            'X_train': X[train_mask], 'y_train': y[train_mask],
            'X_val': X[val_mask], 'y_val': y[val_mask],
            'X_test': X[test_mask], 'y_test': y[test_mask]
        }
    
    def create_xgboost_model(self):
        """Create XGBoost model"""
        return xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    
    def create_lightgbm_model(self):
        """Create LightGBM model"""
        return lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    def create_neural_network(self, input_dim):
        """Create TensorFlow neural network"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_base_models(self, data):
        """Train all base models"""
        print("Training base models...")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = self.create_xgboost_model()
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        self.base_models['xgboost'] = xgb_model
        
        # LightGBM
        print("Training LightGBM...")
        lgb_model = self.create_lightgbm_model()
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        self.base_models['lightgbm'] = lgb_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.base_models['random_forest'] = rf_model
        
        # Extra Trees
        print("Training Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_train, y_train)
        self.base_models['extra_trees'] = et_model
        
        # Neural Network
        print("Training Neural Network...")
        nn_model = self.create_neural_network(X_train.shape[1])
        
        # Normalize data for neural network
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.base_models['neural_network'] = {'model': nn_model, 'scaler': scaler}
        
        # MLP Classifier (scikit-learn)
        print("Training MLP Classifier...")
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp_model.fit(X_train_scaled, y_train)  # Use scaled data
        self.base_models['mlp'] = {'model': mlp_model, 'scaler': scaler}
        
        print("Base models training complete!")
    
    def generate_meta_features(self, data):
        """Generate meta-features using cross-validation"""
        print("Generating meta-features...")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Generate out-of-fold predictions for meta-learning
        meta_features_train = np.zeros((len(X_train), len(self.base_models)))
        meta_features_val = np.zeros((len(X_val), len(self.base_models)))
        meta_features_test = np.zeros((len(X_test), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Generating meta-features for {name}...")
            
            if name in ['neural_network', 'mlp']:
                # Handle models with scalers
                scaler = model['scaler']
                actual_model = model['model']
                
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                if name == 'neural_network':
                    meta_features_train[:, i] = actual_model.predict(X_train_scaled).flatten()
                    meta_features_val[:, i] = actual_model.predict(X_val_scaled).flatten()
                    meta_features_test[:, i] = actual_model.predict(X_test_scaled).flatten()
                else:  # MLP
                    meta_features_train[:, i] = actual_model.predict_proba(X_train_scaled)[:, 1]
                    meta_features_val[:, i] = actual_model.predict_proba(X_val_scaled)[:, 1]
                    meta_features_test[:, i] = actual_model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Regular sklearn-style models
                meta_features_train[:, i] = model.predict_proba(X_train)[:, 1]
                meta_features_val[:, i] = model.predict_proba(X_val)[:, 1]
                meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]
        
        return {
            'meta_X_train': meta_features_train,
            'meta_X_val': meta_features_val,
            'meta_X_test': meta_features_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def train_meta_model(self, meta_data):
        """Train meta-model (stacker)"""
        print("Training meta-model...")
        
        # Combine train and validation for meta-model training
        meta_X = np.vstack([meta_data['meta_X_train'], meta_data['meta_X_val']])
        meta_y = np.hstack([meta_data['y_train'], meta_data['y_val']])
        
        # Use logistic regression as meta-model
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(meta_X, meta_y)
        
        # Evaluate ensemble
        ensemble_preds = self.meta_model.predict_proba(meta_data['meta_X_test'])[:, 1]
        ensemble_binary = (ensemble_preds >= 0.5).astype(int)
        
        accuracy = accuracy_score(meta_data['y_test'], ensemble_binary)
        auc = roc_auc_score(meta_data['y_test'], ensemble_preds)
        logloss = log_loss(meta_data['y_test'], ensemble_preds)
        
        print(f"\nEnsemble Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        
        # Compare with individual models
        print(f"\nIndividual Model Performance:")
        model_names = list(self.base_models.keys())
        for i, name in enumerate(model_names):
            preds = meta_data['meta_X_test'][:, i]
            binary_preds = (preds >= 0.5).astype(int)
            acc = accuracy_score(meta_data['y_test'], binary_preds)
            auc_score = roc_auc_score(meta_data['y_test'], preds)
            print(f"{name}: Accuracy={acc:.4f}, AUC={auc_score:.4f}")
    
    def train_ensemble(self):
        """Train complete ensemble system"""
        print("Loading data...")
        data = self.load_data()
        
        print(f"Training set size: {len(data['X_train'])}")
        print(f"Validation set size: {len(data['X_val'])}")
        print(f"Test set size: {len(data['X_test'])}")
        
        # Train base models
        self.train_base_models(data)
        
        # Generate meta-features
        meta_data = self.generate_meta_features(data)
        
        # Train meta-model
        self.train_meta_model(meta_data)
        
        print("Ensemble training complete!")
    
    def predict_game(self, game_features):
        """Make ensemble prediction for a single game"""
        if isinstance(game_features, pd.DataFrame):
            X = game_features[self.feature_cols].values.reshape(1, -1)
        else:
            X = np.array(game_features).reshape(1, -1)
        
        # Get predictions from all base models
        base_predictions = np.zeros((1, len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if name in ['neural_network', 'mlp']:
                scaler = model['scaler']
                actual_model = model['model']
                X_scaled = scaler.transform(X)
                
                if name == 'neural_network':
                    base_predictions[0, i] = actual_model.predict(X_scaled)[0, 0]
                else:  # MLP
                    base_predictions[0, i] = actual_model.predict_proba(X_scaled)[0, 1]
            else:
                base_predictions[0, i] = model.predict_proba(X)[0, 1]
        
        # Get ensemble prediction
        ensemble_prob = self.meta_model.predict_proba(base_predictions)[0, 1]
        ensemble_pred = int(ensemble_prob >= 0.5)
        
        return {
            'ensemble_probability': ensemble_prob,
            'ensemble_prediction': ensemble_pred,
            'base_predictions': dict(zip(self.base_models.keys(), base_predictions[0]))
        }
    
    def save_ensemble(self, base_name="Ensemble_NBA"):
        """Save entire ensemble system"""
        # Save base models
        for name, model in self.base_models.items():
            if name in ['neural_network', 'mlp']:
                if name == 'neural_network':
                    model['model'].save(f"../../Models/Ensemble_Models/{base_name}_{name}.h5")
                else:
                    joblib.dump(model['model'], f"../../Models/Ensemble_Models/{base_name}_{name}.pkl")
                joblib.dump(model['scaler'], f"../../Models/Ensemble_Models/{base_name}_{name}_scaler.pkl")
            else:
                joblib.dump(model, f"../../Models/Ensemble_Models/{base_name}_{name}.pkl")
        
        # Save meta-model
        joblib.dump(self.meta_model, f"../../Models/Ensemble_Models/{base_name}_meta_model.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_cols, f"../../Models/Ensemble_Models/{base_name}_features.pkl")
        
        print(f"Ensemble saved with base name: {base_name}")

if __name__ == "__main__":
    # Create directory for ensemble models
    import os
    os.makedirs("../../Models/Ensemble_Models", exist_ok=True)
    
    ensemble = EnsembleNBAPredictor()
    ensemble.train_ensemble()
    ensemble.save_ensemble("Ensemble_NBA_v1")

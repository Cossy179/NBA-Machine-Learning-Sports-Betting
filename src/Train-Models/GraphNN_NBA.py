"""
Graph Neural Network for NBA predictions.
Models complex relationships between teams, players, and game dynamics.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class NBAGraphNeuralNetwork:
    def __init__(self, dataset_name="dataset_2012-24_enhanced"):
        self.dataset_name = dataset_name
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.team_encoder = None
        self.num_teams = 30  # NBA teams
        
    def load_data(self):
        """Load and prepare graph data"""
        con = sqlite3.connect("Data/dataset.sqlite")
        
        # Try enhanced dataset first
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.dataset_name,))
        if not cursor.fetchone():
            self.dataset_name = "dataset_2012-24_new"
            print(f"Using base dataset: {self.dataset_name}")
        else:
            print(f"Using enhanced dataset: {self.dataset_name}")
            
        df = pd.read_sql_query(f'select * from "{self.dataset_name}"', con, index_col="index")
        con.close()
        
        # Parse dates and sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Encode teams
        self.team_encoder = LabelEncoder()
        all_teams = list(df['TEAM_NAME'].unique()) + list(df['TEAM_NAME.1'].unique())
        unique_teams = list(set(all_teams))
        self.team_encoder.fit(unique_teams)
        self.num_teams = len(unique_teams)
        
        print(f"Number of teams: {self.num_teams}")
        
        # Target
        y = df["Home-Team-Win"].astype(int)
        
        # Features
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        # Create graph data
        graph_data = self.create_graph_data(df, X, y)
        
        # Time-based splits
        train_mask = df["Date"] < pd.Timestamp("2022-01-01")
        val_mask = (df["Date"] >= pd.Timestamp("2022-01-01")) & (df["Date"] < pd.Timestamp("2023-01-01"))
        test_mask = df["Date"] >= pd.Timestamp("2023-01-01")
        
        return {
            'X_train': graph_data['node_features'][train_mask],
            'edges_train': graph_data['edge_indices'][train_mask],
            'edge_features_train': graph_data['edge_features'][train_mask],
            'y_train': y[train_mask],
            
            'X_val': graph_data['node_features'][val_mask],
            'edges_val': graph_data['edge_indices'][val_mask],
            'edge_features_val': graph_data['edge_features'][val_mask],
            'y_val': y[val_mask],
            
            'X_test': graph_data['node_features'][test_mask],
            'edges_test': graph_data['edge_indices'][test_mask],
            'edge_features_test': graph_data['edge_features'][test_mask],
            'y_test': y[test_mask]
        }
    
    def create_graph_data(self, df, X, y):
        """Create graph representation of games"""
        node_features = []
        edge_indices = []
        edge_features = []
        
        for idx, row in df.iterrows():
            home_team = row['TEAM_NAME']
            away_team = row['TEAM_NAME.1']
            
            # Encode team IDs
            home_id = self.team_encoder.transform([home_team])[0]
            away_id = self.team_encoder.transform([away_team])[0]
            
            # Node features (team embeddings will be learned)
            # For now, use one-hot encoding
            home_features = np.zeros(self.num_teams)
            away_features = np.zeros(self.num_teams)
            home_features[home_id] = 1
            away_features[away_id] = 1
            
            # Combine with game features
            game_features = X.iloc[idx].values
            home_node = np.concatenate([home_features, game_features[:len(game_features)//2]])
            away_node = np.concatenate([away_features, game_features[len(game_features)//2:]])
            
            # Pad to same length if needed
            max_len = max(len(home_node), len(away_node))
            if len(home_node) < max_len:
                home_node = np.pad(home_node, (0, max_len - len(home_node)))
            if len(away_node) < max_len:
                away_node = np.pad(away_node, (0, max_len - len(away_node)))
            
            node_features.append([home_node, away_node])
            
            # Edge between home and away team (bidirectional)
            edge_indices.append([[0, 1], [1, 0]])  # Home->Away, Away->Home
            
            # Edge features (matchup information)
            matchup_features = self.calculate_matchup_features(home_team, away_team, row)
            edge_features.append([matchup_features, matchup_features])  # Same for both directions
        
        return {
            'node_features': np.array(node_features),
            'edge_indices': np.array(edge_indices),
            'edge_features': np.array(edge_features)
        }
    
    def calculate_matchup_features(self, home_team, away_team, game_row):
        """Calculate features specific to team matchup"""
        # This would include head-to-head history, style matchups, etc.
        # For now, use basic features
        features = [
            1.0,  # Home court advantage
            0.0,  # Rivalry indicator (would be calculated from data)
            0.5,  # Historical win rate (would be calculated)
            0.0,  # Rest advantage
            0.5   # Strength differential
        ]
        return np.array(features)
    
    def create_graph_model(self, node_feature_dim, edge_feature_dim):
        """Create Graph Neural Network model"""
        # Input layers
        node_features = keras.Input(shape=(2, node_feature_dim), name='node_features')
        edge_indices = keras.Input(shape=(2, 2), dtype=tf.int32, name='edge_indices')
        edge_features = keras.Input(shape=(2, edge_feature_dim), name='edge_features')
        
        # Graph convolution layers
        x = self.graph_conv_layer(node_features, edge_indices, edge_features, 128)
        x = self.graph_conv_layer(x, edge_indices, edge_features, 64)
        x = self.graph_conv_layer(x, edge_indices, edge_features, 32)
        
        # Global pooling (aggregate node features)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model([node_features, edge_indices, edge_features], outputs, name='nba_graph_nn')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def graph_conv_layer(self, node_features, edge_indices, edge_features, output_dim):
        """Graph convolution layer implementation"""
        # This is a simplified graph convolution
        # In practice, you'd use libraries like DGL or PyTorch Geometric
        
        # Node transformation
        node_transform = layers.Dense(output_dim, activation='relu')
        transformed_nodes = node_transform(node_features)
        
        # Edge transformation
        edge_transform = layers.Dense(output_dim, activation='relu')
        transformed_edges = edge_transform(edge_features)
        
        # Message passing (simplified)
        # In a full implementation, this would aggregate messages from neighbors
        messages = layers.Add()([transformed_nodes, 
                               tf.expand_dims(tf.reduce_mean(transformed_edges, axis=1), axis=1)])
        
        # Layer normalization
        messages = layers.LayerNormalization()(messages)
        
        return messages
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the Graph Neural Network"""
        print("Loading and preparing graph data...")
        data = self.load_data()
        
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        print(f"Test samples: {len(data['X_test'])}")
        
        # Scale node features
        print("Scaling node features...")
        self.scaler = StandardScaler()
        
        # Reshape for scaling
        X_train_reshaped = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        X_val_reshaped = data['X_val'].reshape(-1, data['X_val'].shape[-1])
        X_test_reshaped = data['X_test'].reshape(-1, data['X_test'].shape[-1])
        
        # Fit and transform
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(data['X_train'].shape)
        X_val_scaled = X_val_scaled.reshape(data['X_val'].shape)
        X_test_scaled = X_test_scaled.reshape(data['X_test'].shape)
        
        # Create model
        print("Creating Graph Neural Network...")
        node_feature_dim = X_train_scaled.shape[-1]
        edge_feature_dim = data['edge_features_train'].shape[-1]
        
        self.model = self.create_graph_model(node_feature_dim, edge_feature_dim)
        
        print(f"Node feature dimension: {node_feature_dim}")
        print(f"Edge feature dimension: {edge_feature_dim}")
        print(f"Model parameters: {self.model.count_params():,}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'Models/NN_Models/graph_nba_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        print("Training Graph Neural Network...")
        history = self.model.fit(
            [X_train_scaled, data['edges_train'], data['edge_features_train']],
            data['y_train'],
            validation_data=(
                [X_val_scaled, data['edges_val'], data['edge_features_val']],
                data['y_val']
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_predictions = self.model.predict([
            X_test_scaled, 
            data['edges_test'], 
            data['edge_features_test']
        ])
        
        test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(data['y_test'], test_predictions_binary)
        auc = roc_auc_score(data['y_test'], test_predictions.flatten())
        logloss = log_loss(data['y_test'], test_predictions.flatten())
        
        print(f"\nGraph Neural Network Performance:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Log Loss: {logloss:.4f}")
        
        return {
            'history': history,
            'test_accuracy': accuracy,
            'test_auc': auc,
            'test_logloss': logloss
        }
    
    def predict_game(self, home_team, away_team, game_features):
        """Make prediction for a specific matchup"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Create graph data for this game
        home_id = self.team_encoder.transform([home_team])[0]
        away_id = self.team_encoder.transform([away_team])[0]
        
        # Node features
        home_features = np.zeros(self.num_teams)
        away_features = np.zeros(self.num_teams)
        home_features[home_id] = 1
        away_features[away_id] = 1
        
        # Combine with game features
        home_node = np.concatenate([home_features, game_features[:len(game_features)//2]])
        away_node = np.concatenate([away_features, game_features[len(game_features)//2:]])
        
        # Pad to same length
        max_len = max(len(home_node), len(away_node))
        if len(home_node) < max_len:
            home_node = np.pad(home_node, (0, max_len - len(home_node)))
        if len(away_node) < max_len:
            away_node = np.pad(away_node, (0, max_len - len(away_node)))
        
        node_features = np.array([[home_node, away_node]])
        edge_indices = np.array([[[0, 1], [1, 0]]])
        
        # Edge features
        matchup_features = self.calculate_matchup_features(home_team, away_team, None)
        edge_features = np.array([[matchup_features, matchup_features]])
        
        # Scale node features
        node_features_reshaped = node_features.reshape(-1, node_features.shape[-1])
        node_features_scaled = self.scaler.transform(node_features_reshaped)
        node_features_scaled = node_features_scaled.reshape(node_features.shape)
        
        # Make prediction
        prediction = self.model.predict([
            node_features_scaled,
            edge_indices,
            edge_features
        ], verbose=0)
        
        return {
            'probability': float(prediction[0][0]),
            'prediction': int(prediction[0][0] > 0.5),
            'confidence': abs(prediction[0][0] - 0.5) * 2
        }
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"Models/NN_Models/NBA_GraphNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.team_encoder, f"{filepath}_team_encoder.pkl")
        joblib.dump({
            'feature_cols': self.feature_cols,
            'num_teams': self.num_teams,
            'dataset_name': self.dataset_name
        }, f"{filepath}_metadata.pkl")
        
        print(f"Graph Neural Network saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        import joblib
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}.h5")
        
        # Load scaler and metadata
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.team_encoder = joblib.load(f"{filepath}_team_encoder.pkl")
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        
        self.feature_cols = metadata['feature_cols']
        self.num_teams = metadata['num_teams']
        self.dataset_name = metadata['dataset_name']
        
        print(f"Graph Neural Network loaded from {filepath}")

if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs("Models/NN_Models", exist_ok=True)
    
    # Create and train Graph Neural Network
    gnn = NBAGraphNeuralNetwork()
    
    # Train model
    results = gnn.train_model(epochs=50, batch_size=16)
    
    # Save model
    gnn.save_model()
    
    print("Graph Neural Network training complete!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Final test AUC: {results['test_auc']:.4f}")

"""
Transformer-based NBA prediction model using attention mechanisms.
Captures complex relationships between game features and temporal patterns.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class NBATransformerPredictor:
    def __init__(self, dataset_name="dataset_2012-24_enhanced"):
        self.dataset_name = dataset_name
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.sequence_length = 10  # Look at last 10 games
        
    def load_data(self):
        """Load and prepare sequential data for transformer"""
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
        df = df.sort_values(["Date", "TEAM_NAME"]).reset_index(drop=True)
        
        # Target
        y = df["Home-Team-Win"].astype(int)
        
        # Features
        exclude_cols = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1", "Date.1", "OU", "OU-Cover"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not pd.isna(df[c]).all()]
        X = df[self.feature_cols].fillna(0).astype(float)
        
        print(f"Features: {len(self.feature_cols)}")
        print(f"Samples: {len(X)}")
        
        # Create sequences for transformer
        X_sequences, y_sequences = self.create_sequences(X, y, df)
        
        # Time-based splits
        split_date1 = pd.Timestamp("2022-01-01")
        split_date2 = pd.Timestamp("2023-01-01")
        
        # Get dates for sequences (use the last date in each sequence)
        sequence_dates = df["Date"].iloc[self.sequence_length-1::1][:len(X_sequences)]
        
        train_mask = sequence_dates < split_date1
        val_mask = (sequence_dates >= split_date1) & (sequence_dates < split_date2)
        test_mask = sequence_dates >= split_date2
        
        return {
            'X_train': X_sequences[train_mask], 'y_train': y_sequences[train_mask],
            'X_val': X_sequences[val_mask], 'y_val': y_sequences[val_mask],
            'X_test': X_sequences[test_mask], 'y_test': y_sequences[test_mask]
        }
    
    def create_sequences(self, X, y, df):
        """Create sequences for transformer input"""
        sequences = []
        targets = []
        
        # Group by teams to create sequences
        teams = df['TEAM_NAME'].unique()
        
        for team in teams:
            team_mask = df['TEAM_NAME'] == team
            team_X = X[team_mask].values
            team_y = y[team_mask].values
            
            # Create sequences for this team
            for i in range(self.sequence_length, len(team_X)):
                sequence = team_X[i-self.sequence_length:i]  # Last N games
                target = team_y[i]  # Current game result
                
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def create_transformer_model(self, input_shape):
        """Create transformer model architecture"""
        inputs = keras.Input(shape=input_shape)
        
        # Positional encoding
        x = self.positional_encoding(inputs)
        
        # Multi-head attention layers
        attention_output = self.multi_head_attention_block(x, num_heads=8, key_dim=64)
        attention_output = self.multi_head_attention_block(attention_output, num_heads=8, key_dim=64)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(attention_output)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs, name='nba_transformer')
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def positional_encoding(self, x):
        """Add positional encoding to input sequences"""
        seq_len = tf.shape(x)[1]
        feature_dim = tf.shape(x)[2]
        
        # Create positional encoding
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(feature_dim, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(feature_dim, tf.float32))
        angle_rads = pos * angle_rates
        
        # Apply sin to even indices and cos to odd indices
        pos_encoding = tf.where(
            tf.equal(i % 2, 0),
            tf.sin(angle_rads),
            tf.cos(angle_rads)
        )
        
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return x + pos_encoding
    
    def multi_head_attention_block(self, x, num_heads, key_dim):
        """Multi-head attention block with residual connection"""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim,
            dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        ffn = keras.Sequential([
            layers.Dense(key_dim * 4, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(x.shape[-1])
        ])
        
        ffn_output = ffn(x)
        
        # Add & Norm
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1):
        """Train the transformer model"""
        print("Loading and preparing data...")
        data = self.load_data()
        
        print(f"Training sequences: {len(data['X_train'])}")
        print(f"Validation sequences: {len(data['X_val'])}")
        print(f"Test sequences: {len(data['X_test'])}")
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        
        # Reshape for scaling
        X_train_reshaped = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        X_val_reshaped = data['X_val'].reshape(-1, data['X_val'].shape[-1])
        X_test_reshaped = data['X_test'].reshape(-1, data['X_test'].shape[-1])
        
        # Fit scaler and transform
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled.reshape(data['X_train'].shape)
        X_val_scaled = X_val_scaled.reshape(data['X_val'].shape)
        X_test_scaled = X_test_scaled.reshape(data['X_test'].shape)
        
        # Create model
        print("Creating transformer model...")
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        self.model = self.create_transformer_model(input_shape)
        
        print(f"Model input shape: {input_shape}")
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
                'Models/NN_Models/transformer_nba_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        print("Training transformer model...")
        history = self.model.fit(
            X_train_scaled, data['y_train'],
            validation_data=(X_val_scaled, data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_predictions = self.model.predict(X_test_scaled)
        test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(data['y_test'], test_predictions_binary)
        auc = roc_auc_score(data['y_test'], test_predictions.flatten())
        logloss = log_loss(data['y_test'], test_predictions.flatten())
        
        print(f"\nTransformer Model Performance:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Log Loss: {logloss:.4f}")
        
        return {
            'history': history,
            'test_accuracy': accuracy,
            'test_auc': auc,
            'test_logloss': logloss
        }
    
    def predict_game(self, game_sequence):
        """Make prediction for a game given sequence of previous games"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Ensure sequence has correct shape
        if len(game_sequence.shape) == 2:
            game_sequence = game_sequence.reshape(1, game_sequence.shape[0], game_sequence.shape[1])
        
        # Scale features
        sequence_reshaped = game_sequence.reshape(-1, game_sequence.shape[-1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(game_sequence.shape)
        
        # Make prediction
        prediction = self.model.predict(sequence_scaled, verbose=0)
        
        return {
            'probability': float(prediction[0][0]),
            'prediction': int(prediction[0][0] > 0.5),
            'confidence': abs(prediction[0][0] - 0.5) * 2
        }
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f"Models/NN_Models/NBA_Transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump({
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'dataset_name': self.dataset_name
        }, f"{filepath}_metadata.pkl")
        
        print(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        import joblib
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}.h5")
        
        # Load scaler and metadata
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        
        self.feature_cols = metadata['feature_cols']
        self.sequence_length = metadata['sequence_length']
        self.dataset_name = metadata['dataset_name']
        
        print(f"Transformer model loaded from {filepath}")

if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs("Models/NN_Models", exist_ok=True)
    
    # Create and train transformer
    transformer = NBATransformerPredictor()
    
    # Train model
    results = transformer.train_model(epochs=50, batch_size=16)
    
    # Save model
    transformer.save_model()
    
    print("Transformer training complete!")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Final test AUC: {results['test_auc']:.4f}")

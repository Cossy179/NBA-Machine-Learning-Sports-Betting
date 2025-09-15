import copy
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Add, Multiply, Lambda
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc
import warnings
warnings.filterwarnings('ignore')

init()

_model = None
_ou_model = None
_advanced_models = {}
_scalers = {}

def _load_models():
    global _model, _ou_model
    if _model is None:
        _model = load_model('Models/NN_Models/Trained-Model-ML-1699315388.285516')
    if _ou_model is None:
        _ou_model = load_model("Models/NN_Models/Trained-Model-OU-1699315414.2268295")

def create_advanced_neural_network(input_dim, model_type='ensemble', num_outputs=2):
    """Create advanced neural network architectures for NBA prediction"""
    
    if model_type == 'ensemble':
        # Multi-branch ensemble architecture
        input_layer = Input(shape=(input_dim,), name='input')
        
        # Branch 1: Deep learning path
        branch1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(input_layer)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.3)(branch1)
        
        branch1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.3)(branch1)
        
        branch1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Dropout(0.2)(branch1)
        
        # Branch 2: Wide learning path
        branch2 = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(input_layer)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.4)(branch2)
        
        branch2 = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Dropout(0.3)(branch2)
        
        # Branch 3: Residual learning path
        branch3 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(input_layer)
        branch3 = BatchNormalization()(branch3)
        branch3 = Dropout(0.2)(branch3)
        
        # Residual block 1
        residual1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(branch3)
        residual1 = BatchNormalization()(residual1)
        residual1 = Dropout(0.2)(residual1)
        residual1 = Dense(256, activation='linear', kernel_regularizer=l1_l2(0.001, 0.001))(residual1)
        branch3 = Add()([branch3, residual1])
        branch3 = tf.keras.activations.relu(branch3)
        
        # Residual block 2
        residual2 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(branch3)
        residual2 = BatchNormalization()(residual2)
        residual2 = Dropout(0.2)(residual2)
        residual2 = Dense(256, activation='linear', kernel_regularizer=l1_l2(0.001, 0.001))(residual2)
        branch3 = Add()([branch3, residual2])
        branch3 = tf.keras.activations.relu(branch3)
        
        # Combine branches
        combined = Concatenate()([branch1, branch2, branch3])
        
        # Attention mechanism
        attention_weights = Dense(combined.shape[-1], activation='softmax', name='attention')(combined)
        attended_features = Multiply()([combined, attention_weights])
        
        # Final layers
        final = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(attended_features)
        final = BatchNormalization()(final)
        final = Dropout(0.3)(final)
        
        final = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(final)
        final = BatchNormalization()(final)
        final = Dropout(0.2)(final)
        
        final = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(final)
        final = BatchNormalization()(final)
        final = Dropout(0.1)(final)
        
        # Output layer
        if num_outputs == 2:
            output = Dense(2, activation='softmax', name='prediction')(final)
        else:
            output = Dense(num_outputs, activation='linear', name='prediction')(final)
        
        model = Model(inputs=input_layer, outputs=output)
        
    elif model_type == 'transformer':
        # Transformer-inspired architecture
        input_layer = Input(shape=(input_dim,), name='input')
        
        # Embedding layer
        embedded = Dense(512, activation='linear')(input_layer)
        embedded = BatchNormalization()(embedded)
        
        # Multi-head attention simulation
        attention1 = Dense(256, activation='relu')(embedded)
        attention1 = Dense(512, activation='softmax')(attention1)
        attended1 = Multiply()([embedded, attention1])
        
        attention2 = Dense(256, activation='relu')(embedded)
        attention2 = Dense(512, activation='softmax')(attention2)
        attended2 = Multiply()([embedded, attention2])
        
        # Combine attention heads
        multi_head = Concatenate()([attended1, attended2])
        multi_head = Dense(512, activation='relu')(multi_head)
        multi_head = BatchNormalization()(multi_head)
        multi_head = Dropout(0.3)(multi_head)
        
        # Feed-forward network
        ff1 = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(multi_head)
        ff1 = BatchNormalization()(ff1)
        ff1 = Dropout(0.4)(ff1)
        
        ff2 = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(ff1)
        ff2 = BatchNormalization()(ff2)
        ff2 = Dropout(0.3)(ff2)
        
        # Output layer
        if num_outputs == 2:
            output = Dense(2, activation='softmax', name='prediction')(ff2)
        else:
            output = Dense(num_outputs, activation='linear', name='prediction')(ff2)
        
        model = Model(inputs=input_layer, outputs=output)
        
    elif model_type == 'cnn_like':
        # CNN-like architecture for sequential patterns
        input_layer = Input(shape=(input_dim,), name='input')
        
        # Reshape for 1D convolution
        reshaped = Lambda(lambda x: tf.expand_dims(x, -1))(input_layer)
        
        # 1D Convolution layers
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(reshaped)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        conv3 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(0.2)(conv3)
        
        # Global max pooling
        pooled = tf.keras.layers.GlobalMaxPooling1D()(conv3)
        
        # Dense layers
        dense1 = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        if num_outputs == 2:
            output = Dense(2, activation='softmax', name='prediction')(dense2)
        else:
            output = Dense(num_outputs, activation='linear', name='prediction')(dense2)
        
        model = Model(inputs=input_layer, outputs=output)
    
    else:  # Default advanced architecture
        input_layer = Input(shape=(input_dim,), name='input')
        
        # Deep network with skip connections
        x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Skip connection
        skip1 = Dense(512, activation='linear')(x)
        
        x = Dense(512, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Add skip connection
        x = Add()([x, skip1])
        x = tf.keras.activations.relu(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        if num_outputs == 2:
            output = Dense(2, activation='softmax', name='prediction')(x)
        else:
            output = Dense(num_outputs, activation='linear', name='prediction')(x)
        
        model = Model(inputs=input_layer, outputs=output)
    
    return model

def train_advanced_neural_network(X, y, model_type='ensemble', validation_split=0.2, epochs=200):
    """Train advanced neural network with sophisticated techniques"""
    
    # Prepare data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create model
    input_dim = X_train_scaled.shape[1]
    num_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 2
    
    model = create_advanced_neural_network(input_dim, model_type, num_outputs)
    
    # Compile model with advanced optimizer
    if model_type == 'transformer':
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    else:
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    if num_outputs == 2:
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
        ModelCheckpoint(f'best_model_{model_type}.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, scaler, history

def load_advanced_models():
    """Load advanced neural network models"""
    global _advanced_models, _scalers
    
    try:
        # Try to load pre-trained advanced models
        model_types = ['ensemble', 'transformer', 'cnn_like', 'default']
        
        for model_type in model_types:
            try:
                model = load_model(f'Models/NN_Models/advanced_{model_type}_model.h5')
                scaler = StandardScaler()
                # Load scaler (would need to be saved separately)
                _advanced_models[model_type] = model
                _scalers[model_type] = scaler
                print(f"Loaded advanced {model_type} model")
            except:
                print(f"Advanced {model_type} model not found, will train if needed")
                
    except Exception as e:
        print(f"Error loading advanced models: {e}")

def predict_with_advanced_models(data, model_type='ensemble'):
    """Make predictions using advanced neural network models"""
    global _advanced_models, _scalers
    
    if model_type not in _advanced_models:
        print(f"Advanced {model_type} model not available")
        return None
    
    model = _advanced_models[model_type]
    scaler = _scalers.get(model_type, None)
    
    if scaler:
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data
    
    predictions = model.predict(data_scaled)
    
    # Calculate uncertainty using Monte Carlo Dropout
    if hasattr(model, 'layers'):
        # Enable dropout during inference for uncertainty estimation
        for layer in model.layers:
            if isinstance(layer, Dropout):
                layer.training = True
        
        # Multiple forward passes for uncertainty
        mc_predictions = []
        for _ in range(10):  # 10 Monte Carlo samples
            mc_pred = model.predict(data_scaled, verbose=0)
            mc_predictions.append(mc_pred)
        
        mc_predictions = np.array(mc_predictions)
        mean_predictions = np.mean(mc_predictions, axis=0)
        uncertainty = np.std(mc_predictions, axis=0)
        
        # Disable dropout
        for layer in model.layers:
            if isinstance(layer, Dropout):
                layer.training = False
        
        return {
            'predictions': mean_predictions,
            'uncertainty': uncertainty,
            'confidence': 1 - uncertainty  # Higher uncertainty = lower confidence
        }
    
    return {
        'predictions': predictions,
        'uncertainty': np.zeros_like(predictions),
        'confidence': np.ones_like(predictions)
    }

def advanced_nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """Advanced neural network runner with multiple architectures and uncertainty quantification"""
    _load_models()
    load_advanced_models()
    
    print("=" * 60)
    print("ADVANCED NEURAL NETWORK PREDICTIONS")
    print("=" * 60)
    
    # Get predictions from all available models
    model_predictions = {}
    
    # Original models
    ml_predictions_array = []
    for row in data:
        ml_predictions_array.append(_model.predict(np.array([row])))
    
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    ou_data = frame_uo.values
    ou_data = ou_data.astype(float)
    ou_data = tf.keras.utils.normalize(ou_data, axis=1)
    
    ou_predictions_array = []
    for row in ou_data:
        ou_predictions_array.append(_ou_model.predict(np.array([row])))
    
    model_predictions['original_ml'] = ml_predictions_array
    model_predictions['original_ou'] = ou_predictions_array
    
    # Advanced models
    advanced_model_types = ['ensemble', 'transformer', 'cnn_like', 'default']
    
    for model_type in advanced_model_types:
        if model_type in _advanced_models:
            try:
                # ML predictions
                ml_result = predict_with_advanced_models(data, model_type)
                if ml_result:
                    model_predictions[f'{model_type}_ml'] = ml_result
                
                # OU predictions
                ou_result = predict_with_advanced_models(ou_data, model_type)
                if ou_result:
                    model_predictions[f'{model_type}_ou'] = ou_result
                    
            except Exception as e:
                print(f"Error with {model_type} model: {e}")
    
    # Display predictions for each game
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{home_team}{Style.RESET_ALL} vs {Fore.RED}{away_team}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        # Original model predictions
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count][0][1] if winner == 1 else ml_predictions_array[count][0][0]
        un_confidence = ou_predictions_array[count][0][1] if under_over == 1 else ou_predictions_array[count][0][0]
        
        print(f"\n{Fore.MAGENTA}🏆 ORIGINAL NEURAL NETWORK:{Style.RESET_ALL}")
        winner_name = home_team if winner == 1 else away_team
        winner_color = Fore.GREEN if winner == 1 else Fore.RED
        print(f"   Winner: {winner_color}{winner_name}{Style.RESET_ALL} ({winner_confidence:.1%})")
        
        ou_recommendation = "OVER" if under_over == 1 else "UNDER"
        ou_color = Fore.BLUE if under_over == 1 else Fore.MAGENTA
        print(f"   O/U: {ou_color}{ou_recommendation} {todays_games_uo[count]}{Style.RESET_ALL} ({un_confidence:.1%})")
        
        # Advanced model predictions
        for model_type in advanced_model_types:
            ml_key = f'{model_type}_ml'
            ou_key = f'{model_type}_ou'
            
            if ml_key in model_predictions and ou_key in model_predictions:
                ml_result = model_predictions[ml_key]
                ou_result = model_predictions[ou_key]
                
                ml_pred = ml_result['predictions'][count]
                ou_pred = ou_result['predictions'][count]
                ml_uncertainty = ml_result['uncertainty'][count]
                ou_uncertainty = ou_result['uncertainty'][count]
                ml_confidence = ml_result['confidence'][count]
                ou_confidence = ou_result['confidence'][count]
                
                print(f"\n{Fore.BLUE}🧠 {model_type.upper()} NEURAL NETWORK:{Style.RESET_ALL}")
                
                # ML prediction
                if len(ml_pred.shape) > 1:
                    winner_adv = int(np.argmax(ml_pred))
                    winner_prob_adv = ml_pred[0][1] if winner_adv == 1 else ml_pred[0][0]
                else:
                    winner_adv = 1 if ml_pred[0] > 0.5 else 0
                    winner_prob_adv = ml_pred[0] if winner_adv == 1 else 1 - ml_pred[0]
                
                winner_name_adv = home_team if winner_adv == 1 else away_team
                winner_color_adv = Fore.GREEN if winner_adv == 1 else Fore.RED
                print(f"   Winner: {winner_color_adv}{winner_name_adv}{Style.RESET_ALL} ({winner_prob_adv:.1%})")
                print(f"   Confidence: {ml_confidence:.1%}")
                print(f"   Uncertainty: {ml_uncertainty:.3f}")
                
                # OU prediction
                if len(ou_pred.shape) > 1:
                    ou_adv = int(np.argmax(ou_pred))
                    ou_prob_adv = ou_pred[0][1] if ou_adv == 1 else ou_pred[0][0]
                else:
                    ou_adv = 1 if ou_pred[0] > 0.5 else 0
                    ou_prob_adv = ou_pred[0] if ou_adv == 1 else 1 - ou_pred[0]
                
                ou_recommendation_adv = "OVER" if ou_adv == 1 else "UNDER"
                ou_color_adv = Fore.BLUE if ou_adv == 1 else Fore.MAGENTA
                print(f"   O/U: {ou_color_adv}{ou_recommendation_adv} {todays_games_uo[count]}{Style.RESET_ALL} ({ou_prob_adv:.1%})")
                print(f"   Confidence: {ou_confidence:.1%}")
                print(f"   Uncertainty: {ou_uncertainty:.3f}")
        
        count += 1
    
    # Advanced betting analysis
    if kelly_criterion:
        print(f"\n{Fore.YELLOW}💰 ADVANCED BETTING ANALYSIS:{Style.RESET_ALL}")
        print("=" * 60)
        
        count = 0
        for game in games:
            home_team = game[0]
            away_team = game[1]
            
            print(f"\n{home_team} vs {away_team}:")
            
            # Original model analysis
            if home_team_odds[count] and away_team_odds[count]:
                ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                kelly_home = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])
                kelly_away = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])
                
                print(f"   Original Model:")
                print(f"     {home_team} EV: {Fore.GREEN if ev_home > 0 else Fore.RED}{ev_home:+.3f}{Style.RESET_ALL}, Kelly: {kelly_home:.1f}%")
                print(f"     {away_team} EV: {Fore.GREEN if ev_away > 0 else Fore.RED}{ev_away:+.3f}{Style.RESET_ALL}, Kelly: {kelly_away:.1f}%")
            
            # Advanced model analysis
            for model_type in advanced_model_types:
                ml_key = f'{model_type}_ml'
                if ml_key in model_predictions and home_team_odds[count] and away_team_odds[count]:
                    ml_result = model_predictions[ml_key]
                    ml_pred = ml_result['predictions'][count]
                    ml_uncertainty = ml_result['uncertainty'][count]
                    
                    if len(ml_pred.shape) > 1:
                        home_prob = ml_pred[0][1]
                        away_prob = ml_pred[0][0]
                    else:
                        home_prob = ml_pred[0]
                        away_prob = 1 - ml_pred[0]
                    
                    # Adjust probabilities based on uncertainty
                    uncertainty_factor = max(0.1, 1 - ml_uncertainty)
                    home_prob_adj = home_prob * uncertainty_factor + 0.5 * (1 - uncertainty_factor)
                    away_prob_adj = away_prob * uncertainty_factor + 0.5 * (1 - uncertainty_factor)
                    
                    ev_home_adv = float(Expected_Value.expected_value(home_prob_adj, int(home_team_odds[count])))
                    ev_away_adv = float(Expected_Value.expected_value(away_prob_adj, int(away_team_odds[count])))
                    kelly_home_adv = kc.calculate_kelly_criterion(home_team_odds[count], home_prob_adj)
                    kelly_away_adv = kc.calculate_kelly_criterion(away_team_odds[count], away_prob_adj)
                    
                    print(f"   {model_type.title()} Model:")
                    print(f"     {home_team} EV: {Fore.GREEN if ev_home_adv > 0 else Fore.RED}{ev_home_adv:+.3f}{Style.RESET_ALL}, Kelly: {kelly_home_adv:.1f}%")
                    print(f"     {away_team} EV: {Fore.GREEN if ev_away_adv > 0 else Fore.RED}{ev_away_adv:+.3f}{Style.RESET_ALL}, Kelly: {kelly_away_adv:.1f}%")
            
            count += 1
    
    deinit()

def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """Original neural network runner (backward compatibility)"""
    _load_models()
    
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(_model.predict(np.array([row])))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(_ou_model.predict(np.array([row])))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                      Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                      Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                      Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                      Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        count += 1
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED, 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1

    deinit()

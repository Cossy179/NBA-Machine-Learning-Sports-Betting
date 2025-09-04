"""
AI-Powered Parlay Prediction System using player statistics, correlations, and machine learning.
Generates optimal parlay combinations with risk assessment and expected value calculations.
"""
import pandas as pd
import numpy as np
import sqlite3
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ParlayPredictor:
    def __init__(self):
        self.player_models = {}
        self.correlation_matrix = None
        self.scaler = StandardScaler()
        self.prop_models = {
            'points': None,
            'rebounds': None,
            'assists': None,
            'threes': None,
            'steals_blocks': None
        }
        
    def load_player_data(self):
        """Load comprehensive player statistics"""
        try:
            con = sqlite3.connect("Data/PlayerStats.sqlite")
            
            # Get player stats with game logs
            query = """
            SELECT * FROM player_stats_comprehensive
            WHERE GP > 10  -- Only players with significant games played
            """
            
            player_data = pd.read_sql_query(query, con)
            con.close()
            
            if player_data.empty:
                print("No player data found. Please run PlayerStatsProvider first.")
                return pd.DataFrame()
                
            # Clean and prepare data
            numeric_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'FGA', 'FG_PCT', 'MIN', 'GP']
            for col in numeric_cols:
                if col in player_data.columns:
                    player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)
            
            return player_data
            
        except Exception as e:
            print(f"Error loading player data: {e}")
            return pd.DataFrame()
    
    def calculate_player_correlations(self, player_data):
        """Calculate correlations between player stats for parlay optimization"""
        print("Calculating player stat correlations...")
        
        # Focus on key stats for props
        stat_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M', 'MIN']
        available_cols = [col for col in stat_cols if col in player_data.columns]
        
        if len(available_cols) < 2:
            print("Insufficient stat columns for correlation analysis")
            return pd.DataFrame()
        
        correlation_data = player_data[available_cols].copy()
        
        # Calculate correlation matrix
        self.correlation_matrix = correlation_data.corr()
        
        print("Correlation matrix calculated:")
        print(self.correlation_matrix.round(3))
        
        return self.correlation_matrix
    
    def train_player_prop_models(self, player_data):
        """Train ML models for individual player prop predictions"""
        print("Training player prop prediction models...")
        
        if player_data.empty:
            return
        
        # Prepare features for prediction
        feature_cols = ['MIN', 'FGA', 'FG_PCT', 'GP']  # Available features
        available_features = [col for col in feature_cols if col in player_data.columns]
        
        if len(available_features) < 2:
            print("Insufficient features for model training")
            return
        
        X = player_data[available_features].copy()
        
        # Train models for different prop types
        prop_targets = {
            'points': 'PTS',
            'rebounds': 'REB', 
            'assists': 'AST',
            'threes': 'FG3M'
        }
        
        for prop_name, target_col in prop_targets.items():
            if target_col in player_data.columns:
                y = player_data[target_col]
                
                # Remove rows with missing target values
                mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 50:  # Need sufficient data
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )
                
                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                print(f"{prop_name.title()} model RMSE: {rmse:.3f}")
                
                self.prop_models[prop_name] = {
                    'model': model,
                    'features': available_features,
                    'rmse': rmse
                }
    
    def predict_player_props(self, player_stats, prop_lines):
        """Predict player props and compare to betting lines"""
        predictions = {}
        
        for prop_type, model_info in self.prop_models.items():
            if model_info is None:
                continue
                
            model = model_info['model']
            features = model_info['features']
            
            # Prepare features for prediction
            try:
                X = np.array([[player_stats.get(feat, 0) for feat in features]])
                prediction = model.predict(X)[0]
                
                # Get betting line for this prop
                line = prop_lines.get(prop_type, prediction)
                
                # Calculate edge
                edge = prediction - line
                confidence = min(abs(edge) / model_info['rmse'], 1.0)  # Normalize by model error
                
                predictions[prop_type] = {
                    'prediction': prediction,
                    'line': line,
                    'edge': edge,
                    'confidence': confidence,
                    'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'PASS'
                }
                
            except Exception as e:
                print(f"Error predicting {prop_type}: {e}")
                continue
        
        return predictions
    
    def generate_parlay_combinations(self, game_predictions, player_predictions, max_legs=4, min_confidence=0.6):
        """Generate optimal parlay combinations"""
        print("Generating parlay combinations...")
        
        all_bets = []
        
        # Add game predictions
        for game, pred in game_predictions.items():
            if pred.get('confidence', 0) >= min_confidence:
                all_bets.append({
                    'type': 'game',
                    'description': f"{game} - {pred.get('recommendation', 'ML')}",
                    'probability': pred.get('probability', 0.5),
                    'confidence': pred.get('confidence', 0),
                    'edge': pred.get('edge', 0)
                })
        
        # Add player prop predictions
        for player, props in player_predictions.items():
            for prop_type, pred in props.items():
                if pred.get('confidence', 0) >= min_confidence and pred.get('recommendation') != 'PASS':
                    all_bets.append({
                        'type': 'player_prop',
                        'description': f"{player} {prop_type} {pred.get('recommendation')} {pred.get('line')}",
                        'probability': self.edge_to_probability(pred.get('edge', 0)),
                        'confidence': pred.get('confidence', 0),
                        'edge': pred.get('edge', 0)
                    })
        
        if len(all_bets) < 2:
            print("Insufficient high-confidence bets for parlays")
            return []
        
        # Generate combinations
        parlay_combinations = []
        
        for num_legs in range(2, min(max_legs + 1, len(all_bets) + 1)):
            for combo in combinations(all_bets, num_legs):
                parlay = self.evaluate_parlay(combo)
                if parlay['expected_value'] > 0:  # Only positive EV parlays
                    parlay_combinations.append(parlay)
        
        # Sort by expected value
        parlay_combinations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return parlay_combinations[:10]  # Return top 10 parlays
    
    def edge_to_probability(self, edge):
        """Convert edge to implied probability"""
        # Simple conversion - can be improved with more sophisticated modeling
        base_prob = 0.5
        adjusted_prob = base_prob + (edge * 0.1)  # Edge factor
        return max(0.1, min(0.9, adjusted_prob))
    
    def evaluate_parlay(self, bet_combination):
        """Evaluate a parlay combination"""
        # Calculate combined probability
        combined_prob = 1.0
        total_confidence = 0
        total_edge = 0
        descriptions = []
        
        for bet in bet_combination:
            combined_prob *= bet['probability']
            total_confidence += bet['confidence']
            total_edge += bet['edge']
            descriptions.append(bet['description'])
        
        avg_confidence = total_confidence / len(bet_combination)
        
        # Estimate parlay odds (simplified)
        if combined_prob > 0:
            decimal_odds = 1 / combined_prob
            american_odds = self.decimal_to_american_odds(decimal_odds)
        else:
            decimal_odds = 100
            american_odds = 9900
        
        # Calculate expected value (simplified)
        # In reality, you'd need actual sportsbook parlay odds
        expected_payout = decimal_odds - 1  # Profit multiplier
        expected_value = (combined_prob * expected_payout) - (1 - combined_prob)
        
        return {
            'legs': descriptions,
            'num_legs': len(bet_combination),
            'combined_probability': combined_prob,
            'decimal_odds': decimal_odds,
            'american_odds': american_odds,
            'confidence': avg_confidence,
            'total_edge': total_edge,
            'expected_value': expected_value,
            'kelly_bet_size': max(0, min(0.25, expected_value / (decimal_odds - 1))) if decimal_odds > 1 else 0
        }
    
    def decimal_to_american_odds(self, decimal_odds):
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def analyze_game_day_parlays(self, games_data, player_data_today):
        """Analyze and generate parlays for today's games"""
        print(f"Analyzing parlays for {len(games_data)} games...")
        
        # Mock game predictions (integrate with your main prediction system)
        game_predictions = {}
        for game in games_data:
            # This would integrate with your main prediction system
            game_key = f"{game['away_team']} @ {game['home_team']}"
            game_predictions[game_key] = {
                'probability': 0.65,  # Mock probability
                'confidence': 0.75,
                'edge': 0.1,
                'recommendation': 'ML'
            }
        
        # Mock player predictions
        player_predictions = {}
        for player_name, stats in player_data_today.items():
            # Mock prop lines
            prop_lines = {
                'points': stats.get('avg_points', 20),
                'rebounds': stats.get('avg_rebounds', 8),
                'assists': stats.get('avg_assists', 5)
            }
            
            player_predictions[player_name] = self.predict_player_props(stats, prop_lines)
        
        # Generate parlay combinations
        parlays = self.generate_parlay_combinations(game_predictions, player_predictions)
        
        return parlays
    
    def save_parlay_models(self):
        """Save trained models"""
        import os
        os.makedirs("Models/Parlay_Models", exist_ok=True)
        
        for prop_type, model_info in self.prop_models.items():
            if model_info is not None:
                joblib.dump(model_info, f"Models/Parlay_Models/prop_model_{prop_type}.pkl")
        
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv("Models/Parlay_Models/player_correlations.csv")
        
        print("Parlay models saved successfully")
    
    def load_parlay_models(self):
        """Load trained models"""
        try:
            for prop_type in self.prop_models.keys():
                try:
                    model_info = joblib.load(f"Models/Parlay_Models/prop_model_{prop_type}.pkl")
                    self.prop_models[prop_type] = model_info
                except FileNotFoundError:
                    continue
            
            try:
                self.correlation_matrix = pd.read_csv("Models/Parlay_Models/player_correlations.csv", index_col=0)
            except FileNotFoundError:
                pass
            
            print("Parlay models loaded successfully")
        except Exception as e:
            print(f"Error loading parlay models: {e}")

def create_mock_player_data():
    """Create mock player data for testing"""
    players = [
        "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
        "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic"
    ]
    
    mock_data = {}
    for player in players:
        mock_data[player] = {
            'avg_points': np.random.normal(25, 5),
            'avg_rebounds': np.random.normal(8, 3),
            'avg_assists': np.random.normal(6, 2),
            'MIN': np.random.normal(35, 5),
            'FGA': np.random.normal(18, 4),
            'FG_PCT': np.random.normal(0.45, 0.05),
            'GP': 70
        }
    
    return mock_data

if __name__ == "__main__":
    # Test the parlay predictor
    predictor = ParlayPredictor()
    
    # Load player data
    player_data = predictor.load_player_data()
    
    if not player_data.empty:
        # Calculate correlations
        predictor.calculate_player_correlations(player_data)
        
        # Train models
        predictor.train_player_prop_models(player_data)
        
        # Save models
        predictor.save_parlay_models()
        
        print("Parlay prediction system initialized successfully!")
    else:
        print("Using mock data for testing...")
        
        # Create mock data for testing
        mock_games = [
            {'away_team': 'LAL', 'home_team': 'GSW'},
            {'away_team': 'BOS', 'home_team': 'MIL'}
        ]
        
        mock_player_data = create_mock_player_data()
        
        # Test parlay generation
        parlays = predictor.analyze_game_day_parlays(mock_games, mock_player_data)
        
        print(f"\nGenerated {len(parlays)} parlay combinations:")
        for i, parlay in enumerate(parlays[:3], 1):
            print(f"\nParlay {i}:")
            print(f"  Legs: {len(parlay['legs'])}")
            for leg in parlay['legs']:
                print(f"    - {leg}")
            print(f"  Combined Odds: {parlay['american_odds']:+d}")
            print(f"  Probability: {parlay['combined_probability']:.3f}")
            print(f"  Expected Value: {parlay['expected_value']:+.3f}")
            print(f"  Kelly Bet Size: {parlay['kelly_bet_size']:.1%}")

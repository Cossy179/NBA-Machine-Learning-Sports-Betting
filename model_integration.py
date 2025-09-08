#!/usr/bin/env python3
"""
Model Integration Script for GoonSteen Web Platform
Connects the web platform with existing NBA prediction models
"""

import sys
import os
import sqlite3
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional

# Add the project root to Python path to import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing prediction modules
try:
    from src.Predict.Advanced_Prediction_Runner import AdvancedPredictionRunner
    from src.Predict.XGBoost_Runner import XGBoostRunner
    from src.Predict.NN_Runner import NeuralNetworkRunner
    from src.Utils.tools import load_config
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import existing models: {e}")
    MODELS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelIntegration:
    """Integration class for NBA prediction models"""
    
    def __init__(self, database_path: str = 'web_database.db'):
        self.database_path = database_path
        self.models = {}
        
        if MODELS_AVAILABLE:
            self.initialize_models()
    
    def initialize_models(self):
        """Initialize prediction models"""
        try:
            # Initialize your existing models
            self.models['xgboost'] = XGBoostRunner()
            self.models['neural_network'] = NeuralNetworkRunner()
            self.models['advanced'] = AdvancedPredictionRunner()
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.models = {}
    
    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_today_games(self) -> List[Dict[str, Any]]:
        """Get today's NBA games from database"""
        conn = self.get_db_connection()
        try:
            cursor = conn.execute('''
                SELECT 
                    g.id, g.game_date, g.game_time,
                    g.home_team_id, g.away_team_id,
                    ht.name as home_team_name, ht.abbreviation as home_abbr,
                    at.name as away_team_name, at.abbreviation as away_abbr
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE g.game_date = ? AND g.status = 'scheduled'
                ORDER BY g.game_time
            ''', [date.today().isoformat()])
            
            games = [dict(row) for row in cursor.fetchall()]
            return games
            
        finally:
            conn.close()
    
    def store_prediction(self, game_id: int, model_name: str, prediction_data: Dict[str, Any]):
        """Store prediction in database"""
        conn = self.get_db_connection()
        try:
            # Check if prediction already exists
            existing = conn.execute(
                'SELECT id FROM predictions WHERE game_id = ? AND model_name = ?',
                [game_id, model_name]
            ).fetchone()
            
            if existing:
                # Update existing prediction
                conn.execute('''
                    UPDATE predictions SET
                        prediction_type = ?, predicted_winner = ?, 
                        predicted_home_score = ?, predicted_away_score = ?,
                        predicted_total = ?, predicted_spread = ?,
                        confidence = ?, probability = ?, expected_value = ?,
                        features = ?, created_at = ?
                    WHERE id = ?
                ''', [
                    prediction_data.get('prediction_type', 'moneyline'),
                    prediction_data.get('predicted_winner'),
                    prediction_data.get('predicted_home_score'),
                    prediction_data.get('predicted_away_score'),
                    prediction_data.get('predicted_total'),
                    prediction_data.get('predicted_spread'),
                    prediction_data.get('confidence', 50.0),
                    prediction_data.get('probability'),
                    prediction_data.get('expected_value'),
                    json.dumps(prediction_data.get('features', {})),
                    datetime.now().isoformat(),
                    existing['id']
                ])
            else:
                # Insert new prediction
                conn.execute('''
                    INSERT INTO predictions (
                        game_id, model_name, model_version, prediction_type,
                        predicted_winner, predicted_home_score, predicted_away_score,
                        predicted_total, predicted_spread, confidence, probability,
                        expected_value, features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    game_id, model_name, prediction_data.get('model_version', '1.0'),
                    prediction_data.get('prediction_type', 'moneyline'),
                    prediction_data.get('predicted_winner'),
                    prediction_data.get('predicted_home_score'),
                    prediction_data.get('predicted_away_score'),
                    prediction_data.get('predicted_total'),
                    prediction_data.get('predicted_spread'),
                    prediction_data.get('confidence', 50.0),
                    prediction_data.get('probability'),
                    prediction_data.get('expected_value'),
                    json.dumps(prediction_data.get('features', {}))
                ])
            
            conn.commit()
            logger.info(f"Stored prediction for game {game_id} using {model_name}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error storing prediction: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def run_predictions_for_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Run predictions for a single game"""
        predictions = {}
        
        if not MODELS_AVAILABLE:
            # Return mock predictions if models are not available
            return self.generate_mock_prediction(game)
        
        # Prepare game data for models
        game_data = {
            'home_team': game['home_abbr'],
            'away_team': game['away_abbr'],
            'game_date': game['game_date'],
            'game_time': game['game_time']
        }
        
        # Run each available model
        for model_name, model in self.models.items():
            try:
                prediction = self.run_single_model_prediction(model, model_name, game_data)
                if prediction:
                    predictions[model_name] = prediction
                    
            except Exception as e:
                logger.error(f"Error running {model_name} model: {e}")
                continue
        
        # Create ensemble prediction if multiple models available
        if len(predictions) > 1:
            ensemble_prediction = self.create_ensemble_prediction(predictions, game)
            predictions['ensemble'] = ensemble_prediction
        
        return predictions
    
    def run_single_model_prediction(self, model, model_name: str, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run prediction for a single model"""
        try:
            # This is where you'd call your existing model's prediction method
            # The exact implementation depends on your model interfaces
            
            if model_name == 'xgboost':
                # Example for XGBoost model
                result = model.predict_game(
                    home_team=game_data['home_team'],
                    away_team=game_data['away_team']
                )
                
            elif model_name == 'neural_network':
                # Example for Neural Network model
                result = model.predict_game(
                    home_team=game_data['home_team'],
                    away_team=game_data['away_team']
                )
                
            elif model_name == 'advanced':
                # Example for Advanced model
                result = model.run_prediction(
                    home_team=game_data['home_team'],
                    away_team=game_data['away_team']
                )
            
            else:
                return None
            
            # Convert model result to standardized format
            return self.standardize_prediction_result(result, model_name)
            
        except Exception as e:
            logger.error(f"Error in {model_name} prediction: {e}")
            return None
    
    def standardize_prediction_result(self, result: Any, model_name: str) -> Dict[str, Any]:
        """Standardize prediction result format"""
        # This function should convert your model's output format
        # to the standardized format expected by the web platform
        
        if isinstance(result, dict):
            return {
                'model_version': '1.0',
                'prediction_type': 'moneyline',
                'predicted_winner': result.get('winner_team_id'),
                'predicted_home_score': result.get('home_score'),
                'predicted_away_score': result.get('away_score'),
                'predicted_total': result.get('total_points'),
                'predicted_spread': result.get('spread'),
                'confidence': result.get('confidence', 50.0),
                'probability': result.get('win_probability'),
                'expected_value': result.get('expected_value'),
                'features': result.get('features', {})
            }
        
        # Handle other result formats as needed
        return {
            'model_version': '1.0',
            'prediction_type': 'moneyline',
            'confidence': 50.0
        }
    
    def create_ensemble_prediction(self, predictions: Dict[str, Dict[str, Any]], game: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models"""
        # Simple ensemble - average confidence and use majority vote
        confidences = [pred['confidence'] for pred in predictions.values() if 'confidence' in pred]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 50.0
        
        # Majority vote for winner
        winner_votes = {}
        for pred in predictions.values():
            winner = pred.get('predicted_winner')
            if winner:
                winner_votes[winner] = winner_votes.get(winner, 0) + 1
        
        predicted_winner = max(winner_votes.items(), key=lambda x: x[1])[0] if winner_votes else None
        
        # Average scores
        home_scores = [pred['predicted_home_score'] for pred in predictions.values() 
                      if pred.get('predicted_home_score')]
        away_scores = [pred['predicted_away_score'] for pred in predictions.values() 
                      if pred.get('predicted_away_score')]
        
        avg_home_score = sum(home_scores) / len(home_scores) if home_scores else None
        avg_away_score = sum(away_scores) / len(away_scores) if away_scores else None
        
        return {
            'model_version': '1.0',
            'prediction_type': 'moneyline',
            'predicted_winner': predicted_winner,
            'predicted_home_score': avg_home_score,
            'predicted_away_score': avg_away_score,
            'predicted_total': (avg_home_score + avg_away_score) if (avg_home_score and avg_away_score) else None,
            'confidence': min(avg_confidence * 1.1, 95.0),  # Boost ensemble confidence slightly
            'features': {'ensemble_models': list(predictions.keys())}
        }
    
    def generate_mock_prediction(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock prediction when models are not available"""
        import random
        
        # Generate realistic mock data
        home_score = random.randint(95, 125)
        away_score = random.randint(95, 125)
        confidence = random.randint(60, 90)
        
        # Determine winner based on scores
        predicted_winner = game['home_team_id'] if home_score > away_score else game['away_team_id']
        
        mock_prediction = {
            'xgboost': {
                'model_version': '1.0',
                'prediction_type': 'moneyline',
                'predicted_winner': predicted_winner,
                'predicted_home_score': home_score,
                'predicted_away_score': away_score,
                'predicted_total': home_score + away_score,
                'predicted_spread': abs(home_score - away_score),
                'confidence': confidence,
                'probability': confidence / 100.0,
                'expected_value': random.uniform(0.05, 0.15),
                'features': {'mock_data': True}
            }
        }
        
        return mock_prediction
    
    def run_daily_predictions(self):
        """Run predictions for all today's games"""
        logger.info("Starting daily predictions run")
        
        games = self.get_today_games()
        if not games:
            logger.info("No games scheduled for today")
            return
        
        logger.info(f"Found {len(games)} games for today")
        
        for game in games:
            logger.info(f"Running predictions for {game['away_team_name']} @ {game['home_team_name']}")
            
            predictions = self.run_predictions_for_game(game)
            
            # Store each model's prediction
            for model_name, prediction_data in predictions.items():
                self.store_prediction(game['id'], model_name, prediction_data)
        
        logger.info("Daily predictions completed")
    
    def update_model_performance(self, model_name: str, game_id: int, actual_result: Dict[str, Any]):
        """Update model performance based on actual game results"""
        conn = self.get_db_connection()
        try:
            # Get the prediction
            prediction = conn.execute(
                'SELECT * FROM predictions WHERE game_id = ? AND model_name = ?',
                [game_id, model_name]
            ).fetchone()
            
            if not prediction:
                return
            
            # Calculate accuracy
            predicted_winner = prediction['predicted_winner']
            actual_winner = actual_result.get('winner_team_id')
            
            is_correct = predicted_winner == actual_winner
            
            # Update or insert performance record
            today = date.today().isoformat()
            
            perf_record = conn.execute(
                'SELECT * FROM model_performance WHERE model_name = ? AND date_from = ? AND date_to = ?',
                [model_name, today, today]
            ).fetchone()
            
            if perf_record:
                # Update existing record
                new_total = perf_record['total_predictions'] + 1
                new_correct = perf_record['correct_predictions'] + (1 if is_correct else 0)
                new_accuracy = (new_correct / new_total) * 100
                
                conn.execute('''
                    UPDATE model_performance SET
                        total_predictions = ?, correct_predictions = ?, accuracy = ?, updated_at = ?
                    WHERE id = ?
                ''', [new_total, new_correct, new_accuracy, datetime.now().isoformat(), perf_record['id']])
            else:
                # Insert new record
                conn.execute('''
                    INSERT INTO model_performance (
                        model_name, model_version, prediction_type, total_predictions,
                        correct_predictions, accuracy, date_from, date_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    model_name, prediction['model_version'], prediction['prediction_type'],
                    1, 1 if is_correct else 0, 100.0 if is_correct else 0.0,
                    today, today
                ])
            
            conn.commit()
            logger.info(f"Updated performance for {model_name}: {'Correct' if is_correct else 'Incorrect'}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error updating performance: {e}")
        finally:
            conn.close()

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Prediction Model Integration')
    parser.add_argument('--run-predictions', action='store_true', 
                       help='Run predictions for today\'s games')
    parser.add_argument('--database', default='web_database.db',
                       help='Database file path')
    
    args = parser.parse_args()
    
    integration = ModelIntegration(args.database)
    
    if args.run_predictions:
        integration.run_daily_predictions()
    else:
        print("Use --run-predictions to generate predictions for today's games")

if __name__ == '__main__':
    main()

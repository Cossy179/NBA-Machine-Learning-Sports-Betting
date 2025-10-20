"""
NBA Model Runner - Loads and runs the best prediction model
"""
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NBAModelRunner:
    """Runs NBA prediction models"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_count = 106  # Default feature count
        
        self.load_models()
    
    def load_models(self):
        """Load the LightGBM model and associated files"""
        try:
            logger.info("Loading LightGBM model...")
            
            # Load LightGBM model
            model_path = os.path.join(self.model_dir, 'SuperAdvanced_XGB_v1_lightgbm.txt')
            if os.path.exists(model_path):
                try:
                    self.model = lgb.Booster(model_file=model_path)
                    logger.info(f"✅ Loaded LightGBM model from {model_path}")
                    
                    # Get feature names directly from the model
                    try:
                        self.features = self.model.feature_name()
                        self.feature_count = len(self.features)
                        logger.info(f"✅ Got {self.feature_count} feature names from model")
                    except Exception as feat_error:
                        logger.warning(f"Could not get features from model: {feat_error}")
                        # Use model's expected feature count
                        self.feature_count = self.model.num_feature()
                        self.features = [f"f{i}" for i in range(self.feature_count)]
                        logger.info(f"ℹ️ Using {self.feature_count} generic feature names")
                        
                except Exception as model_error:
                    logger.error(f"❌ Error loading LightGBM model: {model_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.warning("⚠️ Using fallback prediction mode")
                    self.model = None
            else:
                logger.warning(f"⚠️ Model file not found at {model_path}")
                self.model = None
            
            # If model loaded but no features, try loading from file
            if self.model and not self.features:
                features_path = os.path.join(self.model_dir, 'SuperAdvanced_XGB_v1_features.pkl')
                if os.path.exists(features_path):
                    try:
                        self.features = joblib.load(features_path)
                        self.feature_count = len(self.features)
                        logger.info(f"✅ Loaded {self.feature_count} feature names from file")
                    except Exception as features_error:
                        logger.warning(f"Could not load features file: {features_error}")
            
            # Try to load scaler (not all models have this)
            scaler_path = os.path.join(self.model_dir, 'SuperAdvanced_XGB_v1_scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("✅ Loaded feature scaler")
                except:
                    self.scaler = None
                    logger.info("ℹ️ No scaler needed for this model")
            
            logger.info("Model loading complete!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model = None
    
    def create_game_features(self, home_team: str, away_team: str, 
                           home_odds: Optional[float] = None,
                           away_odds: Optional[float] = None,
                           spread: Optional[float] = None,
                           total: Optional[float] = None) -> np.ndarray:
        """
        Create feature vector for a game
        
        This is a simplified version that creates baseline features.
        In a production system, this would fetch real team statistics.
        """
        try:
            # Team strength mappings (simplified - based on general team quality)
            team_ratings = {
                # Championship contenders
                'Boston Celtics': 0.85, 'Denver Nuggets': 0.82, 'Milwaukee Bucks': 0.80,
                'Phoenix Suns': 0.78, 'Philadelphia 76ers': 0.77, 'Miami Heat': 0.75,
                'Los Angeles Lakers': 0.74, 'Golden State Warriors': 0.73,
                'LA Clippers': 0.72, 'Dallas Mavericks': 0.71,
                # Playoff teams
                'New York Knicks': 0.68, 'Cleveland Cavaliers': 0.67, 
                'Sacramento Kings': 0.66, 'Memphis Grizzlies': 0.65,
                'Los Angeles Clippers': 0.72, 'Oklahoma City Thunder': 0.64,
                'New Orleans Pelicans': 0.63, 'Minnesota Timberwolves': 0.62,
                'Brooklyn Nets': 0.61, 'Atlanta Hawks': 0.60,
                # Middle tier
                'Indiana Pacers': 0.58, 'Toronto Raptors': 0.57, 
                'Chicago Bulls': 0.56, 'Utah Jazz': 0.55,
                'Washington Wizards': 0.54, 'Orlando Magic': 0.53,
                # Rebuilding
                'Portland Trail Blazers': 0.50, 'Charlotte Hornets': 0.48,
                'Houston Rockets': 0.47, 'San Antonio Spurs': 0.46,
                'Detroit Pistons': 0.45
            }
            
            # Get team ratings with fallback
            home_rating = team_ratings.get(home_team, 0.50)
            away_rating = team_ratings.get(away_team, 0.50)
            
            # Base features
            features = []
            
            # 1. Team strength features (0-9)
            features.append(home_rating)
            features.append(away_rating)
            features.append(home_rating - away_rating)  # Differential
            features.append(home_rating + 0.03)  # Home court advantage (~3 points)
            features.append(away_rating)
            features.append((home_rating + away_rating) / 2)  # Average strength
            features.append(home_rating * away_rating)  # Interaction
            features.append(max(home_rating, away_rating))  # Max strength
            features.append(min(home_rating, away_rating))  # Min strength
            features.append(abs(home_rating - away_rating))  # Strength gap
            
            # 2. Odds-based features (10-19)
            if home_odds is not None and away_odds is not None:
                # Convert to implied probabilities
                if home_odds < 0:
                    home_prob = abs(home_odds) / (abs(home_odds) + 100)
                else:
                    home_prob = 100 / (home_odds + 100)
                
                if away_odds < 0:
                    away_prob = abs(away_odds) / (abs(away_odds) + 100)
                else:
                    away_prob = 100 / (away_odds + 100)
                
                features.append(home_prob)
                features.append(away_prob)
                features.append(home_prob - away_prob)
                features.append((home_prob + away_prob) / 2)
                features.append(home_prob * away_prob)
                features.append(max(home_prob, away_prob))
                features.append(min(home_prob, away_prob))
                features.append(abs(home_prob - away_prob))
                features.append(1 - home_prob)  # Implied away prob
                features.append(1 - away_prob)  # Implied home prob
            else:
                features.extend([0.5] * 10)
            
            # 3. Spread features (20-29)
            if spread is not None:
                features.append(spread)
                features.append(abs(spread))
                features.append(spread / 10.0)  # Normalized
                features.append(1 if spread < 0 else 0)  # Home favorite
                features.append(1 if spread > 0 else 0)  # Away favorite
                features.append(spread * home_rating)
                features.append(spread * away_rating)
                features.append(spread * (home_rating - away_rating))
                features.append(abs(spread) * (home_rating + away_rating))
                features.append(spread ** 2)  # Spread squared
            else:
                features.extend([0] * 10)
            
            # 4. Total features (30-39)
            if total is not None:
                features.append(total)
                features.append(total / 220.0)  # Normalized (220 is average)
                features.append(total * home_rating)
                features.append(total * away_rating)
                features.append(total * (home_rating + away_rating) / 2)
                features.append(total - 220.0)  # Deviation from average
                features.append((total - 220.0) / 10.0)  # Normalized deviation
                features.append(total ** 2)
                features.append(np.log(total) if total > 0 else 0)
                features.append(total * (1 if spread and spread < 0 else 0))
            else:
                features.extend([220.0, 1.0, 220 * 0.5, 220 * 0.5, 220 * 0.5, 0, 0, 220**2, np.log(220), 0])
            
            # 5. Interaction features (40-59)
            features.append(home_rating * (spread if spread else 0))
            features.append(away_rating * (spread if spread else 0))
            features.append(home_rating * (total if total else 220))
            features.append(away_rating * (total if total else 220))
            features.append((home_rating - away_rating) * (spread if spread else 0))
            features.append((home_rating + away_rating) * (total if total else 220))
            features.append(home_rating ** 2)
            features.append(away_rating ** 2)
            features.append((home_rating - away_rating) ** 2)
            features.append(np.sqrt(home_rating) if home_rating > 0 else 0)
            features.append(np.sqrt(away_rating) if away_rating > 0 else 0)
            features.append(np.log(home_rating + 1))
            features.append(np.log(away_rating + 1))
            features.append(home_rating * away_rating * (spread if spread else 0))
            features.append(home_rating * away_rating * (total if total else 220))
            features.append((home_rating + away_rating) * (spread if spread else 0))
            features.append((home_rating - away_rating) * (total if total else 220))
            features.append(home_rating / (away_rating + 0.01))
            features.append(away_rating / (home_rating + 0.01))
            features.append((spread if spread else 0) / (total if total and total > 0 else 220))
            
            # 6. Advanced statistical features (60-79)
            features.append(home_rating * 110)  # Projected home score
            features.append(away_rating * 110)  # Projected away score
            features.append((home_rating * 110) + (away_rating * 110))  # Projected total
            features.append(abs((home_rating * 110) - (away_rating * 110)))  # Projected margin
            features.append(home_rating * 0.475)  # Win probability (simplified)
            features.append(away_rating * 0.475)
            features.append((home_rating + 0.03) * 0.5)  # Home win prob with HCA
            features.append(away_rating * 0.5)  # Away win prob
            features.append(home_rating * (spread if spread and spread < 0 else 0))
            features.append(away_rating * (spread if spread and spread > 0 else 0))
            features.append((total if total else 220) - (home_rating + away_rating) * 110)
            features.append(home_rating * 48)  # Projected pace
            features.append(away_rating * 48)
            features.append((home_rating + away_rating) * 24)  # Average pace
            features.append(home_rating * 0.55)  # Offensive rating proxy
            features.append(away_rating * 0.55)
            features.append(home_rating * 0.45)  # Defensive rating proxy
            features.append(away_rating * 0.45)
            features.append((home_rating - away_rating + 0.03) * 10)  # Spread estimate
            features.append((home_rating + away_rating) * 110)  # Total estimate
            
            # 7. Contextual features (80-99)
            features.extend([
                0.03,  # Home court advantage
                1.0,   # Rest days (assumed)
                0.0,   # Back-to-back indicator
                0.5,   # Time of season (mid-season)
                0.0,   # Conference game indicator
                0.0,   # Division game indicator
                0.5,   # Recent form home
                0.5,   # Recent form away
                0.0,   # Win streak home
                0.0,   # Win streak away
                home_rating * 41,  # Projected home wins (82 game season)
                away_rating * 41,  # Projected away wins
                0.5,   # Home offensive efficiency
                0.5,   # Home defensive efficiency
                0.5,   # Away offensive efficiency
                0.5,   # Away defensive efficiency
                100.0, # Pace
                0.50,  # True shooting % home
                0.50,  # True shooting % away
                0.25   # Turnover rate
            ])
            
            # 8. Final padding features to reach 106 (100-105)
            remaining = self.feature_count - len(features)
            if remaining > 0:
                features.extend([0.0] * remaining)
            
            # Convert to numpy array and ensure correct length
            features_array = np.array(features[:self.feature_count]).reshape(1, -1)
            
            # Apply scaling if available
            if self.scaler is not None:
                try:
                    features_array = self.scaler.transform(features_array)
                except:
                    pass  # If scaling fails, use unscaled features
            
            # Convert to DataFrame with feature names for LightGBM
            if self.features and len(self.features) == features_array.shape[1]:
                features_df = pd.DataFrame(features_array, columns=self.features)
                return features_df
            else:
                return features_array
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            # Return baseline features
            return np.random.randn(1, self.feature_count) * 0.1
    
    def predict_game(self, home_team: str, away_team: str,
                    home_odds: Optional[float] = None,
                    away_odds: Optional[float] = None,
                    spread: Optional[float] = None,
                    total: Optional[float] = None) -> Dict:
        """
        Predict game outcome
        
        Returns:
            Dict with prediction details including winner, confidence, scores, etc.
        """
        try:
            # Create features
            features = self.create_game_features(
                home_team, away_team, home_odds, away_odds, spread, total
            )
            
            # Make prediction
            if self.model is not None:
                # LightGBM prediction
                prediction_prob = self.model.predict(features)[0]
                
                # Convert to win probability (assuming model outputs home win probability)
                home_win_prob = float(prediction_prob)
                
                # Ensure probability is in valid range
                home_win_prob = max(0.1, min(0.9, home_win_prob))
                
            else:
                # Fallback prediction based on odds if model not loaded
                if home_odds is not None and away_odds is not None:
                    if home_odds < 0:
                        home_win_prob = abs(home_odds) / (abs(home_odds) + 100)
                    else:
                        home_win_prob = 100 / (home_odds + 100)
                else:
                    home_win_prob = 0.52  # Slight home advantage
            
            # Calculate confidence (distance from 50-50)
            confidence = abs(home_win_prob - 0.5) * 2  # Scale to 0-1
            confidence = confidence * 100  # Convert to percentage
            
            # Predict winner
            winner = home_team if home_win_prob > 0.5 else away_team
            winner_prob = home_win_prob if home_win_prob > 0.5 else (1 - home_win_prob)
            
            # Estimate scores (simplified - based on average NBA score of 110)
            base_score = 110
            home_score = int(base_score + (home_win_prob - 0.5) * 10)
            away_score = int(base_score - (home_win_prob - 0.5) * 10)
            
            # Predict spread
            predicted_spread = home_score - away_score
            
            # Predict total
            predicted_total = home_score + away_score
            
            return {
                'winner': winner,
                'winner_probability': round(winner_prob, 3),
                'home_win_probability': round(home_win_prob, 3),
                'away_win_probability': round(1 - home_win_prob, 3),
                'confidence': round(confidence, 1),
                'home_score': home_score,
                'away_score': away_score,
                'spread_prediction': predicted_spread,
                'total_prediction': predicted_total,
                'recommendation': 'HOME' if home_win_prob > 0.55 else 'AWAY' if home_win_prob < 0.45 else 'PASS'
            }
            
        except Exception as e:
            logger.error(f"Error predicting game: {e}")
            return {
                'winner': home_team,
                'confidence': 50.0,
                'error': str(e)
            }
    
    def generate_parlays(self, game_predictions: List[Dict], 
                        max_parlays: int = 15,
                        min_confidence: float = 55.0) -> List[Dict]:
        """
        Generate parlay suggestions from game predictions
        
        Args:
            game_predictions: List of game prediction dicts
            max_parlays: Maximum number of parlays to return
            min_confidence: Minimum confidence threshold for including a game
        
        Returns:
            List of parlay suggestions
        """
        try:
            # Filter games by confidence
            qualified_games = [
                g for g in game_predictions 
                if g['prediction'].get('confidence', 0) >= min_confidence
            ]
            
            if len(qualified_games) < 2:
                logger.warning(f"Not enough qualified games for parlays (need 2, have {len(qualified_games)})")
                return []
            
            parlays = []
            
            # Generate 2-leg parlays
            for i in range(len(qualified_games)):
                for j in range(i + 1, len(qualified_games)):
                    if len(parlays) >= max_parlays:
                        break
                    
                    game1 = qualified_games[i]
                    game2 = qualified_games[j]
                    
                    # Create parlay
                    parlay = self._create_parlay([game1, game2])
                    if parlay:
                        parlays.append(parlay)
            
            # Generate 3-leg parlays (if enough games)
            if len(qualified_games) >= 3:
                for i in range(min(3, len(qualified_games) - 2)):
                    for j in range(i + 1, min(i + 3, len(qualified_games) - 1)):
                        for k in range(j + 1, min(j + 2, len(qualified_games))):
                            if len(parlays) >= max_parlays:
                                break
                            
                            game1 = qualified_games[i]
                            game2 = qualified_games[j]
                            game3 = qualified_games[k]
                            
                            parlay = self._create_parlay([game1, game2, game3])
                            if parlay:
                                parlays.append(parlay)
            
            # Sort by combined odds (higher is better)
            parlays.sort(key=lambda x: x['combined_odds'], reverse=True)
            
            return parlays[:max_parlays]
            
        except Exception as e:
            logger.error(f"Error generating parlays: {e}")
            return []
    
    def _create_parlay(self, games: List[Dict]) -> Optional[Dict]:
        """Create a parlay from multiple games"""
        try:
            legs = []
            combined_prob = 1.0
            total_confidence = 0
            
            for game in games:
                pred = game['prediction']
                winner = pred['winner']
                
                # Determine bet type
                if winner == game['home_team']:
                    bet_desc = f"{game['home_team']} ML"
                    win_prob = pred['home_win_probability']
                else:
                    bet_desc = f"{game['away_team']} ML"
                    win_prob = pred['away_win_probability']
                
                legs.append(bet_desc)
                combined_prob *= win_prob
                total_confidence += pred['confidence']
            
            # Calculate odds
            if combined_prob > 0:
                decimal_odds = 1 / combined_prob
                
                if decimal_odds >= 2.0:
                    american_odds = int((decimal_odds - 1) * 100)
                else:
                    american_odds = int(-100 / (decimal_odds - 1))
            else:
                decimal_odds = 100
                american_odds = 9900
            
            avg_confidence = total_confidence / len(games)
            
            return {
                'legs': legs,
                'num_legs': len(games),
                'combined_probability': round(combined_prob, 3),
                'combined_odds': decimal_odds,
                'american_odds': american_odds,
                'confidence': round(avg_confidence, 1),
                'games': [g['id'] for g in games]
            }
            
        except Exception as e:
            logger.error(f"Error creating parlay: {e}")
            return None


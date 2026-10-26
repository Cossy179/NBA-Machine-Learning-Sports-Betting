"""
Home-win probabilities from the LightGBM v2 model (Models/LightGBM_V2/).

Train or refresh the model with:  python src/Train-Models/LightGBM_V2.py
"""
import json
import os

import lightgbm as lgb
import pandas as pd

from LiveFeatureBuilder import REPO_ROOT, FeaturePipeline

MODEL_DIR = os.path.join(REPO_ROOT, "Models", "LightGBM_V2")


class LightGBMV2Predictor:
    def __init__(self, model_dir=MODEL_DIR, verbose=True):
        model_path = os.path.join(model_dir, "LightGBM_V2.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found. Train it first: python src/Train-Models/LightGBM_V2.py")
        self.model = lgb.Booster(model_file=model_path)
        with open(os.path.join(model_dir, "metadata.txt")) as fh:
            self.metadata = json.load(fh)
        self.features = self.metadata["features"]
        self.pipeline = FeaturePipeline(verbose=verbose)

    def describe(self):
        m = self.metadata
        test = m.get("evaluation", {}).get("test_2025_26", {})
        return (f"LightGBM v2: {m['games']} games from {m['first_game']} to {m['last_game']}, "
                f"{len(self.features)} features; 2025-26 test accuracy "
                f"{test.get('accuracy', float('nan')):.1%}, log loss {test.get('log_loss', float('nan')):.3f}")

    def predict(self, home_team, away_team, game_day, season_type=2):
        """Home-win probability for a game on `game_day` (US date)."""
        f = self.pipeline.features_for(home_team, away_team, game_day, season_type)
        row = pd.DataFrame([f])[self.features]
        return {
            "probability": float(self.model.predict(row)[0]),
            "home_games_played": int(f["GP"]),
            "away_games_played": int(f["GP.1"]),
            "home_elo": f["home_elo_overall"],
            "away_elo": f["away_elo_overall"],
        }

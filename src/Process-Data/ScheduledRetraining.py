"""
Scheduled Retraining Pipeline for 2025-26 Season
Automates periodic retraining of models after new data is collected.
Uses time-series cross-validation to evaluate whether retraining is warranted.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.Utils.time_series_validation import create_season_based_splits
from src.Utils.metrics_and_calibration import CalibrationEvaluator


class ScheduledRetrainer:
    """Manages scheduled retraining of NBA prediction models"""
    
    def __init__(self, model_version_dir: str = "Models"):
        self.model_version_dir = model_version_dir
        self.retraining_log_path = "Data/retraining_log.json"
        self.init_retraining_log()
    
    def init_retraining_log(self):
        """Initialize retraining log file"""
        if not os.path.exists(self.retraining_log_path):
            log = {
                "last_retraining": None,
                "retraining_history": [],
                "model_versions": {}
            }
            with open(self.retraining_log_path, 'w') as f:
                json.dump(log, f, indent=2)
    
    def load_retraining_log(self) -> dict:
        """Load retraining log"""
        try:
            with open(self.retraining_log_path, 'r') as f:
                return json.load(f)
        except:
            return {
                "last_retraining": None,
                "retraining_history": [],
                "model_versions": {}
            }
    
    def save_retraining_log(self, log: dict):
        """Save retraining log"""
        with open(self.retraining_log_path, 'w') as f:
            json.dump(log, f, indent=2)
    
    def check_if_retraining_needed(
        self,
        min_days_since_last: int = 7,
        min_new_games: int = 10
    ) -> bool:
        """
        Check if retraining is needed based on time and new data.
        
        Args:
            min_days_since_last: Minimum days since last retraining
            min_new_games: Minimum new games since last retraining
        
        Returns:
            True if retraining is needed
        """
        log = self.load_retraining_log()
        
        # Check time since last retraining
        if log["last_retraining"]:
            last_date = datetime.fromisoformat(log["last_retraining"])
            days_since = (datetime.now() - last_date).days
            
            if days_since < min_days_since_last:
                print(f"⏭️  Skipping retraining: Only {days_since} days since last retraining")
                return False
        
        # Check for new games
        new_games_count = self.count_new_games_since_last_retraining()
        
        if new_games_count < min_new_games:
            print(f"⏭️  Skipping retraining: Only {new_games_count} new games available")
            return False
        
        print(f"✅ Retraining needed: {new_games_count} new games, {days_since if log['last_retraining'] else 'never'} days since last")
        return True
    
    def count_new_games_since_last_retraining(self) -> int:
        """Count new games added since last retraining"""
        log = self.load_retraining_log()
        
        if not log["last_retraining"]:
            # Count all 2025-26 games
            return self.count_season_games("2025-26")
        
        last_date = datetime.fromisoformat(log["last_retraining"])
        
        try:
            con = sqlite3.connect("Data/dataset.sqlite")
            
            # Count games after last retraining date
            query = """
                SELECT COUNT(*) FROM "dataset_2012-26_new"
                WHERE Date > ?
            """
            count = pd.read_sql_query(query, con, params=[last_date.strftime('%Y-%m-%d')])
            con.close()
            
            return count.iloc[0, 0] if not count.empty else 0
            
        except Exception as e:
            print(f"⚠️  Error counting new games: {e}")
            return 0
    
    def count_season_games(self, season: str) -> int:
        """Count total games for a season"""
        try:
            con = sqlite3.connect("Data/dataset.sqlite")
            
            # Try to count games for the season
            year = season.split('-')[0]
            query = f"""
                SELECT COUNT(*) FROM "dataset_2012-26_new"
                WHERE Date >= '{year}-10-01' AND Date < '{int(year)+1}-07-01'
            """
            count = pd.read_sql_query(query, con)
            con.close()
            
            return count.iloc[0, 0] if not count.empty else 0
            
        except Exception as e:
            print(f"⚠️  Error counting season games: {e}")
            return 0
    
    def evaluate_model_performance(
        self,
        model_path: str,
        test_data: pd.DataFrame
    ) -> dict:
        """
        Evaluate model performance on recent data.
        
        Args:
            model_path: Path to model file
            test_data: Test dataset
        
        Returns:
            Dictionary with performance metrics
        """
        # This would load and evaluate the model
        # Implementation depends on your model format
        print(f"📊 Evaluating model: {model_path}")
        
        # Placeholder metrics
        metrics = {
            "accuracy": 0.0,
            "roi": 0.0,
            "calibration_error": 0.0
        }
        
        return metrics
    
    def retrain_models(
        self,
        model_types: List[str] = None,
        use_time_series_cv: bool = True
    ) -> dict:
        """
        Retrain models with updated data.
        
        Args:
            model_types: List of model types to retrain (None = all)
            use_time_series_cv: Whether to use time-series cross-validation
        
        Returns:
            Dictionary with retraining results
        """
        if model_types is None:
            model_types = ["xgboost", "neural", "ensemble"]
        
        print("=" * 70)
        print("SCHEDULED MODEL RETRAINING")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = {}
        
        for model_type in model_types:
            print(f"\n🔄 Retraining {model_type} models...")
            
            try:
                # Call training script
                result = self._train_model_type(model_type, use_time_series_cv)
                results[model_type] = result
                
                if result["success"]:
                    print(f"✅ {model_type} retraining completed")
                else:
                    print(f"❌ {model_type} retraining failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error retraining {model_type}: {e}")
                results[model_type] = {"success": False, "error": str(e)}
        
        # Update retraining log
        log = self.load_retraining_log()
        log["last_retraining"] = datetime.now().isoformat()
        log["retraining_history"].append({
            "date": datetime.now().isoformat(),
            "results": results
        })
        
        # Store model versions
        for model_type, result in results.items():
            if result.get("success"):
                version = result.get("version", datetime.now().strftime("%Y%m%d_%H%M%S"))
                log["model_versions"][model_type] = {
                    "version": version,
                    "date": datetime.now().isoformat(),
                    "metrics": result.get("metrics", {})
                }
        
        self.save_retraining_log(log)
        
        print("\n" + "=" * 70)
        print("✅ RETRAINING COMPLETE")
        print("=" * 70)
        
        return results
    
    def _train_model_type(
        self,
        model_type: str,
        use_time_series_cv: bool
    ) -> dict:
        """Train a specific model type"""
        # Map model types to training scripts
        script_map = {
            "xgboost": "train.py --xgboost",
            "neural": "train.py --neural",
            "ensemble": "train.py --ensemble",
            "all": "train.py --all"
        }
        
        if model_type not in script_map:
            return {"success": False, "error": f"Unknown model type: {model_type}"}
        
        command = script_map[model_type]
        
        try:
            # Run training script
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Parse output for metrics
                metrics = self._parse_training_output(result.stdout)
                
                return {
                    "success": True,
                    "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "metrics": metrics
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Training timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_training_output(self, output: str) -> dict:
        """Parse training script output for metrics"""
        # This would parse the actual output format
        # Placeholder implementation
        return {
            "accuracy": 0.0,
            "roi": 0.0
        }
    
    def run_backtest_after_retraining(self) -> dict:
        """Run backtest on recent games after retraining"""
        print("\n📊 Running backtest on recent games...")
        
        try:
            # Run backtest script
            result = subprocess.run(
                ["py", "backtest.py"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Backtest completed")
                return {"success": True, "output": result.stdout}
            else:
                print(f"⚠️  Backtest had issues: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            return {"success": False, "error": str(e)}


def scheduled_retraining_workflow(
    force: bool = False,
    model_types: List[str] = None
):
    """
    Complete workflow for scheduled retraining.
    
    Args:
        force: Force retraining even if checks suggest skipping
        model_types: Specific model types to retrain (None = all)
    """
    retrainer = ScheduledRetrainer()
    
    # Check if retraining is needed
    if not force and not retrainer.check_if_retraining_needed():
        print("\n⏭️  Retraining not needed at this time")
        return
    
    # Retrain models
    results = retrainer.retrain_models(model_types=model_types)
    
    # Run backtest
    backtest_results = retrainer.run_backtest_after_retraining()
    
    # Print summary
    print("\n" + "=" * 70)
    print("RETRAINING SUMMARY")
    print("=" * 70)
    
    for model_type, result in results.items():
        status = "✅" if result.get("success") else "❌"
        print(f"{status} {model_type}: {result.get('version', 'N/A')}")
    
    print(f"\nBacktest: {'✅' if backtest_results.get('success') else '❌'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scheduled NBA Model Retraining")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    parser.add_argument("--models", nargs="+", help="Specific model types to retrain")
    
    args = parser.parse_args()
    
    scheduled_retraining_workflow(
        force=args.force,
        model_types=args.models
    )


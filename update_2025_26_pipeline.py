"""
Complete Pipeline Update for 2025-26 NBA Season
Automates data collection, transaction tracking, feature rebuilding, and model retraining.
Run this script daily or weekly to keep the pipeline up to date.
"""
import os
import sys
from datetime import datetime, timedelta
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.DataProviders.HoopRDataCollector import collect_2025_26_season_data
from src.DataProviders.TransactionTracker import track_2025_26_transactions

# Import modules with dashes in directory names
import importlib.util
import os

# RebuildFeatures_2025_26
rebuild_spec = importlib.util.spec_from_file_location(
    "RebuildFeatures_2025_26",
    os.path.join("src", "Process-Data", "RebuildFeatures_2025_26.py")
)
rebuild_module = importlib.util.module_from_spec(rebuild_spec)
rebuild_spec.loader.exec_module(rebuild_module)
rebuild_features_2025_26 = rebuild_module.rebuild_features_2025_26

# ScheduledRetraining
retrain_spec = importlib.util.spec_from_file_location(
    "ScheduledRetraining",
    os.path.join("src", "Process-Data", "ScheduledRetraining.py")
)
retrain_module = importlib.util.module_from_spec(retrain_spec)
retrain_spec.loader.exec_module(retrain_module)
scheduled_retraining_workflow = retrain_module.scheduled_retraining_workflow


def update_pipeline(
    date_from: str = None,
    date_to: str = None,
    skip_collection: bool = False,
    skip_transactions: bool = False,
    skip_features: bool = False,
    skip_retraining: bool = False,
    force_retraining: bool = False
):
    """
    Complete pipeline update workflow.
    
    Args:
        date_from: Start date for data collection (defaults to last collected date)
        date_to: End date for data collection (defaults to today)
        skip_collection: Skip data collection step
        skip_transactions: Skip transaction tracking step
        skip_features: Skip feature rebuilding step
        skip_retraining: Skip model retraining step
        force_retraining: Force retraining even if checks suggest skipping
    """
    print("=" * 70)
    print("NBA PREDICTION PIPELINE UPDATE - 2025-26 SEASON")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")
    
    if date_from is None:
        # Default to 2025-26 opening night
        date_from = "2025-10-22"
    
    # Step 1: Collect new game data
    if not skip_collection:
        print("STEP 1: Collecting new game data...")
        print("-" * 70)
        try:
            game_data = collect_2025_26_season_data(
                date_from=date_from,
                date_to=date_to,
                save_to_db=True
            )
            if game_data.empty:
                print("⚠️  No new game data collected")
            else:
                print(f"✅ Collected {len(game_data)} new game records\n")
        except Exception as e:
            print(f"❌ Error collecting game data: {e}\n")
    else:
        print("⏭️  Skipping data collection\n")
    
    # Step 2: Track transactions
    if not skip_transactions:
        print("STEP 2: Tracking transactions and roster changes...")
        print("-" * 70)
        try:
            transactions = track_2025_26_transactions(
                date_from=date_from,
                date_to=date_to
            )
            if transactions.empty:
                print("⚠️  No new transactions found\n")
            else:
                print(f"✅ Tracked {len(transactions)} transactions\n")
        except Exception as e:
            print(f"❌ Error tracking transactions: {e}\n")
    else:
        print("⏭️  Skipping transaction tracking\n")
    
    # Step 3: Rebuild features
    if not skip_features:
        print("STEP 3: Rebuilding feature matrix...")
        print("-" * 70)
        try:
            features_df = rebuild_features_2025_26()
            if features_df.empty:
                print("⚠️  Feature rebuilding produced no data\n")
            else:
                print(f"✅ Rebuilt features for {len(features_df)} records\n")
        except Exception as e:
            print(f"❌ Error rebuilding features: {e}\n")
    else:
        print("⏭️  Skipping feature rebuilding\n")
    
    # Step 4: Retrain models
    if not skip_retraining:
        print("STEP 4: Retraining models...")
        print("-" * 70)
        try:
            scheduled_retraining_workflow(force=force_retraining)
            print()
        except Exception as e:
            print(f"❌ Error retraining models: {e}\n")
    else:
        print("⏭️  Skipping model retraining\n")
    
    print("=" * 70)
    print("✅ PIPELINE UPDATE COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update NBA prediction pipeline for 2025-26 season"
    )
    parser.add_argument(
        "--date-from",
        type=str,
        help="Start date for data collection (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date-to",
        type=str,
        help="End date for data collection (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection step"
    )
    parser.add_argument(
        "--skip-transactions",
        action="store_true",
        help="Skip transaction tracking step"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature rebuilding step"
    )
    parser.add_argument(
        "--skip-retraining",
        action="store_true",
        help="Skip model retraining step"
    )
    parser.add_argument(
        "--force-retraining",
        action="store_true",
        help="Force model retraining even if checks suggest skipping"
    )
    
    args = parser.parse_args()
    
    update_pipeline(
        date_from=args.date_from,
        date_to=args.date_to,
        skip_collection=args.skip_collection,
        skip_transactions=args.skip_transactions,
        skip_features=args.skip_features,
        skip_retraining=args.skip_retraining,
        force_retraining=args.force_retraining
    )


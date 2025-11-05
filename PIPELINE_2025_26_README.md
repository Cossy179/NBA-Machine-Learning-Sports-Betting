# NBA Prediction Pipeline Update for 2025-26 Season

This document describes the updated data pipeline for the 2025-26 NBA season, including automated data collection, transaction tracking, feature rebuilding, and scheduled retraining.

## Overview

The pipeline has been updated to:
1. **Automate data collection** using the hoopR package
2. **Track transactions** (trades, signings, retirements) 
3. **Rebuild features** with rolling windows including current season data
4. **Schedule periodic retraining** with time-series cross-validation
5. **Monitor and backtest** model performance

## Installation

### Required Packages

Install the hoopR package for data collection:

```bash
pip install hoopR
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

The hoopR package documentation is available at: https://hoopr.sportsdataverse.org/

## Components

### 1. HoopR Data Collector (`src/DataProviders/HoopRDataCollector.py`)

Automated collection of NBA player game logs using the hoopR package.

**Features:**
- Collects player game logs for specific date ranges
- Supports incremental data collection (only new data since last run)
- Aggregates player logs to team-level statistics
- Saves data to SQLite database

**Usage:**

```python
from src.DataProviders.HoopRDataCollector import collect_2025_26_season_data

# Collect data from opening night to today
df = collect_2025_26_season_data(
    date_from="2025-10-22",
    date_to="2025-11-03",
    save_to_db=True
)
```

**Command Line:**

```bash
py src/DataProviders/HoopRDataCollector.py
```

### 2. Transaction Tracker (`src/DataProviders/TransactionTracker.py`)

Tracks NBA transactions (trades, signings, retirements) from Basketball Reference.

**Features:**
- Scrapes transactions from Basketball Reference
- Tracks player team history
- Identifies when team-dependent features should be reset
- Stores transaction data in SQLite database

**Usage:**

```python
from src.DataProviders.TransactionTracker import track_2025_26_transactions

transactions = track_2025_26_transactions(
    date_from="2025-10-22",
    date_to="2025-11-03"
)
```

**Command Line:**

```bash
py src/DataProviders/TransactionTracker.py
```

### 3. Feature Rebuilder (`src/Process-Data/RebuildFeatures_2025_26.py`)

Rebuilds all engineered features with updated data including current season games.

**Features:**
- Loads updated game data including 2025-26 season
- Handles team changes (resets team-dependent features)
- Rebuilds rolling averages (3, 5, 10, 15, 20 game windows)
- Rebuilds advanced features (ELO, momentum, efficiency metrics)
- Saves updated dataset

**Usage:**

```python
from src.Process-Data.RebuildFeatures_2025_26 import rebuild_features_2025_26

features_df = rebuild_features_2025_26()
```

**Command Line:**

```bash
py src/Process-Data/RebuildFeatures_2025_26.py
```

### 4. Scheduled Retraining (`src/Process-Data/ScheduledRetraining.py`)

Automates periodic model retraining with time-series cross-validation.

**Features:**
- Checks if retraining is needed (based on time and new data)
- Retrains models with updated data
- Uses time-series cross-validation
- Tracks model versions
- Runs backtests after retraining
- Maintains retraining log

**Usage:**

```python
from src.Process-Data.ScheduledRetraining import scheduled_retraining_workflow

scheduled_retraining_workflow(
    force=False,
    model_types=["xgboost", "neural", "ensemble"]
)
```

**Command Line:**

```bash
# Normal retraining (checks if needed)
py src/Process-Data/ScheduledRetraining.py

# Force retraining
py src/Process-Data/ScheduledRetraining.py --force

# Retrain specific models
py src/Process-Data/ScheduledRetraining.py --models xgboost neural
```

### 5. Complete Pipeline Update (`update_2025_26_pipeline.py`)

Main script that runs the complete pipeline update workflow.

**Features:**
- Runs all pipeline steps in sequence
- Collects new game data
- Tracks transactions
- Rebuilds features
- Retrains models
- Provides detailed progress output

**Usage:**

```bash
# Run complete pipeline update
py update_2025_26_pipeline.py

# Run with specific date range
py update_2025_26_pipeline.py --date-from 2025-10-22 --date-to 2025-11-03

# Skip specific steps
py update_2025_26_pipeline.py --skip-collection --skip-retraining

# Force retraining
py update_2025_26_pipeline.py --force-retraining
```

## Configuration

The `config.toml` file has been updated to include the 2025-26 season:

```toml
[get-data.2025-26]
    start_date = "2025-10-22"
    end_date = "2026-06-22"
    start_year = "2025"
    end_year = "2026"

[get-odds-data.2025-26]
    start_date = "2025-10-22"
    end_date = "2026-06-22"
    start_year = "2025"
    end_year = "2026"

[create-games.2025-26]
    start_date = "2025-10-22"
    end_date = "2026-06-22"
    start_year = "2025"
    end_year = "2026"
```

## Scheduled Automation

### Windows Task Scheduler

Create a scheduled task to run the pipeline daily:

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to Daily
4. Set action to run: `py update_2025_26_pipeline.py`
5. Set start time (e.g., 2:00 AM)

### Cron (Linux/Mac)

Add to crontab for daily execution:

```bash
# Run pipeline update daily at 2 AM
0 2 * * * cd /path/to/project && py update_2025_26_pipeline.py
```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Update NBA Pipeline

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run pipeline update
        run: py update_2025_26_pipeline.py
```

## Workflow

### Daily/Weekly Updates

1. **Data Collection**
   - Collects new game logs using hoopR
   - Updates from last collected date to today
   - Saves to database

2. **Transaction Tracking**
   - Scrapes Basketball Reference for transactions
   - Updates player team history
   - Identifies players who changed teams

3. **Feature Rebuilding**
   - Loads updated dataset including new games
   - Resets team-dependent features for traded players
   - Rebuilds rolling averages (includes current season)
   - Rebuilds advanced features (ELO, momentum, etc.)

4. **Model Retraining**
   - Checks if retraining is needed (time + new data thresholds)
   - Retrains models with time-series cross-validation
   - Evaluates performance on recent games
   - Stores model versions

5. **Backtesting**
   - Runs backtests on recent games
   - Validates accuracy and ROI
   - Monitors model calibration

## Monitoring

### Retraining Log

The retraining log is stored at `Data/retraining_log.json`:

```json
{
  "last_retraining": "2025-11-03T10:00:00",
  "retraining_history": [
    {
      "date": "2025-11-03T10:00:00",
      "results": {
        "xgboost": {"success": true, "version": "20251103_100000"},
        "neural": {"success": true, "version": "20251103_100001"}
      }
    }
  ],
  "model_versions": {
    "xgboost": {
      "version": "20251103_100000",
      "date": "2025-11-03T10:00:00",
      "metrics": {"accuracy": 0.65, "roi": 0.12}
    }
  }
}
```

### Checking Pipeline Status

```python
from src.Process-Data.ScheduledRetraining import ScheduledRetrainer

retrainer = ScheduledRetrainer()
log = retrainer.load_retraining_log()

print(f"Last retraining: {log['last_retraining']}")
print(f"New games since last: {retrainer.count_new_games_since_last_retraining()}")
```

## Troubleshooting

### hoopR Package Issues

If hoopR is not available:

```bash
pip install hoopR
```

If installation fails, check: https://hoopr.sportsdataverse.org/

### Transaction Scraping Fails

Basketball Reference may block requests. Options:
- Use a VPN or proxy
- Add delays between requests
- Use alternative data sources

### Feature Rebuilding Errors

If feature rebuilding fails:
- Check that new data was collected successfully
- Verify database connections
- Check that all required columns exist in the dataset

### Model Retraining Timeout

If retraining times out:
- Reduce number of models trained
- Use `--models` flag to train specific models only
- Increase timeout in `ScheduledRetraining.py`

## Best Practices

1. **Run regularly**: Schedule daily or weekly updates
2. **Monitor logs**: Check retraining log for issues
3. **Validate data**: Verify collected data before rebuilding features
4. **Version models**: Keep track of model versions for predictions
5. **Backtest**: Always run backtests after retraining
6. **Handle errors**: Implement error handling and notifications

## Next Steps

1. Install hoopR: `pip install hoopR`
2. Run initial data collection: `py update_2025_26_pipeline.py --date-from 2025-10-22`
3. Set up scheduled task (Task Scheduler or Cron)
4. Monitor retraining log for model updates
5. Run backtests to validate performance

## Support

For issues or questions:
- Check hoopR documentation: https://hoopr.sportsdataverse.org/
- Review error logs in `Data/retraining_log.json`
- Check transaction database: `Data/Transactions.sqlite`


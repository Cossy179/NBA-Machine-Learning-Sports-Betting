#!/bin/bash
# Linux/Mac shell script to run the 2025-26 pipeline update
# Can be scheduled using cron

echo "========================================"
echo "NBA Pipeline Update - 2025-26 Season"
echo "========================================"
echo "Started: $(date)"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run the pipeline update
py update_2025_26_pipeline.py

echo ""
echo "========================================"
echo "Pipeline Update Complete"
echo "========================================"
echo "Finished: $(date)"

# Log output to file
echo "$(date) - Pipeline update completed" >> pipeline_update.log


@echo off
REM Windows batch script to run the 2025-26 pipeline update
REM Can be scheduled using Windows Task Scheduler

echo ========================================
echo NBA Pipeline Update - 2025-26 Season
echo ========================================
echo Started: %date% %time%
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the pipeline update
py update_2025_26_pipeline.py

echo.
echo ========================================
echo Pipeline Update Complete
echo ========================================
echo Finished: %date% %time%

REM Log output to file
echo %date% %time% - Pipeline update completed >> pipeline_update.log

pause


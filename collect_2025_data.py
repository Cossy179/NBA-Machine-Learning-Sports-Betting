#!/usr/bin/env python3
"""
Collect 2024-25 season data only
"""
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
import toml

sys.path.insert(1, os.path.join(sys.path[0], 'src'))
from src.Utils.tools import get_json_data, to_data_frame

def collect_2025_data():
    """Collect only 2024-25 season data"""
    config = toml.load("config.toml")
    url = config['data_url']
    
    # Only collect 2024-25 data
    season_config = config['get-data']['2024-25']
    
    print("Collecting 2024-25 NBA Season Data")
    print("=" * 50)
    print(f"Start Date: {season_config['start_date']}")
    print(f"End Date: {season_config['end_date']}")
    print(f"Season: {season_config['start_year']}-{season_config['end_year']}")
    print("=" * 50)
    
    con = sqlite3.connect("Data/TeamData.sqlite")
    
    date_pointer = datetime.strptime(season_config['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(season_config['end_date'], "%Y-%m-%d").date()
    
    collected_days = 0
    failed_days = 0
    
    while date_pointer <= end_date:
        print(f"Getting data: {date_pointer}")
        
        try:
            # Format the URL with the season parameters
            formatted_url = url.format(
                date_pointer.month, 
                date_pointer.day, 
                season_config['start_year'], 
                date_pointer.year, 
                '2024-25'
            )
            
            print(f"URL: {formatted_url}")
            
            raw_data = get_json_data(formatted_url)
            
            if raw_data and 'resultSets' in raw_data and len(raw_data['resultSets']) > 0:
                df = to_data_frame(raw_data)
                
                if not df.empty:
                    df['Date'] = str(date_pointer)
                    df.to_sql(date_pointer.strftime("%Y-%m-%d"), con, if_exists="replace")
                    collected_days += 1
                    print(f"Successfully collected data for {date_pointer}")
                else:
                    print(f"No data available for {date_pointer}")
                    failed_days += 1
            else:
                print(f"No data available for {date_pointer}")
                failed_days += 1
                
        except Exception as e:
            print(f"Error collecting data for {date_pointer}: {e}")
            failed_days += 1
        
        date_pointer = date_pointer + timedelta(days=1)
        time.sleep(random.randint(1, 3))  # Be respectful to the API
    
    con.close()
    
    print("\n" + "=" * 50)
    print("Collection Summary")
    print("=" * 50)
    print(f"Days collected: {collected_days}")
    print(f"Days failed: {failed_days}")
    print(f"Total days attempted: {collected_days + failed_days}")
    
    if collected_days > 0:
        print("Data collection completed successfully!")
    else:
        print("No data was collected. This is normal if the 2024-25 season hasn't started yet.")
        print("The season typically begins in October 2024.")

if __name__ == "__main__":
    collect_2025_data()

import sqlite3
import pandas as pd

# Connect to database
con = sqlite3.connect("Data/dataset.sqlite")

# Get table names
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Available tables:", [t[0] for t in tables])

# Check the main dataset
try:
    df = pd.read_sql_query('SELECT * FROM "dataset_2012-24_new" LIMIT 1', con)
    print(f"\nDataset columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    print(f"\nNumeric columns ({len(df.select_dtypes(include=['number']).columns)}):")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for i, col in enumerate(numeric_cols):
        print(f"{i+1:2d}. {col}")
    
    print(f"\nString columns ({len(df.select_dtypes(include=['object']).columns)}):")
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    for i, col in enumerate(string_cols):
        print(f"{i+1:2d}. {col}")
        
except Exception as e:
    print(f"Error: {e}")

con.close()

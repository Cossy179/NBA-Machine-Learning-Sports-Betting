import sqlite3

con = sqlite3.connect('Data/dataset.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Available tables:", tables)

# Check the structure of the main dataset
if tables:
    main_table = tables[0]  # Get the first table
    print(f"\nChecking structure of table: {main_table}")
    cursor.execute(f'PRAGMA table_info("{main_table}")')
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # Get a sample row
    cursor.execute(f'SELECT * FROM "{main_table}" LIMIT 1')
    sample = cursor.fetchone()
    print(f"\nSample row: {sample}")

con.close()

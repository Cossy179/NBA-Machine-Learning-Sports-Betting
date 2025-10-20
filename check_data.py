import sqlite3

# Check TeamData database
con = sqlite3.connect('Data/TeamData.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC LIMIT 10")
tables = [row[0] for row in cursor.fetchall()]
print("Recent TeamData tables:", tables)
con.close()

# Check main dataset database
con = sqlite3.connect('Data/dataset.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC LIMIT 10")
tables = [row[0] for row in cursor.fetchall()]
print("Recent dataset tables:", tables)
con.close()

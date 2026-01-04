'''
simple script for retrieving data from dolt db

use dolt to start the server in Desktop/Work/Dolt before running script
'''

from mysql import connector as cnc
import pandas as pd

#get connection
conn = cnc.connect(host = "127.0.0.1", 
                   port = 3306, 
                   user = "root", 
                   password = "", 
                   database = "earnings")

#get the names of the tables in the database
cursor = conn.cursor(buffered = True)
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()
print([table[0] for table in tables])
cursor.close()

#query for getting the 1st 5 entries of Ford from balance_sheet_assets
query = """
SELECT *
FROM balance_sheet_assets
WHERE act_symbol = 'F'
LIMIT 5;
"""

df = pd.read_sql(query, conn)
print(df.head())

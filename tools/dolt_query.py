'''
simple script for retrieving data from dolt db

use dolt to start the server in Desktop/Work/Dolt before running script
'''

from mysql import connector as cnc
import pandas as pd
from pdb import set_trace as st

#get connection
conn = cnc.connect(host = "127.0.0.1", 
                   port = 3306, 
                   user = "root", 
                   password = "", 
                   database = None)

#get the names of the tables in the database
cursor = conn.cursor(buffered = True)
"""cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()
print([table[0] for table in tables])"""
#cursor.close()

#query for getting the 1st 5 entries of Ford from balance_sheet_assets
'''query = """
SELECT *
FROM cash_flow_statement
WHERE act_symbol = 'F'
    AND date > '2015-01-01'
        AND date < '2019-03-14'
            AND period = "Quarter";
LIMIT = 1000;
"""'''
#query for getting stock data from ford
query = """
SELECT * 
FROM ohlcv
WHERE act_symbol = 'F'
    AND date > '2015-01-01'
        AND date < '2019-03-14';
LIMIT = 1000;
"""

#query for viewing tables in the database
#query = "USE stocks; SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE';"
query = "SHOW DATABASES;"
query = "SELECT * FROM earnings.cash_flow_statement WHERE act_symbol = 'PFE' AND date > '2015-01-01' AND date < '2025-10-01' AND period = 'Quarter';"
#query = "SELECT * FROM cash_flow_statement"
query = "SELECT * FROM stocks.ohlcv WHERE act_symbol = 'F' AND date > '2015-01-01' AND date < '2019-03-14'; LIMIT = 1000;"
df = pd.read_sql(query, conn)
print(df.head())

st()

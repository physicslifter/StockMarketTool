'''
The goal of this script is to identify stocks which are well-suited for our model

Goal end result:
    dataframe showing:
        volume
        price floor
        market cap
        Hurst exponent
        entropy


Filter data for each month (Monthly universe selection)
    - get all stocks with price > $5 and avg daily volume > $5M over last month
    - calc Hurst exponent for prior year
    - filter top 500 hurst exponents
    - this is stocks for the month

So end result should be a dataframe going back to 2011,
Each entry should be a month, then the top 500 picks

Since 2011:
Filter out 
For first of month, get data from previous year for all stocks
    - for each stock: 
        calculate avg market cap over previous month
        calculate Hurst exponent over previous year
Select top 5000 market caps
From this, select top 10 pct of Hurst exponents
'''
import pandas as pd
from pdb import set_trace as st

has_data = True

if has_data == False:
    from mysql import connector as cnc
    #1. get all price data for all stocks
    #get connection
    conn = cnc.connect(host = "127.0.0.1", 
                       port = 3306, 
                       user = "root", 
                       password = "", 
                       database = None)
    query = "SELECT * FROM stocks.ohlcv"
    all_data = pd.read_sql(query, conn)
    all_data.to_feather("Data/all_ohlcv.feather")
else:
    all_data = pd.read_feather("Data/all_ohlcv.feather")

#2. loop through months
start_month = 1
start_year = 2012
end_month = 12
end_year = 2025

at_end = False
while at_end == False:
    year += 1 if month == 12 else year
    month += 1 if month < 12 else 1
    if month == end_month and year == end_year:
        at_end = True




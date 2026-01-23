'''
returns all data from stock finder
'''


import pandas as pd
from pdb import set_trace as st
from DoltReader import DataReader

"""
Old slow method

ticker_data = pd.read_feather("Data/universe_data.feather")

for c, row in enumerate(ticker_data.iterrows()):
    if c < 2:
        month = row[1].month
        year = row[1].year
        prev_month = month - 1 if month != 1 else 12
        prev_year = year if month != 1 else year - 1
        for c2, ticker in enumerate(row[1].stocks):
            dr = DataReader()
            dr.get_all_data(stock = ticker, start_date = f"{prev_year}-{prev_month}-01", end_date = f"{year}-{month}-01")
            if c + c2 == 0:            
                data = dr.stock_data["ohlcv"]
            else: 
                data = pd.concat([data, dr.stock_data["ohlcv"]], axis = 0, ignore_index = True)
        
st()"""

#Faster method from Gemini
'''
returns all data from stock finder - Optimized
'''
import pandas as pd
from DoltReader import DataReader
from tqdm import tqdm # Recommended for progress tracking
import gc # Garbage collection

ticker_data = pd.read_feather("Data/universe_data.feather")
all_monthly_data = []

# Initialize Reader ONCE to reuse connection
dr = DataReader()

for c, row in tqdm(enumerate(ticker_data.itertuples()), total=len(ticker_data)):
    # Limit for testing as requested (remove this check for full run)
    # if c >= 2: break 
    
    month = row.month
    year = row.year
    prev_month = month - 1 if month != 1 else 12
    prev_year = year if month != 1 else year - 1
    
    start_date = f"{prev_year}-{prev_month}-01"
    end_date = f"{year}-{month}-01"
    
    tickers = row.stocks # List of 1000 stocks
    
    # OPTIMIZATION 1: fetch data for ALL tickers in this month in one go?
    # Since DoltReader doesn't support batching yet, we must loop.
    # But we can reuse the DR instance to save connection time.
    
    monthly_dfs = []
    
    # We can try to modify the query dynamically or just loop faster.
    # Given the existing code, looping with a single instance is the safest immediate fix.
    
    print(f"Processing {len(tickers)} stocks for {month}/{year}...")
    
    for ticker in tickers:
        try:
            # Reusing 'dr' prevents reconnecting to MySQL 1000 times
            dr.get_all_data(stock=ticker, start_date=start_date, end_date=end_date)
            
            # Extract the specific DF we need
            if "ohlcv" in dr.stock_data and not dr.stock_data["ohlcv"].empty:
                df = dr.stock_data["ohlcv"]
                # Tag it so we know which stock/month it belongs to if needed
                df['act_symbol'] = ticker 
                monthly_dfs.append(df)
        except Exception as e:
            # print(f"Error fetching {ticker}: {e}")
            continue
            
    if monthly_dfs:
        # Concatenate this month's data
        month_combined = pd.concat(monthly_dfs, ignore_index=True)
        all_monthly_data.append(month_combined)
        
    # periodic garbage collection to prevent memory ballooning
    if c % 5 == 0:
        gc.collect()

# Combine everything at the end
if all_monthly_data:
    final_df = pd.concat(all_monthly_data, ignore_index=True)
    final_df.to_feather("Data/all_training_data.feather")
    print("Done!")
else:
    print("No data collected.")
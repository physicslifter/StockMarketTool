'''
returns all data from stock finder
'''

"""
import pandas as pd
from pdb import set_trace as st
from DoltReader import DataReader
"""

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

"""
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
"""


'''
Optimized Data Fetcher
- High CPU Usage (Parallel Processing via Joblib)
- Low RAM Usage (Writes monthly chunks to disk immediately)
- Adds Hurst Exponent (1-Year Lookback)
'''

import pandas as pd
import numpy as np
from DoltReader import DataReader
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import os
import gc
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TEMP_DIR = "Data/temp_monthly_chunks"
FINAL_FILE = "Data/all_training_data.feather"
UNIVERSE_FILE = "Data/universe_data.feather"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- HELPER FUNCTIONS ---

def calculate_hurst_scalar(series):
    """
    Calculates a single Hurst exponent for the given time series.
    Input: Array-like price series (1 year of data)
    Output: Scalar Float (Hurst Exponent)
    """
    prices = series.values
    if len(prices) < 30:
        return np.nan

    try:
        # Use log prices
        log_prices = np.log(prices)
        
        # Range of lags to test (e.g., 2 days to 100 days)
        lags = range(2, min(len(prices)//2, 100))
        
        # Calculate volatility (std dev) for different lags
        # Variance ratio method approximation: Vol(tau) ~ tau^H
        tau = []
        for lag in lags:
            # Vectorized difference
            diff = log_prices[lag:] - log_prices[:-lag]
            tau.append(np.std(diff))
        
        # Avoid log(0) or NaNs
        tau = [t if t > 0 else 1e-8 for t in tau]
        
        # Linear regression on log-log plot
        # Slope = Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return np.nan

def process_single_ticker(ticker, target_year, target_month):
    """
    Worker function to fetch data, calc Hurst, and return target month DF.
    """
    try:
        # 1. Define Dates
        # Target End Date: The 1st of the target month (Data is usually fetched up to this point)
        # However, universe_data usually implies we want data FOR that month.
        # Adjusted Logic: Fetch 1 year prior to the END of the target month.
        
        # Date logic: 
        # Target Month: e.g., 2024-02
        # Start Date (Fetch): 2023-02-01 (1 year lookback for Hurst)
        # End Date (Fetch): 2024-03-01 (Include the full target month)
        # Slice Keep: 2024-02-01 to 2024-03-01
        
        dt_target = pd.Timestamp(year=target_year, month=target_month, day=1)
        dt_fetch_end = dt_target + relativedelta(months=1)
        dt_fetch_start = dt_fetch_end - relativedelta(years=1)
        
        # 2. Fetch Data (New instance per process for thread safety)
        dr = DataReader()
        dr.get_all_data(stock=ticker, start_date=dt_fetch_start, end_date=dt_fetch_end)
        
        if "ohlcv" not in dr.stock_data or dr.stock_data["ohlcv"].empty:
            return None

        df = dr.stock_data["ohlcv"]
        df['date'] = pd.to_datetime(df['date'])
        
        # 3. Calculate Hurst (Using full 1-year window)
        # We calculate one static Hurst value representing the "State" of the stock
        # entering this month based on previous year's action.
        hurst_val = calculate_hurst_scalar(df['close'])
        
        # 4. Slice to Target Month Only
        # We only want to save the training data for the specific month requested
        mask_target = (df['date'] >= dt_target) & (df['date'] < dt_fetch_end)
        df_final = df.loc[mask_target].copy()
        
        if df_final.empty:
            return None

        # 5. Add columns
        df_final['act_symbol'] = ticker
        df_final['hurst_1y'] = hurst_val
        
        return df_final

    except Exception as e:
        # print(f"Err {ticker}: {e}")
        return None

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"Reading universe from {UNIVERSE_FILE}...")
    try:
        ticker_data = pd.read_feather(UNIVERSE_FILE)
    except Exception as e:
        print(f"Error reading universe file: {e}")
        exit()

    chunk_files = []

    # Iterate over each month in the universe file
    # processing one month at a time keeps RAM usage predictable
    for i, row in enumerate(ticker_data.itertuples()):
        month = row.month
        year = row.year
        tickers = row.stocks
        
        print(f"[{i+1}/{len(ticker_data)}] Processing {month}/{year} ({len(tickers)} stocks)...")
        
        # Parallel Execution
        # n_jobs=-1 uses all CPU cores. 
        # backend='loky' is default and robust for this.
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_single_ticker)(ticker, year, month) 
            for ticker in tickers
        )
        
        # Filter out None results (failures/empty data)
        valid_dfs = [res for res in results if res is not None]
        
        if valid_dfs:
            # Concatenate just this month
            month_df = pd.concat(valid_dfs, ignore_index=True)
            
            # Save to temporary chunk
            chunk_path = f"{TEMP_DIR}/chunk_{year}_{month}.feather"
            month_df.to_feather(chunk_path)
            chunk_files.append(chunk_path)
            
            # Explicitly delete valid_dfs to free RAM before next loop
            del month_df, valid_dfs, results
            gc.collect()
            print(f"  -> Saved chunk: {chunk_path}")
        else:
            print("  -> No valid data found for this month.")

    # --- FINAL MERGE ---
    print("\nProcessing complete. Merging chunks...")
    
    if chunk_files:
        all_dfs = []
        for cf in chunk_files:
            try:
                all_dfs.append(pd.read_feather(cf))
            except:
                pass
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            # Sort for cleanliness
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values(['act_symbol', 'date'])
            
            print(f"Saving final dataset to {FINAL_FILE}...")
            final_df.to_feather(FINAL_FILE)
            print("Success!")
            
            # Cleanup Temp Files
            print("Cleaning up temp files...")
            for cf in chunk_files:
                if os.path.exists(cf):
                    os.remove(cf)
            os.rmdir(TEMP_DIR)
            
        else:
            print("Error: Chunks were empty on read-back.")
    else:
        print("No chunks were generated.")
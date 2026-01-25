'''
Optimized Training Data Generator (High CPU / Low RAM focus)
Criteria:
1. Liquidity: Top 50% of Average Daily Dollar Volume (1-month lookback)
2. Predictability: Top 500 stocks with Hurst Exponent furthest from 0.5 (1-year lookback)
'''

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import multiprocessing
import gc
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
START_YEAR = 2012
END_YEAR = 2025
START_MONTH = 1
END_MONTH = 12

SAVE_PATH = "Data/hurst_training_universe.feather"
OHLCV_PATH = "Data/all_ohlcv.feather"
ETF_PATH = "Data/ETFs.feather"

# --- OPTIMIZED FUNCTIONS ---

def calculate_hurst_single(ticker, prices):
    """
    Standalone function to calculate Hurst for a single ticker.
    Designed to be lightweight for parallel processing.
    """
    if len(prices) < 30: 
        return (ticker, np.nan)
    
    try:
        # Use Log Prices
        log_prices = np.log(prices)
        
        # Lags to test (2 to 20 days)
        lags = range(2, 20)
        
        # Vectorized volatility calculation
        # std(t) ~ t^H
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        
        # Linear regression slope
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return (ticker, slope)
    except:
        return (ticker, np.nan)

def get_liquid_universe_vectorized(df_month):
    """
    Fast, vectorized calculation of dollar volume.
    """
    # Calculate Dollar Volume directly
    # We use .values to work in numpy (faster, less overhead)
    closes = df_month['close'].values
    volumes = df_month['volume'].values
    symbols = df_month['act_symbol'].values
    
    dollar_vols = closes * volumes
    
    # Create a lightweight temporary DF for grouping
    # This is cheaper than modifying the original huge DF
    temp_df = pd.DataFrame({'s': symbols, 'dv': dollar_vols})
    
    # Group and Mean
    avg_vol = temp_df.groupby('s')['dv'].mean()
    
    # Filter Top 50%
    median_vol = avg_vol.median()
    liquid_symbols = avg_vol[avg_vol >= median_vol].index.tolist()
    
    return liquid_symbols

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"Detected {multiprocessing.cpu_count()} CPU cores.")
    print("Loading Data...")
    
    try:
        all_data = pd.read_feather(OHLCV_PATH)
        etf_data = pd.read_feather(ETF_PATH)
    except FileNotFoundError:
        print("Error: Data files not found.")
        exit()

    # Pre-processing
    # Drop columns we don't need immediately to save RAM? 
    # No, we need OHLCV for the final output. 
    # But we can convert date immediately.
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # Exclude ETFs
    etf_list = set(etf_data['act_symbol'].unique()) # Set is faster for lookup
    all_data = all_data[~all_data['act_symbol'].isin(etf_list)]
    
    # Sort once globally (Crucial for slicing performance)
    all_data = all_data.sort_values(['date', 'act_symbol'])
    
    # Explicit Indexing for faster slicing
    # Setting date as index allows partial string indexing, but boolean masks are robust.
    # We will stick to boolean masks on sorted data for simplicity.

    training_batches = []
    years = range(START_YEAR, END_YEAR + 1)
    months = range(1, 13)

    print("Starting Optimized Universe Generation...")

    for year in years:
        for month in months:
            target_start_date = pd.Timestamp(year, month, 1)
            target_end_date = target_start_date + relativedelta(months=1)
            
            # Stop if out of bounds
            if target_start_date > all_data['date'].max():
                break

            # Define Lookback Windows
            liq_start = target_start_date - relativedelta(months=1)
            hurst_start = target_start_date - relativedelta(years=1)
            
            print(f"Processing {target_start_date.date()}...")

            # --- STEP 1: LIQUIDITY (Low CPU, High RAM efficiency) ---
            # Slice only relevant rows
            mask_liq = (all_data['date'] >= liq_start) & (all_data['date'] < target_start_date)
            df_liq = all_data.loc[mask_liq, ['act_symbol', 'close', 'volume']] # LOAD ONLY NEEDED COLS
            
            if df_liq.empty: continue
            
            liquid_candidates = get_liquid_universe_vectorized(df_liq)
            
            # Clean up immediately
            del df_liq, mask_liq
            
            # --- STEP 2: HURST (High CPU - Parallelized) ---
            # We need 1 year of data, but ONLY for the liquid candidates
            # And ONLY the 'close' price column.
            mask_hurst = (all_data['date'] >= hurst_start) & (all_data['date'] < target_start_date)
            mask_candidate = all_data['act_symbol'].isin(liquid_candidates)
            
            # Extract lightweight subset (Symbol + Close only)
            df_hurst = all_data.loc[mask_hurst & mask_candidate, ['act_symbol', 'close']].copy()
            
            if df_hurst.empty: continue

            # Prepare Data for Parallel Processing
            # We group by symbol and extract the price arrays
            # dict comprehension is fast
            grouped = df_hurst.groupby('act_symbol')['close']
            tasks = [(name, group.values) for name, group in grouped]
            
            # RUN PARALLEL
            # n_jobs=-1 uses all available cores
            # prefer="threads" might be faster for numpy, "processes" better for heavy math. 
            # For Hurst, processes is usually safer to avoid GIL lock.
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(calculate_hurst_single)(ticker, prices) for ticker, prices in tasks
            )
            
            # Process Results
            hurst_scores = []
            for ticker, h_val in results:
                if not np.isnan(h_val):
                    # Calculate distance from 0.5
                    score = abs(h_val - 0.5)
                    hurst_scores.append((ticker, score))
            
            # Sort and Pick Top 500
            hurst_scores.sort(key=lambda x: x[1], reverse=True) # Sort by score desc
            top_500_tuples = hurst_scores[:500]
            final_universe_symbols = [x[0] for x in top_500_tuples]
            
            # Clean up large objects
            del df_hurst, grouped, tasks, results, hurst_scores
            gc.collect() # Force RAM release

            # --- STEP 3: FETCH DATA ---
            # Retrieve full OHLCV for the target month
            mask_target = (all_data['date'] >= target_start_date) & \
                          (all_data['date'] < target_end_date) & \
                          (all_data['act_symbol'].isin(final_universe_symbols))
            
            df_target = all_data.loc[mask_target].copy()
            
            if not df_target.empty:
                training_batches.append(df_target)
            
            print(f"  > Added {len(final_universe_symbols)} stocks.")

        # Check bounds
        if target_start_date > all_data['date'].max():
            break

    # --- SAVE ---
    if training_batches:
        print("Concatenating...")
        final_df = pd.concat(training_batches, ignore_index=True)
        final_df = final_df.sort_values(['act_symbol', 'date'])
        
        print(f"Saving to {SAVE_PATH}...")
        final_df.to_feather(SAVE_PATH)
        print("Done.")
    else:
        print("No data found.")
'''
Universe Generator: Institutional Liquidity + Hurst Feature
Criteria:
1. Liquidity: Top 1000 stocks by Dollar Volume (matches Index quality).
2. Feature: Adds 'hurst_1y' (Hurst Exponent over previous year) as a column.
3. Junk Filter: Excludes stocks < $5.00 or those down >60% in the last year.
'''

import pandas as pd
import numpy as np
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

# Filters
TOP_N_LIQUIDITY = 1000
MIN_PRICE = 5.00           # Exclude penny stocks
MAX_ANNUAL_DROP = -0.60    # Exclude stocks that lost >60% value in 1 year

# --- HELPER FUNCTIONS ---

def calculate_metrics_single(ticker, prices):
    """
    Calculates Hurst Exponent and Annual Return for a single ticker.
    Input: Array of 1 year of daily close prices.
    Output: (ticker, hurst_value, annual_return, last_price)
    """
    if len(prices) < 30: 
        return (ticker, np.nan, np.nan, np.nan)
    
    current_price = prices[-1]
    
    # 1. Calculate Annual Return (Momentum check for "trending to 0")
    # Simple (End / Start) - 1
    annual_ret = (current_price / prices[0]) - 1
    
    # 2. Calculate Hurst Exponent
    try:
        log_prices = np.log(prices)
        lags = range(2, 20) # Lags to test
        
        # Volatility calculation: std(t) ~ t^H
        # Vectorized difference of log prices at specific lags
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        
        # Avoid log(0) error
        tau = [t if t > 0 else 1e-8 for t in tau]
        
        # Slope of log-log plot
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except:
        hurst = np.nan
        
    return (ticker, hurst, annual_ret, current_price)

def get_top_liquid_tickers(df_month, n=1000):
    """
    Selects top N stocks by Dollar Volume.
    """
    # Vectorized Dollar Volume Calculation
    dollar_vol = df_month['close'].values * df_month['volume'].values
    
    # Temp DataFrame for sorting
    temp = pd.DataFrame({'s': df_month['act_symbol'].values, 'dv': dollar_vol})
    avg_dv = temp.groupby('s')['dv'].mean()
    
    # Get Top N
    return avg_dv.nlargest(n).index.tolist()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    cpu_cores = multiprocessing.cpu_count()
    print(f"Detected {cpu_cores} CPU cores.")
    print("Loading OHLCV Data...")
    
    try:
        all_data = pd.read_feather(OHLCV_PATH)
        # Ensure date is datetime
        all_data['date'] = pd.to_datetime(all_data['date'])
        
        # Load and exclude ETFs
        if True: # Wrap in block to allow collapsing
            try:
                etf_data = pd.read_feather(ETF_PATH)
                etf_list = set(etf_data['act_symbol'].unique())
                all_data = all_data[~all_data['act_symbol'].isin(etf_list)]
                print(f"Excluded {len(etf_list)} ETFs.")
            except:
                print("ETF file not found, proceeding without ETF exclusion.")

        # Sort globally once
        all_data = all_data.sort_values(['date', 'act_symbol'])
        
    except FileNotFoundError:
        print("Error: 'Data/all_ohlcv.feather' not found.")
        exit()

    training_batches = []
    
    # Time Loop
    years = range(START_YEAR, END_YEAR + 1)
    months = range(1, 13)

    print(f"Starting Universe Generation (Top {TOP_N_LIQUIDITY} Liq | Avoid Crashes)...")

    for year in years:
        for month in months:
            print(month, year)
            # 1. Define Time Windows
            # Target Month: The month we want to generate training data FOR
            target_start = pd.Timestamp(year, month, 1)
            target_end = target_start + relativedelta(months=1)
            
            # Bounds Check
            if target_start > all_data['date'].max():
                break
                
            # Liquidity Window (Previous Month)
            liq_start = target_start - relativedelta(months=1)
            
            # Feature Window (Previous Year for Hurst/Momentum)
            feat_start = target_start - relativedelta(years=1)
            
            print(f"Processing {target_start.date()}...", end=" ")

            # --- STEP 1: LIQUIDITY FILTER ---
            mask_liq = (all_data['date'] >= liq_start) & (all_data['date'] < target_start)
            df_liq = all_data.loc[mask_liq, ['act_symbol', 'close', 'volume']]
            
            if df_liq.empty: 
                print("No data.")
                continue
                
            liquid_candidates = get_top_liquid_tickers(df_liq, n=TOP_N_LIQUIDITY)
            
            # Clean up
            del df_liq, mask_liq
            
            # --- STEP 2: CALCULATE FEATURES (HURST + MOMENTUM) ---
            # We need 1 year of history for the liquid candidates
            mask_feat = (all_data['date'] >= feat_start) & (all_data['date'] < target_start)
            mask_sym = all_data['act_symbol'].isin(liquid_candidates)
            
            df_feat = all_data.loc[mask_feat & mask_sym, ['act_symbol', 'close']].copy()
            
            if df_feat.empty:
                print("No history.")
                continue

            # Group for Parallel Processing
            grouped = df_feat.groupby('act_symbol')['close']
            tasks = [(name, group.values) for name, group in grouped]
            
            # Run Parallel Calculation
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(calculate_metrics_single)(t, p) for t, p in tasks
            )
            
            # --- STEP 3: APPLY JUNK FILTERS & MAP FEATURES ---
            valid_tickers = []
            hurst_map = {} # To map calculated Hurst to the target month data
            
            dropped_count = 0
            
            for ticker, hurst, ann_ret, price in results:
                # Check 1: Valid Calculation
                if np.isnan(hurst): continue
                
                # Check 2: Penny Stock Filter
                if price < MIN_PRICE:
                    dropped_count += 1
                    continue
                
                # Check 3: "Trending to 0" Filter (Death Spiral)
                # If stock lost > 60% in the last year
                if ann_ret < MAX_ANNUAL_DROP:
                    dropped_count += 1
                    continue
                    
                # If passed, store
                valid_tickers.append(ticker)
                hurst_map[ticker] = hurst
            
            # Clean up
            del df_feat, grouped, tasks, results
            
            # --- STEP 4: FETCH TARGET DATA ---
            # Get data for the month we are going to trade
            mask_target = (all_data['date'] >= target_start) & \
                          (all_data['date'] < target_end) & \
                          (all_data['act_symbol'].isin(valid_tickers))
            
            df_target = all_data.loc[mask_target].copy()
            
            if not df_target.empty:
                # Add the static Hurst feature (computed from past year) to this month's data
                # Map is fast
                df_target['hurst_1y'] = df_target['act_symbol'].map(hurst_map)
                training_batches.append(df_target)
                print(f"Selected {len(valid_tickers)} stocks (Dropped {dropped_count} junk).")
            else:
                print("Target data empty.")

            # Periodic Garbage Collection
            if month % 3 == 0:
                gc.collect()

        if target_start > all_data['date'].max():
            break

    # --- SAVE ---
    if training_batches:
        print("\nConcatenating final universe...")
        final_df = pd.concat(training_batches, ignore_index=True)
        final_df = final_df.sort_values(['act_symbol', 'date'])
        
        print(f"Saving to {SAVE_PATH}...")
        final_df.to_feather(SAVE_PATH)
        print("Done.")
    else:
        print("No data generated.")
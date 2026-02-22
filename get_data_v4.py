'''
Universe Generator: Institutional Liquidity + Hurst Feature + Low Volatility
Criteria:
1. Liquidity: Top 250 stocks by Dollar Volume (The S&P 250).
2. Junk Filter: Excludes < $15.00, Crash > 60%.
3. Volatility Filter: Selects the bottom 50 (most stable) from the liquid list.
4. Feature: Adds 'hurst_1y' (Hurst Exponent over previous year).
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
START_YEAR = 2019
END_YEAR = 2025
START_MONTH = 1
END_MONTH = 12

SAVE_PATH = "Data/test_training_universe.feather"
OHLCV_PATH = "Data/all_ohlcv_w_splits.feather"
ETF_PATH = "Data/ETFs.feather"

# --- UPDATED FILTERS FOR 50 STOCKS ---
TOP_N_LIQUIDITY = 3000      # Broad Pool: Only look at the top 250 most liquid stocks
BOTTOM_N_VOLATILITY = 1000   # Final Selection: Pick the 50 most stable from that list
MIN_PRICE = 5          # Raised to $20 to ensure institutional quality
MAX_ANNUAL_DROP = -0.50    # Stricter: Avoid anything that dropped >50%

# --- HELPER FUNCTIONS ---

def calculate_metrics_single(ticker, prices):
    """
    Calculates Hurst, Annual Return, and Volatility for a single ticker.
    Input: Array of 1 year of daily close prices.
    Output: (ticker, hurst_value, annual_return, volatility, last_price)
    """
    if len(prices) < 30: 
        return (ticker, np.nan, np.nan, np.nan, np.nan)
    
    current_price = prices[-1]
    
    # 1. Calculate Annual Return (Momentum check for "trending to 0")
    annual_ret = (current_price / prices[0]) - 1
    
    # 2. Calculate Annualized Volatility (Standard Deviation of Log Returns)
    # used for the "Stability" filter
    try:
        log_rets = np.diff(np.log(prices))
        # Annualized Vol (assuming daily data)
        volatility = np.std(log_rets) * np.sqrt(252)
    except:
        volatility = np.nan

    # 3. Calculate Hurst Exponent
    try:
        log_prices = np.log(prices)
        lags = range(2, 20) # Lags to test
        
        # Volatility calculation: std(t) ~ t^H
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        
        # Avoid log(0) error
        tau = [t if t > 0 else 1e-8 for t in tau]
        
        # Slope of log-log plot
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except:
        hurst = np.nan
        
    return (ticker, hurst, annual_ret, volatility, current_price)

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
        all_data['date'] = pd.to_datetime(all_data['date'])
        
        if True: 
            try:
                etf_data = pd.read_feather(ETF_PATH)
                etf_list = set(etf_data['act_symbol'].unique())
                all_data = all_data[~all_data['act_symbol'].isin(etf_list)]
                print(f"Excluded {len(etf_list)} ETFs.")
            except:
                print("ETF file not found, proceeding without ETF exclusion.")

        all_data = all_data.sort_values(['date', 'act_symbol'])
        
    except FileNotFoundError:
        print("Error: 'Data/all_ohlcv.feather' not found.")
        exit()

    training_batches = []
    
    # Time Loop
    years = range(START_YEAR, END_YEAR + 1)
    months = range(1, 13)

    print(f"Starting Universe Generation (Top {TOP_N_LIQUIDITY} Liq -> Filter Junk -> Bottom {BOTTOM_N_VOLATILITY} Vol)...")

    for year in years:
        for month in months:
            # 1. Define Time Windows
            target_start = pd.Timestamp(year, month, 1)
            target_end = target_start + relativedelta(months=1)
            
            # Bounds Check
            if target_start > all_data['date'].max():
                break
                
            # Liquidity Window (Previous Month)
            liq_start = target_start - relativedelta(months=1)
            
            # Feature Window (Previous Year for Hurst/Momentum/Vol)
            feat_start = target_start - relativedelta(years=1)
            
            print(f"Processing {target_start.date()}...", end=" ")

            # --- STEP 1: LIQUIDITY FILTER ---
            mask_liq = (all_data['date'] >= liq_start) & (all_data['date'] < target_start)
            df_liq = all_data.loc[mask_liq, ['act_symbol', 'close', 'volume']]
            
            if df_liq.empty: 
                print("No data.")
                continue
            
            # Get broad pool of liquid stocks
            liquid_candidates = get_top_liquid_tickers(df_liq, n=TOP_N_LIQUIDITY)
            
            del df_liq, mask_liq
            
            # --- STEP 2: CALCULATE METRICS ---
            mask_feat = (all_data['date'] >= feat_start) & (all_data['date'] < target_start)
            mask_sym = all_data['act_symbol'].isin(liquid_candidates)
            
            df_feat = all_data.loc[mask_feat & mask_sym, ['act_symbol', 'close', 'date']].copy()
            
            if df_feat.empty:
                print("No history.")
                continue
            
            grouped = df_feat.groupby('act_symbol')['close']
            tasks = [(name, group.values) for name, group in grouped]
            
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(calculate_metrics_single)(t, p) for t, p in tasks
            )
            
            # --- STEP 3: APPLY FILTERS ---
            temp_universe = []
            dropped_junk = 0
            
            for ticker, hurst, ann_ret, vol, price in results:
                # Filter A: Calculation Failed
                if np.isnan(hurst) or np.isnan(vol): continue
                #if np.isnan(vol): continue

                # Filter B: Penny Stock (Strict)
                if price < MIN_PRICE:
                    dropped_junk += 1
                    continue
                
                # Filter C: Death Spiral (>50% drop)
                if ann_ret < MAX_ANNUAL_DROP:
                    dropped_junk += 1
                    continue
                
                temp_universe.append({
                    'act_symbol': ticker,
                    'hurst_1y': hurst,
                    'volatility': vol
                })
            
            # --- STEP 4: VOLATILITY SORT (THE "SAFE 50") ---
            if not temp_universe:
                print("No candidates passed junk filter.")
                continue
                
            df_candidates = pd.DataFrame(temp_universe)
            
            # Sort by Volatility (Low to High) and take top 50
            df_candidates = df_candidates.sort_values('volatility', ascending=True)
            final_selection = df_candidates.head(BOTTOM_N_VOLATILITY).copy()
            
            valid_tickers = final_selection['act_symbol'].tolist()
            hurst_map = dict(zip(final_selection['act_symbol'], final_selection['hurst_1y']))
            
            del df_feat, grouped, tasks, results, df_candidates
            
            # --- STEP 5: FETCH TARGET DATA ---
            mask_target = (all_data['date'] >= target_start) & \
                          (all_data['date'] < target_end) & \
                          (all_data['act_symbol'].isin(valid_tickers))
            
            df_target = all_data.loc[mask_target].copy()
            
            if not df_target.empty:
                df_target['hurst_1y'] = df_target['act_symbol'].map(hurst_map)
                training_batches.append(df_target)
                print(f"Selected {len(valid_tickers)} stocks (Dropped {dropped_junk} junk).")
            else:
                print("Target data empty.")

            if month % 3 == 0:
                gc.collect()

        if target_start > all_data['date'].max():
            break

    # --- SAVE ---
    if training_batches:
        print("\nConcatenating final universe...")
        final_df = pd.concat(training_batches, ignore_index=True)
        final_df = final_df.sort_values(['act_symbol', 'date'], ascending = True)
        final_df = final_df.reset_index(drop=True)
        
        print(f"Saving to {SAVE_PATH}...")
        final_df.to_feather(SAVE_PATH)
        print("Done.")
    else:
        print("No data generated.")
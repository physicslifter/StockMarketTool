'''
Portfolio Optimizer: Inverse Volatility Weighting Backtest & Execution
1. Re-runs the test period using Inverse Volatility Weighting logic.
2. Plots performance vs SPY (Open-to-Open) using Analysis.py.
3. Prints the target allocation for the LATEST date in the file.
'''

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
from Analysis import Stock  # Import the Stock class

plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# --- CONFIG ---
MODEL_PATH = "Data/lgbm_model.txt"
DATA_PATH = "Data/v3_training_universe.feather"
ACCOUNT_SIZE = 2000.00
MAX_POSITIONS = 1

# --- FUNCTIONS ---

def inverse_volatility_weighting(df, vol_col='lag_20d_volatility'):
    """
    Calculates weights based on Inverse Volatility.
    CRITICAL: Must use LAGGED volatility (known at Open) to avoid look-ahead bias.
    """
    if vol_col not in df.columns:
        # Fallback to equal weight if column missing
        print(f"Warning: {vol_col} not found, using Equal Weight.")
        return np.ones(len(df)) / len(df)
    
    # Get volatility (handle zeros to avoid division by zero)
    vol_values = df[vol_col].replace(0, 1e-6)
    
    # Invert volatility (Lower vol = Higher score)
    inv_vol = 1.0 / vol_values
    
    # Normalize to sum to 1
    weights = inv_vol / inv_vol.sum()
    return weights

def cross_sectional_zscore(group):
    """Helper for Cross-Sectional Z-Scoring matching Training Script"""
    std = group.std()
    if std == 0: return 0
    return (group - group.mean()) / std

def backtest_strategy(test_data):
    results = []
    dates = sorted(test_data['date'].unique())
    print(f"Backtesting over {len(dates)} days...")
    
    for d in dates:
        day_df = test_data[test_data['date'] == d].copy()
        
        # Rank by model prediction
        day_df['rank'] = day_df['pred'].rank(ascending=False)
        
        # Select top candidates with positive signal (Threshold > 0.5 implied by model confidence)
        candidates = day_df[(day_df['rank'] <= MAX_POSITIONS) & (day_df['pred'] > 0.5)].copy()
        
        if candidates.empty:
            results.append({'date': d, 'strat_ret': 0.0})
            continue
            
        # Sanity Clip for bad data/splits in feather file
        candidates['target_open_to_open'] = candidates['target_open_to_open'].clip(upper=0.15, lower=-0.15)
        
        # Weighting
        weights = inverse_volatility_weighting(candidates, vol_col='lag_20d_volatility')
        
        daily_ret = np.sum(weights * candidates['target_open_to_open'])
        results.append({'date': d, 'strat_ret': daily_ret})
        
    return pd.DataFrame(results)

# --- MAIN EXECUTION ---

print("1. Loading Universe Data...")
try:
    universe_df = pd.read_feather(DATA_PATH)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found.")
    exit()

universe_df['date'] = pd.to_datetime(universe_df['date'])
universe_df = universe_df.sort_values(['act_symbol', 'date'])

print("2. Re-Engineering Features (Exact Match to lgbm_v5.py)...")

# A. Daily Log Return & Close
universe_df['prev_close'] = universe_df.groupby('act_symbol')['close'].shift(1)
universe_df['log_ret'] = np.log(universe_df['close'] / universe_df['prev_close'])

# B. Momentum & Volatility
grouper = universe_df.groupby('act_symbol')['log_ret']
for window in [5, 20]:
    col_name = f"{window}_day_log_ret"
    universe_df[col_name] = grouper.transform(lambda x: x.rolling(window).sum())

universe_df["20d_volatility"] = grouper.transform(lambda x: x.rolling(20).std())

# --- TARGET CREATION ---
universe_df['next_open'] = universe_df.groupby('act_symbol')['open'].shift(-1)
universe_df['target_open_to_open'] = np.log(universe_df['next_open'] / universe_df['open'])

# --- FEATURE GENERATION ---
model_features = []

# A. Gap Feature
universe_df['morning_gap'] = np.log(universe_df['open'] / universe_df['prev_close'])
universe_df['lag_20d_volatility'] = universe_df.groupby('act_symbol')['20d_volatility'].shift(1) 
universe_df['gap_sigma'] = universe_df['morning_gap'] / (universe_df['lag_20d_volatility'] + 1e-8)
model_features.append('gap_sigma')

# B. Lagged Features
raw_features = ["5_day_log_ret", "20_day_log_ret", "20d_volatility"]

for feat in raw_features:
    lag_col = f"lag_{feat}"
    universe_df[lag_col] = universe_df.groupby('act_symbol')[feat].shift(1)
    
    z_col = f"{lag_col}_z"
    universe_df[z_col] = universe_df.groupby('date')[lag_col].transform(cross_sectional_zscore)
    model_features.append(z_col)

# C. Volume / Liquidity
universe_df['volume_lagged'] = universe_df.groupby('act_symbol')['volume'].shift(1)
universe_df['vol_ma_20_lagged'] = universe_df.groupby('act_symbol')['volume_lagged'].transform(lambda x: x.rolling(20).mean())

universe_df['rel_volume_prev_day'] = universe_df['volume_lagged'] / (universe_df['vol_ma_20_lagged'] + 1e-8)
universe_df['rel_vol_z'] = universe_df.groupby('date')['rel_volume_prev_day'].transform(cross_sectional_zscore)
model_features.append('rel_vol_z')

# D. Price Action / Candle Shape
universe_df['prev_high'] = universe_df.groupby('act_symbol')['high'].shift(1)
universe_df['prev_low'] = universe_df.groupby('act_symbol')['low'].shift(1)
universe_df['prev_close_raw'] = universe_df.groupby('act_symbol')['close'].shift(1)

universe_df['prev_range'] = universe_df['prev_high'] - universe_df['prev_low']
universe_df['prev_close_loc'] = (universe_df['prev_close_raw'] - universe_df['prev_low']) / (universe_df['prev_range'] + 1e-8)
model_features.append('prev_close_loc')

# E. Interaction
universe_df['ret_vol_interaction'] = universe_df['lag_5_day_log_ret'] * universe_df['lag_20d_volatility']
universe_df['interaction_z'] = universe_df.groupby('date')['ret_vol_interaction'].transform(cross_sectional_zscore)
model_features.append('interaction_z')

# F. Hurst Exponent (ADDED BACK TO MATCH lgbm_v5.py)
if 'hurst_1y' in universe_df.columns:
    universe_df['hurst_lagged'] = universe_df.groupby('act_symbol')['hurst_1y'].shift(1)
    universe_df['hurst_z'] = universe_df.groupby('date')['hurst_lagged'].transform(cross_sectional_zscore)
    model_features.append('hurst_z')

print(f"Features Generated: {model_features}")

print("3. Loading Model...")
try:
    model = lgb.Booster(model_file=MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- PART A: BACKTEST & PLOT ---

print("4. Preparing Test Split...")
# Drop NAs
data_clean = universe_df.dropna(subset=model_features + ['target_open_to_open'])

# Define test split (last 15%)
unique_dates = sorted(data_clean['date'].unique())
val_cutoff = unique_dates[int(len(unique_dates) * 0.9)]
test_df = data_clean[data_clean['date'] >= val_cutoff].copy()

print(f"Generating predictions for {len(test_df)} rows...")
test_df['pred'] = model.predict(test_df[model_features])

print("5. Running Backtest...")
backtest_res = backtest_strategy(test_df)
backtest_res['cum_ret'] = backtest_res['strat_ret'].cumsum()

print("6. Fetching SPY Benchmark via Analysis.py...")
try:
    spy_stock = Stock("SPY", data_type="dolt")
    # Align SPY to the test period
    spy_stock.chop_data(backtest_res['date'].min(), backtest_res['date'].max())

    # Calculate Open-to-Open for SPY Benchmark
    spy_data = spy_stock.data.copy()
    spy_data['next_open'] = spy_data['Open'].shift(-1)
    spy_data['open_ret'] = np.log(spy_data['next_open'] / spy_data['Open'])
    spy_data = spy_data.dropna(subset=['open_ret'])
    spy_data['cum_ret'] = spy_data['open_ret'].cumsum()
    
    spy_avail = True
except Exception as e:
    print(f"SPY Benchmark unavailable: {e}")
    spy_avail = False

# Plot
plt.figure(figsize=(12,5))
plt.plot(backtest_res['date'], backtest_res['cum_ret'], label=f'MODEL', color='lime', linewidth = 3)
if spy_avail:
    plt.plot(spy_data['Date'], spy_data['cum_ret'], label="SPY", c="magenta", linewidth = 3)

plt.title(f"# picks = {MAX_POSITIONS}")
plt.axhline(0, color='white', linewidth=0.8, linestyle='--')
plt.legend(fontsize  ="xx-large")
plt.grid(True, alpha=0.1)
plt.show()

# --- PART B: EXECUTION FOR LATEST DATE ---

print("\n--- GENERATING ORDERS FOR LATEST DATE ---")
latest_date = universe_df['date'].max()
print(f"Date: {latest_date.date()}")

# Filter for today
today_df = universe_df[universe_df['date'] == latest_date].copy()
today_df = today_df.dropna(subset=model_features)

if today_df.empty:
    print("No data available for latest date (Check if previous day data exists for lags).")
else:
    today_df['pred'] = model.predict(today_df[model_features])
    today_df['rank'] = today_df['pred'].rank(ascending=False)
    
    # Filter: Top N and Signal > 0.5 (Model Threshold)
    candidates = today_df[(today_df['rank'] <= MAX_POSITIONS) & (today_df['pred'] > 0.5)].copy()
    
    if candidates.empty:
        print("No positive buy signals for today (>0.5 probability).")
    else:
        # Weighting
        candidates['weight'] = inverse_volatility_weighting(candidates, vol_col='lag_20d_volatility')
        candidates['allocation'] = candidates['weight'] * ACCOUNT_SIZE
        candidates['shares'] = candidates['allocation'] / candidates['open']
        
        output_cols = ['act_symbol', 'open', 'pred', 'weight', 'allocation', 'shares']
        print(candidates[output_cols].sort_values('weight', ascending=False).to_string(index=False, formatters={
            'open': '${:,.2f}'.format,
            'pred': '{:.4f}'.format,
            'weight': '{:.2%}'.format,
            'allocation': '${:,.2f}'.format,
            'shares': '{:.1f}'.format
        }))
        print(f"\nTotal Invested: ${candidates['allocation'].sum():,.2f}")
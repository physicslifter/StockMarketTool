'''
Portfolio Optimizer: Finds the optimal N (Number of Stocks)
by running a vectorised backtest with different N values.
'''

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
from pdb import set_trace as st

warnings.filterwarnings('ignore')

# --- CONFIG ---
MODEL_PATH = "Data/lgbm_model.txt"
DATA_PATH = "Data/v3_training_universe.feather"
COMMISSION_BPS = 0.0005  # 5 basis points slippage per trade
INITIAL_CAPITAL = 10000

# --- HELPER FUNCTIONS ---
def get_metrics(daily_returns):
    if daily_returns.std() == 0: return 0, 0, 0
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) == 0:
        sortino = 0
    else:
        sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        
    cum_ret = (1 + daily_returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return sharpe, sortino, max_dd

"""def run_backtest(df, n_positions, weight_scheme='inv_vol', get_info:bool = False):
    results = []
    dates = sorted(df['date'].unique())
    
    if get_info == True:
        predicted_returns = []
        tickers = []
    for d in dates:
        day_df = df[df['date'] == d].copy()
        
        # Rank by prediction
        day_df['rank'] = day_df['pred'].rank(ascending=False)
        
        # Select Top N
        candidates = day_df[day_df['rank'] <= n_positions].copy()
        
        if candidates.empty:
            results.append(0.0)
            if get_info == True:
                tickers.append("NONE")
                predicted_returns.append(np.nan)
        else:
            if get_info == True:
                pred = day_df[day_df.act_symbol in candidates]["pred"]
                predicted_returns.append(pred)
                tickers.append(candidates)
            continue
            
        # Weighting Logic
        if weight_scheme == 'inv_vol':
            # Use lagged volatility to avoid lookahead
            vol = candidates['lag_20d_volatility'].replace(0, 1e-6)
            inv_vol = 1.0 / vol
            weights = inv_vol / inv_vol.sum()
        else:
            weights = 1.0 / len(candidates)
            
        # Calculate Gross Return
        gross_ret = np.sum(weights * candidates['target_open_to_open'])
        
        # Apply Slippage (Entry + Exit)
        net_ret = gross_ret - (COMMISSION_BPS * 2)
    if get_info == True:
        print(len(dates), len(predicted_returns), len(tickers))
        info_df = pd.DataFrame({"date":dates, "ticker": tickers, "pred_ret":predicted_returns})
        return pd.Series(results, index=dates), info_df
    else:
        return pd.Series(results, index=dates)"""

def run_backtest(df, n_positions, weight_scheme='inv_vol', get_info: bool = False):
    """
    Vectorized backtest that guarantees aligned array lengths for debugging.
    """
    dates = sorted(df['date'].unique())
    
    # 1. Initialize output lists
    results = []
    
    # Only initialize these if we need them, but we will fill them inside the loop
    # to guarantee length alignment with 'dates'
    tickers_log = []
    preds_log = []
    
    for d in dates:
        day_df = df[df['date'] == d].copy()
        
        # Rank by prediction
        day_df['rank'] = day_df['pred'].rank(ascending=False)
        
        # Select Top N
        candidates = day_df[day_df['rank'] <= n_positions].copy()
        
        # --- Variables to store for this specific day ---
        daily_net_ret = 0.0
        daily_tickers = "NONE" # Default if empty
        daily_pred_val = np.nan # Default if empty

        # --- LOGIC BRANCH ---
        if candidates.empty:
            # No trades today, defaults remain (0.0 return, "NONE" ticker)
            pass
            
        else:
            # 1. Calculate Weights
            if weight_scheme == 'inv_vol':
                # Use lagged volatility to avoid lookahead
                vol = candidates['lag_20d_volatility'].replace(0, 1e-6)
                inv_vol = 1.0 / vol
                weights = inv_vol / inv_vol.sum()
            else:
                weights = 1.0 / len(candidates)
                
            # 2. Calculate Returns
            gross_ret = np.sum(weights * candidates['target_open_to_open'])
            daily_net_ret = gross_ret - (COMMISSION_BPS * 2)
            
            # 3. Capture Info (If requested)
            if get_info:
                if n_positions == 1:
                    daily_tickers = candidates['act_symbol'].iloc[0]
                    daily_pred_val = candidates['pred'].iloc[0]
                else:
                    daily_tickers = candidates['act_symbol'].tolist()
                    daily_pred_val = candidates['pred'].mean()

        # --- STORAGE ---
        # We append exactly one item per date to ensure alignment
        results.append(daily_net_ret)
        
        if get_info:
            tickers_log.append(daily_tickers)
            preds_log.append(daily_pred_val)

    # --- FINAL RETURN ---
    results_series = pd.Series(results, index=dates)
    
    if get_info:
        # Create detailed DataFrame
        info_df = pd.DataFrame({
            "date": dates, 
            "ticker": tickers_log, 
            "pred_ret": preds_log
        })
        return results_series, info_df
    else:
        return results_series

def cross_sectional_zscore(group):
    std = group.std()
    if std == 0: return 0
    return (group - group.mean()) / std

# --- MAIN EXECUTION ---

print("1. Loading Data...")
try:
    df = pd.read_feather(DATA_PATH)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])

print("2. Re-Engineering Features...")

# A. Daily Log Return & Close
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])

# B. Momentum & Volatility
grouper = df.groupby('act_symbol')['log_ret']
for window in [5, 20]:
    col_name = f"{window}_day_log_ret"
    df[col_name] = grouper.transform(lambda x: x.rolling(window).sum())

df["20d_volatility"] = grouper.transform(lambda x: x.rolling(20).std())

# Target (Needed for Backtest)
df['next_open'] = df.groupby('act_symbol')['open'].shift(-1)
df['target_open_to_open'] = np.log(df['next_open'] / df['open'])

# --- FEATURE GENERATION (Must match Training) ---
model_features = []

# Gap Feature
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
df['lag_20d_volatility'] = df.groupby('act_symbol')['20d_volatility'].shift(1) 
df['gap_sigma'] = df['morning_gap'] / (df['lag_20d_volatility'] + 1e-8)
model_features.append('gap_sigma')

# Lagged Features
raw_features = ["5_day_log_ret", "20_day_log_ret", "20d_volatility"]
for feat in raw_features:
    lag_col = f"lag_{feat}"
    df[lag_col] = df.groupby('act_symbol')[feat].shift(1)
    z_col = f"{lag_col}_z"
    df[z_col] = df.groupby('date')[lag_col].transform(cross_sectional_zscore)
    model_features.append(z_col)

# Volume
df['volume_lagged'] = df.groupby('act_symbol')['volume'].shift(1)
df['vol_ma_20_lagged'] = df.groupby('act_symbol')['volume_lagged'].transform(lambda x: x.rolling(20).mean())
df['rel_volume_prev_day'] = df['volume_lagged'] / (df['vol_ma_20_lagged'] + 1e-8)
df['rel_vol_z'] = df.groupby('date')['rel_volume_prev_day'].transform(cross_sectional_zscore)
model_features.append('rel_vol_z')

# Price Action
df['prev_high'] = df.groupby('act_symbol')['high'].shift(1)
df['prev_low'] = df.groupby('act_symbol')['low'].shift(1)
df['prev_close_raw'] = df.groupby('act_symbol')['close'].shift(1)
df['prev_range'] = df['prev_high'] - df['prev_low']
df['prev_close_loc'] = (df['prev_close_raw'] - df['prev_low']) / (df['prev_range'] + 1e-8)
model_features.append('prev_close_loc')

# Interaction
df['ret_vol_interaction'] = df['lag_5_day_log_ret'] * df['lag_20d_volatility']
df['interaction_z'] = df.groupby('date')['ret_vol_interaction'].transform(cross_sectional_zscore)
model_features.append('interaction_z')

# Hurst (Optional check if column exists)
if 'hurst_1y' in df.columns:
    df['hurst_lagged'] = df.groupby('act_symbol')['hurst_1y'].shift(1)
    df['hurst_z'] = df.groupby('date')['hurst_lagged'].transform(cross_sectional_zscore)
    model_features.append('hurst_z')

print(f"Features: {model_features}")

# Drop NaNs before prediction
# We need rows that have features AND a target to backtest
df_clean = df.dropna(subset=model_features + ['target_open_to_open']).copy()

print("3. Generating Predictions...")
try:
    model = lgb.Booster(model_file=MODEL_PATH)
    # THIS LINE CREATES THE 'pred' COLUMN
    df_clean['pred'] = model.predict(df_clean[model_features])
except Exception as e:
    print(f"Error loading model or predicting: {e}")
    exit()

# Filter for Test Period (Last 15% to match training)
unique_dates = sorted(df_clean['date'].unique())
test_cutoff = unique_dates[int(len(unique_dates) * 0.85)]
test_df = df_clean[df_clean['date'] >= test_cutoff].copy()

print(f"Testing on {len(test_df)} rows from {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# --- SIMULATION LOOP ---
n_values = [1, 2, 3, 5, 10, 20, 50]
stats = []

print("\nRESULTS (Sorted by Sharpe):")
print(f"{'N':<5} | {'Ann. Ret':<10} | {'Sharpe':<8} | {'Sortino':<8} | {'Max DD':<8}")
print("-" * 55)

for n in n_values:
    # Run Equal Weight
    # rets_eq = run_backtest(test_df, n, weight_scheme='equal')
    
    # Run Inverse Vol Weight (Your preferred method)
    get_info = True if n == 1 else False
    if n == 1:
        rets_iv, info_df = run_backtest(test_df, n, weight_scheme='inv_vol', get_info = get_info)
        info_1 = info_df
    else:
        rets_iv = run_backtest(test_df, n, weight_scheme='inv_vol', get_info = get_info)
    sharpe_iv, sort_iv, dd_iv = get_metrics(rets_iv)
    ann_ret_iv = (1 + rets_iv.mean())**252 - 1
    
    print(f"{n:<5} | {ann_ret_iv:.2%}    | {sharpe_iv:.2f}     | {sort_iv:.2f}     | {dd_iv:.2%}")
    
    stats.append({
        'N': n,
        'Sharpe': sharpe_iv,
        'Sortino': sort_iv,
        'MaxDD': dd_iv,
        'AnnRet': ann_ret_iv
    })

# --- VISUALIZATION ---
stats_df = pd.DataFrame(stats)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:green'
ax1.set_xlabel('Number of Positions (N)')
ax1.set_ylabel('Sharpe Ratio', color=color, fontweight='bold')
ax1.plot(stats_df['N'], stats_df['Sharpe'], color=color, marker='o', linewidth=2, label='Sharpe')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Max Drawdown', color=color, fontweight='bold')  
ax2.plot(stats_df['N'], stats_df['MaxDD'], color=color, marker='x', linestyle='--', linewidth=2, label='Max DD')
ax2.tick_params(axis='y', labelcolor=color)
ax2.invert_yaxis() # Invert so higher up is better (smaller drawdown)

plt.title('Optimization: Risk vs Diversification (N)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

st()
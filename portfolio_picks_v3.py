import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
from pdb import set_trace as st
from Analysis import Stock

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# --- CONFIG ---
MODEL_PATH = "Data/lgbm_model.txt"
DATA_PATH = "Data/test_training_universe.feather"
COMMISSION_BPS = 0.0005 

def cross_sectional_zscore(group):
    std = group.std()
    return (group - group.mean()) / (std + 1e-8) if std != 0 else 0

# 1. LOAD AND PREP DATA
print("Loading and Engineering Data...")
df = pd.read_feather(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])

# Basic technicals
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])
df['5_day_log_ret'] = df.groupby('act_symbol')['log_ret'].transform(lambda x: x.rolling(5).sum())
df['20_day_log_ret'] = df.groupby('act_symbol')['log_ret'].transform(lambda x: x.rolling(20).sum())
df['20d_volatility'] = df.groupby('act_symbol')['log_ret'].transform(lambda x: x.rolling(20).std())

# Target: Open-to-Open
df['next_open'] = df.groupby('act_symbol')['open'].shift(-1)
df['target_open_to_open'] = np.log(df['next_open'] / df['open'])

# Feature Engineering (Matching Training)
model_features = []
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
df['lag_20d_volatility'] = df.groupby('act_symbol')['20d_volatility'].shift(1)
df['gap_sigma'] = df['morning_gap'] / (df['lag_20d_volatility'] + 1e-8)
model_features.append('gap_sigma')

for feat in ["5_day_log_ret", "20_day_log_ret", "20d_volatility"]:
    lag_col = f"lag_{feat}"
    df[lag_col] = df.groupby('act_symbol')[feat].shift(1)
    df[f"{lag_col}_z"] = df.groupby('date')[lag_col].transform(cross_sectional_zscore)
    model_features.append(f"{lag_col}_z")

df['volume_lagged'] = df.groupby('act_symbol')['volume'].shift(1)
df['vol_ma_20_lagged'] = df.groupby('act_symbol')['volume_lagged'].transform(lambda x: x.rolling(20).mean())
# 1. Create the raw calculation as a column first
df['rel_vol_raw'] = df['volume_lagged'] / (df['vol_ma_20_lagged'] + 1e-8)

# 2. Now group by the column name 'rel_vol_raw'
df['rel_vol_z'] = df.groupby('date')['rel_vol_raw'].transform(cross_sectional_zscore)
model_features.append('rel_vol_z')

df['prev_high'], df['prev_low'], df['prev_close_raw'] = df.groupby('act_symbol')['high'].shift(1), df.groupby('act_symbol')['low'].shift(1), df.groupby('act_symbol')['close'].shift(1)
df['prev_close_loc'] = (df['prev_close_raw'] - df['prev_low']) / ((df['prev_high'] - df['prev_low']) + 1e-8)
model_features.append('prev_close_loc')

# 1. Create the interaction column
df['ret_vol_interaction'] = df['lag_5_day_log_ret'] * df['lag_20d_volatility']

# 2. Z-Score it
df['interaction_z'] = df.groupby('date')['ret_vol_interaction'].transform(cross_sectional_zscore)
model_features.append('interaction_z')

# F. Hurst Exponent (MUST MATCH TRAINING)
if 'hurst_1y' in df.columns:
    # Shift to avoid lookahead, just like in training
    df['hurst_lagged'] = df.groupby('act_symbol')['hurst_1y'].shift(1)
    df['hurst_z'] = df.groupby('date')['hurst_lagged'].transform(cross_sectional_zscore)
    model_features.append('hurst_z')

# PREDICT
df_clean = df.dropna(subset=model_features + ['target_open_to_open']).copy()
model = lgb.Booster(model_file=MODEL_PATH)
df_clean['pred'] = model.predict(df_clean[model_features])

# FILTER TEST PERIOD (Last 15%)
test_dates = sorted(df_clean['date'].unique())
test_cutoff = test_dates[int(len(test_dates) * 0.9)]
test_df = df_clean[df_clean['date'] >= test_cutoff].copy()

# 2. GENERATE N=1 BASE RESULTS
print("Running N=1 Base Backtest...")
picks = test_df.groupby('date').apply(lambda x: x.nlargest(1, 'pred')).reset_index(drop=True)
picks['net_return'] = picks['target_open_to_open'] - (COMMISSION_BPS * 2)

# 3. ANALYSIS: THRESHOLDS (SNIPER TEST)
thresholds = [0.50, 0.51, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55]
thresh_results = []

for t in thresholds:
    subset = picks[picks['pred'] > t]
    if len(subset) > 5:
        avg_ret = subset['net_return'].mean()
        std_ret = subset['net_return'].std()
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        cum_ret = (1 + subset['net_return']).prod() - 1
        thresh_results.append({'Threshold': t, 'Sharpe': sharpe, 'Trades': len(subset), 'TotalRet': cum_ret})

thresh_df = pd.DataFrame(thresh_results)

# 4. ANALYSIS: MARKET REGIME
print("Fetching SPY for Regime Analysis...")
spy = yf.download("SPY", start=test_df['date'].min(), end=test_df['date'].max(), progress=False)
spy_close = spy['Close'].iloc[:, 0] if isinstance(spy['Close'], pd.DataFrame) else spy['Close']
spy_ma = spy_close.rolling(200).mean()
picks = picks.set_index('date')
picks['spy_bullish'] = (spy_close > spy_ma).reindex(picks.index, method='ffill')

# 5. VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# A. Sharpe vs Threshold
axes[0, 0].plot(thresh_df['Threshold'], thresh_df['Sharpe'], marker='o', color='blue', linewidth=2)
axes[0, 0].set_title("The Sniper Test: Sharpe vs. Confidence", fontsize=12)
axes[0, 0].set_xlabel("Model Probability Threshold")
axes[0, 0].set_ylabel("Annualized Sharpe Ratio")

# B. Cumulative Returns by Threshold
for t in [0.50, 0.525, 0.53, 0.535, 0.54, 0.545]:
    curve = (1 + picks[picks['pred'] > t]['net_return']).cumprod()
    #curve = picks[picks["pred"] > t]["log_ret"].cumsum()
    axes[0, 1].plot(curve, label=f"Threshold > {t}")
axes[0, 1].set_title("Cumulative Growth by Confidence Level", fontsize=12)
axes[0, 1].legend()

#B.1 Plot SPY log return for this time period
SPY = Stock("SPY")
min_date = str(picks.index.min()).split(" ")[0]
max_date = str(picks.index.max()).split(" ")[0]
SPY.chop_data(min_date, max_date)
spy_data = SPY.data
spy_data['next_open'] = spy_data['Open'].shift(-1)
spy_data['open_ret'] = np.log(spy_data['next_open'] / spy_data['Open'])
spy_data = spy_data.dropna(subset=['open_ret'])
spy_data['cum_ret'] = spy_data['open_ret'].cumsum() + 1
axes[0, 1].plot(spy_data.Date, spy_data.cum_ret, linestyle = "--", linewidth = 3, label = "SPY")


# C. Regime Analysis (Bar Chart)
bull_avg = picks[picks['spy_bullish'] == True]['net_return'].mean()
bear_avg = picks[picks['spy_bullish'] == False]['net_return'].mean()
axes[1, 0].bar(['Bull (SPY > 200MA)', 'Bear (SPY < 200MA)'], [bull_avg, bear_avg], color=['green', 'red'])
axes[1, 0].set_title("Regime Performance (Avg Daily Return)", fontsize=12)
axes[1, 0].axhline(0, color='black', lw=1)

# D. Worst Trades Table (Top 5)
worst = picks.sort_values('net_return').head(5)[['act_symbol', 'pred', 'net_return']]
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=worst.values, colLabels=['Ticker', 'Prob', 'Return'], 
                         rowLabels=worst.index.date, loc='center', cellLoc='center')
axes[1, 1].set_title("Failure Analysis: Top 5 Blowups", fontsize=12)


plt.tight_layout()
plt.show()

#print("\n--- SUMMARY STATS ---")
#print(thresh_df.to_string(index=False))
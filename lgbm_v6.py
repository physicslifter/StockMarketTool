'''
Global LGBM Model v6: Enhanced (Ensemble + Time + Ranking + Dynamic Target)
Decision Point: Market Open
Execution: Buy at Open, Sell at Next Open
Target: Probability(Alpha > Volatility Threshold)
Strategy: Top N Ranked Stocks
'''
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')

UNIVERSE_DATA = "Data/v3_training_universe.feather"
MODEL_SAVE_DIR = "Data/models_v6"

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# 1. LOAD DATA
print("Loading Universe data...")
try:
    df = pd.read_feather(UNIVERSE_DATA)
except FileNotFoundError:
    print(f"Error: '{UNIVERSE_DATA}' not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])

# 2. GENERATE BASE TECHNICALS
print("Calculating base technical indicators...")

# A. Daily Log Return & Close
# LEAK PROTECTION: These are raw inputs. We lag them later.
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])

# B. Momentum & Volatility
grouper = df.groupby('act_symbol')['log_ret']
for window in [5, 20]:
    col_name = f"{window}_day_log_ret"
    df[col_name] = grouper.transform(lambda x: x.rolling(window).sum())

df["20d_volatility"] = grouper.transform(lambda x: x.rolling(20).std())

# 3. FEATURE ENGINEERING
print("Engineering features (Z-Scores, Time, & Interaction)...")

# --- TARGET CREATION (LEAK PROTECTION: LABELS ONLY) ---
# We predict the move from TODAY'S Open to TOMORROW'S Open.
df['next_open'] = df.groupby('act_symbol')['open'].shift(-1)
df['target_open_to_open'] = np.log(df['next_open'] / df['open'])

# Market Neutral Target (Alpha)
df['market_ret'] = df.groupby('date')['target_open_to_open'].transform('mean')
df['target_alpha'] = df['target_open_to_open'] - df['market_ret']

# DYNAMIC BINARY TARGET: Only 1 if Alpha > 10% of Daily Volatility
# This removes noise. We don't want to learn 0.01% moves.
vol_threshold = df['20d_volatility'] * 0.1 
df['target_binary'] = (df['target_alpha'] > vol_threshold).astype(int)

# --- FEATURE GENERATION ---
model_features = []

def cross_sectional_zscore(group):
    std = group.std()
    if std == 0: return 0
    return (group - group.mean()) / std

# A. Time Embeddings (Safe: Known in advance)
df['dow'] = df['date'].dt.dayofweek.astype('category')
df['month'] = df['date'].dt.month.astype('category')
df['is_month_end'] = df['date'].dt.is_month_end.astype(int).astype('category')
model_features.extend(['dow', 'month', 'is_month_end'])

# B. Gap Feature (Safe: Known at Market Open)
# Uses (Open_Today / Close_Yesterday)
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
# LEAK PROTECTION: Volatility must be from YESTERDAY
df['volatility_lagged'] = df.groupby('act_symbol')['20d_volatility'].shift(1)
df['gap_sigma'] = df['morning_gap'] / (df['volatility_lagged'] + 1e-8)
model_features.append('gap_sigma')

# C. Lagged Features (LEAK PROTECTION: EXPLICIT SHIFT)
raw_features = ["5_day_log_ret", "20_day_log_ret", "20d_volatility"]
for feat in raw_features:
    lag_col = f"lag_{feat}"
    # LEAK PROTECTION: Shift(1) ensures we use data ending Yesterday
    df[lag_col] = df.groupby('act_symbol')[feat].shift(1)
    
    # Z-Score compares "Yesterday's Momentum" across all stocks
    z_col = f"{lag_col}_z"
    df[z_col] = df.groupby('date')[lag_col].transform(cross_sectional_zscore)
    model_features.append(z_col)

# D. Liquidity Features
df['volume_lagged'] = df.groupby('act_symbol')['volume'].shift(1)
df['vol_ma_20_lagged'] = df.groupby('act_symbol')['volume_lagged'].transform(lambda x: x.rolling(20).mean())
df['rel_volume_prev_day'] = df['volume_lagged'] / (df['vol_ma_20_lagged'] + 1e-8)
df['rel_vol_z'] = df.groupby('date')['rel_volume_prev_day'].transform(cross_sectional_zscore)
model_features.append('rel_vol_z')

# E. Price Action
df['prev_high'] = df.groupby('act_symbol')['high'].shift(1)
df['prev_low'] = df.groupby('act_symbol')['low'].shift(1)
df['prev_close_raw'] = df.groupby('act_symbol')['close'].shift(1) 
df['prev_range'] = df['prev_high'] - df['prev_low']
df['prev_close_loc'] = (df['prev_close_raw'] - df['prev_low']) / (df['prev_range'] + 1e-8)
model_features.append('prev_close_loc')

# F. Interaction
df['ret_vol_interaction'] = df['lag_5_day_log_ret'] * df['lag_20d_volatility']
df['interaction_z'] = df.groupby('date')['ret_vol_interaction'].transform(cross_sectional_zscore)
model_features.append('interaction_z')

# G. Hurst Exponent (Assuming input is already raw hurst)
if 'hurst_1y' in df.columns:
    df['hurst_lagged'] = df.groupby('act_symbol')['hurst_1y'].shift(1)
    df['hurst_z'] = df.groupby('date')['hurst_lagged'].transform(cross_sectional_zscore)
    model_features.append('hurst_z')

# 4. CLEANING
# Drop NaNs created by lags or missing targets
data = df.dropna(subset=model_features + ['target_alpha', 'target_binary'])

# 5. SPLITTING
print("Splitting data chronologically...")
unique_dates = sorted(data['date'].unique())
train_cutoff = unique_dates[int(len(unique_dates) * 0.7)]
val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

train_df = data[data['date'] < train_cutoff]
val_df = data[(data['date'] >= train_cutoff) & (data['date'] < val_cutoff)]
test_df = data[data['date'] >= val_cutoff]

X_train, y_train = train_df[model_features], train_df['target_binary']
X_val, y_val = val_df[model_features], val_df['target_binary']
X_test, y_test_bin = test_df[model_features], test_df['target_binary']

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 6. OPTIMIZATION
def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc", 
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "seed": 42
    }
    
    # Weight logic: Focus on larger moves to separate signal from noise
    train_weights = np.log1p(np.abs(train_df['target_alpha']) * 100)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    gbm = lgb.train(
        param, dtrain, valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20), 
            optuna.integration.LightGBMPruningCallback(trial, "auc") 
        ]
    )
    
    preds = gbm.predict(X_val)
    return roc_auc_score(y_val, preds)

print("Tuning hyperparameters...")
study = optuna.create_study(direction="maximize") 
study.optimize(objective, n_trials=20)

# 7. ENSEMBLE TRAINING
print("Training ensemble models...")
best_params = study.best_params
best_params["objective"] = "binary"
best_params["metric"] = "binary_logloss" 

models = []
N_SEEDS = 5

for i in range(N_SEEDS):
    print(f"Training Seed {i+1}/{N_SEEDS}...")
    p = best_params.copy()
    p['seed'] = 42 + i
    
    dtrain = lgb.Dataset(X_train, label=y_train, weight=np.log1p(np.abs(train_df['target_alpha']) * 100))
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    m = lgb.train(
        p, dtrain, valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    m.save_model(f"{MODEL_SAVE_DIR}/model_seed_{i}.txt")
    models.append(m)

# 8. TESTING & EVALUATION
print("\n--- TEST REGION PERFORMANCE ---")

# Average predictions across ensemble
raw_preds = np.zeros(len(X_test))
for m in models:
    raw_preds += m.predict(X_test)

test_probs = raw_preds / N_SEEDS
test_df['prob_up'] = test_probs

# A. Metrics
auc = roc_auc_score(y_test_bin, test_probs)

# Directional Accuracy Calculation
# We check: If prob > 0.5, did the stock actually go up (return > 0)?
test_df['pred_direction'] = (test_df['prob_up'] > 0.5).astype(int)
test_df['true_direction'] = (test_df['target_open_to_open'] > 0).astype(int)
dir_acc = accuracy_score(test_df['true_direction'], test_df['pred_direction'])

print(f"Ensemble ROC AUC Score: {auc:.4f}")
print(f"Directional Accuracy (Raw Up/Down): {dir_acc:.2%}")

# 9. RANKING BACKTEST
TOP_N = 20
COMMISSION = 0.0010 # 10bps conservative slippage

# Rank: 1 is highest prob
test_df['daily_rank'] = test_df.groupby('date')['prob_up'].rank(ascending=False)

# Signal
test_df['signal'] = (test_df['daily_rank'] <= TOP_N).astype(int)

# Calculate Daily Returns
trades = test_df[test_df['signal'] == 1].copy()
trades['net_ret'] = trades['target_open_to_open'] - (COMMISSION * 2) 

# Equal weight daily return
daily_strat = trades.groupby('date')['net_ret'].mean()

# Benchmark (Equal weight of entire test universe per day)
daily_bench = test_df.groupby('date')['target_open_to_open'].mean()

# Align dates (fill days with no trades as 0 return)
perf_df = pd.DataFrame(index=test_df['date'].unique()).sort_index()
perf_df['Strategy'] = daily_strat
perf_df['Benchmark'] = daily_bench
perf_df['Strategy'] = perf_df['Strategy'].fillna(0)

# Cumulative
perf_df['Cum_Strat'] = np.exp(np.log1p(perf_df['Strategy']).cumsum())
perf_df['Cum_Bench'] = np.exp(np.log1p(perf_df['Benchmark']).cumsum())

# Stats
sharpe = (perf_df['Strategy'].mean() / perf_df['Strategy'].std()) * np.sqrt(252)
total_ret = perf_df['Cum_Strat'].iloc[-1] - 1

print(f"\n--- STRATEGY RETURNS (Top {TOP_N} per day) ---")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Total Return: {total_ret:.2%}")

# 10. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(perf_df['Cum_Strat'], label=f'Top {TOP_N} Ensemble (Sharpe: {sharpe:.2f})', color='green', linewidth=2)
plt.plot(perf_df['Cum_Bench'], label='Benchmark (Universe Avg)', linestyle='--', color='gray', alpha=0.7)
plt.title(f'Performance: LGBM Ensemble (Top {TOP_N} Ranked)')
plt.xlabel('Date')
plt.ylabel('Cumulative Growth ($1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

lgb.plot_importance(models[0], importance_type='gain', figsize=(10, 8), title='Feature Importance (Gain - Seed 0)')
plt.tight_layout()
plt.show()
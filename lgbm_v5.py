'''
Global LGBM Model: Open-to-Open Strategy (Revised: Classifier & Z-Scores)
Decision Point: Market Open
Execution: Buy at Open, Sell at Next Open
Target: Probability(Alpha > 0)
Strategy: Long if Probability > 0.5
'''
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import warnings
import os
from pdb import set_trace as st

warnings.filterwarnings('ignore')

UNIVERSE_DATA = "Data/test_training_universe.feather"

# 1. LOAD DATA
print("Loading Hurst Universe data...")
try:
    df = pd.read_feather(UNIVERSE_DATA)
except FileNotFoundError:
    print(f"Error: '{UNIVERSE_DATA}' not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])
daily_market_ret = df.groupby('date')['close'].mean() # Proxy for market index
market_trend = daily_market_ret.rolling(200).mean()

# 2. GENERATE BASE TECHNICALS
print("Calculating base technical indicators...")

# A. Daily Log Return & Close
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])

# B. Momentum & Volatility
grouper = df.groupby('act_symbol')['log_ret']
for window in [5, 20]:
    col_name = f"{window}_day_log_ret"
    df[col_name] = grouper.transform(lambda x: x.rolling(window).sum())

df["20d_volatility"] = grouper.transform(lambda x: x.rolling(20).std())

# 3. FEATURE ENGINEERING
print("Engineering features (Z-Scores & Interaction)...")

# --- TARGET CREATION ---
df['next_open'] = df.groupby('act_symbol')['open'].shift(-1)
df['target_open_to_open'] = np.log(df['next_open'] / df['open'])

# Market Neutral Target (Alpha)
df['market_ret'] = df.groupby('date')['target_open_to_open'].transform('mean')
df['target_alpha'] = df['target_open_to_open'] - df['market_ret']

# BINARY TARGET (For Classification)
df['target_binary'] = (df['target_alpha'] > 0).astype(int)

# --- FEATURE GENERATION ---
model_features = []

# Helper for Cross-Sectional Z-Scoring
def cross_sectional_zscore(group):
    std = group.std()
    if std == 0: return 0
    return (group - group.mean()) / std

# A. Gap Feature (SAFE)
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
# Fix Volatility: Shift it so it relies on YESTERDAY'S rolling volatility
df['volatility_lagged'] = df.groupby('act_symbol')['20d_volatility'].shift(1)
df['gap_sigma'] = df['morning_gap'] / (df['volatility_lagged'] + 1e-8)
model_features.append('gap_sigma')

# B. Lagged Features (SAFE)
raw_features = ["5_day_log_ret", "20_day_log_ret", "20d_volatility"]

for feat in raw_features:
    lag_col = f"lag_{feat}"
    # CRITICAL: Shift(1) ensures we use yesterday's data
    df[lag_col] = df.groupby('act_symbol')[feat].shift(1)
    
    z_col = f"{lag_col}_z"
    df[z_col] = df.groupby('date')[lag_col].transform(cross_sectional_zscore)
    model_features.append(z_col)

# C. Volume / Liquidity Features (FIXED LEAK)
df['volume_lagged'] = df.groupby('act_symbol')['volume'].shift(1)
df['vol_ma_20_lagged'] = df.groupby('act_symbol')['volume_lagged'].transform(lambda x: x.rolling(20).mean())

df['rel_volume_prev_day'] = df['volume_lagged'] / (df['vol_ma_20_lagged'] + 1e-8)
df['rel_vol_z'] = df.groupby('date')['rel_volume_prev_day'].transform(cross_sectional_zscore)
model_features.append('rel_vol_z')

# D. Price Action / Candle Shape (FIXED LEAK)
df['prev_high'] = df.groupby('act_symbol')['high'].shift(1)
df['prev_low'] = df.groupby('act_symbol')['low'].shift(1)
df['prev_close_raw'] = df.groupby('act_symbol')['close'].shift(1) 

df['prev_range'] = df['prev_high'] - df['prev_low']
df['prev_close_loc'] = (df['prev_close_raw'] - df['prev_low']) / (df['prev_range'] + 1e-8)
model_features.append('prev_close_loc')

# E. Interaction (FIXED LEAK)
df['ret_vol_interaction'] = df['lag_5_day_log_ret'] * df['lag_20d_volatility']
df['interaction_z'] = df.groupby('date')['ret_vol_interaction'].transform(cross_sectional_zscore)
model_features.append('interaction_z')

# F. Hurst Exponent (ADDED & LAGGED)
if 'hurst_1y' in df.columns:
    # Must shift Hurst because the calculation usually includes the current close
    df['hurst_lagged'] = df.groupby('act_symbol')['hurst_1y'].shift(1)
    df['hurst_z'] = df.groupby('date')['hurst_lagged'].transform(cross_sectional_zscore)
    model_features.append('hurst_z')

st()

# 4. CLEANING
# Drop NaNs created by lags or missing volatility
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
# We keep continuous targets for backtesting later
y_test_rets = test_df['target_open_to_open']

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Features: {model_features}")

# 6. OPTIMIZATION (CLASSIFICATION)
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
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "seed": 42
    }
    
    # Weights logic
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

print("Tuning hyperparameters (AUC)...")
study = optuna.create_study(direction="maximize") 
study.optimize(objective, n_trials=20)

# 7. TRAINING
print("Training final model...")
best_params = study.best_params
best_params["objective"] = "binary"
best_params["metric"] = "binary_logloss" 
best_params["seed"] = 42

model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 7.5 SAVE MODEL
print("Saving model to Data/lgbm_model.txt...")
model.save_model("Data/lgbm_model.txt")

# 8. TESTING & EVALUATION
print("\n--- TEST REGION PERFORMANCE ---")

test_probs = model.predict(X_test)
test_df['prob_up'] = test_probs

# A. Classification Metrics
test_preds_class = (test_probs > 0.5).astype(int)

acc = accuracy_score(y_test_bin, test_preds_class)
auc = roc_auc_score(y_test_bin, test_probs)

print(f"Directional Accuracy: {acc:.2%}")
print(f"ROC AUC Score: {auc:.4f}")

ic = np.corrcoef(test_probs, test_df['target_alpha'])[0, 1]
print(f"Information Coefficient (Prob vs Alpha): {ic:.4f}")

# 9. BACKTESTING THE STRATEGY
threshold = 0.53

test_df['signal'] = (test_df['prob_up'] > threshold).astype(int)
test_df['strat_return'] = test_df['signal'] * test_df['target_open_to_open']

daily_perf = test_df.groupby('date').agg({
    'strat_return': 'mean',
    'target_open_to_open': 'mean' 
})
daily_perf.columns = ['Strategy_Ret', 'Benchmark_Ret']
daily_perf['Cum_Strat'] = (1 + daily_perf['Strategy_Ret']).cumprod()
daily_perf['Cum_Bench'] = (1 + daily_perf['Benchmark_Ret']).cumprod()

sharpe = (daily_perf['Strategy_Ret'].mean() / daily_perf['Strategy_Ret'].std()) * np.sqrt(252)

print(f"\n--- STRATEGY RETURNS ---")
print(f"Strategy Sharpe Ratio: {sharpe:.2f}")
print(f"Total Return: {(daily_perf['Cum_Strat'].iloc[-1] - 1):.2%}")
print(f"Benchmark Return: {(daily_perf['Cum_Bench'].iloc[-1] - 1):.2%}")

# 10. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(daily_perf['Cum_Strat'], label=f'LGBM Classifier (Sharpe: {sharpe:.2f})', color='green')
plt.plot(daily_perf['Cum_Bench'], label='Benchmark (Equal Weight)', linestyle='--', color='gray')
plt.title('Performance: LGBM Classifier (Z-Scored Features)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

lgb.plot_importance(model, importance_type='gain', figsize=(10, 6), title='Feature Importance (Gain)')
plt.tight_layout()
plt.show()
'''
Global LGBM Model using Hurst/Liquidity Universe
Input: Data/hurst_training_universe.feather
Features: Includes 'hurst_1y' (Pre-calculated 1-year Hurst Exponent)
'''
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from DoltReader import DataReader

warnings.filterwarnings('ignore')

# 1. LOAD DATA
print("Loading Hurst Universe data...")
try:
    df = pd.read_feather("Data/hurst_training_universe.feather")
except FileNotFoundError:
    print("Error: 'Data/hurst_training_universe.feather' not found. Run get_data_volume_hurst.py first.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])

# 2. GENERATE BASE TECHNICALS (Missing in raw OHLCV)
print("Calculating base technical indicators...")

# A. Daily Log Return (Close-to-Close)
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])

# B. Previous Day's Return (for momentum)
df['prev_ret'] = df.groupby('act_symbol')['log_ret'].shift(1)

# C. Rolling Returns & Volatility
# Note: These include the *current* day's close. We will shift them later.
grouper = df.groupby('act_symbol')['log_ret']

for window in [5, 20]:
    col_name = f"{window}_day_log_ret"
    # Transform maintains the index alignment
    df[col_name] = grouper.transform(lambda x: x.rolling(window).sum())

df["20d_volatility"] = grouper.transform(lambda x: x.rolling(20).std())

# 3. FEATURE ENGINEERING (Model Inputs & Targets)
print("Engineering model features (Lagging)...")

# --- Target: Intraday Return (Open -> Close) ---
df['target_intraday'] = np.log(df['close'] / df['prev_close'])

# --- Features: Shifted by 1 (Use Yesterday's Final Data) ---
raw_features = ["5_day_log_ret", "20_day_log_ret", "20d_volatility", "prev_ret"]
model_features = []

for feat in raw_features:
    new_col = f"lag_{feat}"
    # GroupBy shift ensures Stock A data doesn't leak to Stock B
    df[new_col] = df.groupby('act_symbol')[feat].shift(1)
    model_features.append(new_col)

# --- Gap Feature (Known at Open) ---
# Gap = Log(Open / Yesterday_Close)
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
#model_features.append('morning_gap')

# --- Hurst Feature (Pre-calculated in Universe Generation) ---
# 'hurst_1y' is calculated based on data PRIOR to the current month.
# It is a static "state" feature for the month, so it is known at the Open.
if 'hurst_1y' in df.columns:
    model_features.append('hurst_1y')
else:
    print("Warning: 'hurst_1y' column not found in data. Running without it.")

# 4. CLEANING
# Drop NaNs from rolling windows (first 20 days) and shifting
# This also drops any rows where targets are missing
data = df.dropna(subset=model_features + ['target_intraday'])

# 5. CHRONOLOGICAL SPLITTING
print("Splitting data chronologically...")
unique_dates = sorted(data['date'].unique())

# 70% Train, 15% Val, 15% Test
train_cutoff = unique_dates[int(len(unique_dates) * 0.7)]
val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

train_df = data[data['date'] < train_cutoff]
val_df = data[(data['date'] >= train_cutoff) & (data['date'] < val_cutoff)]
test_df = data[data['date'] >= val_cutoff]

X_train, y_train = train_df[model_features], train_df['target_intraday']
X_val, y_val = val_df[model_features], val_df['target_intraday']
X_test, y_test = test_df[model_features], test_df['target_intraday']

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 6. OPTUNA OPTIMIZATION
def objective(trial):
    param = {
        "objective": "regression",
        "metric": "rmse",
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
        "seed": 42 # Lock seed for stability
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    gbm = lgb.train(
        param, dtrain, valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=20), optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    )
    return np.sqrt(mean_squared_error(y_val, gbm.predict(X_val)))

print("Tuning hyperparameters (Optuna)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 7. FINAL TRAINING
print("Training final model...")
best_params = study.best_params
best_params["objective"] = "regression"
best_params["metric"] = "rmse"
best_params["seed"] = 42

model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 8. ASSESSMENT
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test.values))

print(f"\n--- RESULTS (Hurst Universe) ---")
print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {dir_acc:.2%}")

# 9. VISUALIZATION
test_df = test_df.copy()
test_df['pred'] = y_pred
test_df['strat_ret'] = np.where(test_df['pred'] > 0, test_df['target_intraday'], 0)

# Aggregate daily portfolio returns
daily_stats = test_df.groupby('date')[['strat_ret', 'target_intraday']].mean()

# Anomaly Check (Data Quality Check)
first_day = daily_stats.index[0]
first_day_drop = daily_stats.loc[first_day, 'target_intraday']
if first_day_drop < -0.02: 
    print(f"\n[WARNING] Massive drop detected on {first_day.date()}: {first_day_drop:.2%}")

# Cumulative Sum & Anchoring
equity_curve = daily_stats.cumsum()
start_date = equity_curve.index[0] - pd.Timedelta(days=1)
day_zero = pd.DataFrame({'strat_ret': [0.0], 'target_intraday': [0.0]}, index=[start_date])
equity_curve = pd.concat([day_zero, equity_curve]).sort_index()

# Get SPY data for benchmark (Dynamic Dates)
start_str = test_df['date'].min().strftime('%Y-%m-%d')
end_str = test_df['date'].max().strftime('%Y-%m-%d')
print(f"Fetching SPY benchmark from {start_str} to {end_str}...")

dr = DataReader()
dr.get_all_data("SPY", start_str, end_str)

# Calculate SPY cumulative returns
if "ohlcv" in dr.stock_data and not dr.stock_data["ohlcv"].empty:
    spy_df = dr.stock_data["ohlcv"].sort_values('date')
    spy_df['log_ret'] = np.log(spy_df['close'] / spy_df['close'].shift(1))
    spy_df = spy_df.fillna(0)
    spy_returns = spy_df.set_index('date')['log_ret'].cumsum()
else:
    print("Warning: Could not fetch SPY data for plot.")
    spy_returns = pd.Series()

plt.figure(figsize=(10,6))
plt.plot(equity_curve['strat_ret'], label='Intraday Strategy (Hurst Feature)', color='lime')
plt.plot(equity_curve['target_intraday'], label='Liquid Universe Avg', color='gray', alpha=0.5)

if not spy_returns.empty:
    # Realign SPY to the equity curve index for cleaner plotting
    aligned_spy = spy_returns.reindex(equity_curve.index, method='ffill').fillna(0)
    # Reset SPY to start at 0
    aligned_spy = aligned_spy - aligned_spy.iloc[0]
    plt.plot(aligned_spy, label='SPY Benchmark', c="magenta", alpha=0.8)

plt.title(f"Liquidity + Hurst Strategy Performance (Acc: {dir_acc:.2%})")
plt.axhline(0, color='white', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True, alpha=0.1)
plt.show()

print("\nFeature Importance:")
lgb.plot_importance(model, max_num_features=10)
plt.show()
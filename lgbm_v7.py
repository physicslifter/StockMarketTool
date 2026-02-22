import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

UNIVERSE_DATA = "Data/v5_training_universe.feather"

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading Universe data...")
try:
    df = pd.read_feather(UNIVERSE_DATA)
except FileNotFoundError:
    print("File not found. Generating dummy data for demonstration...")
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    symbols = [f'SYM_{i}' for i in range(50)]
    data = []
    for sym in symbols:
        tmp = pd.DataFrame({'date': dates, 'act_symbol': sym})
        base = 100 + np.random.randn(len(dates)).cumsum()
        tmp['open'] = base + np.random.randn(len(dates))
        tmp['close'] = base + np.random.randn(len(dates))
        tmp['high'] = base + abs(np.random.randn(len(dates))) + 1
        tmp['low'] = base - abs(np.random.randn(len(dates))) - 1
        tmp['volume'] = np.random.randint(1000, 100000, len(dates))
        data.append(tmp)
    df = pd.concat(data)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['act_symbol', 'date'])

# ==========================================
# 2. ROBUST FEATURE ENGINEERING
# ==========================================
print("Engineering features...")

# --- Helper Functions ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def garman_klass_vol(df):
    # More robust volatility measure using H/L/O/C
    # 0.5 * ln(High/Low)^2 - (2*ln(2)-1) * ln(Close/Open)^2
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    return np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)

# --- A. Base Calculations ---
df['prev_close'] = df.groupby('act_symbol')['close'].shift(1)
df['log_ret'] = np.log(df['close'] / df['prev_close'])

# --- B. Volatility (Garman-Klass) ---
# We calculate GK Vol based on TODAY'S candle, then SHIFT it later to use as a feature
df['gk_vol_raw'] = garman_klass_vol(df)
df['gk_vol_20'] = df.groupby('act_symbol')['gk_vol_raw'].transform(lambda x: x.rolling(20).mean())

# --- C. RSI (Momentum) ---
df['rsi_14'] = df.groupby('act_symbol')['close'].transform(lambda x: calculate_rsi(x, 14))

# --- D. Market Context (The "Tide") ---
# Calculate the average return of the whole universe per day
market_daily = df.groupby('date')['log_ret'].mean().rename('market_ret')
df = df.merge(market_daily, on='date', how='left')

# Relative Strength (Stock vs Market)
df['rel_strength_20'] = df.groupby('act_symbol')['log_ret'].transform(lambda x: x.rolling(20).sum()) - \
                        df.groupby('act_symbol')['market_ret'].transform(lambda x: x.rolling(20).sum())

# --- E. Time Embeddings ---
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# ==========================================
# 3. PREPARE MODEL FEATURES (LAGGING)
# ==========================================
# CRITICAL: All features used for prediction must be LAGGED (shifted) 
# except for the Open price used in the Gap calculation (Decision Point info)

model_features = []

def zscore(x):
    if x.std() == 0: return 0
    return (x - x.mean()) / (x.std() + 1e-8)

# 1. The Gap (Decision Point Info)
df['morning_gap'] = np.log(df['open'] / df['prev_close'])
# Normalize gap by Yesterday's volatility (winsorized to prevent extreme outliers)
df['lag_gk_vol'] = df.groupby('act_symbol')['gk_vol_20'].shift(1)
df['gap_sigma'] = df['morning_gap'] / (df['lag_gk_vol'] + 1e-8)
# Clip extreme gaps (e.g. bio-tech buyouts or crashes) to +/- 4 sigma
df['gap_sigma'] = df['gap_sigma'].clip(-4, 4)
model_features.append('gap_sigma')

# 2. Lagged Technicals
features_to_lag = ['rsi_14', 'rel_strength_20', 'gk_vol_20', 'log_ret']

for feat in features_to_lag:
    lag_name = f"lag_{feat}"
    df[lag_name] = df.groupby('act_symbol')[feat].shift(1)
    
    # Cross-Sectional Z-Score (Rank relative to peers today)
    z_name = f"{lag_name}_z"
    df[z_name] = df.groupby('date')[lag_name].transform(zscore)
    model_features.append(z_name)

# 3. Interaction: Momentum * Volatility
# Fixed: Using correct column name 'lag_gk_vol_20_z'
df['mom_vol_inter'] = df['lag_rel_strength_20_z'] * df['lag_gk_vol_20_z']
model_features.append('mom_vol_inter')

# 4. Context Metadata
model_features.append('day_of_week')

# --- TARGET CREATION ---
# Target: Buy Open, Sell Next Open
df['next_open'] = df.groupby('act_symbol')['open'].shift(-1)
df['target_ret'] = np.log(df['next_open'] / df['open'])

# Market Neutral Target (Alpha)
df['target_mkt_mean'] = df.groupby('date')['target_ret'].transform('mean')
df['target_alpha'] = df['target_ret'] - df['target_mkt_mean']

# Binary Target (Probability Alpha > 0)
df['target_binary'] = (df['target_alpha'] > 0).astype(int)

# Clean
data = df.dropna(subset=model_features + ['target_alpha'])

# ==========================================
# 4. SPLITTING
# ==========================================
unique_dates = sorted(data['date'].unique())
train_cutoff = unique_dates[int(len(unique_dates) * 0.7)]
val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

train_df = data[data['date'] < train_cutoff]
val_df = data[(data['date'] >= train_cutoff) & (data['date'] < val_cutoff)]
test_df = data[data['date'] >= val_cutoff]

X_train, y_train = train_df[model_features], train_df['target_binary']
X_val, y_val = val_df[model_features], val_df['target_binary']
X_test, y_test = test_df[model_features], test_df['target_binary']

print(f"Features Used: {model_features}")

# ==========================================
# 5. OPTIMIZATION & TRAINING
# ==========================================

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 30, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 1000),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "subsample": trial.suggest_float("subsample", 0.4, 0.9),
        "subsample_freq": 1
    }
    
    # Weight samples by the magnitude of the opportunity
    weights = np.log1p(np.abs(train_df['target_alpha']) * 1000)
    
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dval = lgb.Dataset(X_val, label=y_val)
    
    gbm = lgb.train(params, dtrain, valid_sets=[dval], 
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    
    preds = gbm.predict(X_val)
    return roc_auc_score(y_val, preds)

print("Tuning Hyperparameters...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

best_params = study.best_params
best_params.update({
    "objective": "binary", 
    "metric": "binary_logloss",
    "n_estimators": 1000,
    "random_state": 42
})

print("Training Final Model...")
train_weights = np.log1p(np.abs(train_df['target_alpha']) * 1000)
dtrain_final = lgb.Dataset(X_train, label=y_train, weight=train_weights)
dval_final = lgb.Dataset(X_val, label=y_val)

model = lgb.train(
    best_params, 
    dtrain_final, 
    valid_sets=[dval_final],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# ==========================================
# 6. EVALUATION
# ==========================================
print("\n--- RESULTS ---")
preds = model.predict(X_test)

# 1. Global Directional Accuracy
# (Did we predict the correct sign of Alpha?)
dir_preds = (preds > 0.5).astype(int)
acc = accuracy_score(y_test, dir_preds)
auc = roc_auc_score(y_test, preds)

# 2. Trade Signal Accuracy
# (When we actually decide to trade (prob > 0.52), how accurate are we?)
signal_threshold = 0.52
trade_mask = preds > signal_threshold
if trade_mask.sum() > 0:
    trade_acc = accuracy_score(y_test[trade_mask], dir_preds[trade_mask])
else:
    trade_acc = 0.0

print(f"Directional Accuracy (Global): {acc:.2%}")
print(f"Trade Signal Accuracy (> {signal_threshold}): {trade_acc:.2%}")
print(f"Test AUC: {auc:.4f}")

# Backtest Simulation
test_df['prob'] = preds
test_df['signal'] = np.where(test_df['prob'] > signal_threshold, 1, 0)
test_df['strategy_ret'] = test_df['signal'] * test_df['target_ret']

cum_ret = (1 + test_df.groupby('date')['strategy_ret'].mean()).cumprod()

print(f"Strategy Total Return: {(cum_ret.iloc[-1]-1):.2%}")

# Visuals
plt.figure(figsize=(10,5))
plt.plot(cum_ret, label='Strategy (LGBM v7)')
plt.title(f"Backtest Performance (Dir Acc: {acc:.2%})")
plt.legend()
plt.show()

lgb.plot_importance(model, importance_type='gain', max_num_features=10, title='Feature Importance')
plt.tight_layout()
plt.show()
'''
Global LGBM Model for Trade-At-Open Strategy (Correction)
'''
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
df = pd.read_feather("Data/all_training_data.feather")
df['date'] = pd.to_datetime(df['date'])

# 2. FEATURE ENGINEERING
print("Engineering features...")
df = df.sort_values(['act_symbol', 'date'])

# Target: Intraday Return (Open -> Close)
df['target_intraday'] = np.log(df['close'] / df['open'])

# Lagged Features (Yesterday's Data)
raw_features = ["log_ret", "5_day_log_ret", "20_day_log_ret", "20d_volatility", "prev_ret"]
model_features = []

for feat in raw_features:
    new_col = f"lag_{feat}"
    df[new_col] = df.groupby('act_symbol')[feat].shift(1)
    model_features.append(new_col)

# Gap Feature
df['prev_close_raw'] = df.groupby('act_symbol')['close'].shift(1)
df['morning_gap'] = np.log(df['open'] / df['prev_close_raw'])
model_features.append('morning_gap')

# 3. CLEANING
data = df.dropna(subset=model_features + ['target_intraday'])

# 4. SPLITTING
unique_dates = sorted(data['date'].unique())
train_cutoff = unique_dates[int(len(unique_dates) * 0.7)]
val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

train_df = data[data['date'] < train_cutoff]
val_df = data[(data['date'] >= train_cutoff) & (data['date'] < val_cutoff)]
test_df = data[data['date'] >= val_cutoff]

X_train, y_train = train_df[model_features], train_df['target_intraday']
X_val, y_val = val_df[model_features], val_df['target_intraday']
X_test, y_test = test_df[model_features], test_df['target_intraday']

# 5. OPTIMIZATION
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
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    gbm = lgb.train(
        param, dtrain, valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=20), optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    )
    return np.sqrt(mean_squared_error(y_val, gbm.predict(X_val)))

print("Tuning hyperparameters...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20) # Keeping trials low for speed

# 6. FINAL TRAINING
print("Training final model...")
best_params = study.best_params
best_params["objective"] = "regression"
best_params["metric"] = "rmse"

model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 7. RESULTS
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
dir_acc = np.mean(np.sign(y_pred) == np.sign(y_test.values))

print(f"\n--- RESULTS ---")
print(f"RMSE: {rmse:.6f}")
print(f"Directional Accuracy: {dir_acc:.2%}")

# 8. VISUALIZATION (FIXED)
test_df = test_df.copy()
test_df['pred'] = y_pred
test_df['strat_ret'] = np.where(test_df['pred'] > 0, test_df['target_intraday'], 0)

# Aggregate daily portfolio returns
daily_stats = test_df.groupby('date')[['strat_ret', 'target_intraday']].mean()

# Investigate the "Massive Drop" Anomaly
first_day = daily_stats.index[0]
first_day_drop = daily_stats.loc[first_day, 'target_intraday']
if first_day_drop < -0.02: # If average drop is worse than -2%
    print(f"\n[WARNING] Massive drop detected on {first_day.date()}: {first_day_drop:.2%}")
    print("Top 5 worst performers that day (Likely Bad Data/Splits):")
    bad_day_data = test_df[test_df['date'] == first_day].sort_values('target_intraday')
    print(bad_day_data[['act_symbol', 'open', 'close', 'target_intraday']].head(5))

# Calculate Cumulative Sum
equity_curve = daily_stats.cumsum()

# Create a "Day Zero" row to anchor the plot at 0.0
start_date = equity_curve.index[0] - pd.Timedelta(days=1)
day_zero = pd.DataFrame({'strat_ret': [0.0], 'target_intraday': [0.0]}, index=[start_date])

# Concatenate and sort
equity_curve = pd.concat([day_zero, equity_curve]).sort_index()

plt.figure(figsize=(10,6))
plt.plot(equity_curve['strat_ret'], label='Intraday Strategy', color='lime')
plt.plot(equity_curve['target_intraday'], label='Market Average', color='gray', alpha=0.5)
plt.title(f"Strategy Performance (Acc: {dir_acc:.2%})")
plt.axhline(0, color='white', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True, alpha=0.1)
plt.show()
'''
Script for doing a simple LGBM model on a single stock
'''
import lightgbm as lgb
from DoltReader import DataReader
from pdb import set_trace as st
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#get data
dr = DataReader()
dr.get_all_data("PFE", "2012-01-01", "2025-12-31")

df = dr.stock_data["ohlcv"]

#get feature data
data_keys = ["prev_ret", 
                #"days_since_release", 
                "volume",
                #"ROIC",
                #"operating_margin",
                #"FCF_margin",
                #"FCF_yield",
                #"EV/EBITDA",
                #"rev_CAGR",
                #"eps_CAGR",
                "log_ret",
                "5_day_log_ret",
                "20_day_log_ret",
                "20d_volatility"
                ]
keys_to_drop = [key for key in df.keys() if key not in data_keys]
data = df.drop(keys_to_drop, axis = 1)
print(len(data))
data = data.dropna()
print(len(data))

feature_data = data.drop(["log_ret"], axis = 1)

#get target data
keys_to_drop = [key for key in data.keys() if key != "log_ret"]
target_data = data.drop(keys_to_drop, axis = 1)


#partition the data
X, X_test, y, y_test = train_test_split(feature_data, target_data, test_size = 0.4, random_state = 33, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.5, random_state = 33)


lgb_train = lgb.Dataset(X_train, label = y_train, free_raw_data = False)
lgb_test = lgb.Dataset(X_test, label = y_test, free_raw_data = False)
lgb_validation = lgb.Dataset(X_val, label = y_val)
for dataset in [lgb_train, lgb_test]:
    dataset.construct()

"""
#plain lgbm strategy
params = {
    "learning_rate": 0.05,
    "objective": "regression",
    "metric": "l2",
    
    # 1. Allow the tree to learn from fewer samples (outliers)
    "min_data_in_leaf": 5,  # Default is 20. Lowering this allows capturing rare events.
    
    # 2. Reduce regularization so it can fit "noise" (which might be signal)
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    
    # 3. Allow deeper trees to capture complex interactions
    "num_leaves": 64,       # Default is 31
}

model = lgb.train(params=params,
                  train_set = lgb_train,
                  valid_sets = [lgb_validation],
                  callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )
"""

#optuna strategy
# 1. Define an objective function to be maximized.
def objective(trial):
    # -- A. Define the Hyperparameter Search Space --
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        
        # Tree Structure (The most important part)
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        
        # Regularization (To prevent overfitting noise)
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        
        # Sampling (Speed & Generalization)
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        
        # Learning
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_pre_filter": False
    }

    # -- B. Train with Pruning --
    # PruningCallback stops unpromising trials early to save time
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    
    gbm = lgb.train(params=param,
                  train_set = lgb_train,
                  valid_sets = [lgb_validation],
                  callbacks=[lgb.early_stopping(stopping_rounds=50), pruning_callback]
                )

    # -- C. Predict and Return Score --
    preds = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100) 
trial = study.best_trial
model = lgb.train(params=trial.params,
                  train_set = lgb_train,
                  valid_sets = [lgb_validation],
                  callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )
#========
#Assessment from Gemini

# 1. Generate predictions on the test set
# Note: model.predict takes the DataFrame X_test directly
y_pred = model.predict(X_test)

# 2. Calculate standard Regression Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# 3. (Optional) Directional Accuracy
# For stocks, knowing if it goes UP or DOWN is often more important than the exact value.
# We check if the sign of the prediction matches the sign of the actual return.
correct_direction = np.sign(y_pred) == np.sign(y_test.values.flatten())
accuracy = np.mean(correct_direction)
print(f"Directional Accuracy: {accuracy:.2%}")

# 4. Visual Comparison
plt.figure(figsize=(12, 6))
# Plotting a subset (e.g., first 100 days) to make it readable
plt.plot(y_test.index[:100], y_test.values[:100], label='Actual Log Return', alpha=0.7)
plt.plot(y_test.index[:100], y_pred[:100], label='Predicted Log Return', alpha=0.7)
plt.title("Actual vs Predicted Returns (First 100 Test Points)")
plt.legend()
plt.show()

# 5. Feature Importance (To see what drove the decisions)
lgb.plot_importance(model, max_num_features=10)
plt.show()
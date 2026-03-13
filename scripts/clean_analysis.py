'''
Pat LaChapelle
Feb 3, 2026

Simplified script that can hold and test out different strategies

Universe
    - allows user to define and pick universe dynamically

Model
    - allows for dynamic model creation/selection

PortfolioStrategy
    - incorporates the model
'''

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
import warnings
from FeatureEngine import *
from pdb import set_trace as st
import os
from scipy.stats import spearmanr
import re


def calculate_metrics_single(ticker, prices):
    """
    Calculates metrics for AdvancedStatsFilter.
    Input: Array of daily close prices (approx 1 year).
    Output: Tuple of metrics.
    """
    history_len = len(prices)
    
    # Need minimal data to calculate anything meaningful
    if history_len < 30: 
        return (ticker, np.nan, np.nan, np.nan, np.nan, False, history_len)
    
    current_price = prices[-1]
    
    # 1. Annual Return (Momentum/Crash check)
    try:
        annual_ret = (current_price / prices[0]) - 1
    except:
        annual_ret = np.nan
    
    # 2. Annualized Volatility
    try:
        log_rets = np.diff(np.log(prices))
        volatility = np.std(log_rets) * np.sqrt(252)
    except:
        volatility = np.nan

    # 3. Hurst Exponent
    try:
        log_prices = np.log(prices)
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        tau = [t if t > 0 else 1e-8 for t in tau]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except:
        hurst = np.nan

    # 4. Trend (200 Day SMA)
    # Check if current price > average of last 200 days
    is_uptrend = False
    if history_len >= 200:
        sma_200 = np.mean(prices[-200:])
        is_uptrend = current_price > sma_200
    else:
        # If not enough history, strictly speaking it's not in a long-term uptrend
        is_uptrend = False 
        
    return (ticker, hurst, annual_ret, volatility, current_price, is_uptrend, history_len)

#ABSTRACT BASE CLASS
class Filter:
    def __init__(self, name):
        self.name = name

    def apply(self, df: pd.DataFrame, target_date: pd.Timestamp, current_tickers: list):
        """
        Args:
            df: The MASTER dataframe (contains all history)
            target_date: The specific date we are generating the universe for
            current_tickers: The list of tickers surviving the previous filter
                             (If None, this is the first filter)
        Returns:
            List of valid tickers
        """
        raise NotImplementedError

class TopLiquidityFilter(Filter):
    def __init__(self, N: int):
        super().__init__("liquidity")
        self.N = N

    def apply(self, df, target_date):
        # 1. Define Calculation Window (Previous Month)
        start_date = target_date - relativedelta(months = 1)
        
        # 2. Slice for Calculation ONLY (don't overwrite 'df' yet)
        # Note: 'df' is already reduced from previous filters, so this is fast.
        calc_mask = (df['date'] >= start_date) & (df['date'] < target_date)
        period_df = df.loc[calc_mask, ['act_symbol', 'close', 'volume']]
        
        if period_df.empty: return df.iloc[0:0] # Return empty DF

        # 3. Calculate Rank
        dollar_vol = period_df['close'].values * period_df['volume'].values
        temp = pd.DataFrame({'s': period_df['act_symbol'].values, 'dv': dollar_vol})
        avg_dv = temp.groupby('s')['dv'].mean()
        
        # 4. Identify Winners
        winners = avg_dv.nlargest(self.N).index.tolist()
        
        # 5. Filter the INPUT dataframe to keep full history of winners
        # This prepares the data for the next filter
        return df[df['act_symbol'].isin(winners)].copy()

class PriceFilter(Filter):
    def __init__(self, min_price: float = 5.0, method = "close_price"):
        super().__init__("price_filter")
        valid_methods = ["close_price", "avg_price_over_last_year"]
        if method not in valid_methods:
            raise Exception(f"{method} is invalid. Must be one of {valid_methods}")
        self.min_price = min_price
        self.method = method

    def apply(self, df, target_date):
        # Window: Previous Month
        if self.method == "avg_price_over_last_year":
            start_date = target_date - relativedelta(years = 1)

            # Slice for calculation
            calc_mask = (df['date'] >= start_date) & (df['date'] < target_date)
            period_df = df.loc[calc_mask, ['act_symbol', 'close']]
        
            # Calculate
            avg_prices = period_df.groupby('act_symbol')['close'].mean()
        
            # Identify Winners
            winners = avg_prices[avg_prices >= self.min_price].index.tolist()
        
        else:
            #filter_df = df[df.date == df.date.max()]
            #winners = filter_df[filter_df.close > self.min_price]
            #winners = winners.act_symbol.tolist()
            last_prices = df.sort_values('date').groupby('act_symbol')['close'].last()
            winners = last_prices[last_prices >= self.min_price].index.tolist()

        # Return reduced dataframe
        return df[df['act_symbol'].isin(winners)].copy()


class AdvancedStatsFilter(Filter):
    def __init__(self, 
                 min_history_days: int = None,
                 max_crash: float = None, 
                 require_uptrend: bool = None,
                 volatility_n: int = None,
                 ):
        """
        Args:
            min_history_days: Exclude stocks with history < N days (e.g. 252 for 1 yr).
            max_crash: Exclude stocks that dropped > X% (input as negative decimal, e.g. -0.60).
            require_uptrend: If True, exclude stocks below their 200-day SMA.
            volatility_n: If set, keep only the bottom N stocks by volatility.
        """
        super().__init__("advanced_stats")
        self.min_history = min_history_days
        self.max_crash = max_crash
        self.uptrend = require_uptrend
        self.vol_n = volatility_n

    def apply(self, df, target_date):
        # 1. Window: Previous Year
        # We need at least 1 year for these stats to be meaningful
        
        start_date = target_date - relativedelta(years=1)
        
        # 2. Slice for Calculation
        calc_mask = (df['date'] >= start_date) & (df['date'] < target_date)
        hist_df = df.loc[calc_mask, ['act_symbol', 'close']]
        
        # 3. Parallel Calculation
        grouped = hist_df.groupby('act_symbol')['close']
        tasks = [(name, group.values) for name, group in grouped]
        
        if not tasks:
            return df.iloc[0:0]

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(calculate_metrics_single)(t, p) for t, p in tasks
        )
        
        # 4. Filter Logic
        candidates = []
        
        # Unpack the 7 values returned by the helper
        for ticker, hurst, ann_ret, vol, price, is_uptrend, history_len in results:
            
            if np.isnan(hurst) or np.isnan(vol):
                continue

            # CHECK 1: Min History (Avoid IPOs)
            if self.min_history is not None:
                if history_len < self.min_history:
                    continue
            
            # CHECK 2: Max Crash (Avoid falling knives)
            if self.max_crash is not None:
                # If ann_ret is NaN, we usually skip it too
                if np.isnan(ann_ret) or ann_ret < self.max_crash:
                    continue
            
            # CHECK 3: Trend (Avoid value traps)
            if self.uptrend is True:
                if not is_uptrend:
                    continue

            # CHECK 4: Volatility Validity
            # If we plan to sort by Vol, it cannot be NaN
            if self.vol_n is not None and np.isnan(vol):
                continue

            candidates.append({'s': ticker, 'vol': vol})
            
        # 5. Sort and Pick Winners (if volatility ranking is requested)
        df_res = pd.DataFrame(candidates)
        
        if df_res.empty: 
            return df.iloc[0:0]
        
        winners = []
        if self.vol_n is not None:
            # Sort by Volatility (Low to High)
            df_res = df_res.sort_values('vol', ascending=True)
            winners = df_res.head(self.vol_n)['s'].tolist()
        else:
            # If no ranking requested, return all survivors
            winners = df_res['s'].tolist()
        
        # 6. Return Reduced DF
        return df[df['act_symbol'].isin(winners)].copy()

class Universe:
    def __init__(self, master_df: pd.DataFrame, filter_etfs:bool = True):
        # We store the master copy. We never modify this directly.
        self.master_df = master_df
        self.filters = []
        if filter_etfs == True:
            etf_data = pd.read_feather("../Data/ETFs.feather")
            etf_list = set(etf_data['act_symbol'].unique())
            self.master_df = self.master_df[~self.master_df['act_symbol'].isin(etf_list)]

    def add_filter(self, filter: Filter):
        if not isinstance(filter, Filter):
            raise Exception("Invalid filter type")
        self.filters.append(filter)

    def add_filters(self, filters):
        for filter in filters:
            if not isinstance(filter, Filter):
                raise Exception("filters must be valid subclass of Filter")
        self.filters += filters

    def get_universe_for_month(self, target_date):
        target_date = pd.to_datetime(target_date)
        
        # 1. Start with a working copy of the master data
        # We must copy so we don't break the master for future runs
        working_df = self.master_df[(self.master_df.date >= target_date - pd.DateOffset(years = 1)) & (self.master_df.date < target_date)]
        print(f"Generating universe for {target_date.date()}...")
        print(f"  -> Starting Count: {working_df['act_symbol'].nunique()}")
        # 2. Pipeline Loop
        #isolate data for the year and month we'rd on
        for f in self.filters:
            if working_df.empty:
                print("  -> Universe Died (0 stocks). Stopping.")
                break
                
            # Apply Filter: Old DF -> New Smaller DF
            working_df = f.apply(working_df, target_date)
            
            count = working_df['act_symbol'].nunique()
            print(f"  -> {f.name}: {count} stocks remaining")

        survivors = working_df['act_symbol'].unique().tolist()
        next_month = target_date + pd.DateOffset(months=1)
        future_data = self.master_df[
            (self.master_df['date'] >= target_date) & 
            (self.master_df['date'] < next_month) & 
            (self.master_df['act_symbol'].isin(survivors))
        ].copy()

        # 3. Return final list of survivors
        #return working_df['act_symbol'].unique().tolist()
        print(target_date, working_df.date.min(), working_df.date.max())
        return future_data
    
    def get_all_universe_data(self, 
                              save_name:str = None,
                              dates:list = None):
        if type(dates) != type(None):
            for date in dates:
                if type(date) != pd.Timestamp:
                    raise Exception("Dates must be of type pandas.Timestamp")
                if dates[1] < dates[0]:
                    raise Exception("End date must be before start date")
            data = self.master_df[(self.master_df.date >= dates[0]) & (self.master_df.date <= dates[1])]
        else:
            data = self.master_df
        start_year = pd.Timestamp(data.date.values[0]).year
        end_year = pd.Timestamp(data.date.values[-1]).year
        start_month = pd.Timestamp(data.date.values[0]).month
        end_month = pd.Timestamp(data.date.values[-1]).month
        print(data)
        print(start_year, end_year, start_month, end_month)
        years = range(start_year, end_year + 1)
        months = range(1, 13)

        all_data_as_list = []
        for year in years:
            for month in months:
                if year == end_year and month > end_month:
                    break
                elif year == start_year and month < start_month:
                    pass
                else:
                    month_data = self.get_universe_for_month(target_date = pd.Timestamp(year = year, month = month, day = 1))
                    all_data_as_list.append(month_data)
                    print(f"{year}-{month}-1 DATA RETRIEVED")
        final_df = pd.concat(all_data_as_list, ignore_index = True)
        final_df = final_df.sort_values(by=['act_symbol', 'date'], ascending=True)
        final_df = final_df.reset_index(drop=True)
        if type(save_name) != type(None):
            extension = save_name.split(".")[-1]
            if extension == "csv":
                final_df.to_csv(save_name)
            elif extension == "feather":
                final_df.to_feather(save_name)
            else:
                raise Exception("Extension invalid. Write code to save for this file type")
        self.universe_data = final_df
        return final_df

class Model:
    def __init__(self, universe_data):
        self.data = universe_data #all data to train the model on
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.has_features = False #no features upon initialization
        self.has_target = False #no target upon initialization
        self.data_split = False #default whether data has been split to false
        self.params_tuned = False
        self.variables_generated = False

    def add_target(self, target, target_type:str = "regression"):
        valid_target_types = ["classification", "regression"]
        if target_type not in valid_target_types:
            err_msg = "Invalid target_type. Must be one of: "
            for c, i in enumerate(valid_target_types):
                err_msg += i
                if c < len(valid_target_types) - 1:
                    err_msg += ", "
        self.target = target
        self.target_type = target_type
        self.has_target = True

    def add_features(self, features):
        '''
        Adds features to the dataset
        '''
        self.features = features
        self.has_features = True

    def split_data(self,
                   cutoffs:list = None
                   ):
        self.generate_targets_and_features()
        if self.has_features == False:
            raise Exception("No features added")
        if self.has_target == False:
            raise Exception("No target added")
        if len(cutoffs) != 3:
            raise Exception("Must have 3 cutoffs; 1 each for training, validation & test data")
        if cutoffs[0] >= cutoffs[1]:
            raise Exception(f"Validation cutoff {cutoffs[1]} must be larger than training cutoff {cutoffs[0]}")
        elif cutoffs[1] >= cutoffs[2]:
            raise Exception(f"Test cutoff {cutoffs[2]} must be larger than validation cutoff {cutoffs[1]}")
        unique_dates = sorted(self.data['date'].unique())
        print(len(unique_dates), int(len(unique_dates) * cutoffs[0]), int(len(unique_dates) * cutoffs[1]), int(len(unique_dates) * cutoffs[2]))
        #st()
        train_cutoff = unique_dates[int(len(unique_dates) * cutoffs[0])]
        val_cutoff = unique_dates[int(len(unique_dates) * cutoffs[1])]
        test_cutoff = unique_dates[int(len(unique_dates) * cutoffs[2]) - 1]
        self.train_df = self.data[self.data['date'] < train_cutoff]
        self.val_df = self.data[(self.data['date'] >= train_cutoff) & (self.data['date'] < val_cutoff)]
        self.test_df = self.data[(self.data['date'] >= val_cutoff) & (self.data['date'] < test_cutoff)]

        self.features = [key for key in self.data.keys() if "F" in key.split("_")]
        self.targets = [key for key in self.data.keys() if "T" in key.split("_")]
        self.target_key = self.targets[0]
        if len(self.targets) != 1:
            st()
            raise Exception(f"Only one target allowed. Identified targets: {self.targets}")
        
        self.X_train, self.y_train = self.train_df[self.features], self.train_df[self.targets[0]]
        self.X_val, self.y_val = self.val_df[self.features], self.val_df[self.targets[0]]
        self.X_test, self.y_test_bin = self.test_df[self.features], self.test_df[self.targets[0]]

        self.data_split = True

    def tune_params(self):
        if self.data_split == False:
            raise Exception("Data must be split before tuning params")
        print("TUNING PARAMS...")
        #coerce 64-bit to 32-bit to speed up
        float_cols = self.data.select_dtypes(include=['float64']).columns
        self.data[float_cols] = self.data[float_cols].astype('float32')
        #objective function for tuning params
        if self.target_type == "classification":
            direction = "maximize"
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
                train_weights = np.log1p(np.abs(self.train_df[self.target_key]) * 100)

                dtrain = lgb.Dataset(self.X_train, label=self.y_train, weight=train_weights)
                dval = lgb.Dataset(self.X_val, label=self.y_val, reference=dtrain)

                gbm = lgb.train(
                    param, dtrain, valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20), 
                        optuna.integration.LightGBMPruningCallback(trial, "auc") 
                    ]
                )

                preds = gbm.predict(self.X_val)
                return roc_auc_score(self.y_val, preds)
        elif self.target_type == "regression":
            direction = "minimize"
            def objective(trial):
                param = {
                    #RMSE
                    '''
                    "objective": "regression",  # <-- CHANGED: Train trees using Huber
                    "metric": "rmse",     # <-- CHANGED: Evaluate using Huber
                    '''
                    #Huber
                    "objective": "huber",
                    "metric": "huber",
                    "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True), 

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

                dtrain = lgb.Dataset(self.X_train, label=self.y_train)
                dval = lgb.Dataset(self.X_val, label=self.y_val, reference=dtrain)

                #for huber
                
                gbm = lgb.train(
                    param, dtrain, valid_sets=[dval],
                    # <-- CHANGED: Pruning callback tracks "huber"
                    callbacks=[lgb.early_stopping(stopping_rounds=20), optuna.integration.LightGBMPruningCallback(trial, "huber")]
                )
                return_val = mean_absolute_error(self.y_val, gbm.predict(self.X_val))
                

                #for mae
                '''
                gbm = lgb.train(
                    param, dtrain, valid_sets=[dval],
                    # <-- CHANGED: Pruning callback now tracks "l1" (MAE)
                    callbacks=[lgb.early_stopping(stopping_rounds=20), optuna.integration.LightGBMPruningCallback(trial, "l1")]
                )
                return_val = mean_absolute_error(self.y_val, gbm.predict(self.X_val))
                '''

                #for rmse
                '''
                gbm = lgb.train(
                    param, dtrain, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(stopping_rounds=20), optuna.integration.LightGBMPruningCallback(trial, "rmse")]
                )
                return_val =  np.sqrt(mean_squared_error(self.y_val, gbm.predict(self.X_val)))
                '''

                return return_val
                

        study = optuna.create_study(direction=direction) 
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_params
        self.params_tuned = True

    def generate_targets_and_features(self):
        if self.has_features == False:
            raise Exception("Model does not have features")
        if self.has_target == False:
            raise Exception("Model does not have a target")
        self.variables_generated = True
        feature_engine = FeatureEngine(feature_requests = self.features + [self.target])
        self.data = feature_engine.compute(self.data)
        #drop rows where targets and features are none
        print(self.data.keys(), len(self.data))
        feature_keys = [key for key in self.data.keys() if "F" in key.split("_")]
        target_key = [key for key in self.data.keys() if "T" in key.split("_")][0]
        self.data = self.data.dropna(subset = feature_keys + [target_key]) #drop rows with nan for features or targets

    def train_model(self, save_name:str=None):
        if self.params_tuned == False:
            self.tune_params()
        self.model = lgb.train(
                self.best_params,
                lgb.Dataset(self.X_train, label=self.y_train),
                valid_sets=[lgb.Dataset(self.X_val, label=self.y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
        self.test_model()
        if type(save_name) != type(None):
            self.model.save_model(f"Data/{save_name}.txt")

    def test_model(self):
        """Tests the model after generating
        """
        predictions = self.model.predict(self.X_test)
        
        if self.target_type == "classification":
            self.test_df["prob_up"] = predictions
            test_preds_class = (predictions > 0.5).astype(int)
            print("MODEL ACCURACY\n========")
            acc = accuracy_score(self.y_test_bin, test_preds_class)
            auc = roc_auc_score(self.y_test_bin, predictions)
            print(f"accuracy: {acc}")
            print(f"auc - roc: {auc}")
            
        elif self.target_type == "regression":
            self.test_df["pred_return"] = predictions
            print("MODEL ASSESSMENT\n========")
            rmse = np.sqrt(mean_squared_error(self.y_test_bin, predictions))
            dir_acc = ((self.y_test_bin > 0) == (predictions > 0)).mean()
            
            #old method for getting metrics
            '''def get_daily_ic(group):
                # Need at least 2 stocks to calculate a rank correlation
                if len(group) > 1:
                    # spearmanr returns (correlation, p-value); we just want correlation [0]
                    return spearmanr(group[self.targets[0]], group["pred_return"])[0] 
                return np.nan
            daily_ic_series = self.test_df.groupby('date').apply(get_daily_ic)
            mean_rank_ic = daily_ic_series.mean()
            std_rank_ic = daily_ic_series.std()
        
            # 2. Prevent division by zero and calculate IC-IR
            if std_rank_ic != 0 and not np.isnan(std_rank_ic):
                ic_ir = mean_rank_ic / std_rank_ic           # Calculate Daily IC-IR
                annualized_ic_ir = ic_ir * np.sqrt(252)      # Annualize it (assuming daily data)
                n_days = len(daily_ic_series.dropna()) 
                ic_t_stat = mean_rank_ic / (std_rank_ic / np.sqrt(n_days))
            else:
                ic_ir = np.nan
                annualized_ic_ir = np.nan
            print(f"rmse: {rmse}")
            print(f"directional accuracy: {dir_acc}")
            print(f"Mean Daily Rank IC: {mean_rank_ic:.4f}")
            print(f"IC Std Dev: {std_rank_ic:.4f}")                 # NEW
            print(f"Daily IC-IR: {ic_ir:.4f}")                      # NEW
            print(f"Annualized IC-IR: {annualized_ic_ir:.4f}")
            print(f"IC T-Statistic: {ic_t_stat:.4f}")'''

            mean_ic, std_ic, ic_ir, ann_ic_ir, t_stat = self._calculate_robust_ic_metrics()
            
            print(f"rmse: {rmse:.4f}")
            print(f"directional accuracy: {dir_acc:.4f}")
            print(f"Robust Mean Rank IC: {mean_ic:.4f}")
            print(f"Robust IC Std Dev: {std_ic:.4f}")                 
            print(f"Robust IC-IR: {ic_ir:.4f}")                      
            print(f"Robust Annualized IC-IR: {ann_ic_ir:.4f}")
            print(f"Robust IC T-Statistic: {t_stat:.4f}")

        lgb.plot_importance(self.model, importance_type = 'gain', figsize = (10, 6), title = "Feature Importance")
        plt.tight_layout()
        plt.show()

    def _calculate_robust_ic_metrics(self):
        """
        Calculates Information Coefficient metrics adjusting for overlapping returns.
        Uses Effective Sample Size to prevent overlapping forward returns 
        from artificially inflating the t-statistic.
        """
        # 1. Extract N (overlap period) specifically from FeatureEngine naming convention
        # This explicitly looks for the parameter right after FWD_LOG_RET or TARGET_SHARPE
        match = re.search(r'(?:FWD_LOG_RET|TARGET_SHARPE)_(\d+)', self.target_key)
        
        if match:
            N = int(match.group(1)) # Extracts the '5' from 'FWD_LOG_RET_5_...'
        else:
            # Fallback in case you manually assign an alias like "Target_5d_T"
            match_alias = re.search(r'(\d+)d', self.target_key)
            N = int(match_alias.group(1)) if match_alias else 1

        # 2. Helper to calculate daily spearman rank correlation
        def get_daily_ic(group):
            if len(group) > 1:
                return spearmanr(group[self.targets[0]], group["pred_return"])[0] 
            return np.nan

        # 3. Calculate daily rank ICs for ALL days
        daily_ics = self.test_df.groupby('date').apply(get_daily_ic).dropna()
        
        if len(daily_ics) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # 4. Calculate core aggregate metrics
        mean_ic = daily_ics.mean()
        std_ic = daily_ics.std()
        n_days = len(daily_ics)
        
        # 5. Apply Overlap Corrections
        if std_ic != 0 and not np.isnan(std_ic):
            ic_ir = mean_ic / std_ic
            
            # Annualize based on non-overlapping frequency
            ann_ic_ir = ic_ir * np.sqrt(252 / N) 
            
            # Robust T-Statistic uses EFFECTIVE Sample Size (n_days / N)
            # This perfectly adjusts the standard error for an MA(N-1) autocorrelation process
            effective_n_days = n_days / N
            t_stat = mean_ic / (std_ic / np.sqrt(effective_n_days))
            
        else:
            ic_ir, ann_ic_ir, t_stat = np.nan, np.nan, np.nan
            
        return mean_ic, std_ic, ic_ir, ann_ic_ir, t_stat
    
    def evaluate_quantile_spread(self, quantiles=10, plot=True):
        """
        Evaluates the model by dividing daily predictions into cross-sectional quantiles.
        Calculates the actual target spread between the top (long) and bottom (short) quantiles.
        """
        import re
        
        if "pred_return" not in self.test_df.columns:
            raise Exception("Model has not generated predictions yet. Run test_model() first.")
            
        df = self.test_df.copy()
        target_col = self.targets[0]
        
        print(f"\nQUANTILE SPREAD ANALYSIS (Top {100/quantiles:.1f}% vs Bottom {100/quantiles:.1f}%)")
        print("========================")
        
        # 1. Assign cross-sectional quantiles daily (1 = Worst, 10 = Best)
        # We use rank(method='first') to ensure qcut doesn't fail on duplicate predictions
        df['quantile'] = df.groupby('date')['pred_return'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=quantiles, labels=False) + 1
        )
        
        # 2. Calculate the mean actual target value for each quantile per day
        # Shape: (Dates as index, Quantiles 1-10 as columns)
        daily_quantile_returns = df.groupby(['date', 'quantile'])[target_col].mean().unstack()
        
        # Drop days where we couldn't form a full set of quantiles
        daily_quantile_returns = daily_quantile_returns.dropna()
        
        if daily_quantile_returns.empty:
            print("Not enough data to form quantiles.")
            return
            
        # 3. Calculate Long-Short Spread (Top Quantile - Bottom Quantile)
        daily_spread = daily_quantile_returns[quantiles] - daily_quantile_returns[1]
        
        # 4. Summary Metrics
        mean_spread = daily_spread.mean()
        win_rate = (daily_spread > 0).mean()
        
        # Calculate Effective Sharpe (adjusting for N-day overlap)
        match = re.search(r'(?:FWD_LOG_RET|TARGET_SHARPE)_(\d+)', self.target_key)
        N = int(match.group(1)) if match else 1
        
        std_spread = daily_spread.std()
        if std_spread != 0 and not np.isnan(std_spread):
            # Annualize and adjust standard error for overlapping paths
            sharpe = (mean_spread / std_spread) * np.sqrt(252 / N)
        else:
            sharpe = np.nan
            
        print(f"Mean Daily Target Spread : {mean_spread:.4f}")
        print(f"Spread Win Rate          : {win_rate:.2%}")
        print(f"Estimated Annual Sharpe  : {sharpe:.4f}")
        
        # 5. Plotting
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Mean Target by Quantile (Check for monotonicity)
            overall_q_mean = daily_quantile_returns.mean()
            axes[0].bar(overall_q_mean.index, overall_q_mean.values, color='skyblue', edgecolor='black')
            axes[0].set_title("Average Target Value by Prediction Quantile")
            axes[0].set_xlabel("Quantile (1 = Worst Prediction, 10 = Best Prediction)")
            axes[0].set_ylabel(f"Mean Actual {target_col}")
            axes[0].set_xticks(range(1, quantiles + 1))
            axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
            
            # Plot 2: Cumulative Long-Short Spread (Alpha Generation)
            cumulative_spread = daily_spread.cumsum()
            axes[1].plot(cumulative_spread.index, cumulative_spread.values, color='purple', linewidth=2)
            axes[1].set_title("Cumulative Long-Short Spread (Alpha)")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Cumulative Target Spread")
            axes[1].axhline(0, color='black', linewidth=1)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        return daily_quantile_returns, daily_spread

class PortfolioStrat:
    def __init__(self):
        pass


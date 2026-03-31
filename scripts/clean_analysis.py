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
from FundamentalEngine import *
from FundamentalEngine import *
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
import pickle


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
    
class SecurityTypeFilter(Filter):
    def __init__(self, valid_tickers_path: str = "../Data/valid_equity_tickers.feather"):
        """
        Filters to valid equities (common stock, ADR, REIT) using a pre-built
        ticker list generated by name_sort_test.py.
        
        Run name_sort_test.py first to generate the valid tickers file.

        Removes tickers identified from DataRetrieval/get_valid_tickers.py
        """
        super().__init__("security_type")
        valid_df = pd.read_feather(valid_tickers_path)
        self.valid_tickers = set(valid_df['act_symbol'])
        print(f"  SecurityTypeFilter loaded {len(self.valid_tickers)} valid tickers")

    def apply(self, df, target_date):
        return df[df['act_symbol'].isin(self.valid_tickers)].copy()
    

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
        # Diagnostic: for a month that flatlines, check the overlap
        price_tickers_in_window = set(working_df['act_symbol'].unique())
        overlap = price_tickers_in_window & self.filters[0].valid_tickers  # SecurityTypeFilter
        print(f"  Price tickers in window: {len(price_tickers_in_window)}")
        print(f"  Valid tickers loaded:    {len(self.filters[0].valid_tickers)}")
        print(f"  Overlap:                 {len(overlap)}")
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
    def __init__(self, universe_path, fund_data_path=None, model_folder=None):
        '''
        universe_path should be the path to the universe file, saved as feather
        fund_data_path (optional) path to quarterly fundamental data
        '''
        self.universe_path = universe_path
        self.fund_data_path = fund_data_path  # <-- Store the path, but don't load the data yet
        
        self.data = pd.read_feather(universe_path) # Load price data immediately
        self.data["date"] = pd.to_datetime(self.data["date"])
        
        self.has_features = False 
        self.has_target = False 
        self.data_split = False 
        self.params_tuned = False
        self.variables_generated = False
        
        if type(model_folder) == type(None):
            self.has_folder = False
        else:
            self.has_folder = True
            os.makedirs(model_folder, exist_ok=True)
        self.model_folder = model_folder
        self.data_generated = False

    def add_target(self, target, target_type:str = "regression", save:bool = True):
        valid_target_types = ["classification", "regression"]
        if target_type not in valid_target_types:
            err_msg = "Invalid target_type. Must be one of: "
            for c, i in enumerate(valid_target_types):
                err_msg += i
                if c < len(valid_target_types) - 1:
                    err_msg += ", "
        self.target = target
        self.target_type = target_type
        if save == True:
            if self.has_folder == True:
                with open(f'{self.model_folder}/target.pkl', 'wb') as f:
                    pickle.dump(target, f, pickle.HIGHEST_PROTOCOL)
        self.has_target = True

    def add_features(self, features, save:bool = True):
        '''
        Adds features to the dataset
        '''
        self.features = features
        self.has_features = True
        if save == True:
            if self.has_folder == True:
                os.makedirs(f"{self.model_folder}/features", exist_ok = True)
                for c, feature in enumerate(features):
                    with open(f'{self.model_folder}/features/{c}.pkl', 'wb') as f:
                        pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)

    def split_data(self,
                   cutoffs:list = None
                   ):
        if self.data_generated == False:
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
        #print(len(unique_dates), int(len(unique_dates) * cutoffs[0]), int(len(unique_dates) * cutoffs[1]), int(len(unique_dates) * cutoffs[2]))
        #st()
        train_cutoff = unique_dates[int(len(unique_dates) * cutoffs[0])]
        val_cutoff = unique_dates[int(len(unique_dates) * cutoffs[1])]
        test_cutoff = unique_dates[int(len(unique_dates) * cutoffs[2]) - 1]

        #Old, not purging the data
        #self.train_df = self.data[self.data['date'] < train_cutoff]
        #self.val_df = self.data[(self.data['date'] >= train_cutoff) & (self.data['date'] < val_cutoff)]
        #self.test_df = self.data[(self.data['date'] >= val_cutoff) & (self.data['date'] < test_cutoff)]

        # ADD A PURGE GAP (e.g., 5 days for a 5-day forward target)
        # Extract N from target name
        match = re.search(r'_(\d+)', self.target.name)
        purge_gap_days = int(match.group(1)) if match else 1

        # ADD AN EMBARGO GAP (1% of unique dates, minimum 5 days) to prevent autocorrelation
        embargo_days = max(5, int(len(unique_dates) * 0.01))
        
        # Shift the validation and test start dates forward by both the purge AND embargo gap
        gap_offset = pd.Timedelta(days=purge_gap_days + embargo_days + 2) # +2 for weekends

        self.train_df = self.data[self.data['date'] < train_cutoff]
        # VALIDATION starts AFTER the purge & embargo gap
        self.val_df = self.data[(self.data['date'] >= (train_cutoff + gap_offset)) & (self.data['date'] < val_cutoff)]
        # TEST starts AFTER the purge & embargo gap
        self.test_df = self.data[(self.data['date'] >= (val_cutoff + gap_offset)) & (self.data['date'] < test_cutoff)]
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

        if self.has_folder == True:
            # --- REPLACE YOUR EXISTING 'dates' DICTIONARY WITH THIS ---
            dates = {
                "train_start": [self.train_df['date'].min()],
                "train_end":[self.train_df['date'].max()],
                "val_start": [self.val_df['date'].min()],
                "val_end": [self.val_df['date'].max()],
                "test_start": [self.test_df['date'].min()],
                "test_end":[self.test_df['date'].max()]
            }
            # ----------------------------------------------------------
            dates_df = pd.DataFrame(dates)
            dates_df.to_csv(f"{self.model_folder}/dates.csv")

    def split_data_by_dates(self, train_start, val_start, test_start, test_end):
        """
        Splits data using explicit date boundaries, retaining purge and embargo gaps.
        Dates should be passed as strings (e.g., '2015-01-01') or pd.Timestamp.
        """
        if self.data_generated == False:
            self.generate_targets_and_features()
        if not self.has_features or not self.has_target:
            raise Exception("Features or target missing.")

        train_start = pd.to_datetime(train_start)
        val_start = pd.to_datetime(val_start)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)

        unique_dates = sorted(self.data['date'].unique())

        # Extract Purge
        match = re.search(r'_(\d+)', self.target_key)
        purge_gap_days = int(match.group(1)) if match else 1

        # Calculate Embargo (1% of total dataset dates)
        embargo_days = max(5, int(len(unique_dates) * 0.01))
        gap_offset = pd.Timedelta(days=purge_gap_days + embargo_days + 2)

        # Split with gaps applied to the BEGINNING of val and test sets
        self.train_df = self.data[(self.data['date'] >= train_start) & (self.data['date'] < val_start)]
        self.val_df = self.data[(self.data['date'] >= (val_start + gap_offset)) & (self.data['date'] < test_start)]
        self.test_df = self.data[(self.data['date'] >= (test_start + gap_offset)) & (self.data['date'] < test_end)]

        self.features =[key for key in self.data.keys() if "F" in key.split("_")]
        self.targets =[key for key in self.data.keys() if "T" in key.split("_")]

        if len(self.targets) != 1:
            raise Exception(f"Only one target allowed. Identified: {self.targets}")

        self.X_train, self.y_train = self.train_df[self.features], self.train_df[self.target_key]
        self.X_val, self.y_val = self.val_df[self.features], self.val_df[self.target_key]
        self.X_test, self.y_test_bin = self.test_df[self.features], self.test_df[self.target_key]

        self.data_split = True

        if self.has_folder:
            dates = {
                "train_start": [self.train_df['date'].min()],
                "train_end":[self.train_df['date'].max()],
                "val_start": [self.val_df['date'].min()],
                "val_end": [self.val_df['date'].max()],
                "test_start": [self.test_df['date'].min()],
                "test_end":[self.test_df['date'].max()]
            }
            pd.DataFrame(dates).to_csv(f"{self.model_folder}/dates.csv")

    def tune_params(self, n_trials:int = 50):
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
                    "seed": 42,

                    #for categorical features to prevent overfitting
                    "cat_smooth": trial.suggest_float("cat_smooth", 10.0, 100.0), # Reduces impact of noisy sectors
                    "cat_l2": trial.suggest_float("cat_l2", 1.0, 50.0),           # L2 penalty specifically for categorical splits
                    "min_data_per_group": trial.suggest_int("min_data_per_group", 50, 200), # Prevents isolating tiny sectors
                    
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
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.study = study
        self.params_tuned = True

    def generate_targets_and_features(self):
        if self.has_features == False:
            raise Exception("Model does not have features")
        if self.has_target == False:
            raise Exception("Model does not have a target")
        self.variables_generated = True

        # 1. SEPARATE REQUESTS
        price_requests = [req for req in self.features if not isinstance(req, FundamentalRequest)]
        fund_requests = [req for req in self.features if isinstance(req, FundamentalRequest)]

        # 2. COMPUTE PRICE FEATURES (Always runs)
        # We assume target is always a price-derived feature (e.g., TARGET_SHARPE)
        feature_engine = FeatureEngine(feature_requests=price_requests + [self.target])
        self.data = feature_engine.compute(self.data)

        # 3. LAZY LOAD & COMPUTE FUNDAMENTALS (Only runs if requested)
        if fund_requests:
            # Defensive check: Did they provide a path?
            if self.fund_data_path is None:
                raise Exception("Fundamental features requested, but 'fund_data_path' was not provided to the Model.")
            
            # Defensive check: Does the file actually exist?
            if not os.path.exists(self.fund_data_path):
                raise Exception(f"Fundamental data file not found at: {self.fund_data_path}")

            print("Loading and computing fundamental features...")
            
            # Lazy Load: Read into memory ONLY now
            fund_df = pd.read_feather(self.fund_data_path)
            fund_df["date"] = pd.to_datetime(fund_df["date"])
            
            # Compute Sparse Features
            fund_engine = FundamentalEngine(requests=fund_requests)
            fund_df = fund_engine.compute(fund_df)

            # Extract generated columns
            fund_cols = [req.alias for req in fund_requests]
            
            # Isolate and Clean for the Merge
            fund_df = fund_df[['act_symbol', 'date'] + fund_cols].copy()
            fund_df = fund_df.dropna(subset=fund_cols, how='all')
            
            # CRITICAL: strictly sort by date and drop intra-day duplicates for merge_asof
            fund_df = fund_df.sort_values('date').drop_duplicates(subset=['act_symbol', 'date'], keep='last')
            self.data = self.data.sort_values('date')

            print("Merging fundamental features into daily price data...")
            self.data = pd.merge_asof(
                left=self.data,
                right=fund_df,
                on='date',
                by='act_symbol',
                direction='backward'
            )

        # 4. FINAL CLEANUP
        # Because FundamentalRequest.alias defaults to 'F_{name}', 
        # this naturally cleans both Price and Fundamental features flawlessly!
        feature_keys = [key for key in self.data.keys() if "F" in key.split("_")]
        target_key = [key for key in self.data.keys() if "T" in key.split("_")][0]
        
        # Drop rows with NaN for features or targets
        self.data = self.data.dropna(subset=feature_keys + [target_key]) 
        self.feature_keys = feature_keys
        self.target_key = target_key
        self.data_generated = True

    def train_model(self, 
                    perturb_hyperparameters:bool = False,
                    inject_noise:bool = False,
                    show_feature_importance:bool = True):
        '''
        perturb_hyperparameters: adds 
        inject_noise: 
        '''
    
        if self.params_tuned == False:
            self.tune_params()

        #===================
        #Perturbing hyperparameters
        if perturb_hyperparameters:
            params = {}
            # 1. Initialize rng once outside the loop
            rng = np.random.default_rng()
            
            # Iterate through both keys and values using .items()
            for param, val in self.best_params.items():
                
                # Calculate the multiplier: e.g., 1.07 or 0.93
                multiplier = 1 + (rng.choice([1, -1]) * rng.uniform(low=0.05, high=0.1))
                
                # Apply the multiplier to the original value
                new_val = val * multiplier
                
                # 2. Keep integers as integers (e.g., max_depth, n_estimators)
                if isinstance(val, int) and not isinstance(val, bool):
                    params[param] = int(round(new_val))
                else:
                    params[param] = new_val
        else:
            # Good practice to copy so you don't accidentally mutate best_params later
            params = self.best_params.copy()
        #====================
        
        #====================
        #Injecting Noise to features & targets
        if inject_noise == True:
            bps_3 = 0.0003
            n_rows = len(self.data)
            n_features = len(self.feature_keys)

            # Generate a 2D matrix of noise for ALL features at once
            feature_noise = np.random.normal(loc=1.0, scale=bps_3, size=(n_rows, n_features))
            self.data[self.feature_keys] = self.data[self.feature_keys] * feature_noise

            # Add noise to target
            target_noise = np.random.normal(loc=1.0, scale=bps_3, size=n_rows)
            self.data[self.target_key] = self.data[self.target_key] * target_noise
        #====================
        self.model = lgb.train(
                params,
                lgb.Dataset(self.X_train, label=self.y_train),
                valid_sets=[lgb.Dataset(self.X_val, label=self.y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
        self.test_model(show_feature_importance = show_feature_importance)
        if self.has_folder == True:
            self.model.save_model(f"{self.model_folder}/model.txt")
            info = {
                "universe_path": [self.universe_path]
            }
            info_df = pd.DataFrame(info)
            info_df.to_csv(f"{self.model_folder}/info.csv")

    def test_model(self, show_feature_importance:bool = True):
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
        if show_feature_importance == True:
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
        import re
        
        if "pred_return" not in self.test_df.columns:
            raise Exception("Model has not generated predictions yet. Run test_model() first.")
            
        df = self.test_df.copy()
        target_col = self.targets[0]
        
        print(f"\nQUANTILE SPREAD ANALYSIS (Top {100/quantiles:.1f}% vs Bottom {100/quantiles:.1f}%)")
        print("========================")
        
        df['quantile'] = df.groupby('date')['pred_return'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=quantiles, labels=False) + 1
        )
        
        daily_quantile_returns = df.groupby(['date', 'quantile'])[target_col].mean().unstack().dropna()
        if daily_quantile_returns.empty:
            print("Not enough data to form quantiles.")
            return
            
        # This daily_spread is mathematically an N-day return realized over N days
        daily_spread = daily_quantile_returns[quantiles] - daily_quantile_returns[1]
        
        mean_spread = daily_spread.mean()
        win_rate = (daily_spread > 0).mean()
        
        # Extract N for overlap adjustment
        match = re.search(r'(?:FWD_LOG_RET|TARGET_SHARPE)_(\d+)', self.target_key)
        N = int(match.group(1)) if match else 1
        
        # Calculate PSR and DSR using the unannualized N-day spread
        sr_hat, psr, dsr, sr0 = self._calculate_psr_dsr(daily_spread, N_horizon=N)
        
        # Annualize for human-readable output
        # Because the spread is an N-day return, there are (252/N) periods in a year
        ann_factor = np.sqrt(252 / N)
        annualized_sr = sr_hat * ann_factor
        annualized_sr0 = sr0 * ann_factor
        
        print(f"Mean Target Spread       : {mean_spread:.4f}")
        print(f"Spread Win Rate          : {win_rate:.2%}")
        print("-" * 40)
        print("ROBUST PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Standard Annualized Sharpe : {annualized_sr:.4f}")
        print(f"Probabilistic Sharpe (PSR) : {psr:.2%}  <-- (Target: > 95%)")
        
        if not np.isnan(dsr):
            print(f"Experiments Run (Optuna N) : {len(self.study.trials)}")
            print(f"Expected Max Sharpe (SR0)  : {annualized_sr0:.4f} (Annualized)")
            print(f"Deflated Sharpe (DSR)      : {dsr:.2%}  <-- (Target: > 95%)")
        else:
             print("* Tune params first to calculate DSR *")
        print("-" * 40)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            overall_q_mean = daily_quantile_returns.mean()
            axes[0].bar(overall_q_mean.index, overall_q_mean.values, color='skyblue', edgecolor='black')
            axes[0].set_title("Average Target Value by Prediction Quantile")
            axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
            
            # Since daily_spread represents overlapping N-day returns, taking cumulative sum
            # directly drastically exaggerates the equity curve. We divide by N to approximate daily PnL.
            approx_daily_pnl = daily_spread / N
            cumulative_spread = approx_daily_pnl.cumsum()
            
            axes[1].plot(cumulative_spread.index, cumulative_spread.values, color='purple', linewidth=2)
            axes[1].set_title(f"Cumulative Long-Short Spread (Scaled by 1/{N})")
            axes[1].axhline(0, color='black', linewidth=1)
            plt.tight_layout()
            plt.show()
            
        return daily_quantile_returns, daily_spread

    def apply_kuhn_johnson_reduction(self, threshold=0.75):
        """
        Automates the Kuhn & Johnson feature reduction algorithm.

        Args:
            df: Pandas DataFrame containing your data.
            feature_cols: List of column names to consider for reduction (your 'X' variables).
            threshold: The absolute correlation threshold to trigger removal.

        Returns:
            List of features to keep.
        """

        # Step 1: Calculate the absolute correlation matrix of the predictors
        # (We use absolute because -0.90 is just as correlated as +0.90)
        feature_cols = [key for key in self.data.keys() if "F" in key.split("_")]
        corr_matrix = self.data[feature_cols].corr().abs()

        dropped_features =[]

        while True:
            # Mask the diagonal and lower triangle so we don't compare features to themselves
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find the single highest absolute correlation value in the matrix
            max_corr = upper_tri.max().max()

            # Step 5: Repeat until no absolute correlations are above the threshold
            if pd.isna(max_corr) or max_corr < threshold:
                break

            # Step 2: Determine the two predictors (A and B) with the largest correlation
            # .stack() creates a series of all pairs, .idxmax() grabs the names of the top pair
            feature_A, feature_B = upper_tri.stack().idxmax()

            # Step 3: Determine the average correlation between A/B and the *other* variables
            # We drop A and B from the column so we are comparing them strictly to the REST of the dataset
            avg_corr_A = corr_matrix[feature_A].drop([feature_A, feature_B]).mean()
            avg_corr_B = corr_matrix[feature_B].drop([feature_A, feature_B]).mean()

            # Step 4: If A has a larger average correlation, remove it; otherwise, remove B
            if avg_corr_A > avg_corr_B:
                to_drop = feature_A
            else:
                to_drop = feature_B

            dropped_features.append(to_drop)

            # Update the correlation matrix by removing the dropped feature for the next loop iteration
            corr_matrix = corr_matrix.drop(index=to_drop, columns=to_drop)

            # Optional: Print progress so you can see what is happening under the hood
            # print(f"Max Corr: {max_corr:.3f} | Pair: ({feature_A}, {feature_B}) | Dropped: {to_drop}")

        # Final reporting
        features_to_keep =[col for col in feature_cols if col not in dropped_features]
        print(f"Dropped {len(dropped_features)} features due to collinearity > {threshold}.")
        print(f"Remaining features: {len(features_to_keep)}")

        features_to_drop = [feature for feature in feature_cols if feature not in features_to_keep]
        self.data.drop(features_to_drop, axis = 1, inplace = True)

    def apply_block_pca_reduction(self, correlation_threshold=0.75, max_pcs=1):
        """
        Groups highly correlated features into clusters and replaces them with 
        their Principal Components (Block PCA), retaining orthogonal signal 
        while reducing noise and multicollinearity.
        
        Args:
            correlation_threshold (float): The absolute correlation above which 
                                           features will be grouped together.
            max_pcs (int): How many principal components to keep per cluster.
                           (1 is usually best to capture the core 'factor')
        """
        import pandas as pd
        import numpy as np
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.decomposition import PCA
        
        # 1. Identify feature columns using your class's existing convention
        feature_cols =[key for key in self.data.keys() if "F" in key.split("_")]
        if len(feature_cols) < 2:
            print("Not enough features to perform reduction.")
            return

        # 2. Create condensed distance matrix based on absolute correlation
        corr_matrix = self.data[feature_cols].corr().abs()
        dist_array = 1.0 - corr_matrix.values
        
        # Mathematical checks to ensure strict matrix symmetry and 0 diagonal 
        # (required by scipy's squareform)
        dist_array = (dist_array + dist_array.T) / 2.0  
        np.fill_diagonal(dist_array, 0.0)               
        dist_array = np.clip(dist_array, 0.0, 1.0)      
        
        condensed_dist = squareform(dist_array)

        # 3. Hierarchical clustering
        # 'complete' linkage ensures all features in a cluster are correlated 
        # to each other by AT LEAST the threshold
        Z = linkage(condensed_dist, method='complete')
        distance_threshold = 1.0 - correlation_threshold
        labels = fcluster(Z, t=distance_threshold, criterion='distance')

        # Group features by their assigned cluster label
        clusters = {}
        for feature, label in zip(feature_cols, labels):
            clusters.setdefault(label,[]).append(feature)

        features_to_drop =[]
        new_features_df = pd.DataFrame(index=self.data.index)

        # 4. Perform PCA on correlated clusters
        for label, group in clusters.items():
            if len(group) == 1:
                continue  # Leave isolated/uncorrelated features alone
                
            features_to_drop.extend(group)
            
            # Standardize before PCA (critical for PCA on variables with different scales)
            subset_data = self.data[group].copy()
            subset_scaled = (subset_data - subset_data.mean()) / (subset_data.std() + 1e-8)
            subset_scaled = subset_scaled.fillna(0) # Safety against NaNs
            
            # Fit PCA
            pca = PCA()
            pca.fit(subset_scaled)
            
            # Calculate how many PCs to keep (do not exceed the original variable count)
            n_comps = min(max_pcs, len(group) - 1)
            n_comps = max(1, n_comps) 
            
            pca_transformed = pca.transform(subset_scaled)[:, :n_comps]
            
            # 5. Name the new features combining the grouped feature names
            # We strip out the "F_" part of the original names to prevent "F_F_F_" stacking
            clean_names =[f.replace('F_', '') for f in group]
            combined_name = "_".join(clean_names)
            
            for i in range(n_comps):
                # Ensure the name starts with F_ so your class regex logic successfully 
                # identifies it as a feature during split_data()
                new_col_name = f"F_PCA_PC{i+1}_{combined_name}"
                
                # Pandas handles long names, but we'll cap them at 100 characters 
                # to prevent unreadable dataframe outputs and plot axes
                if len(new_col_name) > 100:
                    new_col_name = new_col_name[:90] + "_TRUNC"
                    
                new_features_df[new_col_name] = pca_transformed[:, i]

        # 6. Apply back to self.data
        if features_to_drop:
            self.data.drop(columns=features_to_drop, inplace=True)
            self.data = pd.concat([self.data, new_features_df], axis=1)

        print(f"--- Block PCA Reduction Summary ---")
        print(f"Original features evaluated: {len(feature_cols)}")
        print(f"Features dropped due to collinearity > {correlation_threshold}: {len(features_to_drop)}")
        print(f"New PCA features generated: {new_features_df.shape[1]}")
        print(f"Final feature count: {len(feature_cols) - len(features_to_drop) + new_features_df.shape[1]}")

    def apply_shadow_feature_selection(self, num_iterations=5, importance_type='gain'):
        """
        Uses a Boruta-style 'Shadow Feature' approach to drop noise.
        Creates randomly shuffled copies of every feature and drops any real feature
        that cannot outperform the best randomized shadow feature.
        """
        if not self.data_split:
            raise Exception("You must call split_data() before running shadow feature selection.")
            
        print("\n--- Running Target-Aware Shadow Feature Selection ---")
        
        real_features = self.features.copy()
        features_to_drop = set()
        
        # We run this a few times to ensure stability (random seeds matter)
        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}...")
            
            # 1. Create a temporary dataframe with real features
            X_train_temp = self.X_train.copy()
            
            # 2. Add 'shadow' (randomly shuffled) versions of every feature
            shadow_names =[]
            for feat in real_features:
                shadow_name = f"SHADOW_{feat}"
                shadow_names.append(shadow_name)
                # Randomly permute the column
                X_train_temp[shadow_name] = np.random.permutation(X_train_temp[feat].values)
            
            # 3. Train a quick, shallow LightGBM model
            # We use shallow trees to prevent massive overfitting to the noise
            params = {
                'objective': 'regression' if self.target_type == 'regression' else 'binary',
                'boosting_type': 'gbdt',
                'max_depth': 4,
                'num_leaves': 15,
                'learning_rate': 0.05,
                'verbosity': -1,
                'seed': 42 + iteration
            }
            
            dtrain = lgb.Dataset(X_train_temp, label=self.y_train)
            model = lgb.train(params, dtrain, num_boost_round=100)
            
            # 4. Extract Feature Importances
            importance_df = pd.DataFrame({
                'feature': X_train_temp.columns,
                'importance': model.feature_importance(importance_type=importance_type)
            })
            
            # 5. Find the maximum importance achieved by ANY shadow feature
            shadow_importances = importance_df[importance_df['feature'].str.startswith('SHADOW_')]
            max_shadow_importance = shadow_importances['importance'].max()
            
            # 6. Identify real features that scored LOWER than the best shadow feature
            real_importances = importance_df[~importance_df['feature'].str.startswith('SHADOW_')]
            failed_features = real_importances[real_importances['importance'] <= max_shadow_importance]['feature'].tolist()
            
            # Add to our set of features to drop
            features_to_drop.update(failed_features)
            
        # 7. Apply the reduction
        features_to_keep =[f for f in real_features if f not in features_to_drop]
        
        # Update class variables
        self.features = features_to_keep
        self.X_train = self.X_train[self.features]
        self.X_val = self.X_val[self.features]
        self.X_test = self.X_test[self.features]
        
        print(f"\n--- Shadow Feature Selection Complete ---")
        print(f"Original feature count : {len(real_features)}")
        print(f"Features dropped       : {len(features_to_drop)}")
        print(f"Remaining clean features: {len(self.features)}")

    def run_permutation_test(self, n_repeats=5):
        """
        Shuffles the target variable randomly and retrains. 
        If the model still finds a signal (positive Sharpe/IC), your features are 
        leaking the target directly (e.g., using today's close to predict today's return).
        """
        if not self.data_split:
            raise Exception("Split data first.")
            
        print(f"\n--- Running Permutation Test ({n_repeats} Iterations) ---")
        original_y_train = self.y_train.copy()
        leakage_detected = False
        
        for i in range(n_repeats):
            # 1. Randomly shuffle the training targets
            self.y_train = np.random.permutation(original_y_train)
            
            # 2. Train a fast model on the garbage data
            params = {
                'objective': 'regression' if self.target_type == 'regression' else 'binary',
                'verbosity': -1, 'seed': 42 + i, 'boosting_type': 'gbdt'
            }
            dtrain = lgb.Dataset(self.X_train, label=self.y_train)
            gbm = lgb.train(params, dtrain, num_boost_round=50)
            
            # 3. Predict on un-shuffled test data
            preds = gbm.predict(self.X_test)
            
            # 4. Check correlation
            ic, _ = spearmanr(self.y_test_bin, preds)
            print(f"  -> Shuffle {i+1} Rank IC: {ic:.4f}")
            
            # If a model trained on random noise achieves an IC > 0.02, something is leaking
            if abs(ic) > 0.02:
                leakage_detected = True
                
        # Restore actual target
        self.y_train = original_y_train
        
        if leakage_detected:
            print("\n🚨 WARNING: Model found a strong signal in randomized noise!")
            print("Your features are leaking future data. Check your FeatureEngine logic.")
        else:
            print("\n✅ Permutation test passed. No obvious target leakage detected.")


    def evaluate_with_transaction_costs(self, quantiles=10, bps_fee=5):
        """
        Recalculates the Long-Short spread accounting for daily portfolio turnover and slippage.
        bps_fee = basis points charged per trade (5 bps = 0.05% slippage/fee per trade)
        """
        if "pred_return" not in self.test_df.columns:
            raise Exception("Run test_model() first.")
            
        print(f"\n--- TRANSACTION COST ANALYSIS ({bps_fee} bps fee) ---")
        df = self.test_df.copy()
        
        # 1. Assign daily quantiles
        df['quantile'] = df.groupby('date')['pred_return'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=quantiles, labels=False) + 1
        )
        
        # 2. Isolate the Long (Top Q) and Short (Bottom Q) portfolios
        longs = df[df['quantile'] == quantiles].copy()
        shorts = df[df['quantile'] == 1].copy()
        
        # 3. Calculate Daily Turnover
        # How many stocks from yesterday's top decile are NO LONGER in today's top decile?
        def calc_turnover(portfolio_df):
            daily_symbols = portfolio_df.groupby('date')['act_symbol'].apply(set)
            turnover_pct =[]
            dates = daily_symbols.index
            for i in range(1, len(dates)):
                prev_basket = daily_symbols.iloc[i-1]
                curr_basket = daily_symbols.iloc[i]
                # % of new stocks = (stocks in current but not in prev) / total current
                new_stocks = len(curr_basket - prev_basket)
                turnover_pct.append(new_stocks / max(len(curr_basket), 1))
            return pd.Series([0] + turnover_pct, index=dates)

        long_turnover = calc_turnover(longs)
        short_turnover = calc_turnover(shorts)
        
        # 4. Calculate Gross vs Net Returns
        target_col = self.targets[0]
        daily_long_ret = longs.groupby('date')[target_col].mean()
        daily_short_ret = shorts.groupby('date')[target_col].mean()
        
        gross_spread = daily_long_ret - daily_short_ret
        
        # Apply fees (Turnover * bps_fee/10000). Multiply by 2 because we pay to buy AND sell.
        fee_decimal = bps_fee / 10000.0
        daily_fees = (long_turnover * fee_decimal * 2) + (short_turnover * fee_decimal * 2)
        
        net_spread = gross_spread - daily_fees
        
        # 5. Extract N-day horizon for Sharpe annualization
        match = re.search(r'(?:FWD_LOG_RET|TARGET_SHARPE)_(\d+)', self.target_key)
        N = int(match.group(1)) if match else 1
        
        gross_sharpe = (gross_spread.mean() / gross_spread.std()) * np.sqrt(252 / N)
        net_sharpe = (net_spread.mean() / net_spread.std()) * np.sqrt(252 / N)
        
        print(f"Average Daily Turnover (Longs) : {long_turnover.mean():.2%}")
        print(f"Average Daily Turnover (Shorts): {short_turnover.mean():.2%}")
        print(f"Gross Annual Sharpe            : {gross_sharpe:.4f}")
        print(f"NET Annual Sharpe (After Fees) : {net_sharpe:.4f}")
        
        if net_sharpe < 0:
            print("🚨 Your model's returns are entirely consumed by transaction costs.")
        return net_spread

    def check_feature_leakage(self):
        """Flags features that are 'too important'."""
        print("\n--- Feature Importance Sanity Check ---")
        importances = self.model.feature_importance(importance_type='gain')
        total_gain = importances.sum()
        
        features = self.model.feature_name()
        
        for feat, imp in zip(features, importances):
            pct_importance = imp / total_gain
            if pct_importance > 0.25:
                print(f"🚨 WARNING: '{feat}' accounts for {pct_importance:.1%} of model's predictive power.")
                print("   This almost always indicates a look-ahead bias (data leakage).")

    def _calculate_psr_dsr(self, spread_returns, N_horizon):
        '''
        Method for calculating probabilistic and deflated sharpe ratios
        '''
        from scipy.stats import skew, kurtosis, norm
        
        # 1. Unannualized Stats
        returns = np.asarray(spread_returns.dropna())
        if len(returns) < 2 or np.std(returns, ddof=1) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sr_hat = mean_ret / std_ret  
        
        # 2. Adjust Degrees of Freedom (T) for overlapping N-day holds
        T_total = len(returns)
        T_effective = T_total / N_horizon 
        
        skewness = skew(returns, bias=False)
        kurt = kurtosis(returns, fisher=False, bias=False)
        
        psr_denom = np.sqrt(1 - skewness * sr_hat + ((kurt - 1) / 4) * (sr_hat ** 2))
        
        # 3. Calculate PSR (Benchmark = 0)
        psr_stat = (sr_hat - 0.0) * np.sqrt(T_effective - 1) / psr_denom
        psr = norm.cdf(psr_stat)
        
        # 4. Calculate DSR using Optuna Trials
        dsr = np.nan
        sr0 = 0.0
        
        # Extract N_trials and variance from Optuna if tuned
        if hasattr(self, 'study'):
            trials = [t.value for t in self.study.trials if t.value is not None]
            N_trials = len(trials)
            
            if N_trials > 1:
                # Approximate Sharpe variance from the Optuna metric variance
                var_trials = np.var(trials) 
                
                euler_gamma = np.euler_gamma
                term1 = (1 - euler_gamma) * norm.ppf(1 - 1/N_trials)
                term2 = euler_gamma * norm.ppf(1 - (1/N_trials) * np.exp(-1))
                
                sr0 = np.sqrt(var_trials) * (term1 + term2)
                
                dsr_stat = (sr_hat - sr0) * np.sqrt(T_effective - 1) / psr_denom
                dsr = norm.cdf(dsr_stat)
                
        return sr_hat, psr, dsr, sr0
    

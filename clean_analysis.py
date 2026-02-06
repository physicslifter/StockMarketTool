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

'''#helper function for advanced stats filter
def calculate_metrics_single(ticker, prices):
    """
    Calculates Hurst, Annual Return, and Volatility for a single ticker.
    Input: Array of 1 year of daily close prices.
    Output: (ticker, hurst_value, annual_return, volatility, last_price)
    """
    if len(prices) < 30: 
        return (ticker, np.nan, np.nan, np.nan, np.nan)
    
    current_price = prices[-1]
    
    # 1. Calculate Annual Return (Momentum check for "trending to 0")
    annual_ret = (current_price / prices[0]) - 1
    
    # 2. Calculate Annualized Volatility (Standard Deviation of Log Returns)
    # used for the "Stability" filter
    try:
        log_rets = np.diff(np.log(prices))
        # Annualized Vol (assuming daily data)
        volatility = np.std(log_rets) * np.sqrt(252)
    except:
        volatility = np.nan

    # 3. Calculate Hurst Exponent
    try:
        log_prices = np.log(prices)
        lags = range(2, 20) # Lags to test
        
        # Volatility calculation: std(t) ~ t^H
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        
        # Avoid log(0) error
        tau = [t if t > 0 else 1e-8 for t in tau]
        
        # Slope of log-log plot
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except:
        hurst = np.nan
        
    return (ticker, hurst, annual_ret, volatility, current_price)'''

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
            filter_df = df[df.date == df.date.max()]
            winners = filter_df[filter_df.close > self.min_price]
            winners = winners.act_symbol.tolist()
            
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
            etf_data = pd.read_feather("Data/ETFs.feather")
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
        working_df = self.master_df[(self.master_df.date > target_date - pd.DateOffset(years = 1)) & (self.master_df.date < target_date)]
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
        final_df = final_df.sort_values(by='date', ascending=True)
        final_df = final_df.reset_index(drop=True)
        if type(save_name) != type(None):
            extension = save_name.split(".")[-1]
            if extension == "csv":
                final_df.to_csv(save_name)
            elif extension == "feather":
                final_df.to_feather(save_name)
            else:
                raise Exception("Extension invalid. Write code to save for this file type")
        return final_df
                


class Model:
    def __init__(self):
        pass

class PortfolioStrat:
    def __init__(self):
        pass


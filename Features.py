'''
All features and targets we might want for the dataset
& implementation methods
'''
import pandas as pd
import numpy as np
import talib

#list of valid feature string names
valid_features = ["log_ret",
                  "volatility",
                  "SMA"
                  ]

#list of valid target string names
valid_targets = ["log_ret",
                 "volatility"
                 ]

talib_functions = {
    "SMA": talib.SMA,
    "EMA": talib.EMA,
    "bollinger": talib.BBANDS
}

def test_valid_df(df):
    is_valid = True
    for valid_key in ["date", "open", "high", "low", "close", "act_symbol", "volume"]:
        if valid_key not in df.keys():
            print(f"{valid_key} not found in data")
            is_valid = False
    return is_valid

def get_basic_attributes(df):
    '''
    Adds the following to the dataset
    
    log returns
    '''
    pass

#parent class for features and targets
class Variable:
    def __init__(self, name:str, num_days:int):
        self.name = name
        self.num_days = num_days
        self.is_feature = False if num_days > 0 else True
        if self.num_days == 0:
            if name not in talib_functions.keys():  
                raise Exception("num_days cannot be 0")

    def retrieve_data(self, df, return_data:bool = False):
        """Placeholder function for getting the data

        if num_days is negative, we are getting a feature
        if num_days is positive, we are getting a target

        Args:
            df (_type_): dataframe to get feature on
            num_days (_type_): # of days to look back

        Returns:
            pd.Series: pandas series of the feature data
        """
        #ensure df has the necessary keys
        if test_valid_df(df) == False:
            raise Exception("DataFrame is invalid")
        
        #sort dataframe only if it has not already been sorted
        if not df.attrs.get("is_sorted_by_symbol_date", False):
            df.sort_values(by=["act_symbol", "date"], inplace=True)
            df.attrs["is_sorted_by_symbol_date"] = True

        self.get_detailed_name()

        result = self.get(df, self.num_days)
        df[self.detailed_name] = result
        if return_data == True:
            return df
    
    def get_detailed_name(self):
        appendix = "T" if self.is_feature == False else "F"
        penultimate_appendix = "" if self.name in talib_functions.keys() else f"_{np.abs(self.num_days)}d_"
        self.detailed_name = f"{self.name}{penultimate_appendix}{appendix}"
    
class LogReturn(Variable):
    def __init__(self, num_days):
        super().__init__("log_ret", num_days)

    def get(self, df:pd.DataFrame, num_days:int):
        #shifted_close = df.groupby("act_symbol")["close"].shift(-num_days)
        grouped_close = df.groupby("act_symbol")["close"]
        if num_days < 0: #if we're getting a feature
            end_close = grouped_close.shift(1) #close @ end will be yesterday's
            start_close = grouped_close.shift(1 - num_days) #close @ start will be from n+1 days ago
        else:  #if we're getting a target
            end_close = grouped_close.shift(-num_days) #close @ end will be from 5 days in the future
            start_close = grouped_close.shift(0)
        result = np.log(end_close/start_close)
        return result
    
class Volatility(Variable):
    def __init__(self, num_days):
        super().__init__("volatility", num_days)

    def get(self, df:pd.DataFrame, num_days:int):
        grouped_close = df.groupby("act_symbol")["close"]
        log_ret = np.log(df["close"] / grouped_close.shift(1))
        def roll_std(x):
            return x.rolling(window = np.abs(num_days)).std()
        result = log_ret.groupby(df["act_symbol"]).transform(roll_std)
        if num_days > 0: #shift forward if a target
            result = result.groupby(df["act_symbol"]).shift(-num_days)
        else:
            result = result.groupby(df["act_symbol"]).shift(1)
        return result

#Original TALibVar classes
'''class SMA(Variable):
    def __init__(self, num_days):
        super().__init__("SMA", num_days)

    def get(self, df:pd.DataFrame, num_days:int):
        result = talib(talib.SMA)

class TALibVar(Variable):
    def __init__(self, num_days, name):
        super().__init__(name, num_days)

    def get(self, df:pd.DataFrame, num_days:int):
        if self.name not in talib_functions.keys():
            raise Exception("Function not present in talib function dict")
        my_function = talib_functions[self.name]
        timeperiod = np.abs(num_days)
        def calc_and_shift(close): #function for calculating the result & shifting
            
        #result = my_function(df.close, timeperiod = np.abs(num_days))
        """
        if num_days > 0: #shift forward if a target
            result = result.groupby(df["act_symbol"]).shift(-num_days)
        else:
            result = result.groupby(df["act_symbol"]).shift(1)
        return result"""'''
class TALibVarTimePeriod(Variable):
    def __init__(self, name, num_days, timeperiod):
        # num_days: How far to shift (e.g., -5 for 5-day lag, +5 for 5-day future target)
        super().__init__(name, num_days)
        # timeperiod: The indicator setting (e.g., 200 for 200-day SMA)
        self.timeperiod = timeperiod

    def get_detailed_name(self):
        # Override the naming convention to include the timeperiod
        # Example: SMA_200_5d_F
        appendix = "T" if self.is_feature == False else "F"
        self.detailed_name = f"{self.name}_{self.timeperiod}_{np.abs(self.num_days)}d_{appendix}"

    def get(self, df: pd.DataFrame, num_days: int):
        if self.name not in talib_functions.keys():
            raise Exception("Function not present in talib function dict")
        
        my_function = talib_functions[self.name]
        
        # We capture self.timeperiod locally for the closure
        tp = self.timeperiod

        def calculate_and_shift(x):
            # 1. Calculate the Indicator using the specified TIME PERIOD
            # (e.g., 200-day SMA)
            try:
                # talib requires float inputs
                vals = my_function(x.astype(float).values, timeperiod=tp)
                series = pd.Series(vals, index=x.index)
            except Exception:
                return pd.Series([np.nan] * len(x), index=x.index)
            if num_days < 0:
                series = series.shift(1)

            # 2. Apply the Lag/Future Shift using NUM_DAYS
            # if num_days = -5 (Feature): shift(5) -> Moves past data (t-5) to current row (t)
            # if num_days = +5 (Target):  shift(-5) -> Moves future data (t+5) to current row (t)

            return series.shift(-num_days)

        #if log_ret not in dataframe, add it
        if "log_ret" not in df.keys():
            df["log_ret"] = np.log(df["close"] / df.close.shift(1))

        # Apply to each stock individually
        result = df.groupby("act_symbol")["log_ret"].transform(calculate_and_shift)
        
        return result

#bollinger bands from talib
class BollingerBands(TALibVarTimePeriod):
    def __init__(self, num_days, timeperiod=20, nbdevup=2, nbdevdn=2):
        super().__init__("BBANDS", num_days, timeperiod)
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn

    def retrieve_data(self, df, return_data: bool = False):
        # 1. Standard Validation & Sorting (Inherited logic)
        if test_valid_df(df) == False:
            raise Exception("DataFrame is invalid")
        
        if not df.attrs.get("is_sorted_by_symbol_date", False):
            df.sort_values(by=["act_symbol", "date"], inplace=True)
            df.attrs["is_sorted_by_symbol_date"] = True

        # 2. Ensure Log Returns exist
        if "log_ret" not in df.keys():
            df["log_ret"] = np.log(df["close"] / df.close.shift(1))

        # 3. Generate the base name (e.g. BBANDS_20_5d_F)
        self.get_detailed_name()

        # 4. Define the calculation logic
        def calculate_bands(x):
            # x is the log_ret series for a single stock
            try:
                u, m, l = talib.BBANDS(x.astype(float).values, 
                                       timeperiod=self.timeperiod,
                                       nbdevup=self.nbdevup, 
                                       nbdevdn=self.nbdevdn)
                # Convert to Series to keep index alignment
                u = pd.Series(u, index=x.index)
                m = pd.Series(m, index=x.index)
                l = pd.Series(l, index=x.index)
            except Exception:
                nan_s = pd.Series([np.nan] * len(x), index=x.index)
                u, m, l = nan_s, nan_s, nan_s

            # 5. Apply Shifting (Must match TALibVarTimePeriod logic)
            if self.num_days < 0: 
                # FEATURE: Shift 1 (avoid lookahead) + Shift lag
                # If num_days is -5, we shift(1) then shift(5)
                u = u.shift(1).shift(-self.num_days)
                m = m.shift(1).shift(-self.num_days)
                l = l.shift(1).shift(-self.num_days)
            else: 
                # TARGET: Shift backward (future data)
                u = u.shift(-self.num_days)
                m = m.shift(-self.num_days)
                l = l.shift(-self.num_days)

            return pd.DataFrame({'u': u, 'm': m, 'l': l})

        # 6. Apply to Groupby
        # This returns a DataFrame with MultiIndex (act_symbol, original_index)
        results = df.groupby("act_symbol")["log_ret"].apply(calculate_bands)

        # 7. Add columns to DataFrame
        # We use .values to bypass index misalignment, relying on the fact that 
        # df and results are sorted identically by symbol/date.
        df[f"{self.detailed_name}_upper"] = results['u'].values
        df[f"{self.detailed_name}_middle"] = results['m'].values
        df[f"{self.detailed_name}_lower"] = results['l'].values

        if return_data:
            return df
        
class BollingerBandsNormalized(TALibVarTimePeriod):
    def __init__(self, num_days, timeperiod=20, nbdevup=2, nbdevdn=2, pct_B:bool = True, width:bool = True):
        super().__init__("BBANDS_Norm", num_days, timeperiod)
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.width = width
        self.pct_B = pct_B

    def retrieve_data(self, df, return_data: bool = False):
        # ... validation code ...
        
        self.get_detailed_name()

        # 1. Use PRICE, not Log Returns
        def calculate_percent_b(x):
            try:
                u, m, l = talib.BBANDS(x.astype(float).values, 
                                       timeperiod=self.timeperiod, 
                                       nbdevup=self.nbdevup, 
                                       nbdevdn=self.nbdevdn)
                u = pd.Series(u, index=x.index)
                l = pd.Series(l, index=x.index)
                
                # Calculate %B (Normalized Position)
                # Avoid division by zero
                width = u - l
                width[width == 0] = np.nan 
                
                percent_b = (x - l) / width
                
                # Also return Bandwidth (Vol measure)
                bandwidth = width / m
                
            except Exception:
                percent_b = pd.Series([np.nan] * len(x), index=x.index)
                bandwidth = pd.Series([np.nan] * len(x), index=x.index)

            # Shift Logic (Feature vs Target)
            if self.num_days < 0:
                percent_b = percent_b.shift(1).shift(-self.num_days)
                bandwidth = bandwidth.shift(1).shift(-self.num_days)
            else:
                percent_b = percent_b.shift(-self.num_days)
                bandwidth = bandwidth.shift(-self.num_days)

            return pd.DataFrame({'pct_b': percent_b, 'width': bandwidth})

        results = df.groupby("act_symbol")["close"].apply(calculate_percent_b)

        if self.pct_B == True:
            df[f"{self.detailed_name}_pct_b"] = results['pct_b'].values
        if self.width == True:
            df[f"{self.detailed_name}_width"] = results['width'].values
        
        if return_data: return df
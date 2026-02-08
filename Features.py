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
    "SMA": talib.SMA
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
        
        df.sort_values(by=["act_symbol", "date"], inplace=True)

        self.get_detailed_name()

        result = self.get(df, self.num_days)
        df[self.detailed_name] = result
        if return_data == True:
            return df
    
    def get_detailed_name(self):
        appendix = "T" if self.is_feature == False else "F"
        self.detailed_name = f"{self.name}_{np.abs(self.num_days)}d_{appendix}"
    
class LogReturn(Variable):
    def __init__(self, num_days):
        super().__init__("log_ret", num_days)

    def get(self, df:pd.DataFrame, num_days:int):
        shifted_close = df.groupby("act_symbol")["close"].shift(-num_days)
        result = np.log(df["close"]/shifted_close)
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
        return result

class SMA(Variable):
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
        result = my_function(df.close, timeperiod = np.abs(num_days))
    



        

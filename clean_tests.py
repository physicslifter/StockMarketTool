'''
Tests for classes in clean_analysis.py
'''
from DoltReader import DataReader
from clean_analysis import *
import pandas as pd
from dateutil.relativedelta import relativedelta
from pdb import set_trace as st
from matplotlib import pyplot as plt

plt.style.use('dark_background')

#=====================================
#Flags

'''
Filter tests

test_top_liquidity_filter: demonstrates liquidity filter is working
test_price_filter: demonstrates price filter is working
test_advanced_stats_filter: demonstrates advanced stats filter is working
test_universe: demonstrates that Universe() class works
test_ta_lib: demonstrates TA-LIB functionality is working
'''
test_top_liquidity_filter = 0 
test_price_filter = 0
test_advanced_stats_filter = 0
test_universe = 0
test_ta_lib = 1

#=====================================
#useful functions
def get_ohlcv(year, month):
    #gets data preceding a specific month, useful for testing filters
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"]) 
    target_end = pd.Timestamp(year, month, 1)
    target_start = target_end - relativedelta(months=12)
    mask = (df["date"] >= target_start) & (df["date"] <= target_end)
    month_df = df[mask]
    ETFs = pd.read_feather("ETFs.feather")
    etf_list = set(ETFs['act_symbol'].unique())
    month_df = month_df[~month_df['act_symbol'].isin(etf_list)]
    return month_df

#=====================================
#Tests

'''
Filter tests
'''
#get data for filter tests
if test_top_liquidity_filter + test_price_filter + test_advanced_stats_filter > 0:
    #get data for November 2025
    month_df = get_ohlcv(2025, 12)
    target_date = pd.Timestamp(2025, 12, 1)

if test_top_liquidity_filter == True:
    #prove liquidity filter works 
    filter = TopLiquidityFilter(N = 10)
    results = filter.apply(df = month_df, target_date = target_date)
    print(results.act_symbol.unique())

if test_price_filter == True:
    filter = PriceFilter(min_price = 15)
    results = filter.apply(month_df, target_date)
    print(results.nsmallest(10, "open"))

if test_advanced_stats_filter == True:
    filter = AdvancedStatsFilter(max_crash = 0.5, volatility_n = 10, require_uptrend = True)
    results = filter.apply(month_df, target_date)
    print(results.act_symbol.unique())

if test_universe == True:
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    filters = [
        TopLiquidityFilter(N = 3000),
        PriceFilter(min_price = 5),
        AdvancedStatsFilter(min_history_days = None,
                            max_crash = -0.5,
                            require_uptrend = None,
                            volatility_n = 1000)
    ]
    universe = Universe(master_df = df)
    universe.add_filters(filters)
    dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
    universe_data = universe.get_all_universe_data(dates = dates)
    test_universe_data = pd.read_feather("Data/test_training_universe.feather")
    test = test_universe_data.drop(columns = ["hurst_1y"])
    print(universe_data.equals(test))

if test_ta_lib == True:
    import talib
    df = pd.read_feather("Data/all_ohlcv.feather")
    df = df[df.act_symbol == "F"]
    ta_lib_sma = talib.SMA(df["close"]) #SMA FROM TA-LIB
    upper, middle, lower = talib.BBANDS(df.close) #Bollinger Bands from ta-lib
    fig = plt.figure(figsize = (9, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(df.date, df.close, c = "lime", label = "Close Price")
    ax1.plot(df.date, ta_lib_sma, label = "SMA", c = "magenta", linestyle = ":")
    ax1.plot(df.date, upper, label = "Upper BB", c = "orange", linestyle = "--")
    ax1.plot(df.date, lower, label = "Lower BB", c = "orange", linestyle = "--")
    ax1.plot(df.date, middle, label = "Middle BB", c = "goldenrod", linestyle = "--")
    ax1.grid(True)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($USD)")
    ax1.legend()
    ax1.set_title("Testing talib functions on F data")
    plt.show()


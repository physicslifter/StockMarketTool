'''
Tests for classes in clean_analysis.py
'''
from DoltReader import DataReader
from clean_analysis import *
import pandas as pd
from dateutil.relativedelta import relativedelta
from pdb import set_trace as st
from matplotlib import pyplot as plt
from Features import *

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
test_basic_features: tests the log return feature & volatility
test_talib_feature: tests general talib feature class to generate talib features/target
    performs on SMA
test_bbands: test for bollinger bands feature
test_bbands_normalized: Gets normalized width and (close price - lower band)/width 
'''
test_top_liquidity_filter = 0 
test_price_filter = 0
test_advanced_stats_filter = 0
test_universe = 0
test_ta_lib = 0
test_basic_features = 0
test_talib_feature = 0
test_bbands = 0
test_bbands_normalized = 1

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
    ta_lib_sma = talib.SMA(df["close"], timeperiod = 30) #SMA FROM TA-LIB
    upper, middle, lower = talib.BBANDS(df.close) #Bollinger Bands from ta-lib
    fig = plt.figure(figsize = (9, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(df.date, df.close, c = "lime", label = "Close Price")
    ax1.plot(df.date, ta_lib_sma, label = "30 day SMA", c = "magenta", linestyle = ":")
    ta_lib_sma = talib.SMA(df["close"], timeperiod = 200) #SMA FROM TA-LIB
    ax1.plot(df.date, ta_lib_sma, label = "200 day SMA", c = "teal", linestyle = ":")
    ax1.plot(df.date, upper, label = "Upper BB", c = "orange", linestyle = "--")
    ax1.plot(df.date, lower, label = "Lower BB", c = "orange", linestyle = "--")
    ax1.plot(df.date, middle, label = "Middle BB", c = "goldenrod", linestyle = "--")
    ax1.grid(True)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($USD)")
    ax1.legend()
    ax1.set_title("Testing talib functions on F data")
    plt.show()

if test_basic_features == True:
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    df.reset_index(drop = True)
    feature = LogReturn(-5)
    feature_df = feature.retrieve_data(df)
    print(df.head(10))
    print(feature.is_feature)
    target = LogReturn(5)
    target_df = target.retrieve_data(df)
    print(df.tail(10))
    print(target.is_feature)
    volatility_target = Volatility(5)
    volatility_target.retrieve_data(df)
    print(df.tail(10))
    volatility_feature = Volatility(-5)
    volatility_feature.retrieve_data(df)
    print("getting sub df...")
    sub_df = df[(df.date >= pd.Timestamp(year = 2020, month = 1, day = 1)) & 
                (df.date <= pd.Timestamp(year = 2020, month = 1, day = 30)) &
                (df.act_symbol == "PFE")
                ]
    print("DONE")
    fig = plt.figure(figsize = (10, 5))
    price_ax = fig.add_subplot(1, 2, 1)
    price_ax2 = fig.add_subplot(1, 2, 2)
    log_ret_ax = price_ax.twinx()
    volatility_ax = price_ax2.twinx()
    log_ret_ax.plot(sub_df.date, sub_df.log_ret_5d_F, label = "5 day lagging log returns")
    log_ret_ax.plot(sub_df.date, sub_df.log_ret_5d_T, label = "5 day future log returns")
    volatility_ax.plot(sub_df.date, sub_df.volatility_5d_T, label = "Volatility 5 day future returns")
    volatility_ax.plot(sub_df.date, sub_df.volatility_5d_F, label = "Lagging 5 day volatility")
    for ax in [price_ax, price_ax2]:
        ax.plot(sub_df.date, sub_df.close, label = "close price", c = "magenta")
    price_ax.set_xlabel("Date")
    price_ax.set_ylabel("Price (USD)")
    log_ret_ax.set_ylabel("Log ret")
    fig.suptitle("Testing 1st 2 classes in Feature()")
    price_ax.set_title("Log ret")
    price_ax2.set_title("Volatility")
    for ax1, ax2 in (zip([price_ax, log_ret_ax], [price_ax2, volatility_ax])):
        for ax, loc in zip([ax1, ax2], ["upper-left", "upper-right"]):
            ax.legend()
    print(sub_df)
    plt.tight_layout()
    plt.show()

if test_talib_feature == True:
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    df.reset_index(drop = True)
    #get 5 day lag 50 day SMA as a feature
    Feature = TALibVarTimePeriod(name = "SMA", num_days = -5, timeperiod = 50)
    #get 5 day lookahead 50 day SMA as a target
    Target = TALibVarTimePeriod(name = "SMA", num_days = 5, timeperiod = 50)
    for var in [Feature, Target]:
        var.retrieve_data(df)
    print("getting sub df...")
    sub_df = df[(df.date >= pd.Timestamp(year = 2020, month = 1, day = 1)) & 
                (df.date <= pd.Timestamp(year = 2020, month = 1, day = 30)) &
                (df.act_symbol == "PFE")
                ]
    print("DONE")
    #Feature should be lagging target by 11 days
    print(sub_df)

if test_bbands == True:
    print("Testing Bollinger Bands on Log Returns...")
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Instantiate the BollingerBands class
    # num_days = -1: We want a Feature (lagged by 1 day)
    # timeperiod = 20: Standard 20-day window
    bb = BollingerBands(num_days=-1, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # 2. Run calculation
    # This will create 'log_ret' if missing, then calc bands on it
    bb.retrieve_data(df)
    
    # 3. Visualization Setup
    # Pick a specific stock and date range to make the plot readable
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2020-06-01"
    
    mask = (df['act_symbol'] == symbol) & (df['date'] >= start_date) & (df['date'] <= end_date)
    sub_df = df.loc[mask].copy()
    
    # Get the dynamic name generated by the class (e.g., BBANDS_20_1d_T)
    base_name = bb.detailed_name
    col_upper = f"{base_name}_upper"
    col_middle = f"{base_name}_middle"
    col_lower = f"{base_name}_lower"

    print(f"Columns Generated: {col_upper}, {col_middle}, {col_lower}")
    print(sub_df[['date', 'log_ret', col_upper, col_lower]].tail())

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Price (Context)
    ax1.plot(sub_df['date'], sub_df['close'], color='white', label='Close Price')
    ax1.set_title(f"{symbol} Price Action")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Log Returns + Bands
    ax2.plot(sub_df['date'], sub_df['log_ret'], color='cyan', linewidth=1, label='Log Returns')
    
    # Plot Bands
    ax2.plot(sub_df['date'], sub_df[col_upper], color='orange', linestyle='--', label='Upper Band (2std)')
    ax2.plot(sub_df['date'], sub_df[col_lower], color='orange', linestyle='--', label='Lower Band (2std)')
    ax2.plot(sub_df['date'], sub_df[col_middle], color='yellow', linestyle=':', alpha=0.7, label='Middle Band (SMA)')
    
    ax2.set_title(f"Log Returns & Bollinger Bands ({bb.timeperiod} period)")
    ax2.set_ylabel("Log Return")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if test_bbands_normalized == True:
    print("Testing Normalized Bollinger Bands (%B and Bandwidth)...")
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Instantiate the Normalized BB Class
    # Feature: Lagged by 1 day (num_days=-1) so we know the state at Open
    # Timeperiod: 20 days
    bb_norm = BollingerBandsNormalized(num_days=-1, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # 2. Run Calculation
    # This generates two columns: ..._pct_b and ..._width
    bb_norm.retrieve_data(df)
    
    # 3. Setup Visualization Data
    symbol = "AAPL"
    # Select a volatile period to see the bands expand/contract
    start_date = "2020-02-01"
    end_date = "2020-05-01"
    
    mask = (df['act_symbol'] == symbol) & (df['date'] >= start_date) & (df['date'] <= end_date)
    sub_df = df.loc[mask].copy()
    
    # Construct Column Names
    base_name = bb_norm.detailed_name # e.g. BBANDS_Norm_20_1d_T
    col_pct_b = f"{base_name}_pct_b"
    col_width = f"{base_name}_width"

    print(f"Columns Generated: {col_pct_b}, {col_width}")
    print(sub_df[['date', 'close', col_pct_b, col_width]].head())

    # 4. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price
    ax1.plot(sub_df['date'], sub_df['close'], color='white', label='Close Price')
    ax1.set_title(f"{symbol} Price Action")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Plot 2: %B (The Oscillator)
    # %B = 1.0 means Price is at Upper Band
    # %B = 0.0 means Price is at Lower Band
    ax2.plot(sub_df['date'], sub_df[col_pct_b], color='cyan', label='%B (Position within Bands)')
    # Add reference lines
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Upper Band (1.0)')
    ax2.axhline(0.0, color='lime', linestyle='--', alpha=0.5, label='Lower Band (0.0)')
    ax2.axhline(0.5, color='white', linestyle=':', alpha=0.3, label='Middle (0.5)')
    ax2.set_ylabel("%B")
    ax2.set_ylim(-0.2, 1.2) # Give a little space above/below
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bandwidth (Volatility)
    # Spikes in bandwidth indicate high volatility
    ax3.plot(sub_df['date'], sub_df[col_width], color='magenta', label='Bandwidth (Volatility)')
    ax3.set_ylabel("Width")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"Normalized Bollinger Bands Feature Test ({bb_norm.timeperiod} period)")
    plt.tight_layout()
    plt.show()

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
#from FeatureEngine import *
from FeatureEngine import *
import math

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

#tests for clean_analysis.py
test_top_liquidity_filter = 0 
test_price_filter = 0
test_advanced_stats_filter = 0
test_universe = 0

#tests for Features.py
test_ta_lib = 0
test_basic_features = 0
test_talib_feature = 0
test_bbands = 0
test_bbands_normalized = 0
test_momentum = 0

#tests for FeatureEngine.py
test_feature_engine = 0
test_feature_engine2 = 0
test_liquidity = 0
test_MA_crossover = 0
test_hurst_autocorr = 0
test_BETA = 0 #test for pandas BETA feature
test_vwap_zscore = 0
test_vol_zscore = 0
test_adx_regime = 0
test_range_features = 0
test_gap_sigma = 0
test_sharpe_target = 0

#testing the model
test_model = 1

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

if test_momentum == True:
    print("\n========================================")
    print("Testing Momentum Feature (Log Price Trend)")
    print("========================================")
    
    # 1. Load Data
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Instantiate the Class
    # We request a 20-day Momentum, lagged by 5 days
    # Expected Logic: 
    #   1. Calculate 20-day change in Log Price (Trend)
    #   2. Shift 1 day (because it's a feature, must be known at Open)
    #   3. Shift 5 days (the requested lag)
    #   Total Shift = 6 days
    LAG_DAYS = -5
    TIMEPERIOD = 20
    
    mom_feature = TALibVarTimePeriod(name="momentum", 
                                     num_days=LAG_DAYS, 
                                     timeperiod=TIMEPERIOD, 
                                     train_on_log=True)
    
    # 3. Run Calculation
    print(f"Generating feature: {mom_feature.name} (Timeperiod: {TIMEPERIOD}, Lag: {LAG_DAYS})...")
    mom_feature.retrieve_data(df)
    
    col_name = mom_feature.detailed_name 
    print(f"Column created: {col_name}")

    # 4. Validation (Manual Calculation)
    symbol = "AAPL"
    # Filter for symbol and sort to ensure shifts work
    mask = df['act_symbol'] == symbol
    sub_df = df.loc[mask].copy().sort_values("date")
    
    # A: Calculate Log Price
    log_price = np.log(sub_df['close'])
    
    # B: Calculate Momentum (Log Price - Log Price t-20)
    # This represents the cumulative log return over the period
    manual_mom = log_price - log_price.shift(TIMEPERIOD)
    
    # C: Apply Shifts
    # Shift 1 for "Feature Availability" (standard in your class for num_days < 0)
    # Shift abs(LAG_DAYS) for the requested specific lag
    total_shift = 1 + abs(LAG_DAYS)
    manual_mom = manual_mom.shift(total_shift)
    
    # 5. Compare
    comparison = pd.DataFrame({
        'Date': sub_df['date'],
        'Class_Output': sub_df[col_name],
        'Manual_Calc': manual_mom,
        'Close_Price': sub_df['close']
    }).dropna().tail(10)
    
    print("\n--- Comparison Table (Last 10 rows) ---")
    print(comparison)
    
    # Check for equality (ignoring NaNs at the start)
    # We use the full series for the check, not just the tail
    valid_indices = ~np.isnan(manual_mom)
    are_close = np.allclose(sub_df.loc[valid_indices, col_name], 
                            manual_mom[valid_indices], 
                            equal_nan=True)
    
    if are_close:
        print("\n[SUCCESS] Class momentum matches manual Log Price momentum.")
        print(f"Verifies: ln(P_t) - ln(P_t-{TIMEPERIOD}) shifted by {total_shift} days.")
    else:
        print("\n[FAILURE] Values do not match.")
        print("Possible causes:")
        print("1. Class is using daily returns (acceleration) instead of log price.")
        print("2. Groupby/Shift logic is misaligned.")

if test_feature_engine == True:
    print("\n========================================")
    print("Testing Unified Feature Engine (Transforms in Requests)")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df['date'] >= "2020-01-01"].copy()
    except Exception as e:
        print(f"Data load error: {e}")
        exit()

    # 2. Define Requests with TRANSFORMS
    requests = []
    
    # A. Standard Features
    requests.append(FeatureRequest(name='SLOPE', params={'timeperiod': 20}, shift=-1, input_type='log_price', alias='trend_slope'))
    
    # B. Cross-Sectional Feature: Rank of Slope
    # Note: This will calculate the slope (base) internally, then rank it. 
    # The output column will be trend_slope_20d_F_RANK
    requests.append(FeatureRequest(name='SLOPE', params={'timeperiod': 20}, shift=-1, input_type='log_price', alias='trend_slope', transform='rank'))
    
    # C. Z-Score (Custom Stat)
    requests.append(FeatureRequest(name='ZSCORE', params={'timeperiod': 20}, shift=-1, input_type='log_ret', alias='vol_zscore'))
    
    # D. Velocity
    requests.append(FeatureRequest(name='RSI', params={'timeperiod': 14}, shift=-1, deriv_order=1, alias='RSI_velocity'))
    
    # E. Targets (Future Return)
    # 1. Binary Target (Positive/Negative return)
    requests.append(FeatureRequest(name='SUM', params={'timeperiod': 5}, shift=5, input_type='log_ret', alias='target_5d', transform='binary'))
    
    # 2. Regime Target (High/Low/Neutral)
    requests.append(FeatureRequest(name='SUM', params={'timeperiod': 5}, shift=5, input_type='log_ret', alias='target_5d', transform='regime', transform_params={'threshold': 0.02}))

    # 3. Run Engine (Everything happens here now)
    engine = FeatureEngine(requests)
    df = engine.compute(df)
    
    # 4. Verify Data
    print("\n--- Columns Created ---")
    # Helper to see what we made
    ignore = ['open','high','low','close','volume','log_close','log_high','log_low','log_ret','act_symbol', 'gap_size', 'log_volume']
    cols_created = [c for c in df.columns if c not in ignore]
    print(cols_created)
    
    # Check specific transformed columns
    print("\nSample Data (Slope Rank & Binary Target):")
    print(df[['date', 'act_symbol', 'trend_slope_20d_F_RANK', 'target_5d_5d_T_BINARY']].tail())
    
    # Pick a symbol
    symbol = "AAPL"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    
    # Filter Data (Last 1 Year of available data)
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)
    
    # Define columns to plot (exclude OHLC and intermediate calcs)
    feature_cols = [c for c in df.keys() if "F" in c.split("_")]
    
    # Calculate Rows/Cols
    num_feats = len(feature_cols)
    ncols = 3
    nrows = math.ceil(num_feats / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        
        # 1. Twin Axis for Price (Context)
        ax_price = ax.twinx()
        ax_price.plot(sub_df['date'], sub_df['close'], color='white', linewidth=1, label='Price')
        ax_price.set_yticks([]) # Hide price labels
        
        # 2. Color Logic
        color = 'cyan'           # Default Feature
        if "_T" in col or "target" in col: 
            color = 'magenta'    # Target
        elif "rank" in col: 
            color = 'lime'       # Rank/Cross-sectional
        elif "zscore" in col:
            color = 'yellow'     # Stats

        # 3. Plot Feature
        # Use simple line plot for most, scatter for binary targets
        if "binary" in col or "regime" in col:
            ax.step(sub_df['date'], sub_df[col], where='post', color=color, linewidth=1.5)
            ax.set_yticks([-1, 0, 1])
        else:
            ax.plot(sub_df['date'], sub_df[col], color=color, linewidth=1.5)
            # Add zero line for oscillators
            if "velocity" in col or "slope" in col or "zscore" in col:
                ax.axhline(0, color='white', linestyle=':', alpha=0.3)
        
        # 4. Styling
        ax.set_title(col, fontsize=10, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # X-Axis formatting
        if i >= num_feats - ncols:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax.set_xticklabels([])

    # Turn off empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"New Feature Analysis: {symbol}", fontsize=16, y=1.02)
    plt.show()

    st()

if test_feature_engine2 == True:
    print("\n========================================")
    print("Testing Unified Feature Engine + SPREAD_AR (Candlesticks)")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
        # Filter for a specific date range
        df = df[df['date'] >= "2020-01-01"].copy()
    except Exception as e:
        print(f"Data load error: {e}")
        exit()

    # 2. Define Requests
    requests = []
    
    # --- LIQUIDITY ---
    # Abdi-Ranaldo Spread (Corrected Math)
    requests.append(FeatureRequest(name='SPREAD_AR', params={'timeperiod': 20}, shift=-1, alias='spread'))
    
    # --- VOLATILITY ---
    requests.append(FeatureRequest(name='ZSCORE', params={'timeperiod': 20}, shift=-1, input_type='log_ret', alias='vol_zscore'))
    
    # --- TREND ---
    requests.append(FeatureRequest(name='SLOPE', params={'timeperiod': 20}, shift=-1, input_type='log_price', alias='trend_slope'))
    
    # --- TRANSFORMS ---
    requests.append(FeatureRequest(name='SPREAD_AR', params={'timeperiod': 20}, shift=-1, alias='spread', transform='rank'))
    
    # --- TARGETS ---
    requests.append(FeatureRequest(name='SUM', params={'timeperiod': 5}, shift=5, input_type='log_ret', alias='target_5d', transform='binary'))

    # 3. Run Engine
    print(f"Computing {len(requests)} features...")
    engine = FeatureEngine(requests)
    df = engine.compute(df)
    
    # 4. Verify Data
    ignore = ['open','high','low','close','volume','log_close','log_high','log_low','log_ret','act_symbol', 'gap_size', 'log_volume', 'CAL_MONTH', 'CAL_DOW', 'CAL_QUARTER', 'CAL_DAY']
    cols_created = [c for c in df.columns if c not in ignore]
    
    print("\n--- Features Created ---")
    print(cols_created)
    
    # =========================================================
    # 5. GRID PLOT VISUALIZATION (CANDLESTICK EDITION)
    # =========================================================
    print("\nGenerating Visualization with Candlesticks...")
    
    # Pick a symbol
    symbol = "AAPL"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    
    # Filter Data (Last 150 days for better candle visibility)
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(150)
    
    # Define columns to plot
    feature_cols = [key for key in sub_df.keys() if "F" in key.split("_")]
    
    # Layout Logic
    num_feats = len(feature_cols)
    ncols = 3
    nrows = math.ceil(num_feats / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        
        # --- 1. PLOT CANDLESTICKS (Background) ---
        ax_price = ax.twinx()
        
        # Split into Up and Down days
        up = sub_df[sub_df.close >= sub_df.open]
        down = sub_df[sub_df.close < sub_df.open]
        
        # Plot Wicks (High to Low)
        ax_price.vlines(up.date, up.low, up.high, color='dimgray', linewidth=0.8, alpha=0.6)
        ax_price.vlines(down.date, down.low, down.high, color='dimgray', linewidth=0.8, alpha=0.6)
        
        # Plot Bodies (Open to Close)
        # Note: Using bar width relative to days. 0.6 is a good standard width.
        ax_price.bar(up.date, up.close - up.open, bottom=up.open, width=0.6, color='green', alpha=0.3)
        ax_price.bar(down.date, down.close - down.open, bottom=down.open, width=0.6, color='red', alpha=0.3)
        
        # Hide Y-ticks for price to keep it clean (context only)
        ax_price.set_yticks([]) 
        
        # --- 2. PLOT FEATURE (Foreground) ---
        # Color Logic
        color = 'cyan'           # Default
        if "target" in col or "_T" in col: 
            color = 'magenta'    # Target
        elif "RANK" in col: 
            color = 'lime'       # Rank
        elif "spread" in col:
            color = 'orange'     # Liquidity
        elif "zscore" in col:
            color = 'yellow'     # Volatility

        # Plot Feature Line / Step
        if "binary" in col or "regime" in col:
            ax.step(sub_df['date'], sub_df[col], where='post', color=color, linewidth=2)
            ax.set_yticks([-1, 0, 1])
        else:
            ax.plot(sub_df['date'], sub_df[col], color=color, linewidth=1.5)
            # Add Zero Line
            if "zscore" in col or "slope" in col or "spread" in col:
                ax.axhline(0, color='white', linestyle=':', alpha=0.5)
        
        # Styling
        ax.set_title(col, fontsize=10, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # X-Axis labels only on bottom row
        if i >= num_feats - ncols:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            ax.set_xticklabels([])

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"Feature Analysis: {symbol} (Candlestick View)", fontsize=16, y=1.02)
    plt.show()

    st()

if test_liquidity == True:
    #get data
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])

    #compute features
    liquidity_feature = FeatureRequest(name='SPREAD_AR', params={'timeperiod': 1}, shift=-1, alias='liquidity')
    feature_engine = FeatureEngine([liquidity_feature])
    df = feature_engine.compute(df)

    #get AAPL data to show for the test
    symbol = "F"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)

    print(sub_df.keys())
    liquidity_key = [key for key in sub_df.keys() if "F" in key.split("_")][0]
    fig = plt.figure()
    feature_ax = fig.add_subplot(1, 1, 1)
    price_ax = feature_ax.twinx()
    feature_ax.plot(sub_df.date, sub_df[liquidity_key], c = "lime", label = "AR liquidity proxy")
    price_ax.plot(sub_df.date, sub_df.close, c = "magenta", label = "close price")
    feature_ax.set_xlabel("Date")
    price_ax.set_ylabel("Price (USD)")
    feature_ax.set_ylabel("Abdi-Ronaldo Liquidity proxy")
    price_ax.legend(loc = "upper right")
    feature_ax.legend(loc = "upper left")
    price_ax.set_title("AR Liquidity + Price On F over past year")
    plt.show()

if test_MA_crossover == True:
    #get data
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])

    #compute features
    liquidity_feature = FeatureRequest(name='MA_crossover', #moving average crossover
                                       params={'timeperiod': 50}, #50 day MA
                                       shift=-1, #make feature from prev day
                                       alias='MA_cross', #
                                       input_type = "raw")
    feature_engine = FeatureEngine([liquidity_feature])
    df = feature_engine.compute(df)

    #get F data to show for the test
    symbol = "F"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)
    st()
    print(sub_df.keys())
    liquidity_key = [key for key in sub_df.keys() if "F" in key.split("_")][0]
    fig = plt.figure()
    feature_ax = fig.add_subplot(1, 2, 1)
    price_ax = fig.add_subplot(1, 2, 2)
    feature_ax.plot(sub_df.date, sub_df[liquidity_key], c = "lime", label = "Crossover bool \n (1 = Above, 2 = Below)")
    price_ax.plot(sub_df.date, sub_df.close, c = "magenta", label = "close price")
    price_ax.plot(sub_df.date, sub_df.close.rolling(50).mean(), c = "teal", label = "50 day MA")
    feature_ax.set_xlabel("Date")
    price_ax.set_xlabel("Date")
    price_ax.set_ylabel("Price (USD)")
    feature_ax.set_ylabel("Above/Below")
    price_ax.legend(loc = "upper right")
    feature_ax.legend(loc = "upper left")
    price_ax.set_title("MA crossover on F over past year")
    plt.tight_layout()
    plt.show()

# ... inside clean_tests.py ...
if test_hurst_autocorr == True:
    print("\n========================================")
    print("Testing Persistence: Autocorrelation vs Hurst")
    print("========================================")

    # 1. Load Data
    print("Loading data...")
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    
    # --- SPEED OPTIMIZATION ---
    # Filter to just a few symbols for the test. 
    # This prevents the engine from calculating features for 5000+ stocks.
    test_symbols = ['TSLA', 'AAPL', 'SPY'] 
    df = df[df['act_symbol'].isin(test_symbols)].copy()
    print(f"Data filtered to {len(test_symbols)} symbols for testing.")
    
    # 2. Define Features
    requests = []
    
    # A. Autocorrelation (Short-term memory)
    # Calculated on Returns. 
    # High +Val = Trending, Negative = Mean Reverting (Choppy)
    requests.append(FeatureRequest(name='AUTOCORR', 
                                   params={'timeperiod': 20}, 
                                   shift=-1, 
                                   alias='auto_corr', 
                                   input_type='log_ret'))
    
    # B. Hurst Exponent (Long-term memory)
    # Calculated on Log Prices (function handles diff internally).
    # > 0.5 = Trending, < 0.5 = Mean Reverting, 0.5 = Random Walk
    requests.append(FeatureRequest(name='HURST', 
                                   params={'timeperiod': 100}, 
                                   shift=-1, 
                                   alias='hurst', 
                                   input_type='log_price'))

    # 3. Compute
    print("Computing features...")
    feature_engine = FeatureEngine(requests)
    df = feature_engine.compute(df)

    # 4. Filter for Visualization
    # We choose TSLA because it often shows distinct trending vs ranging regimes
    symbol = "TSLA"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
        
    # Get last 500 days to see the evolution
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(500)

    # Identify dynamic column names generated by the engine
    keys = sub_df.keys()
    try:
        ac_col = [k for k in keys if "auto_corr" in k][0]
        hurst_col = [k for k in keys if "hurst" in k][0]
    except IndexError:
        print("Error: Could not find feature columns. Check FeatureEngine names.")
        print("Available columns:", keys)
        exit()

    # 5. Plotting
    print(f"Generating plot for {symbol}...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price
    ax1.plot(sub_df['date'], sub_df['close'], color='white', linewidth=1, label='Price')
    ax1.set_title(f"Price Action: {symbol}")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Plot 2: Hurst Exponent (Long Term)
    ax2.plot(sub_df['date'], sub_df[hurst_col], color='cyan', linewidth=1.5, label='Hurst (100d)')
    
    # Reference lines for Hurst
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.8, label='Random Walk (0.5)')
    
    # Fill areas to highlight regimes
    # Cyan fill = Trending (Persistent)
    ax2.fill_between(sub_df['date'], 0.5, sub_df[hurst_col], 
                     where=(sub_df[hurst_col] > 0.5), color='cyan', alpha=0.1)
    # Pink fill = Mean Reverting (Anti-persistent)
    ax2.fill_between(sub_df['date'], 0.5, sub_df[hurst_col], 
                     where=(sub_df[hurst_col] < 0.5), color='magenta', alpha=0.1)
    
    ax2.set_title("Long-Term Memory: Hurst Exponent (Trend Strength)")
    ax2.set_ylabel("Hurst Index")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0.2, 0.8) # Typical range for Hurst

    # Plot 3: Autocorrelation (Short Term)
    ax3.plot(sub_df['date'], sub_df[ac_col], color='lime', linewidth=1, label='Autocorr (20d)')
    ax3.axhline(0.0, color='white', linestyle='--', alpha=0.5)
    
    # Visual aid for significant correlation
    ax3.axhline(0.2, color='white', linestyle=':', alpha=0.2)
    ax3.axhline(-0.2, color='white', linestyle=':', alpha=0.2)
    
    ax3.set_title("Short-Term Persistence: Serial Correlation of Returns")
    ax3.set_ylabel("Correlation")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    print("Test Complete.")

if test_BETA == True:
    df = pd.read_feather("Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    beta_feature = FeatureRequest(name='pandas_beta', #moving average crossover
                                       params={'timeperiod': 50}, #get medium term beta from the past 50 days
                                       shift=-1, #make feature from prev day
                                       alias='Beta')
    feature_engine = FeatureEngine([beta_feature])
    df = feature_engine.compute(df)
    symbol = "F"
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    mask = (df['act_symbol'] == symbol)
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)
    print(sub_df.keys())
    print(sub_df.head(10))
    beta_key = [key for key in sub_df.keys() if "F" in key.split("_")][0]
    fig = plt.figure(figsize = (12, 5))
    feature_ax = fig.add_subplot(1, 2, 1)
    price_ax = fig.add_subplot(1, 2, 2, sharex = feature_ax)
    ref_price_ax = price_ax.twinx()
    feature_ax.plot(sub_df.date, sub_df[beta_key], c = "lime", label = "beta")
    correlation_ax = feature_ax.twinx()
    rolling_corr = sub_df['log_ret'].rolling(50).corr(sub_df['MKT_SPY_RET'])
    correlation_ax.plot(sub_df.date, rolling_corr, c = "magenta", label = "correlation")
    price_ax.plot(sub_df.date, sub_df.log_ret, c = "magenta", label = symbol)
    price_ax.plot(sub_df.date, sub_df.MKT_SPY_RET, label = "SPY")
    feature_ax.set_xlabel("Date")
    price_ax.set_xlabel("Date")
    price_ax.set_ylabel("Log ret")
    feature_ax.set_ylabel("Beta")
    price_ax.legend(loc = "upper right")
    feature_ax.legend(loc = "upper left")
    correlation_ax.legend(loc = "upper right")
    fig.suptitle("50 day Beta on F over the past year")
    plt.tight_layout()
    plt.show()
    st()

if test_vwap_zscore == True:
    print("\n========================================")
    print("Testing VWAP Z-Score Feature")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Compute Feature
    # We use shift=0 for visualization to align Price(t) with VWAP(t)
    TIMEPERIOD = 20
    vwap_req = FeatureRequest(name='VWAP_Z', 
                              params={'timeperiod': TIMEPERIOD}, 
                              shift=-1, 
                              alias='vwap_z')
    
    print(f"Computing VWAP Z-Score ({TIMEPERIOD}d)...")
    engine = FeatureEngine([vwap_req])
    df = engine.compute(df)

    # 3. Filter Data for Visualization
    symbol = "F" # Ford
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    
    mask = df['act_symbol'] == symbol
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)

    # 4. Manual VWAP Calculation (for overlay plot)
    # The engine returns Z-Score, but we want to plot the actual VWAP line too
    sub_df['pv'] = sub_df['close'] * sub_df['volume']
    sub_df['roll_vwap'] = sub_df['pv'].rolling(TIMEPERIOD).sum() / sub_df['volume'].rolling(TIMEPERIOD).sum()

    # 5. Plotting
    # Layout: 1 Row, 2 Columns, Share X
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # --- Plot 1: Price vs VWAP ---
    ax1.plot(sub_df.date, sub_df.close, color='magenta', linewidth=1.5, label='Close Price')
    ax1.plot(sub_df.date, sub_df.roll_vwap, color='cyan', linestyle='--', linewidth=1.5, label=f'{TIMEPERIOD}-Day VWAP')
    
    ax1.set_title(f"{symbol} Price vs VWAP")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: VWAP Z-Score ---
    # Find the feature column name dynamically
    z_col = [c for c in sub_df.columns if 'vwap_z' in c][0]
    
    ax2.plot(sub_df.date, sub_df[z_col], color='lime', linewidth=1.5, label='VWAP Z-Score')
    
    # Reference Lines
    ax2.axhline(2, color='red', linestyle=':', alpha=0.6, label='Overbought (+2)')
    ax2.axhline(-2, color='red', linestyle=':', alpha=0.6, label='Oversold (-2)')
    ax2.axhline(0, color='white', linestyle='-', alpha=0.3)
    
    ax2.set_title(f"Distance from VWAP (Std Devs)")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
    st()

if test_vol_zscore == True:
    print("\n========================================")
    print("Testing Volume Z-Score Feature")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Compute Feature
    # We use shift=-1 to create a feature available for tomorrow (Lagged)
    # input_type="raw" is required because our function expects 'volume'
    TIMEPERIOD = 20
    vol_req = FeatureRequest(name='VOL_ZSCORE', 
                             params={'timeperiod': TIMEPERIOD}, 
                             shift=-1, 
                             input_type='raw',
                             alias='vol_z')
    
    print(f"Computing Volume Z-Score ({TIMEPERIOD}d)...")
    engine = FeatureEngine([vol_req])
    df = engine.compute(df)

    # 3. Filter Data for Visualization
    symbol = "TSLA" # High volume volatility stock
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    
    mask = df['act_symbol'] == symbol
    sub_df = df.loc[mask].copy().sort_values('date').tail(252)

    # 4. Plotting
    # Layout: 3 Rows (Price, Raw Volume, Vol Z-Score)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- Plot 1: Price ---
    ax1.plot(sub_df.date, sub_df.close, color='white', linewidth=1.5, label='Close Price')
    ax1.set_title(f"{symbol} Price Action")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Raw Volume ---
    # Color bars based on price change (Green/Red)
    colors = np.where(sub_df.close >= sub_df.open, 'green', 'red')
    ax2.bar(sub_df.date, sub_df.volume, color=colors, alpha=0.5, label='Volume')
    # Add Moving Average
    vol_ma = sub_df['volume'].rolling(TIMEPERIOD).mean()
    ax2.plot(sub_df.date, vol_ma, color='yellow', linestyle='--', linewidth=1, label=f'{TIMEPERIOD}d MA')
    
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)

    # --- Plot 3: Volume Z-Score ---
    # Find the feature column name dynamically
    z_col = [c for c in sub_df.columns if 'vol_z' in c][0]
    
    ax3.plot(sub_df.date, sub_df[z_col], color='cyan', linewidth=1.5, label='Vol Z-Score (Log Space)')
    
    # Reference Lines
    ax3.axhline(2, color='red', linestyle=':', alpha=0.6, label='Unusual Activity (+2 std)')
    ax3.axhline(0, color='white', linestyle='-', alpha=0.3, label='Average')
    ax3.axhline(-2, color='gray', linestyle=':', alpha=0.6)
    
    # Highlight Spikes
    ax3.fill_between(sub_df.date, 2, sub_df[z_col], where=(sub_df[z_col] > 2), color='red', alpha=0.3)
    
    ax3.set_title(f"Volume Trend (Z-Score of Log Volume)")
    ax3.set_ylabel("Z-Score")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
    st()

if test_adx_regime == True:
    print("\n========================================")
    print("Testing ADX Regime Classification (1 / 0 / -1)")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Compute Feature
    # We use shift=0 to see the "Current" regime for visualization
    # In training, you might use shift=-1 (Future Regime) or shift=0 (Current State)
    req = FeatureRequest(name='ADX_REGIME', 
                         params={'timeperiod': 14, 'threshold': 20}, # Lower threshold captures more trends
                         shift=0, 
                         alias='regime')
    
    engine = FeatureEngine([req])
    df = engine.compute(df)

    # 3. Filter for Visualization
    symbol = "SPY" # Good example of distinct trends vs chop
    if symbol not in df['act_symbol'].values:
        symbol = df['act_symbol'].unique()[0]
    
    mask = df['act_symbol'] == symbol
    sub_df = df.loc[mask].copy().sort_values('date').tail(1000)

    # 4. Plotting
    col_name = [c for c in sub_df.columns if 'regime' in c][0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # -- Plot 1: Price colored by Regime --
    # Hack to color line segments: Scatter plot is easier, but let's do background fill
    ax1.plot(sub_df.date, sub_df.close, color='white', linewidth=1, label='Close Price')
    
    # Fill backgrounds
    # Green for Bull (1), Red for Bear (-1), Gray/None for Chop (0)
    y_min, y_max = ax1.get_ylim()
    
    # Bull Zones
    ax1.fill_between(sub_df.date, sub_df.close.min(), sub_df.close.max(), 
                     where=(sub_df[col_name] == 1), color='green', alpha=0.15, label='Bull Trend')
    
    # Bear Zones
    ax1.fill_between(sub_df.date, sub_df.close.min(), sub_df.close.max(), 
                     where=(sub_df[col_name] == -1), color='red', alpha=0.15, label='Bear Trend')

    ax1.set_title(f"{symbol} Market Regimes (ADX Filter)")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # -- Plot 2: The Regime Signal --
    ax2.step(sub_df.date, sub_df[col_name], where='post', linewidth=2, color='cyan')
    
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Bear (-1)', 'Chop (0)', 'Bull (1)'])
    ax2.grid(True, alpha=0.2)
    ax2.set_title("Regime Signal")

    plt.tight_layout()
    plt.show()
    st()

if test_range_features == True:
    print("\n========================================")
    print("Testing Range Features (Efficiency & Relative)")
    print("========================================")
    
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Data error: {e}")
        exit()

    # Define Requests
    reqs = [
        # Is the candle solid (1) or a doji (0)?
        FeatureRequest(name='RANGE_EFFICIENCY', params={}, shift=0, alias='candle_eff', input_type = "raw"),
        # Is the range expanding (>1) or compressing (<1)?
        FeatureRequest(name='REL_RANGE', params={'timeperiod': 20}, shift=0, alias='vol_ratio')
    ]
    
    engine = FeatureEngine(reqs)
    df = engine.compute(df)

    # Filter for a stock (TSLA is good for volatility examples)
    symbol = "TSLA"
    if symbol not in df['act_symbol'].values: symbol = df['act_symbol'].unique()[0]
    
    sub_df = df[df['act_symbol'] == symbol].copy().sort_values('date').tail(100)
    
    # Extract columns
    eff_col = [c for c in sub_df.columns if 'candle_eff' in c][0]
    vol_col = [c for c in sub_df.columns if 'vol_ratio' in c][0]

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- Plot 1: Price ---
    ax1.plot(sub_df.date, sub_df.close, color='white', label='Price')
    ax1.set_title(f"{symbol} Price Action")
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Range Efficiency (Body / Range) ---
    # High values = Solid Candles (Trendiness)
    # Low values = Wicks (Indecision)
    ax2.plot(sub_df.date, sub_df[eff_col], color='cyan', linewidth=1)
    ax2.axhline(0.5, color='white', linestyle=':', alpha=0.5)
    
    # Highlight Indecision (Dojis)
    ax2.fill_between(sub_df.date, 0, sub_df[eff_col], 
                     where=(sub_df[eff_col] < 0.2), color='red', alpha=0.5, label='Indecision (Doji)')
    # Highlight Conviction (Marubozu)
    ax2.fill_between(sub_df.date, 0, sub_df[eff_col], 
                     where=(sub_df[eff_col] > 0.8), color='lime', alpha=0.5, label='Conviction')

    ax2.set_title("Candle Efficiency (1.0 = Solid Body, 0.0 = All Wick)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)

    # --- Plot 3: Relative Range (Expansion/Compression) ---
    ax3.bar(sub_df.date, sub_df[vol_col], color='orange', alpha=0.6)
    ax3.axhline(1.0, color='white', linestyle='-', alpha=0.5, label='Average')
    ax3.axhline(0.5, color='cyan', linestyle='--', alpha=0.8, label='Squeeze (<0.5)')
    ax3.axhline(2.0, color='red', linestyle='--', alpha=0.8, label='Explosion (>2.0)')
    
    ax3.set_title("Relative Range (Volatility Ratio)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
    st()

if test_gap_sigma == True:
    print("\n========================================")
    print("Testing Gap Sigma (Side-by-Side View)")
    print("========================================")
    
    # 1. Load Data
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Data error: {e}")
        exit()

    # 2. Define Request
    # CRITICAL: We must use input_type='raw' because 'open' price is required
    # and standard log_price data usually drops the open.
    req = FeatureRequest(name='GAP_SIGMA', 
                         params={'timeperiod': 14}, 
                         shift=-1, 
                         alias='gap_sig',
                         input_type='raw')
    
    engine = FeatureEngine([req])
    df = engine.compute(df)

    # 3. Filter for a Volatile Stock (e.g., TSLA or NVDA)
    symbol = "TSLA"
    if symbol not in df['act_symbol'].values: 
        symbol = df['act_symbol'].unique()[0]
    
    # Get last 100 days for clear candle visualization
    sub_df = df[df['act_symbol'] == symbol].copy().sort_values('date').tail(100)
    
    # Identify the specific column name created
    gap_col = [c for c in sub_df.columns if 'gap_sig' in c][0]

    # 4. Plotting (1 Row, 2 Columns)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharex=True)

    # --- PLOT 1 (Left): Candlesticks ---
    # Split Up/Down days
    up = sub_df[sub_df.close >= sub_df.open]
    down = sub_df[sub_df.close < sub_df.open]
    
    # Plot Wicks
    ax1.vlines(up.date, up.low, up.high, color='dimgray', linewidth=0.8)
    ax1.vlines(down.date, down.low, down.high, color='dimgray', linewidth=0.8)
    
    # Plot Bodies
    # width=0.6 days
    ax1.bar(up.date, up.close - up.open, bottom=up.open, color='green', width=0.6, alpha=0.6)
    ax1.bar(down.date, down.close - down.open, bottom=down.open, color='red', width=0.6, alpha=0.6)
    
    ax1.set_title(f"{symbol} Price Action")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.2)

    # --- PLOT 2 (Right): Gap Sigma ---
    # Color logic: Green for Gap Up, Red for Gap Down
    colors = np.where(sub_df[gap_col] >= 0, 'lime', 'magenta')
    
    ax2.bar(sub_df.date, sub_df[gap_col], color=colors, alpha=0.7, width=0.6)
    
    # Reference Lines (Significant Gaps)
    ax2.axhline(0, color='white', linewidth=0.5)
    ax2.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Extreme (+2 ATR)')
    ax2.axhline(-2.0, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(1.0, color='white', linestyle=':', alpha=0.3)
    ax2.axhline(-1.0, color='white', linestyle=':', alpha=0.3)
    
    ax2.set_title("Gap Magnitude (Normalized by Volatility)")
    ax2.set_ylabel("Sigma (Gap / ATR)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
    st()

if test_sharpe_target == True:
    print("Testing Forward Sharpe Target")
    # ... load data ...
    try:
        df = pd.read_feather("Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Data error: {e}")
        exit()

    # Request
    req = FeatureRequest(name='TARGET_SHARPE', params={'timeperiod': 10}, shift=0, alias='sharpe_10d')
    engine = FeatureEngine([req])
    df = engine.compute(df)
    
    # Plot
    symbol = "TSLA"
    sub_df = df[df['act_symbol'] == symbol].tail(200)
    col = [c for c in sub_df.columns if 'sharpe' in c][0]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot Price
    ax1.plot(sub_df.date, sub_df.close, color='gray', alpha=0.5)
    
    # Plot Sharpe Target
    # Green = High Quality Up Trend
    # Red = High Quality Down Trend
    ax2.plot(sub_df.date, sub_df[col], color='blue', linewidth=1)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    # Highlight "Good Trades" (Sharpe > 1 or < -1)
    ax2.fill_between(sub_df.date, 0, sub_df[col], where=(sub_df[col] > 1), color='green', alpha=0.3)
    ax2.fill_between(sub_df.date, 0, sub_df[col], where=(sub_df[col] < -1), color='red', alpha=0.3)
    
    plt.title("Forward Sharpe Ratio (The 'Quality' Target)")
    plt.show()

if test_model == True:
    #get a universe for training the model
    print("GETTING UNIVERSE...")
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

    #define model
    model = Model(universe_data)

    #define features & add
    volatility = FeatureRequest(name='VOL_ZSCORE', 
                                         params={'timeperiod': 20}, 
                                         shift=-1, 
                                         input_type='raw',
                                         alias='vol_z')
    liquidity = FeatureRequest(name='SPREAD_AR', params={'timeperiod': 20}, shift=-1, alias='liquidity', transform='rank')
    autocorrelation = FeatureRequest(name='AUTOCORR', 
                                   params={'timeperiod': 20}, 
                                   shift=-1, 
                                   alias='auto_corr', 
                                   input_type='log_ret')
    z_score = FeatureRequest(name='ZSCORE', params={'timeperiod': 20}, shift=-1, input_type='log_ret', alias='vol_zscore')
    model.add_features([volatility, liquidity, autocorrelation, z_score])

    #define target & add
    target = FeatureRequest(name='SUM', params={'timeperiod': 5}, shift=1, input_type='log_ret', alias='target_5d', transform='binary')
    model.add_target(target)
    #split data for training
    model.split_data(cutoffs = [0.7, 0.85, 1])
    model.train_model(save_name = "clean_model_test")

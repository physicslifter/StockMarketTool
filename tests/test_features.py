'''
Demonstrates all features are working in FeatureEngine.py
'''
# This tells Python: "Look inside the 'scripts' folder for my imports!"

from DoltReader import DataReader
from clean_analysis import *
import pandas as pd
from dateutil.relativedelta import relativedelta
from pdb import set_trace as st
from matplotlib import pyplot as plt
from FeatureEngine import *
import math

plt.style.use('dark_background')

#=====================================
#Flags

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
test_cross_sectional_z = 0
test_rate_features = 1
test_cross_sectional_features = 0

#testing the model
test_model = 0

#=====================================
#useful functions
def get_ohlcv(year, month):
    #gets data preceding a specific month, useful for testing filters
    df = pd.read_feather("../Data/all_ohlcv.feather")
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

if test_ta_lib == True:
    import talib
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
    df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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
        df = pd.read_feather("../Data/all_ohlcv.feather")
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

if test_cross_sectional_z == True:
    # 1. Load Data
    try:
        df = pd.read_feather("../Data/all_ohlcv.feather")
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"Data error: {e}")
        exit()
    
    stocks_to_get = ["AAPL", "TSLA", "PFE", "AMZN", "NFLX", "F", "STLA"]
    df = df[df.act_symbol.isin(stocks_to_get)]

    # 2. Setup requests: 
    # We request the raw forward return AND the cs_zscored forward return
    reqs =[
        FeatureRequest(name='FWD_LOG_RET', shift=1), # Raw baseline
        FeatureRequest(name='FWD_LOG_RET', shift=1, transform='cs_zscore') # New Transform
    ]
    
    # 3. Compute
    engine = FeatureEngine(reqs)
    df_out = engine.compute(df)
    
    # 4. Filter for a specific date to plot the cross-section
    # We pick index 5, ensuring we have forward returns available
    sample_date = df_out['date'].unique()[5]
    df_day = df_out[df_out['date'] == sample_date].copy()
    
    
    target_cols = [key for key in df_out.keys() if "T" in key.split("_")]
    raw_col = target_cols[0]
    zscore_col = target_cols[1]
    
    # Show the math works
    mean_raw = df_day[raw_col].mean()
    std_raw = df_day[raw_col].std()
    print(f"--- Cross Section Stats for {pd.to_datetime(sample_date).date()} ---")
    print(f"Raw Mean: {mean_raw:.4f} | Raw Std: {std_raw:.4f}")
    print(f"Z-Score Mean: {df_day[zscore_col].mean():.4f} | Z-Score Std: {df_day[zscore_col].std():.4f}\n")
    
    # 5. Plotting
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Raw Returns
    bars1 = ax1.bar(df_day['act_symbol'], df_day[raw_col], color='dodgerblue')
    ax1.axhline(mean_raw, color='red', linestyle='--', label=f'Mean ({mean_raw:.4f})')
    ax1.set_title("Raw Forward Returns (1d)")
    ax1.set_ylabel("Log Return")
    ax1.legend()
    
    # Plot 2: CS Z-Score Returns
    # Map colors: green if > 0, red if < 0
    colors =['lime' if val > 0 else 'tomato' for val in df_day[zscore_col]]
    bars2 = ax2.bar(df_day['act_symbol'], df_day[zscore_col], color=colors)
    ax2.axhline(0, color='white', linestyle='--', label='Mean (0.0)')
    ax2.set_title("Cross-Sectional Z-Score Targets")
    ax2.set_ylabel("Z-Score (Standard Deviations)")
    ax2.legend()
    
    plt.suptitle(f"Cross-Sectional Standardization Test for {pd.to_datetime(sample_date).date()}", fontsize=14)
    plt.tight_layout()
    plt.show()

if test_rate_features == True:
    print("--- Starting Rate Feature Test ---")
    
    #get data
    df = pd.read_feather("../Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])
    sample_price_df = df[df.act_symbol == "AAPL"]

    # 1. Define the Rate Requests
    # We will test all 3 calculations: Spread, Butterfly, and Velocity
    rate_requests =[
        RateFeatureRequest(name='SPREAD', term1='10_year', term2='2_year'),
        RateFeatureRequest(name='BUTTERFLY', term1='10_year', term2='2_year', term3='3_month'),
        RateFeatureRequest(name='VELOCITY', term1='10_year', timeperiod=5)
    ]
    
    # 2. Run the Engine
    engine = FeatureEngine(rate_requests)
    
    # Make a copy of the sample data so we don't mutate your original workspace
    test_df = sample_price_df.copy()
    test_df = engine.compute(test_df)
    
    # 3. Define Expected Column Names based on the RateFeatureRequest logic
    col_spread = 'RATE_SPR_10_year_2_year_F'
    col_fly = 'RATE_FLY_10_year_2_year_3_month_F'
    col_vel = 'RATE_VEL_10_year_5d_F'
    
    # We need to temporarily merge the raw rates just to plot them side-by-side
    raw_rates = pd.read_csv("../Data/treasury_rates.csv")
    raw_rates['date'] = pd.to_datetime(raw_rates['date'])
    plot_df = pd.merge(test_df, raw_rates[['date', '10_year', '2_year', '3_month']], on='date', how='left')
    
    # ---------------------------------------------------------
    # 4. ASSERTIONS: Ensure no unexpected missing values
    # ---------------------------------------------------------
    print("\nRunning Assertions...")
    
    # Spread and Butterfly should have ZERO NaNs (since you forward-filled the raw data)
    assert plot_df[col_spread].isna().sum() == 0, f"Error: Found NaNs in {col_spread}!"
    assert plot_df[col_fly].isna().sum() == 0, f"Error: Found NaNs in {col_fly}!"
    
    # Velocity should have EXACTLY 'timeperiod' NaNs at the very beginning of the dataset 
    # (Because you can't look back 5 days on day 1). Everything after day 5 should be non-NaN.
    vel_nans = plot_df[col_vel].isna().sum()
    expected_nans = 5 # matching timeperiod=5
    assert vel_nans == expected_nans, f"Expected {expected_nans} NaNs in Velocity due to lookback, found {vel_nans}!"
    
    print("✅ All assertions passed! No unexpected missing values found.")
    
    # ---------------------------------------------------------
    # 5. PLOTTING
    # ---------------------------------------------------------
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot 1: Raw Rates
    axes[0].plot(plot_df['date'], plot_df['10_year'], label='10 Year', color='navy')
    axes[0].plot(plot_df['date'], plot_df['2_year'], label='2 Year', color='darkorange')
    axes[0].plot(plot_df['date'], plot_df['3_month'], label='3 Month', color='green')
    axes[0].set_title('Raw Treasury Rates')
    axes[0].set_ylabel('Yield (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Spread
    axes[1].plot(plot_df['date'], plot_df[col_spread], label='10Y - 2Y Spread', color='purple')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Rate Spread (10Y - 2Y)')
    axes[1].set_ylabel('Spread (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Butterfly
    axes[2].plot(plot_df['date'], plot_df[col_fly], label='Butterfly (10Y + 3M - 2*2Y)', color='teal')
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Rate Butterfly')
    axes[2].set_ylabel('Curvature')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Velocity
    axes[3].plot(plot_df['date'], plot_df[col_vel], label='10Y Velocity (5-Day Change)', color='firebrick')
    axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('Rate Velocity (5 Trading Day Change of 10Y)')
    axes[3].set_ylabel('Absolute Change')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if test_cross_sectional_features == True:
    print("--- Starting Cross-Sectional & Macro-Derived Feature Test ---")

    # Load full universe (breadth needs multiple stocks)
    df = pd.read_feather("../Data/Universes/7.feather")
    df["date"] = pd.to_datetime(df["date"])

    # Define requests
    cs_requests = [
        FeatureRequest("VIX_ZSCORE", shift=-1, params={"timeperiod": 20}, input_type="raw", alias="vix_zscore"),
        FeatureRequest("DIFFUSION", shift=-1, params={"timeperiod": 21}, input_type="raw", alias="diffusion_21"),
        FeatureRequest("DIFFUSION", shift=-1, params={"timeperiod": 63}, input_type="raw", alias="diffusion_63"),
        FeatureRequest("AD_SPREAD", shift=-1, params={"timeperiod": 5}, input_type="raw", alias="ad_spread"),
        FeatureRequest("CS_DISPERSION", shift=-1, params={"timeperiod": 21}, input_type="raw", alias="cs_dispersion"),
        FeatureRequest("HERFINDAHL", shift=-1, params={"timeperiod": 21}, input_type="raw", alias="herfindahl"),
    ]

    engine = FeatureEngine(cs_requests)
    result_df = engine.compute(df)

    # Extract one row per date (these are market-wide features, same for all stocks)
    col_vix_z = [c for c in result_df.columns if "vix_zscore" in c.lower()][0]
    col_diff_21 = [c for c in result_df.columns if "diffusion_21" in c.lower()][0]
    col_diff_63 = [c for c in result_df.columns if "diffusion_63" in c.lower()][0]
    col_ad = [c for c in result_df.columns if "ad_spread" in c.lower()][0]
    col_disp = [c for c in result_df.columns if "cs_dispersion" in c.lower()][0]
    col_herf = [c for c in result_df.columns if "herfindahl" in c.lower()][0]

    feature_cols = [col_vix_z, col_diff_21, col_diff_63, col_ad, col_disp, col_herf]
    daily = result_df.groupby("date")[feature_cols].first().sort_index()

    print(f"\nDetected columns: {feature_cols}")
    print(f"Date range: {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"\nNaN counts:")
    for col in feature_cols:
        n_nan = daily[col].isna().sum()
        n_total = len(daily)
        print(f"  {col}: {n_nan}/{n_total} ({n_nan/n_total:.1%})")

    # ── Assertions ──
    print("\nRunning Assertions...")

    warmup = 70
    post_warmup = daily.iloc[warmup:]

    for col in feature_cols:
        nan_pct = post_warmup[col].isna().mean()
        threshold = 0.10 if "herfindahl" in col.lower() else 0.05
        assert nan_pct < threshold, f"Error: {col} has {nan_pct:.1%} NaNs after warmup (expected < {threshold:.0%})"

    for col in [col_diff_21, col_diff_63]:
        valid = daily[col].dropna()
        assert valid.min() >= 0, f"Error: {col} has values below 0"
        assert valid.max() <= 1, f"Error: {col} has values above 1"

    valid_ad = daily[col_ad].dropna()
    assert valid_ad.min() >= -1, f"Error: {col_ad} has values below -1"
    assert valid_ad.max() <= 1, f"Error: {col_ad} has values above 1"

    assert daily[col_disp].dropna().min() >= 0, f"Error: {col_disp} has negative values"
    assert daily[col_herf].dropna().min() >= 0, f"Error: {col_herf} has negative values"

    print("✅ All assertions passed!")

    # ── Plots (3 rows x 2 columns) ──
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex=True)

    # Row 1, Left: VIX Z-Score
    axes[0, 0].plot(daily.index, daily[col_vix_z], color="firebrick", linewidth=1)
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0, 0].axhline(2, color="red", linestyle=":", alpha=0.5, label="z=2")
    axes[0, 0].axhline(-2, color="green", linestyle=":", alpha=0.5, label="z=-2")
    axes[0, 0].set_title("VIX Z-Score (20d)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Right: Diffusion Index
    axes[0, 1].plot(daily.index, daily[col_diff_21], color="blue", linewidth=1, label="21d")
    axes[0, 1].plot(daily.index, daily[col_diff_63], color="navy", linewidth=1, alpha=0.7, label="63d")
    axes[0, 1].axhline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].axhline(0.7, color="green", linestyle=":", alpha=0.3, label="70% (broad rally)")
    axes[0, 1].axhline(0.3, color="red", linestyle=":", alpha=0.3, label="30% (broad selloff)")
    axes[0, 1].set_title("Diffusion Index")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Row 2, Left: Advance-Decline Spread
    axes[1, 0].plot(daily.index, daily[col_ad], color="purple", linewidth=1)
    axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Advance-Decline Spread (5d)")
    axes[1, 0].grid(True, alpha=0.3)

    # Row 2, Right: Dispersion
    axes[1, 1].plot(daily.index, daily[col_disp], color="teal", linewidth=1)
    axes[1, 1].set_title("Cross-Sectional Dispersion (21d)")
    axes[1, 1].grid(True, alpha=0.3)

    # Row 3, Left: Herfindahl
    axes[2, 0].plot(daily.index, daily[col_herf], color="darkorange", linewidth=1)
    axes[2, 0].set_title("Herfindahl Concentration (21d)")
    axes[2, 0].grid(True, alpha=0.3)

    # Row 3, Right: Diffusion Divergence
    diff_spread = daily[col_diff_21] - daily[col_diff_63]
    axes[2, 1].fill_between(daily.index, diff_spread, 0,
                            where=diff_spread >= 0, color="green", alpha=0.4, label="Short > Long")
    axes[2, 1].fill_between(daily.index, diff_spread, 0,
                            where=diff_spread < 0, color="red", alpha=0.4, label="Short < Long")
    axes[2, 1].axhline(0, color="black", linewidth=1)
    axes[2, 1].set_title("Breadth Divergence (21d - 63d)")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    #print a test to the terminal before showing the plot
    # ── Manual Verification ──
    print("\n--- Manual Spot-Check ---")
    
    # Pick a date well past warmup
    check_date = daily.index[200]
    print(f"Checking date: {check_date.date()}")
    
    # Get raw data for this date and surrounding window
    raw_df = df.sort_values(["act_symbol", "date"])
    
    # 1. DIFFUSION (21d): manually compute % of stocks with positive 21-day log return
    date_mask = raw_df["date"] == check_date
    date_df = raw_df[date_mask]
    
    positive_count = 0
    total_count = 0
    for ticker in date_df["act_symbol"].unique():
        ticker_data = raw_df[raw_df["act_symbol"] == ticker].set_index("date")["close"]
        if check_date in ticker_data.index:
            current = ticker_data.loc[check_date]
            # Find price 21 trading days ago
            past_prices = ticker_data[ticker_data.index < check_date].tail(21)
            if len(past_prices) >= 21:
                past_price = past_prices.iloc[0]
                log_ret = np.log(current / past_price)
                total_count += 1
                if log_ret > 0:
                    positive_count += 1
    
    manual_diffusion = positive_count / total_count if total_count > 0 else np.nan
    engine_diffusion = daily.loc[check_date, col_diff_21]
    diff_err = abs(manual_diffusion - engine_diffusion)
    print(f"  Diffusion 21d:  manual={manual_diffusion:.6f}  engine={engine_diffusion:.6f}  diff={diff_err:.6f}")
    assert diff_err < 0.01, f"Diffusion mismatch: {diff_err:.6f}"
    
    # 2. AD_SPREAD: manually compute (advances - declines) / total for that date
    daily_rets = []
    for ticker in date_df["act_symbol"].unique():
        ticker_data = raw_df[raw_df["act_symbol"] == ticker].set_index("date")["close"]
        if check_date in ticker_data.index:
            prev_prices = ticker_data[ticker_data.index < check_date].tail(1)
            if len(prev_prices) == 1:
                log_ret = np.log(ticker_data.loc[check_date] / prev_prices.iloc[0])
                daily_rets.append(log_ret)
    
    advances = sum(1 for r in daily_rets if r > 0)
    declines = sum(1 for r in daily_rets if r < 0)
    raw_ad = (advances - declines) / len(daily_rets) if daily_rets else np.nan
    print(f"  AD raw (1 day): advances={advances}  declines={declines}  total={len(daily_rets)}  ratio={raw_ad:.6f}")
    print(f"  (AD_SPREAD is 5d smoothed, so exact match not expected — just sanity check sign & magnitude)")
    
    # 3. CS_DISPERSION: manually compute cross-sectional std of returns for that date
    manual_disp = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else np.nan

    plt.show()


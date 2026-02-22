import pandas as pd
import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

#for viewing exceptions
import traceback
import sys

#0: helper functions
def calc_zscore(real, timeperiod=20):
    """Standard Z-Score"""
    s = pd.Series(real)
    roll_mean = s.rolling(window=timeperiod).mean()
    roll_std = s.rolling(window=timeperiod).std()
    return ((s - roll_mean) / roll_std).values

def calc_volume_zscore(volume, timeperiod=20):
    """
    Calculates Z-Score of Volume (using Log transformation).
    Standardizes volume trends: (LogVol - MeanLogVol) / StdLogVol
    """
    s_vol = pd.Series(volume)
    
    # 1. Log transform 
    # (Phase 1 of engine already sets <=0 volume to NaN, so this is safe)
    log_vol = np.log(s_vol)
    
    # 2. Rolling Stats on Log Volume
    roll_mean = log_vol.rolling(window=timeperiod).mean()
    roll_std = log_vol.rolling(window=timeperiod).std(ddof=1)
    
    # 3. Z-Score
    return ((log_vol - roll_mean) / roll_std).values

def calc_abdi_ranaldo(high, low, close, timeperiod=20):
    """
    Abdi-Ranaldo (2017) Spread Estimator.
    Corrected for 'Mean of Logs' definition and simplified shifting.
    """
    # 1. Safety for Logs
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    close = np.where(close <= 0, np.nan, close)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        # 2. Log Prices
        log_high = np.log(high)
        log_low = np.log(low)
        log_close = np.log(close)
        
        # 3. Mid-Range (Eta) - CORRECTED DEFINITION
        # Paper: "mean of the daily high and low log-prices"
        log_eta = (log_high + log_low) / 2.0
        
        # 4. Calculate Terms based on "What do we know TODAY (Day T)?"
        # The formula links Close(t) with Eta(t) and Eta(t+1).
        # To get a value valid for NOW, we use:
        # Close(t-1) with Eta(t-1) and Eta(t)
        
        # Convert to Series for easy shifting
        s_log_close = pd.Series(log_close)
        s_log_eta = pd.Series(log_eta)
        
        # Yesterday's Close and Yesterday's Eta
        prev_log_close = s_log_close.shift(1)
        prev_log_eta = s_log_eta.shift(1)
        
        # Term 1: (Close_prev - Eta_prev)
        term1 = prev_log_close - prev_log_eta
        
        # Term 2: (Close_prev - Eta_curr)
        term2 = prev_log_close - s_log_eta
        
        # 5. Daily Estimator
        # The paper uses 2 * sqrt(Mean). 
        # We can do sqrt(4 * Mean), so we multiply by 4 here.
        daily_prod = 4 * term1 * term2
        
        # 6. Rolling Expectation (Mean)
        roll_mean = daily_prod.rolling(window=timeperiod).mean()
        
        # 7. Final Calculation: S = sqrt(max(0, mean))
        # We use max(0, x) because variance cannot be negative, 
        # though the estimator can be noisy.
        return np.sqrt(np.maximum(0, roll_mean)).values
    
def moving_avg_crossover(close, timeperiod):
    close = np.where(close <= 0, np.nan, close)
    close = pd.Series(close)
    MA = close.rolling(timeperiod).mean()
    signal = (close >= MA).astype(int)
    return signal.values

def calc_hurst(real, timeperiod=100):
    """
    Vectorized Hurst Exponent (R/S Analysis).
    """
    # 1. Prepare Data
    real = np.array(real)
    
    # Safety check
    if len(real) < timeperiod + 1:
        return np.full(len(real), np.nan)

    # 2. Calculate Log Returns (Diff of Log Prices)
    # The Hurst exponent analyzes the persistence of the *changes* in price
    rets = np.diff(real) # Length becomes N-1
    
    # 3. Create Sliding Windows (Vectorized)
    # This creates a view: shape (num_windows, timeperiod)
    # We use timeperiod as the window size
    try:
        windows = sliding_window_view(rets, window_shape=timeperiod)
    except AttributeError:
        # Fallback for older NumPy versions (< 1.20)
        shape = (rets.shape[0] - timeperiod + 1, timeperiod)
        strides = (rets.strides[0], rets.strides[0])
        windows = np.lib.stride_tricks.as_strided(rets, shape=shape, strides=strides)

    # 4. Perform R/S Analysis on all windows simultaneously (Axis 1)
    
    # Mean of each window
    means = np.mean(windows, axis=1, keepdims=True)
    
    # Mean-Centered series (Y)
    y = windows - means
    
    # Cumulative Deviate series (Z)
    z = np.cumsum(y, axis=1)
    
    # Range (R)
    r = np.max(z, axis=1) - np.min(z, axis=1)
    
    # Standard Deviation (S)
    s = np.std(windows, axis=1, ddof=1)
    
    # Avoid division by zero
    s = np.where(s == 0, 1e-9, s)
    
    # R/S Ratio
    rs = r / s
    
    # Hurst Estimate
    # H = log(R/S) / log(n)
    h = np.log(rs) / np.log(timeperiod)
    
    # 5. Pad the result to match original length
    # We lost 1 point from diff() and (timeperiod-1) points from windowing
    # Total NaN padding needed at the start = timeperiod
    pad = np.full(timeperiod, np.nan)
    
    return np.concatenate((pad, h))

def calc_autocorr(real, timeperiod=20):
    """
    Rolling Lag-1 Autocorrelation.
    Measures persistence: Positive = Trend, Negative = Mean Reversion.
    """
    s = pd.Series(real)
    # Correlation between Series and its Lag-1 over the window
    return s.rolling(window=timeperiod).corr(s.shift(1)).values

#BETA FUNCTION using pandas
def calc_beta(real, market, timeperiod=20):
    """
    Rolling Beta = Covariance(Asset, Market) / Variance(Market)
    """
    # 1. Convert to Series
    s_real = pd.Series(real)
    s_market = pd.Series(market)
    
    # 2. Rolling Covariance (Asset vs Market)
    # The rolling window aligns indices implicitly (0 to N)
    cov = s_real.rolling(window=timeperiod).cov(s_market)
    
    # 3. Rolling Variance (Market)
    var = s_market.rolling(window=timeperiod).var()
    
    # 4. Beta calculation
    # Handle division by zero if variance is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = cov / var
    
    # Fill infinite values with NaN if market variance was 0
    beta = beta.replace([np.inf, -np.inf], np.nan)
    
    return beta.values

#Gives # of standard deviations the VWAP is from the close
def calc_vwap_zscore(close, volume, timeperiod=20):
    """
    Calculates the Z-Score distance from Rolling VWAP.
    Formula: (Close - VWAP) / StdDev(Close)
    """
    # 1. Input Safety
    # Ensure inputs are Series for rolling operations
    s_close = pd.Series(close)
    s_volume = pd.Series(volume)
    
    # Handle NaN or 0 volume to prevent division errors
    s_volume = s_volume.replace(0, np.nan)

    # 2. Calculate VWAP Components
    # PV = Price * Volume
    pv = s_close * s_volume
    # 3. Rolling Sums (The "V" and "W" in VWAP)
    roll_pv = pv.rolling(window=timeperiod).sum()
    roll_vol = s_volume.rolling(window=timeperiod).sum()
    
    # 4. Calculate Rolling VWAP
    vwap = roll_pv / roll_vol
    
    # 5. Calculate Volatility (Standard Deviation of Price)
    # We use this to normalize the distance (Z-score)
    roll_std = s_close.rolling(window=timeperiod).std(ddof=1)
    
    # 6. Calculate Z-Score
    # (Price - VWAP) / Volatility
    z_score = (s_close - vwap) / roll_std
    
    # Fill any initial NaNs (result of rolling window)
    return z_score.values
    
def calc_adx_regime(high, low, close, timeperiod=14, threshold=25):
    """
    Returns:
     1 = Bull Trend (Strong ADX + Positive Direction)
     0 = Chop / Range (Weak ADX)
    -1 = Bear Trend (Strong ADX + Negative Direction)
    """
    # 1. Calculate Component Indicators
    adx = talib.ADX(high, low, close, timeperiod=timeperiod)
    pdi = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
    mdi = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)
    
    # 2. Vectorized Logic
    # Default to 0 (Chop)
    regime = np.zeros_like(adx)
    
    # Identify Trending Indices (ADX > Threshold)
    trending = (adx > threshold)
    
    # Identify Bullish: Trending AND (PDI > MDI)
    bull_mask = trending & (pdi > mdi)
    regime[bull_mask] = 1
    
    # Identify Bearish: Trending AND (MDI > PDI)
    bear_mask = trending & (mdi > pdi)
    regime[bear_mask] = -1
    
    return regime

def calc_range_efficiency(open_p, high, low, close, timeperiod=1):
    """
    Body-to-Range Ratio (Candle Efficiency).
    1.0 = All body (Trend)
    0.0 = All wick (Indecision)
    """
    s_open = pd.Series(open_p)
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    s_close = pd.Series(close)
    
    # Calculate Range and Body
    rng = s_high - s_low
    body = (s_close - s_open).abs()
    
    # Avoid Division by Zero (if High == Low, ratio is 0 or 1 depending on logic, usually 1 if flat)
    # We replace 0 range with NaN or a small epsilon
    rng = rng.replace(0, np.nan) 
    
    return (body / rng).values

def calc_relative_range(high, low, timeperiod=20):
    """
    Current Range / Average Range of last N days.
    """
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    
    # Current Range
    rng = s_high - s_low
    
    # Average Range (ATR is technically different because of gaps, 
    # but simple mean of range is often cleaner for this specific feature)
    avg_rng = rng.rolling(window=timeperiod).mean()
    
    return (rng / avg_rng).values

def calc_gap_sigma(open_p, high, low, close, timeperiod=14):
    """
    Calculates Gap Size normalized by Volatility (ATR).
    Formula: (Open - PrevClose) / PrevATR
    """
    # 1. Calculate ATR (Volatility)
    # Note: We compute ATR on the raw arrays first
    atr = talib.ATR(high, low, close, timeperiod=timeperiod)
    
    # 2. Convert to Series for shifting
    s_atr = pd.Series(atr)
    s_open = pd.Series(open_p)
    s_close = pd.Series(close)
    
    # 3. Get Context (Yesterday's Data)
    # We compare the gap to Yesterday's Volatility, not Today's.
    prev_atr = s_atr.shift(1)
    prev_close = s_close.shift(1)
    
    # 4. Calculate Gap
    raw_gap = s_open - prev_close
    
    # 5. Normalize
    # Handle division by zero/nan if ATR is missing
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = raw_gap / prev_atr
        
    return sigma.values

def calc_forward_sharpe(close, timeperiod=5):
    """
    Calculates the Forward Sharpe Ratio (Target).
    Formula: Future_Return / Future_Volatility
    """
    # 1. Log Returns
    s_close = pd.Series(close)
    log_ret = np.log(s_close / s_close.shift(1))
    
    # 2. Rolling Forward Window
    # We use a trick: Reverse the series, calculate rolling, then reverse back.
    # This allows us to get the "Next 5 days" statistics aligned with "Today".
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=timeperiod)
    
    # Sum of returns over next N days
    fwd_ret = log_ret.rolling(window=indexer).sum()
    
    # Std Dev of returns over next N days
    fwd_std = log_ret.rolling(window=indexer).std()
    
    # 3. Calculate Sharpe
    # Add epsilon to prevent division by zero
    sharpe = fwd_ret / (fwd_std + 1e-6)
    
    return sharpe.values


# ==========================================
# 1. THE REGISTRY
# ==========================================
FEATURE_REGISTRY = {
    # --- TA-LIB INDICATORS ---
    'MOM':      {'type': 'talib', 'fn': talib.MOM,      'outputs': ['real']},
    'ROC':      {'type': 'talib', 'fn': talib.ROC,      'outputs': ['real']},
    'RSI':      {'type': 'talib', 'fn': talib.RSI,      'outputs': ['real']},
    'MACD':     {'type': 'talib', 'fn': talib.MACD,     'outputs': ['macd', 'macdsignal', 'macdhist']},
    'SLOPE':    {'type': 'talib', 'fn': talib.LINEARREG_SLOPE, 'outputs': ['real']},
    'ANGLE':    {'type': 'talib', 'fn': talib.LINEARREG_ANGLE, 'outputs': ['real']},
    'TSF':      {'type': 'talib', 'fn': talib.TSF,      'outputs': ['real']}, 
    'SMA':      {'type': 'talib', 'fn': talib.SMA,      'outputs': ['real']},
    'EMA':      {'type': 'talib', 'fn': talib.EMA,      'outputs': ['real']},
    'ATR':      {'type': 'talib', 'fn': talib.ATR,      'outputs': ['real'], 'inputs': ['high', 'low', 'close']},
    'NATR':     {'type': 'talib', 'fn': talib.NATR,     'outputs': ['real'], 'inputs': ['high', 'low', 'close']},
    'BBANDS':   {'type': 'talib', 'fn': talib.BBANDS,   'outputs': ['upper', 'middle', 'lower']},
    'ADX':      {'type': 'talib', 'fn': talib.ADX,      'outputs': ['real'], 'inputs': ['high', 'low', 'close']},
    'STOCH':    {'type': 'talib', 'fn': talib.STOCH,    'outputs': ['slowk', 'slowd'], 'inputs': ['high', 'low', 'close']},
    'BETA':     {'type': 'talib', 'fn': talib.BETA,     'outputs': ['real'], 'inputs': ['real0', 'real1']},

    # --- ROLLING STATS ---
    'VOLATILITY': {'type': 'rolling', 'fn': 'std',      'inputs': ['real']},
    'MEAN':       {'type': 'rolling', 'fn': 'mean',     'inputs': ['real']},
    'SUM':        {'type': 'rolling', 'fn': 'sum',      'inputs': ['real']},
    'SKEW':       {'type': 'rolling', 'fn': 'skew',     'inputs': ['real']},
    'KURT':       {'type': 'rolling', 'fn': 'kurt',     'inputs': ['real']},
    
    # --- CALENDAR ---
    'MONTH':      {'type': 'calendar', 'attr': 'month'},
    'DAY':        {'type': 'calendar', 'attr': 'day'},
    'DOW':        {'type': 'calendar', 'attr': 'dayofweek'},
    'QUARTER':    {'type': 'calendar', 'attr': 'quarter'},

    #=======================
    #Custom stats
    #=======================
    'ZSCORE': {
        'type': 'custom_stat', 
        'fn': calc_zscore,       # Pass the function directly
        'inputs': ['real'], 
        'outputs': ['real']
    },

    #liquidity proxy
    'SPREAD_AR': {
        'type': 'custom_stat', 
        'fn': calc_abdi_ranaldo,
        'inputs': ['high', 'low', 'close'],
        'outputs': ['real'] # This will be the estimated spread %
    },

    #Moving avg crossovers
    "MA_crossover": {
        "type": "custom_stat",
        "fn": moving_avg_crossover,
        "inputs": ["close"],
        "outputs": ["real"]
    },

    #hurst exponent
    'HURST': {
        'type': 'custom_stat',
        'fn': calc_hurst,
        'inputs': ['real'],   # Pass log_close (the engine default)
        'outputs': ['real']
    },

    #autocorrelation
    'AUTOCORR': {
        'type': 'custom_stat',
        'fn': calc_autocorr,
        'inputs': ['real'],
        'outputs': ['real']
    },

    'pandas_beta': {
        'type': 'custom_stat', 
        'fn': calc_beta,     # Use the new pandas function
        'inputs': ['real'],  # 'real' identifies the asset returns; market is grabbed in _worker
        'outputs': ['real']
    },

    'VWAP_Z': {
        'type': 'custom_stat', 
        'fn': calc_vwap_zscore,
        'inputs': ['close', 'volume'], # Requires raw data
        'outputs': ['real']
    },

    'VOL_ZSCORE': {
        'type': 'custom_stat', 
        'fn': calc_volume_zscore,
        'inputs': ['volume'],   # Requires raw volume column
        'outputs': ['real']
    },

    'ADX_REGIME': {
        'type': 'custom_stat',
        'fn': calc_adx_regime,
        'inputs': ['high', 'low', 'close'],
        'outputs': ['real']
    },
    
    'RANGE_EFFICIENCY': {
        'type': 'custom_stat',
        'fn': calc_range_efficiency,
        'inputs': ['open', 'high', 'low', 'close'],
        'outputs': ['real']
    },
    
    'REL_RANGE': {
        'type': 'custom_stat',
        'fn': calc_relative_range,
        'inputs': ['high', 'low'],
        'outputs': ['real']
    },

    'GAP_SIGMA': {
        'type': 'custom_stat',
        'fn': calc_gap_sigma,
        # We need Open for the Gap, and High/Low/Close for ATR
        'inputs': ['open', 'high', 'low', 'close'], 
        'outputs': ['real']
    },

    'TARGET_SHARPE': {
        'type': 'custom_stat',
        'fn': calc_forward_sharpe,
        'inputs': ['close'], # We calculate returns internally
        'outputs': ['real']
    },
}

# ==========================================
# 2. THE REQUEST OBJECT
# ==========================================
class FeatureRequest:
    def __init__(self, name, params=None, shift=0, input_type='log_price', 
                 market_ref=None, alias=None, deriv_order=0, 
                 transform=None, transform_params=None):
        """
        transform: 'rank', 'demean', 'binary', 'regime'
        transform_params: dict, e.g. {'threshold': 0.02} for targets
        """
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' not found in Registry")
            
        self.name = name
        self.params = params if params else {}
        self.shift = shift
        self.input_type = input_type
        if name in ["MA_crossover", "VWAP_Z"]:
            self.input_type = "raw"
        self.market_ref = market_ref
        if name in ["BETA", "pandas_beta"]:
            self.input_type = "log_ret"
            self.market_ref = "SPY" if type(self.market_ref) == type(None) else self.market_ref
        self.deriv_order = deriv_order
        self.transform = transform
        self.transform_params = transform_params if transform_params else {}
        
        # --- Naming Logic ---
        if shift > 0: type_suffix = "T"
        elif shift < 0: type_suffix = "F"
        else: type_suffix = "C"
        
        period_str = f"{self.params.get('timeperiod', '')}d" if 'timeperiod' in self.params else ""
        
        deriv_suffix = ""
        if self.deriv_order == 1: deriv_suffix = "_VEL"
        elif self.deriv_order == 2: deriv_suffix = "_ACC"

        # 1. Generate the BASE name (The raw calculation)
        base_alias = alias if alias else name
        sep = "_" if period_str else ""
        
        if FEATURE_REGISTRY[name]['type'] == 'calendar':
            self.base_col_name = f"CAL_{name}"
        else:
            shift_str = f"{abs(shift)}d" if shift != 0 else "0d"
            if alias:
                self.base_col_name = f"{alias}{sep}{period_str}{deriv_suffix}_{type_suffix}"
            else:
                param_str = "_".join([str(v) for v in self.params.values()])
                self.base_col_name = f"{name}_{param_str}_{input_type}{deriv_suffix}_{shift_str}_{type_suffix}"

        # 2. Generate the FINAL name (Including transforms)
        self.col_name = self.base_col_name
        if self.transform:
            self.col_name += f"_{self.transform.upper()}"

# ==========================================
# 3. THE UNIFIED ENGINE
# ==========================================
class FeatureEngine:
    def __init__(self, feature_requests):
        self.requests = feature_requests

    def compute(self, df):
        # 0. Safety Sort
        if not df.attrs.get("is_sorted", False):
            df = df.sort_values(['act_symbol', 'date'])
            df.attrs["is_sorted"] = True
        
        # -------------------------------------------------------
        # PHASE 1: GLOBAL CALCULATIONS (Vectorized)
        # -------------------------------------------------------
        print("Phase 1: Computing Global & Calendar Features...")
        
        # A. Clean Data
        cols_to_check = [c for c in ['close', 'high', 'low', 'open', 'volume'] if c in df.columns]
        for col in cols_to_check:
            df.loc[df[col] <= 0, col] = np.nan

        # B. Transforms
        if 'log_close' not in df.columns: df['log_close'] = np.log(df['close'])
        if 'log_high' not in df.columns:  df['log_high'] = np.log(df['high'])
        if 'log_low' not in df.columns:   df['log_low'] = np.log(df['low'])
        if 'log_volume' not in df.columns and 'volume' in df.columns:
            df['log_volume'] = np.log(df['volume'])
        
        # C. Log Returns & Mask
        mask = df['act_symbol'] != df['act_symbol'].shift(1)
        
        if 'log_ret' not in df.columns:
            close_vals = df['close'].values
            prev_close = df['close'].shift(1).values
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ret = np.log(close_vals / prev_close)
            df['log_ret'] = log_ret
            df.loc[mask, 'log_ret'] = np.nan

        # Gap Size
        if 'gap_size' not in df.columns:
             df['gap_size'] = df['open'] - df['close'].shift(1)
             df.loc[mask, 'gap_size'] = np.nan

        # D. Calendar
        calendar_reqs = [r for r in self.requests if FEATURE_REGISTRY[r.name]['type'] == 'calendar']
        other_reqs    = [r for r in self.requests if FEATURE_REGISTRY[r.name]['type'] != 'calendar']
        
        for req in calendar_reqs:
            attr = FEATURE_REGISTRY[req.name]['attr']
            # Calendar features usually don't need grouped calc, direct assign
            df[req.base_col_name] = getattr(df['date'].dt, attr)
            # If there's a transform on a calendar feature (rare but possible), assign to col_name
            if req.transform is None:
                df[req.col_name] = df[req.base_col_name]

        #add a reference if we're doing a beta calculation
        beta_reqs = [r for r in self.requests if r.name in ['BETA', "pandas_beta"]]
        processed_refs = set()

        if beta_reqs:
            print("Processing Market References for Beta...")
            # 1. Create a temp normalized date for safer merging
            df['_merge_date'] = df['date'].dt.normalize()
            
            for req in beta_reqs:
                mkt_ticker = req.market_ref
                
                if mkt_ticker in processed_refs:
                    continue
                
                mkt_col_name = f'MKT_{mkt_ticker}_RET'
                
                if mkt_col_name in df.columns:
                    processed_refs.add(mkt_ticker)
                    continue

                if mkt_ticker not in df['act_symbol'].values:
                    print(f"  [WARNING] Market reference '{mkt_ticker}' not found in DataFrame! Beta will be NaN.")
                    # Continue allows the merge to create NaNs rather than crashing
                
                # 2. Extract market data using the normalized date
                # We extract specific columns: The match key (_merge_date) and the target (log_ret)
                mkt_mask = df['act_symbol'] == mkt_ticker
                mkt_df = df.loc[mkt_mask, ['_merge_date', 'log_ret']].copy()
                
                # Handle potential duplicates in market data (take the last close of the day)
                # This prevents row explosion if SPY has multiple entries per day
                mkt_df = mkt_df.drop_duplicates(subset=['_merge_date'], keep='last')
                
                mkt_df = mkt_df.rename(columns={'log_ret': mkt_col_name})
                
                # 3. Merge on the normalized date
                df = df.merge(mkt_df, on='_merge_date', how='left')
                processed_refs.add(mkt_ticker)
            
            # 4. Clean up
            df = df.drop(columns=['_merge_date'])
            
            # CRITICAL FIX: Re-sort dataframe. 
            df = df.sort_values(['act_symbol', 'date'])

        """
        old
        
        for req in beta_reqs:
            mkt_ticker = req.market_ref
            # Extract market data once
            mkt_df = df.loc[df['act_symbol'] == mkt_ticker, ['date', 'log_ret']]
            mkt_df = mkt_df.rename(columns={'log_ret': f'MKT_{mkt_ticker}_RET'})

            # Merge onto main dataframe
            df = df.merge(mkt_df, on='date', how='left')"""

        # -------------------------------------------------------
        # PHASE 2: GROUPBY LOOP (Time Series Features)
        # -------------------------------------------------------
        print(f"Phase 2: Computing {len(other_reqs)} Grouped Features...")
        
        if other_reqs:
            def _worker(group):
                # 1. Map data
                def get_vals(col):
                    return group[col].values.astype(float) if col in group else np.full(len(group), np.nan)

                data_map = {
                    'raw': {
                        'open': get_vals('open'), 'high': get_vals('high'), 
                        'low': get_vals('low'), 'close': get_vals('close'), 
                        'volume': get_vals('volume')
                    },
                    'log_price': {
                        'real': get_vals('log_close'), 'high': get_vals('log_high'), 
                        'low':  get_vals('log_low'),   'close': get_vals('log_close')
                    },
                    'log_ret': {'real': get_vals('log_ret'), 'close': get_vals('log_ret')},
                    'volume':  {'real': get_vals('volume')},
                    'log_volume': {'real': get_vals('log_volume')}
                }
                
                result_cols = {}

                for req in other_reqs:
                    config = FEATURE_REGISTRY[req.name]
                    ftype = config['type']
                    
                    # --- PREPARE INPUTS ---
                    args = []
                    if req.name in ['BETA', "pandas_beta"]:
                        args.append(data_map[req.input_type]['real'])
                        # Just grab the pre-merged column
                        mkt_col = f"MKT_{req.market_ref}_RET"
                        args.append(group[mkt_col].values)
                    elif 'inputs' in config and len(config['inputs']) > 1:
                        src = data_map['log_price'] if req.input_type == 'log_price' else data_map['raw']
                        for input_name in config['inputs']:
                            args.append(src[input_name])
                    else:
                        if req.input_type == "raw":
                            args.append(data_map[req.input_type][config["inputs"][0]])
                        else:
                            args.append(data_map[req.input_type]['real'])

                    # --- EXECUTE ---
                    try:
                        out = None
                        if ftype == 'talib':
                            out = config['fn'](*args, **req.params)

                        elif ftype == 'rolling':
                            s_in = pd.Series(args[0], index=group.index)
                            window = req.params.get('timeperiod', 10)
                            out = getattr(s_in.rolling(window=window), config['fn'])().values

                        elif ftype == 'custom_stat':
                            # Generalized Custom Logic
                            func = config['fn'] # This is now the Python function (calc_amihud, etc.)

                            # Extract timeperiod from params (default to 20 if missing)
                            tp = req.params.get('timeperiod', 20)

                            # Call the helper function
                            # *args expands to (close, volume) or (high, low, close) etc.
                            out = func(*args, timeperiod=tp)

                    except Exception as e:
                        # Helpful error printing if needed, or just silence
                        traceback.print_exception(e, file=sys.stdout)
                        print(f"Error in {req.name}: {e}")
                        out = [np.full(len(group), np.nan)] * len(config['outputs'])

                    # --- DERIVATIVES ---
                    if not isinstance(out, (list, tuple)): out = [out]
                    if req.deriv_order > 0:
                        processed_out = []
                        for out_arr in out:
                            diff_res = np.diff(out_arr, n=req.deriv_order, prepend=[np.nan]*req.deriv_order)
                            processed_out.append(diff_res)
                        out = processed_out

                    # --- STORE & SHIFT ---
                    for i, out_arr in enumerate(out):
                        # Handle multi-output indicators (like BBANDS)
                        suffix = f"_{config['outputs'][i]}" if len(out) > 1 else ""
                        
                        # Use base_col_name here to store raw values
                        fname = req.base_col_name + suffix
                        
                        s = pd.Series(out_arr, index=group.index)
                        
                        if req.shift == 0: pass # Current
                        elif req.shift < 0: s = s.shift(abs(req.shift)) # Lag
                        elif req.shift > 0: s = s.shift(-req.shift) # Target
                            
                        result_cols[fname] = s

                return pd.DataFrame(result_cols)

            # Execute GroupBy
            new_features = df.groupby('act_symbol').apply(_worker)
            if isinstance(new_features.index, pd.MultiIndex):
                new_features = new_features.reset_index(level=0, drop=True)
            
            df = pd.concat([df, new_features], axis=1)

        # -------------------------------------------------------
        # PHASE 3: TRANSFORMS (Cross-Sectional & Targets)
        # -------------------------------------------------------
        print("Phase 3: Computing Transforms & Targets...")
        
        for req in self.requests:
            if not req.transform:
                continue

            # Identify the columns created in Phase 2
            # Handle multi-output indicators
            config = FEATURE_REGISTRY[req.name]
            outputs = config.get('outputs', ['real'])
            
            for i, output_name in enumerate(outputs):
                suffix = f"_{output_name}" if len(outputs) > 1 else ""
                
                # Input Column (from Phase 2)
                base_col = req.base_col_name + suffix
                
                # Output Column (final name)
                final_col = req.col_name + suffix
                
                if base_col not in df.columns:
                    print(f"Warning: Base column {base_col} missing for transform.")
                    continue

                # --- APPLY TRANSFORMS ---
                if req.transform == 'rank':
                    df[final_col] = df.groupby('date')[base_col].rank(pct=True)
                    
                elif req.transform == 'demean':
                    means = df.groupby('date')[base_col].transform('mean')
                    df[final_col] = df[base_col] - means
                    
                elif req.transform == 'binary':
                    # e.g., Target > 0
                    thresh = req.transform_params.get('threshold', 0)
                    df[final_col] = np.where(df[base_col] > thresh, 1, 0)
                    df.loc[df[base_col].isna(), final_col] = np.nan
                    
                elif req.transform == 'regime':
                    # e.g., > 0.02 (1), < -0.02 (-1), else 0
                    thresh = req.transform_params.get('threshold', 0.02)
                    conditions = [
                        (df[base_col] > thresh), 
                        (df[base_col] < -thresh)
                    ]
                    choices = [1, -1]
                    df[final_col] = np.select(conditions, choices, default=0)
                    df.loc[df[base_col].isna(), final_col] = np.nan

        return df
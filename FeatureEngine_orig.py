import pandas as pd
import numpy as np
import talib

# ==========================================
# 1. THE REGISTRY
#    Maps string names to configuration logic
# ==========================================
FEATURE_REGISTRY = {
    # --- TA-LIB INDICATORS (Type: 'talib') ---
    'MOM':      {'type': 'talib', 'fn': talib.MOM,      'outputs': ['real']},
    'RSI':      {'type': 'talib', 'fn': talib.RSI,      'outputs': ['real']},
    'BETA':     {'type': 'talib', 'fn': talib.BETA,     'outputs': ['real'], 'inputs': ['real0', 'real1']},
    'ADX':      {'type': 'talib', 'fn': talib.ADX,      'outputs': ['real'], 'inputs': ['high', 'low', 'close']},
    'ATR':      {'type': 'talib', 'fn': talib.ATR,      'outputs': ['real'], 'inputs': ['high', 'low', 'close']},
    'BBANDS':   {'type': 'talib', 'fn': talib.BBANDS,   'outputs': ['upper', 'middle', 'lower']},
    'STOCH':    {'type': 'talib', 'fn': talib.STOCH,    'outputs': ['slowk', 'slowd'], 'inputs': ['high', 'low', 'close']},
    'MACD':     {'type': 'talib', 'fn': talib.MACD,     'outputs': ['macd', 'macdsignal', 'macdhist']},
    'SMA':      {'type': 'talib', 'fn': talib.SMA,      'outputs': ['real']},
    'EMA':      {'type': 'talib', 'fn': talib.EMA,      'outputs': ['real']},

    # --- NEW TA-LIB INDICATORS ---
    'ROC':      {'type': 'talib', 'fn': talib.ROC, 'outputs': ['real']},
    'SLOPE':    {'type': 'talib', 'fn': talib.LINEARREG_SLOPE, 'outputs': ['real']}, # For Trend Slope
    'ANGLE':    {'type': 'talib', 'fn': talib.LINEARREG_ANGLE, 'outputs': ['real']},
    'TSF':      {'type': 'talib', 'fn': talib.TSF, 'outputs': ['real']}, # Time Series Forecast
    'NATR':     {'type': 'talib', 'fn': talib.NATR, 'outputs': ['real'], 'inputs': ['high', 'low', 'close']}, # Normalized ATR (Volatility)

    # --- PANDAS ROLLING STATS (Type: 'rolling') ---
    'VOLATILITY': {'type': 'rolling', 'fn': 'std',  'inputs': ['real']},
    'MEAN':       {'type': 'rolling', 'fn': 'mean', 'inputs': ['real']},
    'SUM':        {'type': 'rolling', 'fn': 'sum',  'inputs': ['real']},

    # --- NEW ROLLING STATS ---
    'SKEW':     {'type': 'rolling', 'fn': 'skew', 'inputs': ['real']},
    'KURT':     {'type': 'rolling', 'fn': 'kurt', 'inputs': ['real']},
    'ZSCORE':   {'type': 'custom_stat', 'fn': 'zscore', 'inputs': ['real']}, # Requires custom logic below
    
    # --- CALENDAR FEATURES (Type: 'calendar') ---
    'MONTH':      {'type': 'calendar', 'attr': 'month'},
    'DAY':        {'type': 'calendar', 'attr': 'day'},
    'DOW':        {'type': 'calendar', 'attr': 'dayofweek'},
    'QUARTER':    {'type': 'calendar', 'attr': 'quarter'}
}



# ==========================================
# 2. THE REQUEST OBJECT
# ==========================================
class FeatureRequest:
    def __init__(self, name, params=None, shift=0, input_type='log_price', market_col=None, alias=None, deriv_order=0):
        """
        name:       Key in FEATURE_REGISTRY (e.g. 'RSI', 'ZSCORE')
        params:     Dict for function args (e.g. {'timeperiod': 14})
        shift:      Negative (Lag/Feature), Positive (Future/Target), 0 (Current)
        input_type: 'log_price', 'log_ret', 'raw', 'volume', 'log_volume'
        alias:      Custom name prefix.
        deriv_order: 0 = Value, 1 = Velocity (diff), 2 = Acceleration (diff of diff)
        """
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' not found in Registry")
            
        self.name = name
        self.params = params if params else {}
        self.shift = shift
        self.input_type = input_type
        self.market_col = market_col
        self.deriv_order = deriv_order
        
        # --- Naming Logic ---
        type_suffix = "F" if shift < 0 else "T"
        if shift == 0: type_suffix = "C" # Current
        
        # Determine Period String
        if 'timeperiod' in self.params:
            period_str = f"{self.params['timeperiod']}d"
        else:
            period_str = "" 

        # Add derivative suffix if needed (e.g., _VEL, _ACC)
        deriv_suffix = ""
        if self.deriv_order == 1: deriv_suffix = "_VEL"
        elif self.deriv_order == 2: deriv_suffix = "_ACC"

        if alias:
            # Format: {alias}_{period}_{deriv}_{suffix}
            sep = "_" if period_str else ""
            self.col_name = f"{alias}{sep}{period_str}{deriv_suffix}_{type_suffix}"
        else:
            # Format: {name}_{params}_{input}_{shift}_{deriv}_{suffix}
            if FEATURE_REGISTRY[name]['type'] == 'calendar':
                self.col_name = f"CAL_{name}"
            else:
                shift_str = f"{abs(shift)}d"
                param_str = "_".join([str(v) for v in self.params.values()])
                self.col_name = f"{name}_{param_str}_{input_type}{deriv_suffix}_{shift_str}_{type_suffix}"

# ==========================================
# 3. THE UNIFIED ENGINE
# ==========================================
class FeatureEngine:
    def __init__(self, feature_requests):
        self.requests = feature_requests

    def compute(self, df):
        # Ensure sorting for Time Series ops
        if not df.attrs.get("is_sorted", False):
            df = df.sort_values(['act_symbol', 'date'])
            df.attrs["is_sorted"] = True
        
        # -------------------------------------------------------
        # PHASE 1: GLOBAL CALCULATIONS (Vectorized)
        # -------------------------------------------------------
        print("Phase 1: Computing Global & Calendar Features...")
        
        # A. Clean Data (Prevent log(0) errors)
        cols_to_check = [c for c in ['close', 'high', 'low', 'open', 'volume'] if c in df.columns]
        for col in cols_to_check:
            df.loc[df[col] <= 0, col] = np.nan

        # B. Pre-calculate Transforms
        if 'log_close' not in df.columns: df['log_close'] = np.log(df['close'])
        if 'log_high' not in df.columns:  df['log_high'] = np.log(df['high'])
        if 'log_low' not in df.columns:   df['log_low'] = np.log(df['low'])
        
        # Calculate Log Volume (Add 1 to avoid log(0) if not cleaned above, though we cleaned <=0)
        if 'log_volume' not in df.columns and 'volume' in df.columns:
            df['log_volume'] = np.log(df['volume'])
        
        # C. Pre-calculate Log Returns
        if 'log_ret' not in df.columns:
            close_vals = df['close'].values
            prev_close = df['close'].shift(1).values
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ret = np.log(close_vals / prev_close)
            df['log_ret'] = log_ret
            # Mask crossover
            mask = df['act_symbol'] != df['act_symbol'].shift(1)
            df.loc[mask, 'log_ret'] = np.nan

        # Gap Size (Open - Prev Close)
        if 'gap_size' not in df.columns:
             df['gap_size'] = df['open'] - df['close'].shift(1)
             df.loc[mask, 'gap_size'] = np.nan

        # D. Process Calendar Requests
        calendar_reqs = [r for r in self.requests if FEATURE_REGISTRY[r.name]['type'] == 'calendar']
        other_reqs    = [r for r in self.requests if FEATURE_REGISTRY[r.name]['type'] != 'calendar']
        
        for req in calendar_reqs:
            attr = FEATURE_REGISTRY[req.name]['attr']
            df[req.col_name] = getattr(df['date'].dt, attr)

        # -------------------------------------------------------
        # PHASE 2: GROUPBY LOOP (TA-Lib, Rolling, Stats)
        # -------------------------------------------------------
        print(f"Phase 2: Computing {len(other_reqs)} Grouped Features...")
        
        if not other_reqs:
            return df

        def _worker(group):
            # 1. Map data for easy access
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
                
                # --- A. PREPARE INPUTS ---
                args = []
                
                # Handle Market Comparison (BETA)
                if req.name == 'BETA':
                    args.append(data_map[req.input_type]['real'])
                    if req.market_col in group:
                        args.append(group[req.market_col].values)
                    else:
                        args.append(np.full(len(group), np.nan))
                
                # Handle Multi-Input (ADX, ATR, STOCH)
                elif 'inputs' in config and len(config['inputs']) > 1:
                    src = data_map['raw'] # usually High/Low/Close are raw prices
                    # Some might need log prices, but TA-Lib usually expects OHLC
                    if req.input_type == 'log_price': src = data_map['log_price']
                    
                    for input_name in config['inputs']:
                        args.append(src[input_name])
                
                # Handle Single Input
                else:
                    args.append(data_map[req.input_type]['real'])

                # --- B. EXECUTE ---
                try:
                    out = None
                    
                    if ftype == 'talib':
                        out = config['fn'](*args, **req.params)
                        
                    elif ftype == 'rolling':
                        s_in = pd.Series(args[0], index=group.index)
                        window = req.params.get('timeperiod', 10)
                        # Pandas rolling functions (mean, std, skew, kurt, etc.)
                        out = getattr(s_in.rolling(window=window), config['fn'])().values
                    
                    elif ftype == 'custom_stat':
                        s_in = pd.Series(args[0], index=group.index)
                        window = req.params.get('timeperiod', 20)
                        
                        if config['fn'] == 'zscore':
                            # Z-Score = (x - mean) / std
                            roll_mean = s_in.rolling(window=window).mean()
                            roll_std = s_in.rolling(window=window).std()
                            out = ((s_in - roll_mean) / roll_std).values

                except Exception:
                    out = [np.full(len(group), np.nan)] * len(config['outputs'])

                # --- C. DERIVATIVES (Velocity/Acceleration) ---
                if not isinstance(out, (list, tuple)): out = [out]
                
                # If the user requested Velocity (diff) or Accel (diff of diff)
                if req.deriv_order > 0:
                    processed_out = []
                    for out_arr in out:
                        # n=deriv_order applies diff n times
                        # prepend=NaN to keep array length same
                        diff_res = np.diff(out_arr, n=req.deriv_order, prepend=[np.nan]*req.deriv_order)
                        processed_out.append(diff_res)
                    out = processed_out

                # --- D. STORE & SHIFT ---
                for i, out_arr in enumerate(out):
                    suffix = f"_{config['outputs'][i]}" if len(out) > 1 else ""
                    fname = req.col_name + suffix
                    
                    s = pd.Series(out_arr, index=group.index)
                    
                    # Shift Logic
                    if req.shift < 0:
                        # FEATURE: Shift 1 (Availability) + Lag
                        s = s.shift(1).shift(abs(req.shift) - 1 if abs(req.shift) > 0 else 0)
                    elif req.shift > 0:
                        # TARGET: Shift Backwards
                        s = s.shift(-req.shift)
                        
                    result_cols[fname] = s

            return pd.DataFrame(result_cols)

        # Execute GroupBy
        new_features = df.groupby('act_symbol').apply(_worker)
        
        # Align and Merge
        if isinstance(new_features.index, pd.MultiIndex):
            new_features = new_features.reset_index(level=0, drop=True)
            
        return pd.concat([df, new_features], axis=1)

    # -------------------------------------------------------
    # PHASE 3: CROSS-SECTIONAL FEATURES
    # -------------------------------------------------------
    def compute_cross_sectional(self, df, feature_col, new_col_name, method='rank'):
        """
        Computes features based on date groups (comparing stocks).
        method: 'rank' (percentile), 'demean' (value - daily_avg)
        """
        print(f"Phase 3: Computing Cross-Sectional {new_col_name}...")
        
        if feature_col not in df.columns:
            print(f"Warning: {feature_col} missing. Skipping cross-sectional.")
            return df
            
        if method == 'rank':
            # pct=True gives 0.0 to 1.0
            df[new_col_name] = df.groupby('date')[feature_col].rank(pct=True)
            
        elif method == 'demean':
            daily_means = df.groupby('date')[feature_col].transform('mean')
            df[new_col_name] = df[feature_col] - daily_means
            
        return df

    # -------------------------------------------------------
    # PHASE 4: CLASSIFICATION TARGETS
    # -------------------------------------------------------
    def compute_targets(self, df, target_col_prefix='target'):
        """
        Generates binary/class targets from existing FUTURE return columns.
        Assumes columns like 'log_ret_5d_T' exist.
        """
        print("Phase 4: Computing Classification Targets...")
        
        # Find future return columns (ending in _T and containing 'log_ret')
        future_cols = [c for c in df.columns if c.endswith('_T') and 'log_ret' in c]
        
        for col in future_cols:
            # Extract period (e.g., "5d" from "log_ret_5d_T")
            parts = col.split('_')
            period = "next"
            for p in parts:
                if 'd' in p and p[:-1].isdigit(): period = p
            
            # 1. Binary Direction (Up/Down)
            bin_name = f"{target_col_prefix}_binary_{period}"
            df[bin_name] = np.where(df[col] > 0, 1, 0)
            # Set to NaN where target is NaN
            df.loc[df[col].isna(), bin_name] = np.nan
            
            # 2. Multiclass Regime (Bear/Chop/Bull)
            # Threshold: 2% (approx 0.02 log ret)
            reg_name = f"{target_col_prefix}_regime_{period}"
            conditions = [
                (df[col] > 0.02),  # Bull
                (df[col] < -0.02)  # Bear
            ]
            choices = [1, -1]
            # Default 0 (Chop)
            df[reg_name] = np.select(conditions, choices, default=0)
            df.loc[df[col].isna(), reg_name] = np.nan
            
        return df
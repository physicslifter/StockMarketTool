import pandas as pd
import numpy as np
import talib

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
    'ZSCORE':     {'type': 'custom_stat', 'fn': 'zscore', 'inputs': ['real']},
    
    # --- CALENDAR ---
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
        shift: 
            0  = Current Feature (Uses Data available at Close of Day T).
           -1  = Lagged Feature (Uses Data available at Close of Day T-1).
            1  = Target (Uses Data from Day T+1).
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
        # T = Target (Future), F = Feature (Lag), C = Current (No Shift)
        if shift > 0: type_suffix = "T"
        elif shift < 0: type_suffix = "F"
        else: type_suffix = "C"
        
        # Determine Period String
        period_str = f"{self.params.get('timeperiod', '')}d" if 'timeperiod' in self.params else ""

        # Derivative suffix
        deriv_suffix = ""
        if self.deriv_order == 1: deriv_suffix = "_VEL"
        elif self.deriv_order == 2: deriv_suffix = "_ACC"

        if alias:
            sep = "_" if period_str else ""
            self.col_name = f"{alias}{sep}{period_str}{deriv_suffix}_{type_suffix}"
        else:
            if FEATURE_REGISTRY[name]['type'] == 'calendar':
                self.col_name = f"CAL_{name}"
            else:
                shift_str = f"{abs(shift)}d" if shift != 0 else "0d"
                param_str = "_".join([str(v) for v in self.params.values()])
                self.col_name = f"{name}_{param_str}_{input_type}{deriv_suffix}_{shift_str}_{type_suffix}"

# ==========================================
# 3. THE UNIFIED ENGINE
# ==========================================
class FeatureEngine:
    def __init__(self, feature_requests):
        self.requests = feature_requests

    def compute(self, df):
        # 0. Safety Sort
        # Crucial: Data MUST be sorted by Date to ensure shifts work chronologically
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
        # These are Row-Invariant (Row T depends only on Row T)
        if 'log_close' not in df.columns: df['log_close'] = np.log(df['close'])
        if 'log_high' not in df.columns:  df['log_high'] = np.log(df['high'])
        if 'log_low' not in df.columns:   df['log_low'] = np.log(df['low'])
        if 'log_volume' not in df.columns and 'volume' in df.columns:
            df['log_volume'] = np.log(df['volume'])
        
        # C. Pre-calculate Log Returns
        # LEAKAGE CHECK: This calculates log(Close_t / Close_t-1).
        # This value is known at the Close of Day T. (SAFE)
        if 'log_ret' not in df.columns:
            close_vals = df['close'].values
            prev_close = df['close'].shift(1).values
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ret = np.log(close_vals / prev_close)
            df['log_ret'] = log_ret
            # Mask crossover between symbols
            mask = df['act_symbol'] != df['act_symbol'].shift(1)
            df.loc[mask, 'log_ret'] = np.nan

        # Gap Size (Open_t - Close_t-1)
        # Known at Open of Day T. (SAFE)
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
        # PHASE 2: GROUPBY LOOP
        # -------------------------------------------------------
        print(f"Phase 2: Computing {len(other_reqs)} Grouped Features...")
        
        if not other_reqs:
            return df

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
                
                # --- A. PREPARE INPUTS ---
                args = []
                
                # Market Col (BETA) - Requires Market Data aligned on index
                if req.name == 'BETA':
                    args.append(data_map[req.input_type]['real'])
                    if req.market_col in group:
                        args.append(group[req.market_col].values)
                    else:
                        args.append(np.full(len(group), np.nan))
                
                elif 'inputs' in config and len(config['inputs']) > 1:
                    src = data_map['log_price'] if req.input_type == 'log_price' else data_map['raw']
                    for input_name in config['inputs']:
                        args.append(src[input_name])
                else:
                    args.append(data_map[req.input_type]['real'])

                # --- B. EXECUTE CALCULATION ---
                # All calculations here use data up to index T (rolling backwards).
                # Example: SMA(20) at index T uses prices T, T-1 ... T-19.
                # This is SAFE for post-market analysis.
                try:
                    out = None
                    if ftype == 'talib':
                        out = config['fn'](*args, **req.params)
                    elif ftype == 'rolling':
                        s_in = pd.Series(args[0], index=group.index)
                        window = req.params.get('timeperiod', 10)
                        # Pandas Rolling is NOT centered by default. It looks back. (SAFE)
                        out = getattr(s_in.rolling(window=window), config['fn'])().values
                    elif ftype == 'custom_stat':
                        s_in = pd.Series(args[0], index=group.index)
                        window = req.params.get('timeperiod', 20)
                        if config['fn'] == 'zscore':
                            roll_mean = s_in.rolling(window=window).mean()
                            roll_std = s_in.rolling(window=window).std()
                            out = ((s_in - roll_mean) / roll_std).values
                except Exception:
                    out = [np.full(len(group), np.nan)] * len(config['outputs'])

                # --- C. DERIVATIVES ---
                if not isinstance(out, (list, tuple)): out = [out]
                if req.deriv_order > 0:
                    processed_out = []
                    for out_arr in out:
                        # Diff uses current vs previous (SAFE)
                        diff_res = np.diff(out_arr, n=req.deriv_order, prepend=[np.nan]*req.deriv_order)
                        processed_out.append(diff_res)
                    out = processed_out

                # --- D. STORE & SHIFT (DATA LEAKAGE PREVENTION LOGIC) ---
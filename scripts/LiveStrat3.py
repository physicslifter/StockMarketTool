import pandas as pd
import numpy as np
import os
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from datetime import datetime

# Import your existing local modules
from FeatureEngine import *
from Portfolio import retrieve_model, HRPBacktest
from HRP import compute_hrp_weights

from pdb import set_trace as st

# =====================================================================
# NEW: LIVE YANG-ZHANG VOLATILITY ESTIMATOR
# =====================================================================
def get_live_yang_zhang(ohlc_df, window=20):
    """
    Computes the Yang-Zhang volatility across the universe.
    Returns the cross-sectional median annualized volatility.
    """
    df_O = ohlc_df.pivot(index="date", columns="act_symbol", values="open").ffill()
    df_H = ohlc_df.pivot(index="date", columns="act_symbol", values="high").ffill()
    df_L = ohlc_df.pivot(index="date", columns="act_symbol", values="low").ffill()
    df_C = ohlc_df.pivot(index="date", columns="act_symbol", values="close").ffill()

    log_ho = np.log(df_H / df_O)
    log_lo = np.log(df_L / df_O)
    log_co = np.log(df_C / df_O)
    log_oc = np.log(df_O / df_C.shift(1))

    rs_var = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))

    # Variances over the trailing window (we take the final row)
    overnight_var = log_oc.tail(window).var(ddof=1)
    open_close_var = log_co.tail(window).var(ddof=1)
    rs_var_mean = rs_var.tail(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    yz_var = overnight_var + (k * open_close_var) + ((1 - k) * rs_var_mean)
    yz_vol = np.sqrt(yz_var * 252)
    
    return yz_vol.median()


def bootstrap_cold_start(model_folder, universe_path, target_date, lookback_days, starting_capital, strategy_kwargs, scaling_kwargs, raw_data_path="../Data/all_ohlcv.feather"):
    """
    Solves the 'Cold Start' problem by generating a simulated equity curve 
    over the last N days using your most recent universe basket.
    """
    print(f"\n{'='*50}")
    print(f"BOOTSTRAPPING COLD START (Last {lookback_days} Days)")
    print(f"{'='*50}")
    
    target_date = pd.to_datetime(target_date)
    seed_start = target_date - pd.Timedelta(days=lookback_days)
    
    # 1. Retrieve Model
    lgb_model, info, dates, features, target = retrieve_model(model_folder)
    trained_features = lgb_model.feature_name()
    
    # 2. Get latest universe basket from local feather
    print(f"Loading universe from {universe_path}...")
    univ_df = pd.read_feather(universe_path)
    latest_univ_date = univ_df['date'].max()
    valid_tickers = univ_df[univ_df['date'] == latest_univ_date]['act_symbol'].unique().tolist()
    print(f"  -> Extracted {len(valid_tickers)} tickers from latest universe date: {latest_univ_date.date()}")
    
    # We use a slightly wider window to allow features (like 200d SMA) to warm up
    warmup_start = seed_start - pd.Timedelta(days=250)
    
    print(f"Fetching historical data for these tickers from {seed_start.date()} to {target_date.date()}...")
    dataset = ds.dataset(raw_data_path, format="feather")
    filter_cond = (ds.field('act_symbol').isin(valid_tickers)) & (ds.field('date') >= warmup_start) & (ds.field('date') <= target_date)
    raw_df = dataset.to_table(filter=filter_cond).to_pandas()
    
    hist_mask = raw_df[['date', 'act_symbol']].copy()
    hist_mask['date'] = pd.to_datetime(hist_mask['date']).astype('datetime64[ns]')
    
    # 3. Generate Features
    print("Computing historical features...")
    engine = FeatureEngine(feature_requests=features)
    hist_data = engine.compute(input_df=hist_mask, raw_data_path=raw_data_path)
    hist_data = hist_data.dropna(subset=trained_features).copy()
    
    # 4. Generate Predictions
    print("Generating historical predictions...")
    hist_data["pred_return"] = lgb_model.predict(hist_data[trained_features])
    hist_data["pred_zscore"] = hist_data.groupby("date")["pred_return"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # 5. Run HRPBacktest
    print("Running simulated backtest to build equity curve...")
    class DummyPortfolio:
        def __init__(self, df):
            self.data = df
            self.has_data = True
            
    dummy_port = DummyPortfolio(hist_data)
    
    # Map kwargs to ensure bootstrap math matches live math exactly
    bt_kwargs = strategy_kwargs.copy()
    bt_kwargs["volatility_type"] = scaling_kwargs.get("volatility_type", "simple")
    
    bt = HRPBacktest(portfolio=dummy_port, **bt_kwargs)
    
    # Apply dynamic scaling to the cold-start if toggled on
    if scaling_kwargs.get("use_dynamic_scaling", True):
        bt.set_scaling_params(
            target_vol=scaling_kwargs.get("target_vol", 0.10),
            vol_lookback=scaling_kwargs.get("vol_lookback", 20),
            max_vol_leverage=scaling_kwargs.get("max_vol_leverage", 1.0),
            dd_warning_threshold=scaling_kwargs.get("dd_warning_threshold", -0.15),
            dd_penalty=scaling_kwargs.get("dd_penalty", 0.5),
            dd_kill_threshold=scaling_kwargs.get("dd_kill_threshold", -0.25),
            use_ic_scaling=scaling_kwargs.get("use_ic_scaling", False),
            ic_lookback=scaling_kwargs.get("ic_lookback", 20),
            ic_threshold=scaling_kwargs.get("ic_threshold", 0.02),
            use_spread_scaling=scaling_kwargs.get("use_spread_scaling", False),
            spread_lookback=scaling_kwargs.get("spread_lookback", 30),
            spread_floor=scaling_kwargs.get("spread_floor", 0.3)
        )
    else:
        bt.use_dynamic_scaling = False
    
    bt_results = bt.run(verbose=False)
    
    if bt_results.empty:
        raise Exception("Cold Start failed: Backtest returned no results.")
        
    # 6. Scale Equity Curve to Starting Capital
    dr = bt_results["return"].values
    simulated_growth = np.cumprod(1 + dr)
    simulated_equity = starting_capital * (simulated_growth / simulated_growth[-1])
    
    seed_history = pd.Series(simulated_equity, index=bt_results.index)
    
    current_dd = (seed_history.iloc[-1] / seed_history.max()) - 1.0
    print(f"Cold Start Complete! Passed {len(seed_history)} days of history to live pipeline.")
    print(f"Day 1 Assumed Drawdown: {current_dd:.2%}")
    print(f"{'='*50}\n")
    
    return seed_history


class LiveExecutionPipeline:
    def __init__(self, model_folder, live_account_history, strategy_type="HRP", strategy_kwargs=None, scaling_kwargs=None):
        self.model_folder = model_folder
        self.capital = live_account_history.iloc[-1] 
        self.account_history = live_account_history
        self.strategy_type = strategy_type
        self.strategy_kwargs = strategy_kwargs if strategy_kwargs else {}
        self.scaling_kwargs = scaling_kwargs if scaling_kwargs else {}
        
        print(f"Loading Model Artifacts from '{self.model_folder}'...")
        self.lgb_model, self.info, self.dates, self.features, self.target = retrieve_model(self.model_folder)

    def prepare_live_data(self, target_date, universe_path, raw_data_path="../Data/all_ohlcv.feather"):
        target_date = pd.to_datetime(target_date)
        start_date = target_date - pd.Timedelta(days=45)
        
        print(f"--- STEP 1: Preparing Live Data ({start_date.date()} to {target_date.date()}) ---")
        
        # 1. Pull the most recent universe basket
        univ_df = pd.read_feather(universe_path)
        latest_univ_date = univ_df['date'].max()
        valid_tickers = univ_df[univ_df['date'] == latest_univ_date]['act_symbol'].unique().tolist()
        print(f"Using universe from {latest_univ_date.date()} ({len(valid_tickers)} tickers)")
        
        # 2. CREATE A TEMPORARY RAW DATASET
        print(f"Loading raw OHLCV to inject target date {target_date.date()}...")
        data_start = target_date - pd.Timedelta(days=400) 
        
        dataset = ds.dataset(raw_data_path, format="feather")
        filter_cond = (ds.field('act_symbol').isin(valid_tickers)) & (ds.field('date') >= data_start)
        raw_df = dataset.to_table(filter=filter_cond).to_pandas()
        raw_df['date'] = pd.to_datetime(raw_df['date']).astype('datetime64[ns]')
        
        if raw_df['date'].max() < target_date:
            print(f"Injecting {target_date.date()} into temporary raw data to force FeatureEngine output...")
            last_known = raw_df.sort_values('date').groupby('act_symbol').tail(1).copy()
            last_known['date'] = target_date 
            raw_df = pd.concat([raw_df, last_known], ignore_index=True)
            raw_df = raw_df.sort_values(['act_symbol', 'date']).reset_index(drop=True)
            
        temp_raw_path = "../Data/temp_live_ohlcv.feather"
        raw_df.to_feather(temp_raw_path)

        # 3. GENERATE MASK & COMPUTE FEATURES
        mask_df = raw_df[(raw_df['date'] >= start_date) & (raw_df['date'] <= target_date)][['date', 'act_symbol']].copy()

        print("Computing features over the trailing window...")
        engine = FeatureEngine(feature_requests=self.features)
        
        self.live_data = engine.compute(input_df=mask_df, raw_data_path=temp_raw_path)
        self.trained_feature_names = self.lgb_model.feature_name()
        self.live_data = self.live_data.dropna(subset=self.trained_feature_names)
        
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

    def predict(self):
        print("--- STEP 2: Generating Model Predictions ---")
        X_live = self.live_data[self.trained_feature_names]
        self.live_data["pred_return"] = self.lgb_model.predict(X_live)
        
        self.live_data = self.live_data.sort_values(["act_symbol", "date"])
        
        # --- NEW: ON-THE-FLY PREDICTION SMOOTHING ---
        if self.strategy_kwargs.get("smooth_predictions", False):
            print("  Applying 3-day EMA smoothing to predictions...")
            self.live_data["final_pred"] = self.live_data.groupby("act_symbol")["pred_return"].transform(
                lambda x: x.ewm(span=3, min_periods=1).mean()
            )
        else:
            self.live_data["final_pred"] = self.live_data["pred_return"]
            
        self.live_data["fwd_ret"] = self.live_data.groupby("act_symbol")["close"].shift(-1) / self.live_data["close"] - 1.0

    def _compute_dynamic_scalar(self, target_date, raw_data_path):
        print("--- STEP 3: Computing Dynamic Risk Scalars ---")
        
        # --- NEW: UNSCALED TOGGLE BYPASS ---
        if not self.scaling_kwargs.get("use_dynamic_scaling", True):
            print("  Dynamic Scaling DISABLED. Using Unscaled Strategy (1.0x).")
            return 1.0
        
        # 1. DRAWDOWN SCALAR
        dd_scalar = 1.0
        rolling_peak = self.account_history.rolling(252, min_periods=1).max().iloc[-1]
        current_dd = (self.capital / rolling_peak) - 1.0 if rolling_peak > 0 else 0.0
        
        if current_dd <= self.scaling_kwargs.get("dd_kill_threshold", -0.25):
            dd_scalar = 0.0
        elif current_dd <= self.scaling_kwargs.get("dd_warning_threshold", -0.15):
            dd_scalar = self.scaling_kwargs.get("dd_penalty", 0.5)

        # 2. VOLATILITY SCALAR (With Yang-Zhang Support)
        vol_scalar = 1.0
        vol_lookback = self.scaling_kwargs.get("vol_lookback", 20)
        target_vol = self.scaling_kwargs.get("target_vol", 0.10)
        vol_type = self.scaling_kwargs.get("volatility_type", "simple")
        
        if vol_type == "yang_zhang":
            print(f"  Computing Yang-Zhang market volatility (lookback={vol_lookback})...")
            valid_tickers = self.live_data[self.live_data['date'] == target_date]['act_symbol'].unique().tolist()
            start_date = target_date - pd.Timedelta(days=vol_lookback + 20)
            
            dataset = ds.dataset(raw_data_path, format="feather")
            filter_cond = (ds.field('act_symbol').isin(valid_tickers)) & (ds.field('date') >= start_date) & (ds.field('date') <= target_date)
            ohlc_df = dataset.to_table(filter=filter_cond).to_pandas()
            ohlc_df['date'] = pd.to_datetime(ohlc_df['date'])
            
            # FFill dummy row if needed
            if ohlc_df['date'].max() < target_date:
                last_known = ohlc_df.sort_values('date').groupby('act_symbol').tail(1).copy()
                last_known['date'] = target_date
                ohlc_df = pd.concat([ohlc_df, last_known], ignore_index=True)
                
            median_yz_vol = get_live_yang_zhang(ohlc_df, window=vol_lookback)
            if median_yz_vol > 0:
                vol_scalar = min(self.scaling_kwargs.get("max_vol_leverage", 1.0), target_vol / median_yz_vol)
        else:
            if len(self.account_history) >= vol_lookback:
                recent_rets = self.account_history.pct_change().dropna().tail(vol_lookback)
                realized_vol = np.std(recent_rets) * np.sqrt(252)
                if realized_vol > 0:
                    vol_scalar = min(self.scaling_kwargs.get("max_vol_leverage", 1.0), target_vol / realized_vol)

        # 3. SPREAD SCALAR
        spread_scalar = 1.0
        if self.scaling_kwargs.get("use_spread_scaling", False):
            spread_lookback = self.scaling_kwargs.get("spread_lookback", 30)
            spread_floor = self.scaling_kwargs.get("spread_floor", 0.3)
            
            # Ensure we use final_pred for consistency
            daily_spreads = self.live_data.groupby('date')['final_pred'].std().fillna(0.0)
            if target_date in daily_spreads.index:
                today_spread = daily_spreads.loc[target_date]
                recent_spreads = daily_spreads.iloc[-(spread_lookback+1):-1].values 
                
                if len(recent_spreads) > 0 and recent_spreads.std() > 0:
                    spread_pct = np.sum(recent_spreads <= today_spread) / len(recent_spreads)
                    spread_scalar = spread_floor + (1.0 - spread_floor) * spread_pct

        # 4. IC SCALAR
        ic_scalar = 1.0
        if self.scaling_kwargs.get("use_ic_scaling", False):
            ic_lookback = self.scaling_kwargs.get("ic_lookback", 20)
            ic_thresh = self.scaling_kwargs.get("ic_threshold", 0.02)
            
            def calc_safe_ic(group):
                valid = group.dropna(subset=['final_pred', 'fwd_ret'])
                if len(valid) > 10:
                    val, _ = spearmanr(valid['final_pred'], valid['fwd_ret'])
                    return val if not np.isnan(val) else 0.0
                return 0.0
                
            raw_daily_ic = self.live_data.groupby('date').apply(calc_safe_ic)
            historical_ic = raw_daily_ic.shift(1).dropna().tail(ic_lookback)
            
            if len(historical_ic) > 0:
                mean_ic = historical_ic.mean()
                midpoint = ic_thresh * 0.25  
                exponent = np.clip(-1.0/ic_thresh * (mean_ic - midpoint), -100, 100)
                sigmoid = 1.0 / (1.0 + np.exp(exponent))
                ic_scalar = 0.33 + (1.0 - 0.33) * sigmoid

        active_scalar = min(dd_scalar, vol_scalar, spread_scalar, ic_scalar)
        
        print(f"  Live Account Drawdown : {current_dd:.2%}")
        print(f"  DD Scalar     : {dd_scalar:.2f}")
        print(f"  Vol Scalar    : {vol_scalar:.2f}")
        print(f"  Spread Scalar : {spread_scalar:.2f}")
        print(f"  IC Scalar     : {ic_scalar:.2f}")
        print(f"  --> FINAL ACTIVE SCALAR: {active_scalar:.2f}")
        
        return active_scalar

    def allocate(self, target_date, raw_data_path="../Data/all_ohlcv.feather"):
        print(f"--- STEP 4: Allocating Portfolio ({self.strategy_type}) ---")
        
        avail = self.live_data['date']
        prediction_date = avail[avail <= target_date].max()

        if pd.isna(prediction_date):
            raise Exception(f"No predictions available on or before target date: {target_date.date()}")

        active_scalar = self._compute_dynamic_scalar(prediction_date, raw_data_path)
        df_today = self.live_data[self.live_data['date'] == prediction_date].copy()
        
        if df_today.empty:
            raise Exception(f"No predictions generated for prediction date: {prediction_date.date()}")
            
        # Ensure we sort by final_pred (handles smoothed vs unsmoothed seamlessly)
        df_today = df_today.sort_values("final_pred", ascending=False)
        
        if self.strategy_type == "HRP":
            n_longs = self.strategy_kwargs.get("n_longs", 20)
            n_shorts = self.strategy_kwargs.get("n_shorts", 20)
            hrp_lookback = self.strategy_kwargs.get("hrp_lookback", 60)
            holding_period = self.strategy_kwargs.get("holding_period", 5)
            rebalance_days = self.strategy_kwargs.get("rebalance_days", 1)
            sizing_method = self.strategy_kwargs.get("sizing_method", "dollar_neutral")
            
            num_tranches = max(1, int(holding_period / rebalance_days))
            tranche_budget = (self.capital / num_tranches) * active_scalar
            
            longs = df_today.head(n_longs)['act_symbol'].tolist()
            shorts = df_today.tail(n_shorts)['act_symbol'].tolist()
            active_symbols = longs + shorts
            
            min_date = prediction_date - pd.Timedelta(days=hrp_lookback + 30)
            valid_tickers = df_today['act_symbol'].unique().tolist()
            
            dataset = ds.dataset(raw_data_path, format="feather")
            filter_cond = (ds.field('act_symbol').isin(valid_tickers)) & (ds.field('date') >= min_date) & (ds.field('date') <= target_date)
            price_history = dataset.to_table(filter=filter_cond).to_pandas().pivot_table(index="date", columns="act_symbol", values="close").ffill()
            
            returns_pivot = price_history.pct_change().clip(lower=-0.75, upper=1.0).fillna(0)
            today_prices = price_history.iloc[-1]
            
            def get_weights(symbols):
                if not symbols: return {}
                raw_weights = compute_hrp_weights(returns_pivot.tail(hrp_lookback)[symbols], linkage_method="single")
                total = sum(raw_weights.values())
                return {k: v / total for k, v in raw_weights.items()} if total > 0 else {}

            long_weights = get_weights(longs)
            short_weights = get_weights(shorts)
            
            active_sides = (1 if len(long_weights) > 0 else 0) + (1 if len(short_weights) > 0 else 0)
            
            if sizing_method == "dollar_neutral":
                if active_sides == 2:
                    side_budget = tranche_budget / 2.5 
                    long_budget = side_budget
                    short_budget = side_budget
                elif active_sides == 1:
                    long_budget = tranche_budget / 1.0 if len(long_weights) > 0 else 0.0
                    short_budget = tranche_budget / 1.5 if len(short_weights) > 0 else 0.0
                    
            elif sizing_method == "beta_neutral":
                if active_sides == 2:
                    mkt_ret = returns_pivot.mean(axis=1) 
                    mkt_var = mkt_ret.rolling(60, min_periods=20).var()
                    rolling_cov = returns_pivot[active_symbols].rolling(60, min_periods=20).cov(mkt_ret)
                    beta_pivot = rolling_cov.div(mkt_var, axis=0).ffill().fillna(1.0).clip(0.1, 3.0)
                    today_betas = beta_pivot.iloc[-1]
                    
                    basket_beta_long = sum(w * today_betas.get(sym, 1.0) for sym, w in long_weights.items())
                    basket_beta_short = sum(w * today_betas.get(sym, 1.0) for sym, w in short_weights.items())
                    
                    if pd.isna(basket_beta_long) or basket_beta_long == 0: basket_beta_long = 1.0
                    if pd.isna(basket_beta_short) or basket_beta_short == 0: basket_beta_short = 1.0
                    
                    beta_ratio = np.clip(basket_beta_short / basket_beta_long, 0.33, 3.0)
                    short_budget = tranche_budget / (beta_ratio + 1.5)
                    long_budget = short_budget * beta_ratio
                else:
                    long_budget = tranche_budget / 1.0 if len(long_weights) > 0 else 0.0
                    short_budget = tranche_budget / 1.5 if len(short_weights) > 0 else 0.0
            
            positions = []
            for sym, w in long_weights.items():
                target_dollars = w * long_budget
                positions.append({"act_symbol": sym, "side": "LONG", "target_dollars": target_dollars, "target_shares": int(target_dollars / today_prices[sym]), "tranche_budget": tranche_budget})
                
            for sym, w in short_weights.items():
                target_dollars = w * short_budget
                positions.append({"act_symbol": sym, "side": "SHORT", "target_dollars": target_dollars, "target_shares": -int(target_dollars / today_prices[sym]), "tranche_budget": tranche_budget})
                
            return pd.DataFrame(positions)

    def run(self, target_date, universe_path, raw_data_path="../Data/all_ohlcv.feather"):
        self.prepare_live_data(target_date, universe_path, raw_data_path)
        self.predict()
        positions_df = self.allocate(target_date, raw_data_path)
        
        print("\n==============================================")
        print(f"LIVE TARGET POSITIONS FOR {pd.to_datetime(target_date).date()}")
        print("==============================================")
        print(positions_df.to_string(index=False))
        return positions_df


if __name__ == "__main__":
    
    # 1. Configuration
    today = pd.to_datetime(datetime.now().date())
    
    total_capital = 10000
    model_path = "../Data/Models/Model28_wf/fold_2026-01_2027-01" 
    universe_path = "../Data/Universes/7.feather"
    raw_data_path = "../Data/all_ohlcv.feather"
    
    # 2. Strategy & Scaling Settings
    hrp_settings = {
        "n_longs": 4, 
        "n_shorts": 4, 
        "sizing_method": "beta_neutral", 
        "hrp_lookback": 150,
        "holding_period": 5,
        "rebalance_days": 1,
        "smooth_predictions": True 
    }
    
    scaling_settings = {
        "use_dynamic_scaling": True,        # NEW: Set False for Unscaled Strategy
        "volatility_type": "yang_zhang",    # NEW: "yang_zhang" or "simple"
        "target_vol": 0.1,
        "vol_lookback": 3,
        "max_vol_leverage": 1.0,
        "dd_warning_threshold": -1,
        "dd_penalty": 1,
        "dd_kill_threshold": -1,
        "use_ic_scaling": False,
        "ic_lookback": 20,
        "ic_threshold": 0.01,
        "use_spread_scaling": False,
        "spread_lookback": 30,
        "spread_floor": 0.2
    }
    
    # 3. Check for Account History (Cold Start)
    history_file = "../Data/paper_trading/live_account_history.csv"
    
    if os.path.exists(history_file):
        print("Loading existing live account history...")
        live_account_history = pd.read_csv(history_file, index_col=0, parse_dates=True).squeeze()
        
        if today not in live_account_history.index:
            live_account_history.loc[today] = live_account_history.iloc[-1] 
    else:
        live_account_history = bootstrap_cold_start(
            model_folder=model_path,
            universe_path=universe_path,
            target_date=today,
            lookback_days=100,  
            starting_capital=total_capital,
            strategy_kwargs=hrp_settings,
            scaling_kwargs=scaling_settings,
            raw_data_path=raw_data_path
        )
        live_account_history.to_csv(history_file)

    # 4. RUN LIVE PIPELINE
    pipeline = LiveExecutionPipeline(
        model_folder=model_path,
        live_account_history=live_account_history, 
        strategy_type="HRP",
        strategy_kwargs=hrp_settings,
        scaling_kwargs=scaling_settings            
    )
    
    target_positions = pipeline.run(
        target_date=today, 
        universe_path=universe_path,
        raw_data_path=raw_data_path
    )
    
    # Save the output
    out_path = f"../Data/paper_trading/target_positions_{today.date()}.csv"
    target_positions.to_csv(out_path, index=False)
    print(f"\nSaved targets to {out_path}")
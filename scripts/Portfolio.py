#implementing portfolio strategies
import lightgbm as lgb
import pandas as pd
from FeatureEngine import *
import os
import pickle
from clean_analysis import Model
from matplotlib import pyplot as plt

def retrieve_model(model_folder):
    """gets model information from model folder

    Args:
        model_folder (str): path to the model folder

    Raises:
        Exception: feature cannot be found

    Returns:
        tuple: info for the model
    """
    lgb_model = lgb.Booster(model_file = f"{model_folder}/model.txt")
    info = pd.read_csv(f"{model_folder}/info.csv")
    dates = pd.read_csv(f"{model_folder}/dates.csv")
    features = []
    for feature_file in os.listdir(f"{model_folder}/features"):
        try:
            with open(f"{model_folder}/features/{feature_file}", 'rb') as file:
                # Load the object from the file
                feature = pickle.load(file)
        except:
            raise Exception(f"Feature {feature_file} could not be read")
        features.append(feature)
    with open(f"{model_folder}/target.pkl", 'rb') as file:
        target = pickle.load(file)
    return lgb_model, info, dates, features, target

class Portfolio:
    def __init__(self):
        self.has_data = False

    def open(self, portfolio_folder):
        self.data = pd.read_feather(f"{portfolio_folder}/predictions.feather")
        info = pd.read_csv(f"{portfolio_folder}/info.csv")
        self.model_folder = info.model_folder
        self.has_data = True

    def get_model_data(self, model_folder):
        '''
        Pulls data for the model
        '''
        self.model_folder = model_folder
        self.get_model()
        self.get_model_predictions()
        self.drop_unnecessary()
        self.has_data = True

    def get_model(self):
        lgb_model, info, dates, features, target = retrieve_model(self.model_folder)
        self.model = Model(universe_path = info.universe_path[0])
        self.model.add_features(features)
        self.model.add_target(target)
        self.model.model = lgb_model
        self.model.data = self.model.data[(self.model.data.date >= dates.test_start[0]) & 
                                          (self.model.data.date <= dates.test_end[0])]
        self.model.generate_targets_and_features()

    def get_model_predictions(self):
        """
        Generates predictions on the test data using the loaded model.
        Adds 'pred_return' column to self.model.data.
        """
        if not self.model.data_generated:
            self.model.generate_targets_and_features()

        feature_keys = [k for k in self.model.data.columns if "F" in k.split("_")]
        self.model.data["pred_return"] = self.model.model.predict(self.model.data[feature_keys])

        # Cross-sectional prediction z-score (for ranking)
        self.model.data["pred_zscore"] = self.model.data.groupby("date")["pred_return"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

        print(f"Predictions generated: {len(self.model.data)} rows, "
              f"{self.model.data['act_symbol'].nunique()} stocks, "
              f"{self.model.data['date'].min().date()} -> {self.model.data['date'].max().date()}")
        
    def drop_unnecessary(self):
        # Compute actual forward log returns for P&L BEFORE dropping columns
        self.model.data["daily_log_ret"] = self.model.data.groupby("act_symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

        keep_cols = ["date", "act_symbol", "pred_return", "pred_zscore", "fwd_log_ret"]

        for col in ["open", "close"]:
            if col in self.model.data.columns:
                keep_cols.append(col)

        # Still keep target for reference, but NOT for P&L
        target_cols = [k for k in self.model.data.columns if "T" in k.split("_")]
        keep_cols += target_cols

        keep_cols = [c for c in keep_cols if c in self.model.data.columns]

        self.model.data = self.model.data[keep_cols]
        self.data = self.model.data
        del self.model

    def save(self, folder):
        '''
        Saves the portfolio into a folder
        '''
        os.makedirs(f"{folder}", exist_ok = True)
        info = {
            "model_folder": [self.model_folder]
        }
        info_df = pd.DataFrame(info)
        info_df.to_csv(f"{folder}/info.csv")
        self.data.to_feather(f"{folder}/predictions.feather")

class WalkForwardPortfolio:
    def __init__(self):
        self.has_data = False
        self.folds = []

    def _discover_folds(self, wf_folder):
        """
        Scan wf_folder for sub-directories containing dates.csv.
        Sort by test_start and auto-compute gap_end for each fold.
        """
        fold_dirs = []
        for entry in os.scandir(wf_folder):
            if entry.is_dir() and os.path.exists(f"{entry.path}/dates.csv"):
                dates = pd.read_csv(f"{entry.path}/dates.csv")
                fold_dirs.append({
                    "model_folder": entry.path,
                    "test_start":   pd.Timestamp(dates.test_start[0]),
                    "test_end":     pd.Timestamp(dates.test_end[0]),
                })

        if not fold_dirs:
            raise Exception(
                f"No valid fold directories found in '{wf_folder}'. "
                f"Each sub-folder must contain a dates.csv file."
            )

        fold_dirs.sort(key=lambda x: x["test_start"])

        for i, fold in enumerate(fold_dirs):
            fold["gap_end"] = (
                fold_dirs[i + 1]["test_start"] - pd.Timedelta(days=1)
                if i < len(fold_dirs) - 1 else None
            )

        for i, fold in enumerate(fold_dirs):
            print(f"  Fold {i}: {fold['test_start'].date()} -> {fold['test_end'].date()}"
                  + (f" | gap to {fold['gap_end'].date()}" if fold["gap_end"] else " | no gap"))

        return fold_dirs

    def _load_fold_models(self):
        """
        Load the lgb model, features, and target for every fold.
        Raises immediately if any fold's feature set differs from fold 0 —
        a mismatch means features cannot be computed once for all folds.
        """
        fold_models = []
        for i, fold_info in enumerate(self.folds):
            lgb_model, info, _, features, target = retrieve_model(fold_info["model_folder"])
            fold_models.append({
                "lgb_model": lgb_model,
                "features":  features,
                "target":    target,
                "info":      info,
            })

        ref_names = [f.name for f in fold_models[0]["features"]]
        for i, fm in enumerate(fold_models[1:], start=1):
            fold_names = [f.name for f in fm["features"]]
            if fold_names != ref_names:
                raise Exception(
                    f"Feature mismatch between fold 0 and fold {i} — "
                    f"cannot compute features once across all folds.\n"
                    f"Fold 0 : {ref_names}\n"
                    f"Fold {i}: {fold_names}"
                )

        print(f"  Feature set verified: {len(ref_names)} features consistent "
              f"across all {len(fold_models)} folds.")
        return fold_models

    def _compute_features(self, ohlcv_path, features, target):
        """
        Load the full OHLCV dataset and compute features and targets once.
        No date slicing before feature computation — this is the entire point:
        every row gets its full rolling lookback window.
        """
        print(f"\nComputing features on full dataset: '{ohlcv_path}'...")
        master = Model(universe_path=ohlcv_path)
        master.add_features(features)
        master.add_target(target)
        master.generate_targets_and_features()

        feature_keys = [k for k in master.data.columns if "F" in k.split("_")]
        print(f"  {len(feature_keys)} features computed across "
              f"{master.data['date'].nunique()} dates, "
              f"{master.data['act_symbol'].nunique()} stocks "
              f"({len(master.data)} rows total).")

        return master.data, feature_keys

    def get_model_data(self, wf_folder, ohlcv_path="../Data/all_ohlcv.feather"):
        """
        Discover folds, compute features once on the full OHLCV dataset,
        then generate predictions per fold (including gap periods).

        Args:
            wf_folder  : folder containing one sub-directory per fold
            ohlcv_path : path to the full OHLCV feather file used for
                         feature computation (default: ../Data/all_ohlcv.feather)
        """
        # --- 1. Discover fold structure -----------------------------------------
        print(f"Discovering folds in '{wf_folder}'...")
        self.folds = self._discover_folds(wf_folder)

        # --- 2. Load & validate fold models -------------------------------------
        print(f"\nLoading and validating {len(self.folds)} fold models...")
        fold_models = self._load_fold_models()

        # --- 3. Compute features once on full dataset ---------------------------
        full_data, feature_keys = self._compute_features(
            ohlcv_path,
            features=fold_models[0]["features"],
            target=fold_models[0]["target"],
        )

        # --- 4. Predict per fold on pre-computed feature slices -----------------
        print(f"\nGenerating predictions for {len(self.folds)} folds...")
        all_dfs = []

        for i, (fold_info, fm) in enumerate(zip(self.folds, fold_models)):
            test_start = fold_info["test_start"]
            test_end   = fold_info["test_end"]
            gap_end    = fold_info["gap_end"]
            end_date   = gap_end if gap_end is not None else test_end

            # Slice the pre-computed feature matrix — no re-computation
            fold_data = full_data[
                (full_data["date"] >= test_start) &
                (full_data["date"] <= end_date)
            ].copy()

            if fold_data.empty:
                raise Exception(
                    f"Fold {i}: no rows found between "
                    f"{test_start.date()} and {end_date.date()}. "
                    f"Check that ohlcv_path covers this date range."
                )

            fold_data["pred_return"] = fm["lgb_model"].predict(fold_data[feature_keys])
            fold_data["pred_zscore"] = fold_data.groupby("date")["pred_return"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            fold_data["fold"]   = i
            fold_data["is_gap"] = fold_data["date"] > test_end

            n_real = (~fold_data["is_gap"]).sum()
            n_gap  = fold_data["is_gap"].sum()
            print(f"  Fold {i}: {test_start.date()} -> {test_end.date()} | "
                  f"{n_real} real rows"
                  + (f" | {n_gap} gap rows to {end_date.date()}" if n_gap > 0 else ""))

            all_dfs.append(fold_data)

        # --- 5. Combine; real predictions always supersede gap predictions ------
        combined   = pd.concat(all_dfs, ignore_index=True)
        real_dates = set(combined.loc[~combined["is_gap"], "date"].unique())
        combined   = combined[~(combined["is_gap"] & combined["date"].isin(real_dates))]

        keep_cols = ["date", "act_symbol", "pred_return", "pred_zscore", "fold", "is_gap"]
        for col in ["open", "close", "fwd_log_ret"]:
            if col in combined.columns:
                keep_cols.append(col)
        target_cols = [k for k in combined.columns if "T" in k.split("_")]
        keep_cols  += [c for c in target_cols if c not in keep_cols]
        keep_cols   = [c for c in keep_cols if c in combined.columns]

        self.data = (combined[keep_cols]
                     .sort_values(["date", "act_symbol"])
                     .reset_index(drop=True))
        self.has_data = True

        print(f"\nWalkForwardPortfolio ready: "
              f"{self.data['date'].nunique()} dates | "
              f"{self.data['act_symbol'].nunique()} stocks | "
              f"{self.data['date'].min().date()} -> {self.data['date'].max().date()} | "
              f"{self.data['is_gap'].sum()} gap rows across {len(self.folds)} folds")

    def save(self, folder):
        if not self.has_data:
            raise Exception("No data. Run get_model_data() first.")

        os.makedirs(folder, exist_ok=True)

        info_df = pd.DataFrame([{
            "fold":         i,
            "model_folder": f["model_folder"],
            "gap_end":      str(f["gap_end"]) if f["gap_end"] is not None else "",
        } for i, f in enumerate(self.folds)])
        info_df.to_csv(f"{folder}/info.csv", index=False)

        self.data.to_feather(f"{folder}/predictions.feather")

        print(f"Saved {len(self.folds)} folds to '{folder}'")

    def open(self, folder):
        info_df    = pd.read_csv(f"{folder}/info.csv")
        self.folds = []

        for _, row in info_df.iterrows():
            raw_gap = row["gap_end"]
            self.folds.append({
                "model_folder": row["model_folder"],
                "gap_end": pd.Timestamp(raw_gap) if (pd.notna(raw_gap) and raw_gap != "") else None,
            })

        self.data     = pd.read_feather(f"{folder}/predictions.feather")
        self.has_data = True

        print(f"Opened WalkForwardPortfolio: "
              f"{self.data['date'].nunique()} dates | "
              f"{self.data['act_symbol'].nunique()} stocks | "
              f"{self.data['date'].min().date()} -> {self.data['date'].max().date()}")

class BacktestRule:
    def __init__(self, feature, threshold, direction:str = "above"):
        if type(feature) not in [FeatureRequest, RateFeatureRequest]:
            raise Exception("feature must be FeatureRequest or RateFeatureRequest")
        if direction not in ["above", "below"]:
            raise Exception("direction must be 'above' or 'below'")
        self.direction = direction
        self.feature = feature
        self.threshold = threshold

    def set_col_name(self, df):
        """Finds the column name generated by FeatureEngine for this rule's feature."""
        if hasattr(self.feature, 'col_name') and self.feature.col_name in df.columns:
            self.col_name = self.feature.col_name
        elif hasattr(self.feature, 'base_col_name') and self.feature.base_col_name in df.columns:
            self.col_name = self.feature.base_col_name
        else:
            raise KeyError(f"Could not find column for feature '{self.feature.name}' in DataFrame. "
                          f"Tried '{getattr(self.feature, 'col_name', None)}' and "
                          f"'{getattr(self.feature, 'base_col_name', None)}'")
        
    def passes(self, row):
        """Returns True if the rule passes for this row."""
        val = row.get(self.col_name, np.nan)
        if pd.isna(val):
            return False
        if self.direction == "above":
            return val > self.threshold
        else:
            return val < self.threshold

class BacktestRuleset:
    def __init__(self, rules):
        for rule in rules:
            if type(rule) != BacktestRule:
                raise Exception("Each rule must be of type BacktestRule")
        self.rules = rules

    def add_rule_info(self, df):
        engine = FeatureEngine([rule.feature for rule in self.rules])
        df = engine.compute(input_df = df)
        # Map each rule to its computed column name
        for rule in self.rules:
            rule.set_col_name(df)
            print(f"  Rule: {rule.feature.name} -> column '{rule.col_name}', threshold={rule.threshold}")

        return df

class RulesBasedBacktest:
    def __init__(self, portfolio):
        if type(portfolio) not in [Portfolio, WalkForwardPortfolio]:
            raise Exception("Invalid portfolio type. Must be Portfolio of WalkForwardPortfolio")
        self.portfolio = portfolio

    def add_ruleset(self, ruleset: BacktestRuleset):
        self.ruleset = ruleset
        self.df = ruleset.add_rule_info(self.portfolio.model.data)

    def run_backtest(self):
        pass

class SimpleBacktest:
    def __init__(self, portfolio, n_longs=20, n_shorts=20, 
                 rebalance_days=5, cost_bps=5.0, holding_period=5):
        """
        Signal-weighted long-short backtest.
        Weights are proportional to |pred_zscore| — strongest convictions get most capital.
        
        Args:
            portfolio:      Portfolio object with .data containing pred_return, pred_zscore, close
            n_longs:        Number of long positions
            n_shorts:       Number of short positions
            rebalance_days: Rebalance every N trading days
            cost_bps:       Transaction cost per side in basis points
            holding_period: Forward return horizon in days (for Sharpe annualization)
        """
        if not portfolio.has_data:
            raise Exception("Portfolio has no data. Run get_model_data() first.")
        
        self.data = portfolio.data.copy()
        self.n_longs = n_longs
        self.n_shorts = n_shorts
        self.rebalance_days = rebalance_days
        self.cost_bps = cost_bps
        self.holding_period = holding_period

        # Compute daily log returns from close prices
        if "close" not in self.data.columns:
            raise Exception("Portfolio data must contain 'close' column. Don't drop it in drop_unnecessary().")
    
        self.data = self.data.sort_values(["act_symbol", "date"])
        self.data["daily_log_ret"] = self.data.groupby("act_symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        self.pnl_col = "daily_log_ret"
        self.data["daily_log_ret"] = self.data["daily_log_ret"].fillna(0)

        # Precompute
        self.data = self.data.sort_values(["date", "pred_return"], ascending=[True, False])
        self.trading_dates = sorted(self.data["date"].unique())
        self._date_groups = {d: g for d, g in self.data.groupby("date")}

        print(f"SimpleBacktest: {len(self.trading_dates)} days, "
              f"{self.data['act_symbol'].nunique()} stocks, "
              f"P&L column: {self.pnl_col}")

    def run(self):
        """Run the signal-weighted long-short backtest."""
        n_l = self.n_longs
        n_s = self.n_shorts
        rebal_set = set(self.trading_dates[::self.rebalance_days])
        cost_rate = self.cost_bps / 10_000

        weights = {}
        daily_rets = []
        daily_turnover = []

        for date in self.trading_dates:
            cost = 0.0

            if date in rebal_set and date in self._date_groups:
                df = self._date_groups[date]

                if len(df) >= n_l + n_s:
                    sorted_df = df.sort_values("pred_return", ascending=False)
                    longs = sorted_df.head(n_l)
                    shorts = sorted_df.tail(n_s)

                    # Signal-weighted: proportional to |pred_zscore|
                    long_scores = longs["pred_zscore"].abs()
                    long_total = long_scores.sum()
                    long_norm = long_scores / long_total if long_total > 0 else pd.Series(1.0 / n_l, index=long_scores.index)

                    short_scores = shorts["pred_zscore"].abs()
                    short_total = short_scores.sum()
                    short_norm = short_scores / short_total if short_total > 0 else pd.Series(1.0 / n_s, index=short_scores.index)

                    # After computing long_norm and short_norm:
                    # With 150% margin requirement, shorts get 2/3 the capital of longs
                    # because $1 of shorts consumes $1.50 of buying power
                    margin_factor = 1.0 / 1.5  # = 0.667

                    long_budget = capital * (1.0 / (1.0 + margin_factor))   # ~60% to longs
                    short_budget = capital * (margin_factor / (1.0 + margin_factor))  # ~40% to shorts

                    for idx, r in longs.iterrows():
                        new_w[r["act_symbol"]] = long_norm.loc[idx] * long_budget
                    for idx, r in shorts.iterrows():
                        new_w[r["act_symbol"]] = -short_norm.loc[idx] * short_budget

                    # Transaction costs
                    turnover = sum(
                        abs(new_w.get(t, 0) - weights.get(t, 0))
                        for t in set(list(new_w) + list(weights))
                    )
                    cost = turnover * cost_rate
                    weights = new_w
                    daily_turnover.append(turnover)
                else:
                    daily_turnover.append(0.0)
            else:
                daily_turnover.append(0.0)

            # Portfolio return
            if date in self._date_groups and weights:
                rets = self._date_groups[date].set_index("act_symbol")[self.pnl_col]
                port_ret = sum(w * rets.get(t, 0) for t, w in weights.items()) - cost
            else:
                port_ret = 0.0

            daily_rets.append(port_ret)

        self.results = pd.DataFrame({
            "date": self.trading_dates,
            "return": daily_rets,
            "turnover": daily_turnover,
        }).set_index("date")

        self._print_results()
        return self.results

    def _print_results(self):
        dr = self.results["return"].values
        N = self.holding_period

        if len(dr) == 0 or np.std(dr) == 0:
            print("No valid returns.")
            return

        cum = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()
        sharpe = (dr.mean() / dr.std(ddof=1)) * np.sqrt(252)
        ann_ret = dr.mean() * (252 / N)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Drawdown duration
        current_dur = 0
        max_dur = 0
        for uw in cum < peak:
            current_dur = current_dur + 1 if uw else 0
            max_dur = max(max_dur, current_dur)

        print(f"\n{'='*50}")
        print(f"SIMPLE BACKTEST RESULTS (signal-weighted)")
        print(f"{'='*50}")
        print(f"Sharpe           : {sharpe:.4f}")
        print(f"Annual Return    : {ann_ret:.2%}")
        print(f"Total Return     : {cum[-1] - 1:.2%}")
        print(f"Max Drawdown     : {max_dd:.2%}")
        print(f"Max DD Duration  : {max_dur} days")
        print(f"Calmar           : {calmar:.4f}")
        print(f"Win Rate         : {(dr > 0).mean():.2%}")
        print(f"Avg Turnover     : {self.results['turnover'].mean():.4f}")

    def plot(self):
        if not hasattr(self, "results"):
            raise Exception("Run run() first.")

        dr = self.results["return"].values
        dates = self.results.index
        N = self.holding_period

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Cumulative return
        cum = np.cumprod(1 + dr)
        axes[0, 0].plot(dates, cum, color="purple", lw=1.5)
        axes[0, 0].axhline(1, color="black", lw=0.5)
        axes[0, 0].set_title("Cumulative Return")

        # Drawdown
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        axes[0, 1].fill_between(dates, dd, 0, color="red", alpha=0.4)
        axes[0, 1].set_title("Drawdown")

        # Rolling Sharpe
        rs = pd.Series(dr).rolling(60).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252 / N)
        )
        axes[1, 0].plot(dates, rs.values, color="blue", lw=1)
        axes[1, 0].axhline(0, color="red", ls="--", lw=0.5)
        axes[1, 0].set_title("Rolling 60-Day Sharpe")

        # Monthly returns
        monthly = pd.Series(dr, index=dates).resample("ME").sum()
        axes[1, 1].bar(
            range(len(monthly)), monthly.values,
            color=["green" if r > 0 else "red" for r in monthly], alpha=0.7,
        )
        axes[1, 1].set_title("Monthly Returns")
        axes[1, 1].set_xticks([])

        plt.tight_layout()
        plt.show()
    
class ManagedBacktest:
    """
    Signal-weighted long-short backtest with dynamic risk management and ROLLING TRANCHES.
    
    If holding_period = 5, the portfolio is split into 5 equal tranches. 
    Every day, 1/5th of the portfolio is rebalanced based on the newest signals.
    """

    def __init__(self, portfolio, n_longs=20, n_shorts=20, holding_period=5, cost_bps=5.0):
        if not portfolio.has_data:
            raise Exception("Portfolio has no data. Run get_model_data() first.")

        self.data = portfolio.data.copy()
        self.n_longs = n_longs
        self.n_shorts = n_shorts
        self.holding_period = holding_period
        self.cost_bps = cost_bps

        if "close" not in self.data.columns:
            raise Exception("Portfolio data must contain 'close' column.")

        # Compute daily log returns
        self.data = self.data.sort_values(["act_symbol", "date"])
        # Compute open-to-open returns instead
        self.data = self.data.sort_values(["act_symbol", "date"])
        self.data["daily_open_ret"] = self.data.groupby("act_symbol")["open"].transform(
            lambda x: np.log(x / x.shift(1))
            ).fillna(0)
        self.pnl_col = "daily_open_ret"

        self.data = self.data.sort_values(["date", "pred_return"], ascending=[True, False])
        self.trading_dates = sorted(self.data["date"].unique())
        self._date_groups = {d: g for d, g in self.data.groupby("date")}

        print(f"ManagedBacktest (Tranches): {len(self.trading_dates)} days, "
              f"{self.data['act_symbol'].nunique()} stocks")

    def run(self,
            dd_threshold=-0.05,      
            dd_full_cut=-0.15,       
            target_vol=0.10,         
            vol_lookback=20,         
            vol_cap=1.5,             
            max_gross_exposure=2.0,  
            dd_lookback=252          # Changed to 252 (1-year rolling high)
            ):
        
        n_l = self.n_longs
        n_s = self.n_shorts
        cost_rate = self.cost_bps / 10_000
        tranche_wt = 1.0 / self.holding_period

        # Initialize empty tranches
        tranches = {i: {} for i in range(self.holding_period)}
        
        daily_rets = []
        daily_scalars = []
        daily_components = []

        cum_ret = 1.0
        cum_history = []
        recent_rets = []

        # Default scalars
        combined_scalar = 1.0
        dd_scalar = 1.0
        vol_scalar = 1.0
        current_dd = 0.0

        for day_idx, date in enumerate(self.trading_dates):
            
            # =======================================================
            # STEP 1: CALCULATE TODAY'S GROSS P&L 
            # =======================================================
            daily_port_gross_ret = 0.0
            
            if date in self._date_groups:
                rets = self._date_groups[date].set_index("act_symbol")[self.pnl_col]
                
                # Sum the returns of all active tranches
                for i in range(self.holding_period):
                    w = tranches[i]
                    if w:
                        tranche_ret = sum(weight * rets.get(sym, 0.0) for sym, weight in w.items())
                        daily_port_gross_ret += tranche_ret * tranche_wt

            # =======================================================
            # STEP 2: UPDATE RISK METRICS 
            # =======================================================
            temp_cum_ret = cum_ret * (1 + daily_port_gross_ret)
            cum_history.append(temp_cum_ret)
            recent_rets.append(daily_port_gross_ret)

            if len(cum_history) > dd_lookback:
                trailing_peak = max(cum_history[-dd_lookback:])
            else:
                trailing_peak = max(cum_history)
            
            current_dd = (temp_cum_ret - trailing_peak) / trailing_peak

            if current_dd <= dd_threshold:
                dd_scalar = max(0.0, (current_dd - dd_full_cut) / (dd_threshold - dd_full_cut))
            else:
                dd_scalar = 1.0

            if len(recent_rets) >= vol_lookback:
                realized_vol = np.std(recent_rets[-vol_lookback:]) * np.sqrt(252)
                if realized_vol > 0:
                    vol_scalar = min(vol_cap, target_vol / realized_vol)
            
            combined_scalar = dd_scalar * vol_scalar

            # =======================================================
            # STEP 3: REBALANCE EXACTLY ONE TRANCHE FOR TOMORROW
            # =======================================================
            tranche_id = day_idx % self.holding_period
            old_w = tranches[tranche_id]
            new_w = {}
            turnover = 0.0
            
            base_target_gross = 1.0 
            
            if date in self._date_groups:
                df = self._date_groups[date]

                if len(df) >= n_l + n_s:
                    sorted_df = df.sort_values("pred_return", ascending=False)
                    longs = sorted_df.head(n_l)
                    shorts = sorted_df.tail(n_s)

                    long_scores = longs["pred_zscore"].abs()
                    long_total = long_scores.sum()
                    long_norm = long_scores / long_total if long_total > 0 else pd.Series(1.0 / n_l, index=long_scores.index)

                    short_scores = shorts["pred_zscore"].abs()
                    short_total = short_scores.sum()
                    short_norm = short_scores / short_total if short_total > 0 else pd.Series(1.0 / n_s, index=short_scores.index)

                    for idx, r in longs.iterrows():
                        new_w[r["act_symbol"]] = long_norm.loc[idx] * (base_target_gross / 2) * combined_scalar
                    for idx, r in shorts.iterrows():
                        new_w[r["act_symbol"]] = -short_norm.loc[idx] * (base_target_gross / 2) * combined_scalar

                    gross = sum(abs(w) for w in new_w.values())
                    if gross > max_gross_exposure:
                        ratio = max_gross_exposure / gross
                        new_w = {k: v * ratio for k, v in new_w.items()}

                    turnover = sum(
                        abs(new_w.get(t, 0) - old_w.get(t, 0))
                        for t in set(list(new_w) + list(old_w))
                    )
            
            # --- THE CRITICAL MISSING LINE ---
            tranches[tranche_id] = new_w
            # ---------------------------------

            # =======================================================
            # STEP 4: CALCULATE FINAL NET RETURN
            # =======================================================
            port_cost = (turnover * cost_rate) * tranche_wt
            net_port_ret = daily_port_gross_ret - port_cost
            
            cum_ret = cum_ret * (1 + net_port_ret)
            
            cum_history[-1] = cum_ret
            recent_rets[-1] = net_port_ret

            total_gross = sum(
                sum(abs(w) for w in tranches[i].values()) * tranche_wt 
                for i in range(self.holding_period)
            )

            daily_rets.append(net_port_ret)
            daily_scalars.append(combined_scalar)
            daily_components.append({
                "dd_scalar": dd_scalar,
                "vol_scalar": vol_scalar,
                "combined": combined_scalar,
                "drawdown": current_dd,
                "gross_exposure": total_gross,
            })

        self.results = pd.DataFrame({
            "date": self.trading_dates,
            "return": daily_rets,
            "scalar": daily_scalars,
        }).set_index("date")

        self.components = pd.DataFrame(daily_components, index=self.trading_dates)
        self._print_results()
        return self.results

    def _print_results(self):
        dr = self.results["return"].values
        if len(dr) == 0 or np.std(dr) == 0:
            print("No valid returns.")
            return

        cum = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()
        sharpe = (dr.mean() / dr.std(ddof=1)) * np.sqrt(252)
        ann_ret = dr.mean() * 252
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        current_dur = 0
        max_dur = 0
        for uw in cum < peak:
            current_dur = current_dur + 1 if uw else 0
            max_dur = max(max_dur, current_dur)

        scalars = self.results["scalar"].values

        print(f"\n{'='*50}")
        print(f"MANAGED BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Sharpe           : {sharpe:.4f}")
        print(f"Annual Return    : {ann_ret:.2%}")
        print(f"Total Return     : {cum[-1] - 1:.2%}")
        print(f"Max Drawdown     : {max_dd:.2%}")
        print(f"Max DD Duration  : {max_dur} days")
        print(f"Calmar           : {calmar:.4f}")
        print(f"Win Rate         : {(dr > 0).mean():.2%}")
        print(f"Avg Scalar       : {scalars.mean():.3f}")
        print(f"Min Scalar       : {scalars.min():.3f}")
        print(f"Days at <50%     : {(scalars < 0.5).sum()} ({(scalars < 0.5).mean():.1%})")

    def plot(self):
        if not hasattr(self, "results"):
            raise Exception("Run run() first.")

        dr = self.results["return"].values
        dates = self.results.index

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Cumulative return
        cum = np.cumprod(1 + dr)
        axes[0, 0].plot(dates, cum, color="purple", lw=1.5)
        axes[0, 0].axhline(1, color="black", lw=0.5)
        axes[0, 0].set_title("Cumulative Return")

        # Drawdown
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        axes[0, 1].fill_between(dates, dd, 0, color="red", alpha=0.4)
        axes[0, 1].set_title("Drawdown")

        # Rolling Sharpe
        rs = pd.Series(dr).rolling(60).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252)
        )
        axes[1, 0].plot(dates, rs.values, color="blue", lw=1)
        axes[1, 0].axhline(0, color="red", ls="--", lw=0.5)
        axes[1, 0].set_title("Rolling 60-Day Sharpe")

        # Monthly returns
        monthly = pd.Series(dr, index=dates).resample("ME").sum()
        axes[1, 1].bar(
            range(len(monthly)), monthly.values,
            color=["green" if r > 0 else "red" for r in monthly], alpha=0.7,
        )
        axes[1, 1].set_title("Monthly Returns")
        axes[1, 1].set_xticks([])

        # Exposure scalar over time
        axes[2, 0].plot(dates, self.components["combined"], color="teal", lw=1)
        axes[2, 0].fill_between(dates, self.components["dd_scalar"], alpha=0.2, color="red", label="DD scalar")
        axes[2, 0].fill_between(dates, self.components["vol_scalar"], alpha=0.2, color="blue", label="Vol scalar")
        axes[2, 0].set_title("Exposure Scalars")
        axes[2, 0].set_ylim(0, 2.5)
        axes[2, 0].legend(fontsize=8)

        # Gross exposure over time
        axes[2, 1].plot(dates, self.components["gross_exposure"], color="darkorange", lw=1)
        axes[2, 1].set_title("Gross Exposure")

        plt.tight_layout()
        plt.show()

    def compare(self, simple_results):
        """
        Compare managed vs simple backtest side by side.
        Pass in the results DataFrame from SimpleBacktest.run().
        """
        dr_managed = self.results["return"].values
        dr_simple = simple_results["return"].values
        dates = self.results.index

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        cum_m = np.cumprod(1 + dr_managed)
        cum_s = np.cumprod(1 + dr_simple)

        axes[0, 0].plot(dates, cum_m, color="purple", lw=1.5, label="Managed")
        axes[0, 0].plot(dates[:len(cum_s)], cum_s, color="gray", lw=1, alpha=0.7, label="Simple")
        axes[0, 0].axhline(1, color="black", lw=0.5)
        axes[0, 0].set_title("Cumulative Return")
        axes[0, 0].legend()

        peak_m = np.maximum.accumulate(cum_m)
        dd_m = (cum_m - peak_m) / peak_m
        peak_s = np.maximum.accumulate(cum_s)
        dd_s = (cum_s - peak_s) / peak_s
        axes[0, 1].fill_between(dates, dd_m, 0, color="purple", alpha=0.3, label="Managed")
        axes[0, 1].fill_between(dates[:len(dd_s)], dd_s, 0, color="gray", alpha=0.3, label="Simple")
        axes[0, 1].set_title("Drawdown")
        axes[0, 1].legend()

        rs_m = pd.Series(dr_managed).rolling(60).apply(lambda x: x.mean()/(x.std()+1e-8)*np.sqrt(252))
        rs_s = pd.Series(dr_simple).rolling(60).apply(lambda x: x.mean()/(x.std()+1e-8)*np.sqrt(252))
        axes[1, 0].plot(dates, rs_m.values, color="purple", lw=1, label="Managed")
        axes[1, 0].plot(dates[:len(rs_s)], rs_s.values, color="gray", lw=1, alpha=0.7, label="Simple")
        axes[1, 0].axhline(0, color="red", ls="--", lw=0.5)
        axes[1, 0].set_title("Rolling 60-Day Sharpe")
        axes[1, 0].legend()

        # Summary stats comparison
        def calc_stats(dr):
            cum = np.cumprod(1 + dr)
            peak = np.maximum.accumulate(cum)
            max_dd = ((cum - peak) / peak).min()
            sharpe = (dr.mean() / dr.std(ddof=1)) * np.sqrt(252)
            ann_ret = dr.mean() * 252
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
            return sharpe, ann_ret, max_dd, calmar

        s_m = calc_stats(dr_managed)
        s_s = calc_stats(dr_simple)

        labels = ["Sharpe", "Ann Return", "Max DD", "Calmar"]
        managed_vals = [s_m[0], s_m[1], s_m[2], s_m[3]]
        simple_vals = [s_s[0], s_s[1], s_s[2], s_s[3]]

        x = np.arange(len(labels))
        axes[1, 1].bar(x - 0.15, managed_vals, 0.3, color="purple", alpha=0.7, label="Managed")
        axes[1, 1].bar(x + 0.15, simple_vals, 0.3, color="gray", alpha=0.7, label="Simple")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_title("Metrics Comparison")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()
    


    
        


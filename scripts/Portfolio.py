#implementing portfolio strategies
import lightgbm as lgb
import pandas as pd
from FeatureEngine import *
import os
import pickle
from clean_analysis import Model
from matplotlib import pyplot as plt
import numpy as np
import optuna
from scipy.stats import spearmanr
from HRP import compute_hrp_weights

def retrieve_model(model_folder):
    """gets model information from model folder

    Args:
        model_folder (str): path to the model folder

    Raises:
        Exception: feature cannot be found

    Returns:
        tuple: info for the model
    """
    print(model_folder)
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

        # AFTER — ask the booster which columns it was actually trained on
        trained_features = self.model.model.feature_name()
        missing = [c for c in trained_features if c not in self.model.data.columns]
        if missing:
            raise Exception(
                f"Booster was trained on {len(trained_features)} features, but "
                f"{len(missing)} are missing from regenerated data: {missing[:5]}..."
            )
        self.model.data["pred_return"] = self.model.model.predict(self.model.data[trained_features])

        # Cross-sectional prediction z-score (for ranking)
        self.model.data["pred_zscore"] = self.model.data.groupby("date")["pred_return"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

        print(f"Predictions generated: {len(self.model.data)} rows, "
              f"{self.model.data['act_symbol'].nunique()} stocks, "
              f"{self.model.data['date'].min().date()} -> {self.model.data['date'].max().date()}")
        
    def drop_unnecessary(self, ohlcv_path="../Data/all_ohlcv.feather"):
        """
        Drops everything except essential columns. Computes clean open-to-open
        daily log returns from the full contiguous OHLCV dataset to avoid
        gap-induced return errors.
        """
        # Get the stocks and date range we need returns for
        symbols = self.model.data["act_symbol"].unique().tolist()
        min_date = self.model.data["date"].min() - pd.Timedelta(days=10)
        max_date = self.model.data["date"].max()

        # Load only what we need from the full OHLCV
        ohlcv = pd.read_feather(ohlcv_path)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv = ohlcv[
            (ohlcv["act_symbol"].isin(symbols)) &
            (ohlcv["date"] >= min_date) &
            (ohlcv["date"] <= max_date)
        ].sort_values(["act_symbol", "date"])

        # Compute clean open-to-open log returns from contiguous data
        ohlcv["daily_log_ret"] = ohlcv.groupby("act_symbol")["open"].transform(
            lambda x: np.log(x / x.shift(1))
        ).fillna(0)

        # Merge onto portfolio data
        self.model.data = self.model.data.merge(
            ohlcv[["date", "act_symbol", "daily_log_ret"]],
            on=["date", "act_symbol"],
            how="left",
        )
        self.model.data["daily_log_ret"] = self.model.data["daily_log_ret"].fillna(0)

        keep_cols = ["date", "act_symbol", "pred_return", "pred_zscore", 
                     "daily_log_ret", "open", "close", "high", "low"]

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

    def decile_analysis(self, start_date=None, end_date=None,
                    horizon=1, quantiles=10, plot=True,
                    include_gap=True, price_col="open"):
        """
        Decile analysis on portfolio predictions over a specified date range.
    
        For each date, ranks stocks into `quantiles` buckets by pred_return and 
        computes mean realized forward return per bucket. The forward return is 
        computed from price data, independent of the model's training target.
    
        Args:
            start_date  : start of analysis (inclusive). None = earliest in data.
            end_date    : end of analysis (inclusive). None = latest in data.
            horizon     : forward return horizon in trading days. 1 = next-day.
                         Set to your holding period (e.g., 5 for a 5-day 
                         rebalance) to evaluate at the natural strategy horizon.
            quantiles   : number of buckets (10 = deciles).
            plot        : show diagnostic plots.
            include_gap : include extrapolated (gap) predictions. False for 
                         validated-only.
            price_col   : 'open' for open-to-open (matches open-execution 
                         backtest) or 'close' for close-to-close.
    
        Returns:
            dict with decile_means, spread, spread_sharpe, rank_ic_mean,
            rank_ic_ir, daily_returns, n_obs, horizon.
        """
        if not self.has_data:
            raise Exception("No data. Call get_model_data() first.")
    
        df = self.data.copy()
    
        # --- Compute forward returns from price data ---------------------------
        if price_col not in df.columns:
            raise Exception(
                f"Price column '{price_col}' not in self.data. "
                f"Available: {sorted(df.columns)}"
            )
        df = df.sort_values(["act_symbol", "date"])
        df["fwd_ret"] = df.groupby("act_symbol")[price_col].transform(
            lambda x: np.log(x.shift(-horizon) / x)
        )
    
        # --- Filter on gap policy and date range -------------------------------
        if not include_gap and "is_gap" in df.columns:
            df = df[~df["is_gap"]]
        if start_date is not None:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.to_datetime(end_date)]
    
        df = df.dropna(subset=["pred_return", "fwd_ret"])
        if df.empty:
            raise Exception(
                "No valid rows after filtering. Most likely the horizon shift "
                "exceeded the data tail, or the date range is outside what's loaded."
            )
    
        # --- Assign deciles per date -------------------------------------------
        def _assign(g):
            if len(g) < quantiles:
                return pd.Series(np.nan, index=g.index)
            try:
                return pd.qcut(g["pred_return"], q=quantiles, labels=False,
                               duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=g.index)
    
        df["decile"] = df.groupby("date", group_keys=False).apply(_assign)
        df = df.dropna(subset=["decile"])
        df["decile"] = df["decile"].astype(int)
        if df.empty:
            raise Exception(
                f"No dates with >={quantiles} stocks for quantile assignment."
            )
    
        # --- Aggregate ---------------------------------------------------------
        daily_returns = df.groupby(["date", "decile"])["fwd_ret"].mean().unstack()
        top, bot = daily_returns.columns.max(), daily_returns.columns.min()
        spread   = (daily_returns[top] - daily_returns[bot]).dropna()
        decile_means = daily_returns.mean()
    
        # Annualization: sqrt(252) for 1-day non-overlapping, sqrt(252/h) for
        # h-day overlapping. h-day returns at daily frequency overlap, which 
        # inflates apparent sample size; sqrt(252/h) is the correct correction.
        ann_factor = np.sqrt(252 / horizon)
    
        spread_mean   = spread.mean()
        spread_std    = spread.std()
        spread_sharpe = (spread_mean / spread_std) * ann_factor if spread_std > 0 else np.nan
    
        # --- Rank IC (Spearman) -------------------------------------------------
        daily_ic = df.groupby("date").apply(
            lambda g: g["pred_return"].corr(g["fwd_ret"], method="spearman")
        ).dropna()
        rank_ic_mean = daily_ic.mean()
        rank_ic_std  = daily_ic.std()
        rank_ic_ir   = (rank_ic_mean / rank_ic_std) * ann_factor if rank_ic_std > 0 else np.nan
    
        # --- Report -------------------------------------------------------------
        print(f"\n--- Decile Analysis ---")
        print(f"Period          : {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"Trading days    : {df['date'].nunique()}")
        print(f"Observations    : {len(df)}")
        print(f"Horizon         : {horizon} day(s), price_col={price_col}")
        print(f"Include gap     : {include_gap}")
        print(f"Rank IC (mean)  : {rank_ic_mean:.4f}")
        print(f"Rank IC (IR)    : {rank_ic_ir:.4f}")
        print(f"Spread mean     : {spread_mean:.6f}")
        print(f"Spread Sharpe   : {spread_sharpe:.4f}")
        print("\nMean return by decile:")
        for d, r in decile_means.items():
            bar = "#" * max(0, int(r * 5000)) if r > 0 else ""
            neg = "-" * max(0, int(-r * 5000)) if r < 0 else ""
            print(f"  D{int(d)}: {r:>+10.6f}  {neg}{bar}")
    
        # --- Plot --------------------------------------------------------------
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    
            colors = ["red" if r < 0 else "green" for r in decile_means.values]
            axes[0].bar(decile_means.index.astype(int), decile_means.values, color=colors)
            axes[0].axhline(0, color="black", linewidth=0.5)
            axes[0].set_xlabel(f"Decile (0=lowest pred, {quantiles-1}=highest pred)")
            axes[0].set_ylabel(f"Mean {horizon}-day forward log return")
            axes[0].set_title(f"Mean Realized Return by Decile (IC={rank_ic_mean:.3f})")
    
            cum_spread = spread.cumsum()
            axes[1].plot(cum_spread.index, cum_spread.values, color="steelblue")
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Cumulative Spread Return")
            axes[1].set_title(f"Top - Bottom Decile Spread (Sharpe={spread_sharpe:.2f})")
    
            cum_decile = daily_returns.cumsum()
            cmap = plt.cm.RdYlGn
            n = len(cum_decile.columns)
            for idx, d in enumerate(cum_decile.columns):
                axes[2].plot(cum_decile.index, cum_decile[d],
                             label=f"D{int(d)}", color=cmap(idx / max(1, n - 1)),
                             alpha=0.85)
            axes[2].axhline(0, color="black", linewidth=0.5)
            axes[2].set_xlabel("Date")
            axes[2].set_ylabel("Cumulative Return")
            axes[2].set_title("Cumulative Return by Decile")
            axes[2].legend(loc="best", fontsize=8, ncol=2)
    
            plt.tight_layout()
            plt.show()
    
        return {
            "decile_means":   decile_means,
            "spread":         spread,
            "spread_sharpe":  spread_sharpe,
            "rank_ic_mean":   rank_ic_mean,
            "rank_ic_ir":     rank_ic_ir,
            "daily_returns":  daily_returns,
            "n_obs":          len(df),
            "horizon":        horizon,
        }

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
        fold_models = []
        for i, fold_info in enumerate(self.folds):
            lgb_model, info, _, features, target = retrieve_model(fold_info["model_folder"])
            fold_models.append({
                "lgb_model":     lgb_model,
                "features":      features,
                "target":        target,
                "info":          info,
                "universe_path": info.universe_path[0],
            })

        # Validate feature set consistency across folds
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

        # Validate universe path consistency across folds
        ref_universe = fold_models[0]["universe_path"]
        for i, fm in enumerate(fold_models[1:], start=1):
            if fm["universe_path"] != ref_universe:
                raise Exception(
                    f"Universe path mismatch between fold 0 and fold {i} — "
                    f"cannot compute features once on a single dataset.\n"
                    f"Fold 0 : {ref_universe}\n"
                    f"Fold {i}: {fm['universe_path']}"
                )

        print(f"  Feature set verified: {len(ref_names)} features consistent "
              f"across all {len(fold_models)} folds.")
        print(f"  Universe path: '{ref_universe}'")
        return fold_models

    def _compute_features(self, features, target, universe_path):
        """
        Load the full universe dataset and compute features and targets once.
        universe_path is pulled from the fold model info, exactly as Portfolio does.
        """
        print(f"\nComputing features on full dataset: '{universe_path}'...")
        master = Model(universe_path=universe_path)
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
            universe_path=fold_models[0]["universe_path"],
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

            #fold_data["pred_return"] = fm["lgb_model"].predict(fold_data[feature_keys])
            
            # Ask the LightGBM model exactly which features it was trained on
            trained_features = fm["lgb_model"].feature_name()
            
            # Predict using only that exact subset of columns
            fold_data["pred_return"] = fm["lgb_model"].predict(fold_data[trained_features])

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
        for col in ["open", "close", "fwd_log_ret", "high", "low"]:
            if col in combined.columns:
                keep_cols.append(col)
        target_cols = [k for k in combined.columns if "T" in k.split("_")]
        keep_cols  += [c for c in target_cols if c not in keep_cols]
        keep_cols   = [c for c in keep_cols if c in combined.columns]

        self.data = (combined[keep_cols]
                     .sort_values(["date", "act_symbol"])
                     .reset_index(drop=True))
        self.has_data = True

        # --- 6. Compute clean open-to-open returns from contiguous OHLCV ------
        print("Computing clean open-to-open returns from full OHLCV...")
        symbols = self.data["act_symbol"].unique().tolist()
        min_date = self.data["date"].min() - pd.Timedelta(days=10)
        max_date = self.data["date"].max()

        ohlcv = pd.read_feather(ohlcv_path)
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv = ohlcv[
            (ohlcv["act_symbol"].isin(symbols)) &
            (ohlcv["date"] >= min_date) &
            (ohlcv["date"] <= max_date)
        ].sort_values(["act_symbol", "date"])

        ohlcv["daily_log_ret"] = ohlcv.groupby("act_symbol")["open"].transform(
            lambda x: np.log(x / x.shift(1))
        ).fillna(0)

        self.data = self.data.merge(
            ohlcv[["date", "act_symbol", "daily_log_ret"]],
            on=["date", "act_symbol"],
            how="left",
        )
        self.data["daily_log_ret"] = self.data["daily_log_ret"].fillna(0)

        print(f"  Returns computed: {(self.data['daily_log_ret'] != 0).sum()} "
              f"non-zero out of {len(self.data)} rows")

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
        
    def decile_analysis(self, start_date=None, end_date=None,
                    horizon=1, quantiles=10, plot=True,
                    include_gap=True, price_col="open"):
        """
        Decile analysis on portfolio predictions over a specified date range.

        For each date, ranks stocks into `quantiles` buckets by pred_return and 
        computes mean realized forward return per bucket. The forward return is 
        computed from price data, independent of the model's training target.

        Args:
            start_date  : start of analysis (inclusive). None = earliest in data.
            end_date    : end of analysis (inclusive). None = latest in data.
            horizon     : forward return horizon in trading days. 1 = next-day.
                         Set to your holding period (e.g., 5 for a 5-day 
                         rebalance) to evaluate at the natural strategy horizon.
            quantiles   : number of buckets (10 = deciles).
            plot        : show diagnostic plots.
            include_gap : include extrapolated (gap) predictions. False for 
                         validated-only.
            price_col   : 'open' for open-to-open (matches open-execution 
                         backtest) or 'close' for close-to-close.

        Returns:
            dict with decile_means, spread, spread_sharpe, rank_ic_mean,
            rank_ic_ir, daily_returns, n_obs, horizon.
        """
        if not self.has_data:
            raise Exception("No data. Call get_model_data() first.")

        df = self.data.copy()

        # --- Compute forward returns from price data ---------------------------
        if price_col not in df.columns:
            raise Exception(
                f"Price column '{price_col}' not in self.data. "
                f"Available: {sorted(df.columns)}"
            )
        df = df.sort_values(["act_symbol", "date"])
        df["fwd_ret"] = df.groupby("act_symbol")[price_col].transform(
            lambda x: np.log(x.shift(-horizon) / x)
        )

        # --- Filter on gap policy and date range -------------------------------
        if not include_gap and "is_gap" in df.columns:
            df = df[~df["is_gap"]]
        if start_date is not None:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        df = df.dropna(subset=["pred_return", "fwd_ret"])
        if df.empty:
            raise Exception(
                "No valid rows after filtering. Most likely the horizon shift "
                "exceeded the data tail, or the date range is outside what's loaded."
            )

        # --- Assign deciles per date -------------------------------------------
        def _assign(g):
            if len(g) < quantiles:
                return pd.Series(np.nan, index=g.index)
            try:
                return pd.qcut(g["pred_return"], q=quantiles, labels=False,
                               duplicates="drop")
            except ValueError:
                return pd.Series(np.nan, index=g.index)

        df["decile"] = df.groupby("date", group_keys=False).apply(_assign)
        df = df.dropna(subset=["decile"])
        df["decile"] = df["decile"].astype(int)
        if df.empty:
            raise Exception(
                f"No dates with >={quantiles} stocks for quantile assignment."
            )

        # --- Aggregate ---------------------------------------------------------
        daily_returns = df.groupby(["date", "decile"])["fwd_ret"].mean().unstack()
        top, bot = daily_returns.columns.max(), daily_returns.columns.min()
        spread   = (daily_returns[top] - daily_returns[bot]).dropna()
        decile_means = daily_returns.mean()

        # Annualization: sqrt(252) for 1-day non-overlapping, sqrt(252/h) for
        # h-day overlapping. h-day returns at daily frequency overlap, which 
        # inflates apparent sample size; sqrt(252/h) is the correct correction.
        ann_factor = np.sqrt(252 / horizon)

        spread_mean   = spread.mean()
        spread_std    = spread.std()
        spread_sharpe = (spread_mean / spread_std) * ann_factor if spread_std > 0 else np.nan

        # --- Rank IC (Spearman) -------------------------------------------------
        daily_ic = df.groupby("date").apply(
            lambda g: g["pred_return"].corr(g["fwd_ret"], method="spearman")
        ).dropna()
        rank_ic_mean = daily_ic.mean()
        rank_ic_std  = daily_ic.std()
        rank_ic_ir   = (rank_ic_mean / rank_ic_std) * ann_factor if rank_ic_std > 0 else np.nan

        # --- Report -------------------------------------------------------------
        print(f"\n--- Decile Analysis ---")
        print(f"Period          : {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"Trading days    : {df['date'].nunique()}")
        print(f"Observations    : {len(df)}")
        print(f"Horizon         : {horizon} day(s), price_col={price_col}")
        print(f"Include gap     : {include_gap}")
        print(f"Rank IC (mean)  : {rank_ic_mean:.4f}")
        print(f"Rank IC (IR)    : {rank_ic_ir:.4f}")
        print(f"Spread mean     : {spread_mean:.6f}")
        print(f"Spread Sharpe   : {spread_sharpe:.4f}")
        print("\nMean return by decile:")
        for d, r in decile_means.items():
            bar = "#" * max(0, int(r * 5000)) if r > 0 else ""
            neg = "-" * max(0, int(-r * 5000)) if r < 0 else ""
            print(f"  D{int(d)}: {r:>+10.6f}  {neg}{bar}")

        # --- Plot --------------------------------------------------------------
        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

            colors = ["red" if r < 0 else "green" for r in decile_means.values]
            axes[0].bar(decile_means.index.astype(int), decile_means.values, color=colors)
            axes[0].axhline(0, color="black", linewidth=0.5)
            axes[0].set_xlabel(f"Decile (0=lowest pred, {quantiles-1}=highest pred)")
            axes[0].set_ylabel(f"Mean {horizon}-day forward log return")
            axes[0].set_title(f"Mean Realized Return by Decile (IC={rank_ic_mean:.3f})")

            cum_spread = spread.cumsum()
            axes[1].plot(cum_spread.index, cum_spread.values, color="steelblue")
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Cumulative Spread Return")
            axes[1].set_title(f"Top - Bottom Decile Spread (Sharpe={spread_sharpe:.2f})")

            cum_decile = daily_returns.cumsum()
            cmap = plt.cm.RdYlGn
            n = len(cum_decile.columns)
            for idx, d in enumerate(cum_decile.columns):
                axes[2].plot(cum_decile.index, cum_decile[d],
                             label=f"D{int(d)}", color=cmap(idx / max(1, n - 1)),
                             alpha=0.85)
            axes[2].axhline(0, color="black", linewidth=0.5)
            axes[2].set_xlabel("Date")
            axes[2].set_ylabel("Cumulative Return")
            axes[2].set_title("Cumulative Return by Decile")
            axes[2].legend(loc="best", fontsize=8, ncol=2)

            plt.tight_layout()
            plt.show()

        return {
            "decile_means":   decile_means,
            "spread":         spread,
            "spread_sharpe":  spread_sharpe,
            "rank_ic_mean":   rank_ic_mean,
            "rank_ic_ir":     rank_ic_ir,
            "daily_returns":  daily_returns,
            "n_obs":          len(df),
            "horizon":        horizon,
        }

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
                    
                    #==============================
                    #Old ranking logic:
                    #   Get top longs and shorts
                    longs = sorted_df.head(n_l)
                    shorts = sorted_df.tail(n_s)
                    #==============================

                    #==============================
                    #New ranking logic:
                    #   Keep stocks unless they drop out of the top 20
                    #==============================

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
        print(f"Avg Daily Turnover: {avg_daily_turnover:.2%}")
        print(f"Annual Turnover   : {ann_turnover:.2f}x (portfolio flipped this many times/yr)")

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

        self.pnl_col = "daily_log_ret"

        # --- NEW: CALCULATE ROLLING BETA ---
        print("Calculating 60-day rolling Beta against equal-weight universe...")
        
        # 1. Calculate daily universe return (our market proxy)
        self.data['mkt_ret'] = self.data.groupby('date')[self.pnl_col].transform('mean')
        
        # 2. Market Variance
        mkt_var = self.data.groupby('date')['mkt_ret'].first().rolling(60, min_periods=20).var()
        self.data['mkt_var'] = self.data['date'].map(mkt_var)
        
        # 3. Asset Covariance (Fast vectorized calculation)
        def rolling_cov(group):
            return group[self.pnl_col].rolling(60, min_periods=20).cov(group['mkt_ret'])
            
        self.data['cov'] = self.data.groupby('act_symbol').apply(rolling_cov).reset_index(level=0, drop=True)
        
        # 4. Calculate Beta = Covariance / Variance
        self.data['beta'] = (self.data['cov'] / self.data['mkt_var'])
        
        # 5. Clean up: fill NaNs with 1.0 (market neutral), and clip extremes
        self.data['beta'] = self.data.groupby('act_symbol')['beta'].ffill().fillna(1.0).clip(0.1, 3.0)

        self.data = self.data.sort_values(["date", "pred_return"], ascending=[True, False])
        '''
        Smoothed predictionsxf
        '''
        self.data["smoothed_pred"] = self.data.groupby("act_symbol")["pred_return"].transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()
        )

        # Rolling prediction volatility per asset (signal stability)
        self.data["pred_zscore_vol"] = self.data.groupby("act_symbol")["pred_zscore"].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        ).fillna(self.data["pred_zscore"].std())
        
        # Sort by smoothed pred instead of raw pred
        self.data = self.data.sort_values(["date", "smoothed_pred"], ascending=[True, False])
        
        self.trading_dates = sorted(self.data["date"].unique())
        self._date_groups = {d: g for d, g in self.data.groupby("date")}

        print(f"ManagedBacktest (Tranches): {len(self.trading_dates)} days, "
              f"{self.data['act_symbol'].nunique()} stocks")

    def run(self,
            rebalance_days=None,     # <--- NEW ARGUMENT ADDED HERE
            dd_threshold=-0.05,      
            dd_full_cut=-0.15,       
            target_vol=0.10,         
            vol_lookback=20,         
            vol_cap=1.5,             
            max_gross_exposure=2.0,  
            dd_lookback=252,
            verbose:bool = True,
            weight_by_prediction_volatility:bool = False          
            ):
        
        # Default to the target holding period if no specific rebalance day is provided
        if rebalance_days is None:
            rebalance_days = self.holding_period

        cost_rate = self.cost_bps / 10_000

        # Single active portfolio dictionary
        current_weights = {}
        
        daily_rets = []
        daily_scalars = []
        daily_scalar_components = []
        daily_turnover = []
        daily_components = []

        cum_ret = 1.0
        cum_history = []
        recent_rets = []

        # Risk scalars
        smoothed_scalar = 1.0 
        dd_scalar = 1.0
        vol_scalar = 1.0
        current_dd = 0.0

        for day_idx, date in enumerate(self.trading_dates):
            
            # =======================================================
            # STEP 1: CALCULATE TODAY'S GROSS P&L 
            # =======================================================
            daily_port_gross_ret = 0.0
            
            if date in self._date_groups and current_weights:
                rets = self._date_groups[date].set_index("act_symbol")[self.pnl_col]
                daily_port_gross_ret = sum(w * rets.get(sym, 0.0) for sym, w in current_weights.items())

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
            
            # EMA Smoothing of the risk scalar
            raw_scalar = dd_scalar * vol_scalar
            smoothed_scalar = (0.8 * smoothed_scalar) + (0.2 * raw_scalar)

            # =======================================================
            # STEP 3: REBALANCE LOGIC (TRIGGERED EVERY N DAYS)
            # =======================================================
            turnover = 0.0
            
            # USE THE NEW REBALANCE_DAYS ARGUMENT HERE
            if day_idx % rebalance_days == 0 and date in self._date_groups:
                
                df = self._date_groups[date]
                current_universe_size = len(df)
                new_w = {}
                base_target_gross = 1.0 
                
                # --- Dynamic Sizing (Float = %, Int = Count) ---
                if isinstance(self.n_longs, float) and 0.0 < self.n_longs < 1.0:
                    n_l = max(1, int(current_universe_size * self.n_longs))
                else:
                    n_l = int(self.n_longs)
                    
                if isinstance(self.n_shorts, float) and 0.0 < self.n_shorts < 1.0:
                    n_s = max(1, int(current_universe_size * self.n_shorts))
                else:
                    n_s = int(self.n_shorts)

                if current_universe_size >= n_l + n_s:
                    
                    # 1. Identify what we ALREADY hold
                    current_longs = [sym for sym, w in current_weights.items() if w > 0]
                    current_shorts = [sym for sym, w in current_weights.items() if w < 0]

                    # 2. Rank all stocks today
                    pred_col = "smoothed_pred" if "smoothed_pred" in df.columns else "pred_return"
                    df['rank_long'] = df[pred_col].rank(ascending=False)
                    df['rank_short'] = df[pred_col].rank(ascending=True)

                    # 3. Hysteresis Buffer
                    buffer_multiplier = 2.0 
                    
                    kept_longs = df[(df['act_symbol'].isin(current_longs)) & (df['rank_long'] <= n_l * buffer_multiplier)]
                    needed_longs = max(0, n_l - len(kept_longs))
                    new_longs = df[~df['act_symbol'].isin(current_longs)].sort_values('rank_long').head(needed_longs)
                    longs = pd.concat([kept_longs, new_longs])

                    kept_shorts = df[(df['act_symbol'].isin(current_shorts)) & (df['rank_short'] <= n_s * buffer_multiplier)]
                    needed_shorts = max(0, n_s - len(kept_shorts))
                    new_shorts = df[~df['act_symbol'].isin(current_shorts)].sort_values('rank_short').head(needed_shorts)
                    shorts = pd.concat([kept_shorts, new_shorts])

                    

                    # --- LONGS ---
                    if len(longs) > 0:
                        long_scores = longs["pred_zscore"].abs()
                        if weight_by_prediction_volatility:
                            long_scores = long_scores / (longs["pred_zscore_vol"] + 1e-8)
                        long_total = long_scores.sum()
                        if long_total > 0:
                            long_norm = long_scores / long_total
                        else:
                            long_norm = pd.Series(1.0 / len(longs), index=longs.index)
                    else:
                        long_norm = pd.Series(dtype=float)

                    # --- SHORTS ---
                    if len(shorts) > 0:
                        short_scores = shorts["pred_zscore"].abs()
                        if weight_by_prediction_volatility:
                            short_scores = short_scores / (shorts["pred_zscore_vol"] + 1e-8)
                        short_total = short_scores.sum()
                        if short_total > 0:
                            short_norm = short_scores / short_total
                        else:
                            short_norm = pd.Series(1.0 / len(shorts), index=shorts.index)
                    else:
                        short_norm = pd.Series(dtype=float)

                    # --- APPLY WEIGHTS ---
                    # If we only have longs, give them the full target gross. 
                    # If we have both, split the capital evenly (Dollar Neutral).
                    active_sides = (1 if len(longs) > 0 else 0) + (1 if len(shorts) > 0 else 0)
                    side_allocation = base_target_gross / active_sides if active_sides > 0 else 0.0

                    for idx, r in longs.iterrows():
                        new_w[r["act_symbol"]] = long_norm.loc[idx] * side_allocation * smoothed_scalar
                    for idx, r in shorts.iterrows():
                        new_w[r["act_symbol"]] = -short_norm.loc[idx] * side_allocation * smoothed_scalar

                    # 5. Calculate Turnover
                    turnover = sum(
                        abs(new_w.get(t, 0) - current_weights.get(t, 0))
                        for t in set(list(new_w) + list(current_weights))
                    )
                    
                    # Update active portfolio
                    current_weights = new_w

            # =======================================================
            # STEP 4: CALCULATE FINAL NET RETURN
            # =======================================================
            port_cost = turnover * cost_rate 
            net_port_ret = daily_port_gross_ret - port_cost
            
            cum_ret = cum_ret * (1 + net_port_ret)
            
            cum_history[-1] = cum_ret
            recent_rets[-1] = net_port_ret

            total_gross = sum(abs(w) for w in current_weights.values())

            daily_rets.append(net_port_ret)
            daily_scalars.append(smoothed_scalar)
            daily_turnover.append(turnover) 
            daily_components.append({
                "dd_scalar": dd_scalar,
                "vol_scalar": vol_scalar,
                "combined": smoothed_scalar,
                "drawdown": current_dd,
                "gross_exposure": total_gross,
            })

        self.results = pd.DataFrame({
            "date": self.trading_dates,
            "return": daily_rets,
            "scalar": daily_scalars,
            "turnover": daily_turnover
        }).set_index("date")

        self.components = pd.DataFrame(daily_components, index=self.trading_dates)
        
        # ── Compute daily Rank IC (prediction quality monitor) ──
        daily_ic = []
        for date in self.trading_dates:
            if date in self._date_groups:
                day = self._date_groups[date]
                if len(day) > 10:
                    ic, _ = spearmanr(day["pred_return"], day[self.pnl_col])
                    daily_ic.append(ic if not np.isnan(ic) else 0.0)
                else:
                    daily_ic.append(0.0)
            else:
                daily_ic.append(0.0)
        self.results["ic"] = daily_ic
        
        if verbose == True:
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
        avg_daily_turnover = self.results["turnover"].mean()
        ann_turnover = avg_daily_turnover * 252

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
        print(f"Avg Daily Turnover: {avg_daily_turnover:.2%}")
        print(f"Annual Turnover   : {ann_turnover:.2f}x (portfolio flipped this many times/yr)")

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

        # Rolling Sharpe + Rolling IC (dual axis)
        rs = pd.Series(dr).rolling(60).apply(
            lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252)
        )
        axes[1, 0].plot(dates, rs.values, color="blue", lw=1, label="Sharpe")
        axes[1, 0].axhline(0, color="red", ls="--", lw=0.5)
        axes[1, 0].set_ylabel("Sharpe", color="blue")
        axes[1, 0].tick_params(axis="y", labelcolor="blue")

        ax_ic = axes[1, 0].twinx()
        rolling_ic = self.results["ic"].rolling(60).mean()
        ax_ic.plot(dates, rolling_ic.values, color="green", lw=1, alpha=0.8, label="IC")
        ax_ic.axhline(0, color="green", ls=":", lw=0.5, alpha=0.5)
        ax_ic.set_ylabel("Rank IC", color="green")
        ax_ic.tick_params(axis="y", labelcolor="green")

        # Combined legend
        lines1, labels1 = axes[1, 0].get_legend_handles_labels()
        lines2, labels2 = ax_ic.get_legend_handles_labels()
        axes[1, 0].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        axes[1, 0].set_title("Rolling 60-Day Sharpe & IC")

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
    

class StrategyOptimizer:
    def __init__(self, portfolio, strategy_class, param_func, start_date=None, end_date=None):
        """
        A flexible optimizer that works with ANY portfolio strategy.
        
        Args:
            portfolio: The Portfolio object containing the data.
            strategy_class: The uninstantiated class of the strategy (e.g., ManagedBacktest).
            param_func: A function that takes an Optuna 'trial' and returns a dictionary 
                        of parameters to pass into the strategy's __init__ and run() methods.
            start_date: Optional string (e.g. '2016-01-01') to isolate In-Sample data.
            end_date: Optional string to isolate In-Sample data.
        """
        self.strategy_class = strategy_class
        self.param_func = param_func
        
        # Create a dummy portfolio to safely slice data without modifying the original
        class DummyPort: pass
        self.eval_portfolio = DummyPort()
        self.eval_portfolio.has_data = True
        
        mask = pd.Series(True, index=portfolio.data['date'].index) if type(portfolio.data.index) != pd.DatetimeIndex else pd.Series(True, index=portfolio.data.index)
        
        if start_date:
            mask &= (portfolio.data['date'] >= pd.to_datetime(start_date))
        if end_date:
            mask &= (portfolio.data['date'] <= pd.to_datetime(end_date))
            
        self.eval_portfolio.data = portfolio.data[mask].copy()

    def objective(self, trial):
        # 1. Fetch the parameters for this specific trial from the user's custom function
        params = self.param_func(trial)
        init_kwargs = params.get("init_kwargs", {})
        run_kwargs = params.get("run_kwargs", {})
        
        # Ensure the backtest runs silently during optimization
        run_kwargs['verbose'] = False

        # 2. Dynamically instantiate and run the strategy
        try:
            strategy = self.strategy_class(self.eval_portfolio, **init_kwargs)
            results = strategy.run(**run_kwargs)
        except Exception as e:
            return -999.0  # Fail trial if parameters break the strategy

        dr = results['return'].values
        if len(dr) < 100 or np.std(dr) == 0:
            return -999.0

        # =================================================================
        # 3. THE "ALL-WEATHER" OBJECTIVE FUNCTION
        # =================================================================
        
        # Calculate Global Max Drawdown
        cum = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(cum)
        max_dd = abs(((cum - peak) / peak).min())

        # Group returns by Year to ensure consistency across ALL time periods
        df_res = results.copy()
        df_res['year'] = df_res.index.year
        
        # Calculate the Sharpe Ratio for each individual year
        def calc_annual_sharpe(x):
            if x.std() == 0: return 0
            return (x.mean() / x.std()) * np.sqrt(252)
            
        yearly_sharpes = df_res.groupby('year')['return'].apply(calc_annual_sharpe)
        yearly_returns = df_res.groupby('year')['return'].sum()
        
        # Metrics
        mean_yearly_sharpe = yearly_sharpes.mean()
        min_yearly_sharpe = yearly_sharpes.min()
        yearly_win_rate = (yearly_returns > 0).mean() # What % of years were profitable?
        
        # SCORE: Average Yearly Sharpe * Yearly Win Rate * (1 - Drawdown)
        score = mean_yearly_sharpe * yearly_win_rate * (1 - max_dd)
        
        # SEVERE PENALTY: If ANY year has a deeply negative Sharpe, slash the score.
        # This prevents the 2016-2021 flatline from hiding behind the 2024 massive run.
        if min_yearly_sharpe < -0.5:
            score *= 0.2
            
        return score

    def optimize(self, n_trials=50):
        # Suppress Optuna's heavy console logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="maximize")
        print(f"Running flexible optimization for {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        print("\n--- OPTIMIZATION COMPLETE ---")
        print(f"Best Score: {study.best_value:.4f}")
        print("Best Parameters:")
        
        # Reconstruct the optimal dictionary layout
        best_params = self.param_func(optuna.trial.FixedTrial(study.best_params))
        
        for key, val in study.best_params.items():
            print(f"  {key}: {val}")
            
        return best_params

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from HRP import compute_hrp_weights

class HRPBacktest:
    """
    Ledger-Based Long-Short backtest using Hierarchical Risk Parity (HRP).
    Tracks exact cash, shares, and Mark-to-Market value to ensure flawless math.
    """

    def __init__(self, portfolio, n_longs=20, n_shorts=20, holding_period=5, 
                 rebalance_days=1, cost_bps=5.0, hrp_lookback=60, 
                 smooth_predictions=False, buffer_multiplier=1.0, initial_capital=2000.0,
                 sizing_method = "dollar_neutral", volatility_type:str = "simple"):
        
        if not portfolio.has_data:
            raise Exception("Portfolio has no data. Run get_model_data() first.")

        if sizing_method not in ["dollar_neutral", "beta_neutral"]:
            raise ValueError("sizing_method must be 'dollar_neutral' or 'beta_neutral'")
        self.sizing_method = sizing_method

        self.data = portfolio.data.copy()
        self.n_longs = n_longs
        self.n_shorts = n_shorts
        self.holding_period = holding_period
        self.rebalance_days = rebalance_days
        self.cost_bps = cost_bps
        self.hrp_lookback = hrp_lookback
        self.smooth_predictions = smooth_predictions
        self.buffer_multiplier = buffer_multiplier
        self.initial_capital = initial_capital

        self.num_tranches = max(1, int(self.holding_period / self.rebalance_days))

        if volatility_type not in ["simple", "yang_zhang", "ewma"]:
            raise ValueError("volatility_type must be 'simple' or 'yang_zhang'")
        self.volatility_type = volatility_type

        if "close" not in self.data.columns:
            raise Exception("Portfolio data must contain 'close' column.")

        self.data = self.data.sort_values(["date", "act_symbol"])
        
        # Build a master Price History table. 
        # ffill() ensures if a stock halts or drops from the universe, we can still value existing shares.
        # Need open prices for execution and MTM
        if "open" not in self.data.columns:
            raise Exception("Portfolio data must contain 'open' column for open-execution backtest.")

        # =====================================================================
        # CONTIGUOUS HISTORICAL DATA FETCH 
        # (Solves the "New Stock in Universe" missing history problem)
        # =====================================================================
        print("Fetching contiguous historical OHLCV data for warm-up periods...")
        
        # Find exactly what stocks we need and how far back we must look
        all_symbols = self.data['act_symbol'].unique().tolist()
        lookback_days = max(self.hrp_lookback, 60) + 30 
        min_date = self.data['date'].min() - pd.Timedelta(days=lookback_days)
        max_date = self.data['date'].max()

        import pyarrow.dataset as ds
        dataset = ds.dataset("../Data/all_ohlcv.feather", format="feather")
        filter_cond = (ds.field('act_symbol').isin(all_symbols)) & \
                      (ds.field('date') >= min_date) & \
                      (ds.field('date') <= max_date)
        
        raw_ohlcv = dataset.to_table(filter=filter_cond).to_pandas()
        raw_ohlcv['date'] = pd.to_datetime(raw_ohlcv['date'])

        # 1. Build master price history (Open prices for execution and MTM)
        self.price_history = raw_ohlcv.pivot_table(index="date", columns="act_symbol", values="open").ffill()
        self.returns_pivot = self.price_history.pct_change().clip(lower=-0.75, upper=1.0).fillna(0)

        # 2. Build Beta Matrix (if needed)
        if self.sizing_method == "beta_neutral":
            print("Pre-computing 60-day rolling Beta for Beta-Neutral sizing...")
            mkt_ret = self.returns_pivot.mean(axis=1) 
            mkt_var = mkt_ret.rolling(60, min_periods=20).var()
            rolling_cov = self.returns_pivot.rolling(60, min_periods=20).cov(mkt_ret)
            self.beta_pivot = rolling_cov.div(mkt_var, axis=0).ffill().fillna(1.0).clip(0.1, 3.0)

        # 3. Build Vectorized Yang-Zhang/EWMA Volatility Panels
        self.yz_vol_panel = None
        self.ewma_vol_panel = None
        
        if self.volatility_type == "yang_zhang":
            if all(c in raw_ohlcv.columns for c in ["open", "high", "low", "close"]):
                print("Pre-computing Vectorized Yang-Zhang Volatility panel...")
                df_O = self.price_history
                df_H = raw_ohlcv.pivot_table(index="date", columns="act_symbol", values="high").ffill()
                df_L = raw_ohlcv.pivot_table(index="date", columns="act_symbol", values="low").ffill()
                df_C = raw_ohlcv.pivot_table(index="date", columns="act_symbol", values="close").ffill()
                
                log_ho = np.log(df_H / df_O)
                log_lo = np.log(df_L / df_O)
                log_co = np.log(df_C / df_O)
                log_oc = np.log(df_O / df_C.shift(1))
                
                rs_var = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))
                
                window = 20
                overnight_var = log_oc.rolling(window=window).var()
                open_close_var = log_co.rolling(window=window).var()
                rs_var_rolling = rs_var.rolling(window=window).mean()
                
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                yz_var = overnight_var + (k * open_close_var) + ((1 - k) * rs_var_rolling)
                self.yz_vol_panel = np.sqrt(yz_var * 252)
            else:
                print("WARNING: Missing OHLC data. Falling back to 'simple' volatility.")
                self.volatility_type = "simple"
                
        elif self.volatility_type == "ewma":
            print("Pre-computing Vectorized EWMA Volatility panel...")
            # We use the clean, clipped returns_pivot we already built for HRP
            # span=20 applies a half-life roughly equivalent to a 20-day SMA, but with exponential decay
            ewma_var = self.returns_pivot.ewm(span=20, min_periods=10).var()
            self.ewma_vol_panel = np.sqrt(ewma_var * 252)
        # =====================================================================

        # Apply optional smoothing
        if self.smooth_predictions:
            self.data = self.data.sort_values(["act_symbol", "date"])
            self.data["final_pred"] = self.data.groupby("act_symbol")["pred_return"].transform(
                lambda x: x.ewm(span=3, min_periods=1).mean()
            )
        else:
            self.data["final_pred"] = self.data["pred_return"]
        
        self.data = self.data.sort_values(["date", "final_pred"], ascending=[True, False])
        self.trading_dates = sorted(self.data["date"].unique())
        self._date_groups = {d: g for d, g in self.data.groupby("date")}

        print(f"Ledger-Based HRPBacktest Initialized.")
        print(f"Tranches: {self.num_tranches} | Starting Capital: ${self.initial_capital:,.2f}")

    def run(self, verbose=True):
        cost_rate = self.cost_bps / 10_000
        
        # Initialize sub-accounts for each tranche
        self.tranches = []
        for _ in range(self.num_tranches):
            self.tranches.append({
                'cash': self.initial_capital / self.num_tranches,
                'shares': {}
            })
        
        daily_rets = []
        daily_turnover_pct = []
        daily_components = []
        hrp_stats = []

        active_eff_n_long, active_eff_n_short = 0, 0
        active_max_w_long, active_max_w_short = 0, 0
        
        # --- NEW: PRE-COMPUTE DAILY 1-DAY IC FOR CONVICTION SCALING ---
        print("Pre-computing historical Rank IC for conviction scaling...")
        df_ic = self.data.copy()
        
        # Calculate the 1-day forward return to evaluate the prediction
        df_ic['fwd_ret'] = df_ic.groupby('act_symbol')['close'].shift(-1) / df_ic['close'] - 1.0
        
        def calc_safe_ic(group):
            if len(group) > 10:
                val, _ = spearmanr(group['final_pred'], group['fwd_ret'])
                return val if not np.isnan(val) else 0.0
            return 0.0
            
        raw_daily_ic = df_ic.groupby('date').apply(calc_safe_ic)
        
        # CRITICAL: Shift by 1! 
        # On Tuesday evening, we only know the IC of Monday's predictions (which resolved Tuesday).
        # We must shift it so 'date' contains the most recently resolved IC without lookahead bias.
        self.historical_ic = raw_daily_ic.shift(1).fillna(0.0)

        # --- NEW: PRE-COMPUTE CROSS-SECTIONAL PREDICTION SPREAD ---
        print("Pre-computing cross-sectional prediction spread...")
        # Std dev of today's predictions across all assets
        self.daily_spread = self.data.groupby('date')['final_pred'].std().fillna(0.0)
        
        prev_total_nav = self.initial_capital
        peak_nav = self.initial_capital
        # ... (rest of the code continues: daily_scalars = [], etc.)

        prev_total_nav = self.initial_capital

        # --- NEW: Tracking variables for Scaling ---
        peak_nav = self.initial_capital
        daily_scalars = []
        daily_scalar_components = []
        if not hasattr(self, 'use_dynamic_scaling'):
            self.use_dynamic_scaling = False

        for day_idx, date in enumerate(self.trading_dates):
            
            # Fast lookup for today's closing prices for every stock ever traded
            today_prices = self.price_history.loc[date]

            # =========================================================
            # STEP 1: MARK-TO-MARKET PORTFOLIO VALUATION
            # =========================================================
            tranche_navs = []
            for t in self.tranches:
                # Value = Cash + sum(shares * price)
                # (For shorts, shares are negative, so price gains reduce NAV, as they should)
                nav = t['cash']
                for sym, shares in t['shares'].items():
                    nav += shares * today_prices[sym]
                tranche_navs.append(nav)
                
            total_nav = sum(tranche_navs)

            # --- NEW: DYNAMIC SCALING LOGIC ---
            # =========================================================
            # STEP 2: I IF APPLICABLE, IMPLEMENT DYNAMIC 
            # =========================================================
            active_scalar = 1.0
            dd_scalar = 1.0
            vol_scalar = 1.0
            ic_scalar = 1.0
            spread_scalar = 1.0
            
            # Update peak NAV and current drawdown
            """
            Old, using total NAV
            if total_nav > peak_nav:
                peak_nav = total_nav
            current_dd = (total_nav / peak_nav) - 1.0"""

            #new, using rolling NAV
            if not hasattr(self, 'nav_history'):
                self.nav_history = []
            self.nav_history.append(total_nav)
            
            # Use a 252-day rolling peak (1 year)
            rolling_peak = max(self.nav_history[-60:]) 
            current_dd = (total_nav / rolling_peak) - 1.0 if rolling_peak > 0 else 0.0
            
            if self.use_dynamic_scaling and day_idx > max(self.vol_lookback, self.ic_lookback):
                
                # 1. DRAWDOWN CONTROL (Step-Function)
                if current_dd <= self.dd_kill_threshold:
                    dd_scalar = 0.0  # Kill switch
                elif current_dd <= self.dd_warning_threshold:
                    dd_scalar = self.dd_penalty  # Penalty cut
                else:
                    dd_scalar = 1.0
                
                # 2. VOLATILITY SCALING
                # Use daily_rets list (which contains up to yesterday's return)
                # 2. VOLATILITY SCALING
                if self.volatility_type == "yang_zhang" and self.yz_vol_panel is not None and date in self.yz_vol_panel.index:
                    # --- YANG ZHANG METHOD ---
                    today_yz = self.yz_vol_panel.loc[date].dropna()
                    
                    if len(today_yz) > 0:
                        market_regime_vol = today_yz.median()
                        if market_regime_vol > 0:
                            vol_scalar = self.target_vol / market_regime_vol
                            vol_scalar = min(self.max_vol_leverage, vol_scalar)
                        else:
                            vol_scalar = 1.0
                    else:
                        vol_scalar = 1.0

                elif self.volatility_type == "ewma" and getattr(self, 'ewma_vol_panel', None) is not None and date in self.ewma_vol_panel.index:
                    # --- NEW: EWMA METHOD ---
                    today_ewma = self.ewma_vol_panel.loc[date].dropna()
                    if len(today_ewma) > 0:
                        market_regime_vol = today_ewma.median()
                        if market_regime_vol > 0:
                            vol_scalar = self.target_vol / market_regime_vol
                            vol_scalar = min(self.max_vol_leverage, vol_scalar)
                        else:
                            vol_scalar = 1.0
                    else:
                        vol_scalar = 1.0

                else:
                    # --- SIMPLE METHOD (Default) ---
                    # Uses standard deviation of recent portfolio returns
                    recent_rets = daily_rets[-self.vol_lookback:]
                    realized_vol = np.std(recent_rets) * np.sqrt(252) if len(recent_rets) > 0 else 0.0
                    
                    if realized_vol > 0:
                        vol_scalar = self.target_vol / realized_vol
                        vol_scalar = min(self.max_vol_leverage, vol_scalar)
                    else:
                        vol_scalar = 1.0
                    
                # 3. CONVICTION SCALING (Kelly Proxy: Rolling Hit Rate)
                """
                Old, linearly scaling down conviction
                # If the model has been losing consistently for 20 days, scale down smoothly.
                conviction_scalar = 1.0
                if self.use_conviction_scaling:
                    win_rate = sum(1 for r in recent_rets[-self.conviction_lookback:] if r > 0) / self.conviction_lookback
                    # If win rate drops below 40%, start linearly scaling down exposure
                    if win_rate < 0.40:
                        conviction_scalar = max(0.0, win_rate / 0.40)"""
                '''
                New: hard stop on conviction
                
                conviction_scalar = 1.0
                if self.use_conviction_scaling:
                    # Look at the last 5 days
                    recent_5 = daily_rets[-self.conviction_lookback:]
                    win_rate = sum(1 for r in recent_5 if r > 0) / len(recent_5)
                    
                    # If we lost money on 4 out of the last 5 days, the market is broken.
                    # Hard-cut exposure to 0% immediately. 
                    if win_rate <= 0.20:
                        conviction_scalar = 0.0
                '''
                '''
                New: hard stop w/ heartbeat
                '''
                
                '''
                Old Method w/ heartbeat for conviction
                # 3. CONVICTION SCALING (With Heartbeat)
                conviction_scalar = 1.0
                if self.use_conviction_scaling:
                    # Optional: Add a tiny epsilon so exact 0.0 returns (cash days) don't count as losses
                    win_rate = sum(1 for r in recent_rets[-self.conviction_lookback:] if r > -1e-6) / self.conviction_lookback
                    
                    if win_rate < 0.40:
                        # THE FIX: Floor the scalar at 0.10 (10% Heartbeat Exposure)
                        raw_conviction = win_rate / 0.40
                        conviction_scalar = max(0.10, raw_conviction)
                    
                    # 4. APPLY THE MOST RESTRICTIVE SCALAR (The Min Function)
                    active_scalar = min(dd_scalar, vol_scalar, conviction_scalar)
                '''

                # 3. ROLLING IC SCALING (Smoothed S-Curve Alpha Sensor)
                ic_scalar = 1.0
                if self.use_ic_scaling:
                    recent_ics = self.historical_ic.iloc[day_idx - self.ic_lookback : day_idx]
                    mean_ic = recent_ics.mean()
                    
                    # --- SMOOTHED S-CURVE PARAMETERS ---
                    heartbeat = 0.33
                    
                    # Shift the cliff lower. (e.g., if threshold is 0.02, midpoint is now 0.005).
                    # It will only aggressively cut exposure if IC approaches zero.
                    midpoint = self.ic_threshold * 0.25  
                    
                    # Flatten the slope. (Changed from 10.0 to 3.0).
                    # Lower numbers = wider, smoother transition. Higher numbers = binary light-switch.
                    steepness = 1 
                    k = steepness / self.ic_threshold  
                    
                    exponent = np.clip(-k * (mean_ic - midpoint), -100, 100)
                    sigmoid = 1.0 / (1.0 + np.exp(exponent))
                    
                    ic_scalar = heartbeat + (1.0 - heartbeat) * sigmoid
                        
                # 4. APPLY THE MOST RESTRICTIVE SCALAR (The Min Function)
                # --- NEW: PREDICTION SPREAD SCALING ---
                spread_scalar = 1.0
                if self.use_spread_scaling and day_idx > self.spread_lookback:
                    # Get the spread history for the lookback window
                    recent_spreads = self.daily_spread.iloc[day_idx - self.spread_lookback : day_idx].values
                    today_spread = self.daily_spread.iloc[day_idx]
                    
                    if len(recent_spreads) > 0 and recent_spreads.std() > 0:
                        # Calculate rank percentile of today's spread (0.0 to 1.0)
                        # e.g., 0.10 means today's spread is smaller than 90% of recent days
                        spread_pct = np.sum(recent_spreads <= today_spread) / len(recent_spreads)
                        
                        # Map the percentile to our allowed exposure range [floor, 1.0]
                        spread_scalar = self.spread_floor + (1.0 - self.spread_floor) * spread_pct
                    else:
                        spread_scalar = 1.0

                # 4. APPLY THE MOST RESTRICTIVE SCALAR (The Min Function)
                # Now includes spread_scalar!
                active_scalar = min(dd_scalar, vol_scalar, ic_scalar, spread_scalar)
                #active_scalar = min(dd_scalar, vol_scalar, ic_scalar)
                
            daily_scalars.append(active_scalar)

            daily_scalar_components.append({
                "dd_scalar": dd_scalar,
                "vol_scalar": vol_scalar,
                "ic_scalar": ic_scalar,
                "spread_scalar": spread_scalar,
                "active_scalar": active_scalar,
            })

            # --- END DYNAMIC SCALING LOGIC ---

            # =========================================================
            # STEP 2: REBALANCE ACTIVE TRANCHE
            # =========================================================
            turnover_dollars = 0.0
            gross_exposure_dollars = sum(
                abs(shares) * today_prices[sym] 
                for t in self.tranches for sym, shares in t['shares'].items()
            )
            
            if day_idx % self.rebalance_days == 0 and date in self._date_groups:
                tranche_idx = (day_idx // self.rebalance_days) % self.num_tranches
                #budget = tranche_navs[tranche_idx] old, before dynamic scaling
                tranche_total_value = tranche_navs[tranche_idx]
                budget = tranche_total_value * active_scalar
                
                df = self._date_groups[date]
                current_universe_size = len(df)
                
                n_l = max(1, int(current_universe_size * self.n_longs)) if isinstance(self.n_longs, float) and 0.0 < self.n_longs < 1.0 else int(self.n_longs)
                n_s = max(1, int(current_universe_size * self.n_shorts)) if isinstance(self.n_shorts, float) and 0.0 < self.n_shorts < 1.0 else int(self.n_shorts)

                if current_universe_size >= n_l + n_s:
                    
                    # Hysteresis: Look at shares held across ALL tranches
                    current_longs = set(sym for t in self.tranches for sym, sh in t['shares'].items() if sh > 0)
                    current_shorts = set(sym for t in self.tranches for sym, sh in t['shares'].items() if sh < 0)

                    # --- EXPLICIT LONG/SHORT TOGGLES ---
                    if n_l > 0:
                        df['rank_long'] = df["final_pred"].rank(ascending=False)
                        kept_longs = df[(df['act_symbol'].isin(current_longs)) & (df['rank_long'] <= n_l * self.buffer_multiplier)]
                        needed_longs = max(0, n_l - len(kept_longs))
                        new_longs = df[~df['act_symbol'].isin(current_longs)].sort_values('rank_long').head(needed_longs)
                        longs = pd.concat([kept_longs, new_longs])
                    else:
                        longs = pd.DataFrame(columns=df.columns)

                    if n_s > 0:
                        df['rank_short'] = df["final_pred"].rank(ascending=True)
                        kept_shorts = df[(df['act_symbol'].isin(current_shorts)) & (df['rank_short'] <= n_s * self.buffer_multiplier)]
                        needed_shorts = max(0, n_s - len(kept_shorts))
                        new_shorts = df[~df['act_symbol'].isin(current_shorts)].sort_values('rank_short').head(needed_shorts)
                        shorts = pd.concat([kept_shorts, new_shorts])
                    else:
                        shorts = pd.DataFrame(columns=df.columns)

                    # --- HRP WEIGHT CALCULATION ---
                    current_date_idx = self.returns_pivot.index.get_loc(date)
                    start_idx = max(0, current_date_idx - self.hrp_lookback)
                    past_returns = self.returns_pivot.iloc[start_idx:current_date_idx]

                    def get_safe_hrp_weights(symbols):
                        if len(symbols) == 0: return {}, 0.0, 0.0
                        rets = past_returns[symbols]
                        raw_weights = compute_hrp_weights(rets, linkage_method="single")
                        
                        dropped_symbols = [s for s in symbols if s not in raw_weights]
                        if dropped_symbols:
                            fallback_w = min(raw_weights.values()) / 2.0 if raw_weights else 1.0 / len(symbols)
                            for sym in dropped_symbols: raw_weights[sym] = fallback_w
                                
                        total = sum(raw_weights.values())
                        final_w = {k: v / total for k, v in raw_weights.items() if total > 0}
                        
                        w_arr = np.array(list(final_w.values()))
                        return final_w, (1.0 / np.sum(w_arr**2) if len(w_arr) > 0 else 0), (np.max(w_arr) if len(w_arr) > 0 else 0)

                    long_weights, active_eff_n_long, active_max_w_long = get_safe_hrp_weights(longs["act_symbol"].tolist())
                    short_weights, active_eff_n_short, active_max_w_short = get_safe_hrp_weights(shorts["act_symbol"].tolist())

                    # --- DYNAMIC BUYING POWER ALLOCATION ---
                    #allows for beta-neutral or dollar-neutral
                    active_sides = (1 if len(long_weights) > 0 else 0) + (1 if len(short_weights) > 0 else 0)
                    
                    if self.sizing_method == "dollar_neutral":
                        if active_sides == 2:
                            side_budget = budget / 2.5 # 100% Long + 150% Short Margin = 2.5
                            long_budget = side_budget
                            short_budget = side_budget
                        elif active_sides == 1:
                            long_budget = budget / 1.0 if len(long_weights) > 0 else 0.0
                            short_budget = budget / 1.5 if len(short_weights) > 0 else 0.0
                        else:
                            long_budget = 0.0
                            short_budget = 0.0

                    elif self.sizing_method == "beta_neutral":
                        if active_sides == 2:
                            today_betas = self.beta_pivot.loc[date]
                            
                            basket_beta_long = sum(w * today_betas.get(sym, 1.0) for sym, w in long_weights.items())
                            basket_beta_short = sum(w * today_betas.get(sym, 1.0) for sym, w in short_weights.items())
                            
                            if pd.isna(basket_beta_long) or basket_beta_long == 0: basket_beta_long = 1.0
                            if pd.isna(basket_beta_short) or basket_beta_short == 0: basket_beta_short = 1.0
                            
                            beta_ratio = basket_beta_short / basket_beta_long
                            beta_ratio = np.clip(beta_ratio, 0.33, 3.0)
                            
                            short_budget = budget / (beta_ratio + 1.5)
                            long_budget = short_budget * beta_ratio
                        elif active_sides == 1:
                            long_budget = budget / 1.0 if len(long_weights) > 0 else 0.0
                            short_budget = budget / 1.5 if len(short_weights) > 0 else 0.0
                        else:
                            long_budget = 0.0
                            short_budget = 0.0

                    new_shares = {}
                    for sym, w in long_weights.items():
                        new_shares[sym] = (w * long_budget) / today_prices[sym]
                    for sym, w in short_weights.items():
                        new_shares[sym] = -(w * short_budget) / today_prices[sym]

                    # --- EXECUTION & COSTS ---
                    old_shares = self.tranches[tranche_idx]['shares']
                    all_symbols = set(new_shares.keys()).union(old_shares.keys())
                    
                    for sym in all_symbols:
                        old_s = old_shares.get(sym, 0.0)
                        new_s = new_shares.get(sym, 0.0)
                        turnover_dollars += abs(new_s - old_s) * today_prices[sym]
                        
                    cost = turnover_dollars * cost_rate
                    
                    # Update Tranche Sub-Account
                    # Cash = Starting Budget - Cost of buying Longs + Proceeds from selling Shorts
                    # Update Tranche Sub-Account
                    net_spend = sum(new_shares[sym] * today_prices[sym] for sym in new_shares)
                    
                    # THE FIX: Subtract spend from the total starting tranche NAV, not the scaled budget!
                    self.tranches[tranche_idx]['cash'] = tranche_navs[tranche_idx] - net_spend - cost
                    self.tranches[tranche_idx]['shares'] = new_shares
                    
                    # Adjust Total NAV down by the exact cost of the trades made today
                    total_nav -= cost

            # =========================================================
            # STEP 3: RECORD DAILY METRICS
            # =========================================================
            if prev_total_nav <= 0:
                daily_ret = 0.0
                total_nav = 0.0  # Stay at zero
            else:
                daily_ret = (total_nav / prev_total_nav) - 1.0
                
            prev_total_nav = total_nav

            daily_rets.append(daily_ret)
            daily_turnover_pct.append(turnover_dollars / total_nav if total_nav > 0 else 0)
            daily_components.append({"gross_exposure": gross_exposure_dollars / total_nav if total_nav > 0 else 0})
            
            hrp_stats.append({
                "eff_n_long": active_eff_n_long, "eff_n_short": active_eff_n_short,
                "max_w_long": active_max_w_long, "max_w_short": active_max_w_short
            })

        self.results = pd.DataFrame({
            "date": self.trading_dates,
            "return": daily_rets,
            "turnover": daily_turnover_pct
        }).set_index("date")

        self.scaling_history = pd.DataFrame(daily_scalar_components, index=self.trading_dates)

        self.components = pd.DataFrame(daily_components, index=self.trading_dates)
        self.hrp_stats = pd.DataFrame(hrp_stats, index=self.trading_dates)
        
        # Calculate Information Coefficient (IC)
        daily_ic = []
        for date in self.trading_dates:
            if date in self._date_groups:
                day = self._date_groups[date]
                # Because we no longer track a forward return column natively, we build it temporarily for IC tracking
                df_ic = day.copy()
                df_ic['fwd_close'] = self.price_history.shift(-1).loc[date, df_ic['act_symbol']].values
                df_ic['fwd_ret'] = (df_ic['fwd_close'] / df_ic['close']) - 1.0
                
                df_ic = df_ic.dropna(subset=['fwd_ret'])
                if len(df_ic) > 10:
                    ic, _ = spearmanr(df_ic["final_pred"], df_ic["fwd_ret"])
                    daily_ic.append(ic if not np.isnan(ic) else 0.0)
                else:
                    daily_ic.append(0.0)
            else:
                daily_ic.append(0.0)
        self.results["ic"] = daily_ic
        
        if verbose:
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

        current_dur, max_dur = 0, 0
        for uw in cum < peak:
            current_dur = current_dur + 1 if uw else 0
            max_dur = max(max_dur, current_dur)

        avg_daily_turnover = self.results["turnover"].mean()

        print(f"\n{'='*50}")
        print(f"LEDGER-BASED HRP TRANCHE BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Sharpe           : {sharpe:.4f}")
        print(f"Annual Return    : {ann_ret:.2%}")
        print(f"Total Return     : {cum[-1] - 1:.2%}")
        print(f"Max Drawdown     : {max_dd:.2%}")
        print(f"Max DD Duration  : {max_dur} days")
        print(f"Calmar           : {calmar:.4f}")
        print(f"Win Rate         : {(dr > 0).mean():.2%}")
        print(f"Avg Daily Turnover: {avg_daily_turnover:.2%}")
        print(f"Annual Turnover   : {avg_daily_turnover * 252:.2f}x")
        print(f"Avg Eff. N Longs  : {self.hrp_stats['eff_n_long'].mean():.1f} (Target: {self.n_longs})")
        print(f"Avg Max Wt Long   : {self.hrp_stats['max_w_long'].mean():.2%}")

    def plot(self):
        if not hasattr(self, "results"): raise Exception("Run run() first.")

        dr = self.results["return"].values
        dates = self.results.index
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))

        cum = np.cumprod(1 + dr)
        axes[0, 0].plot(dates, cum, color="purple", lw=1.5)
        
        axes[0, 0].axhline(1, color="black", lw=0.5)
        axes[0, 0].set_title("Cumulative Return")

        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        axes[0, 1].fill_between(dates, dd, 0, color="red", alpha=0.4)
        axes[0, 1].set_title("Drawdown")

        axes[0, 2].plot(dates, self.hrp_stats["eff_n_long"], color="blue", lw=1, alpha=0.8, label="Eff. N (Longs)")
        axes[0, 2].plot(dates, self.hrp_stats["eff_n_short"], color="red", lw=1, alpha=0.8, label="Eff. N (Shorts)")
        axes[0, 2].axhline(self.n_longs, color="gray", ls="--", lw=1, label="Target N")
        axes[0, 2].set_title("HRP Diversification (Effective N)")
        axes[0, 2].legend(loc="lower right")

        rs = pd.Series(dr).rolling(60).apply(lambda x: x.mean() / (x.std() + 1e-8) * np.sqrt(252))
        axes[1, 0].plot(dates, rs.values, color="blue", lw=1, label="Sharpe")
        axes[1, 0].axhline(0, color="red", ls="--", lw=0.5)
        axes[1, 0].set_ylabel("Sharpe", color="blue")
        
        ax_ic = axes[1, 0].twinx()
        rolling_ic = self.results["ic"].rolling(60).mean()
        ax_ic.plot(dates, rolling_ic.values, color="green", lw=1, alpha=0.8, label="IC")
        ax_ic.axhline(0, color="green", ls=":", lw=0.5, alpha=0.5)
        ax_ic.set_ylabel("Rank IC", color="green")
        axes[1, 0].set_title("Rolling 60-Day Sharpe & IC")

        monthly = pd.Series(dr, index=dates).resample("ME").sum()
        axes[1, 1].bar(range(len(monthly)), monthly.values, color=["green" if r > 0 else "red" for r in monthly], alpha=0.7)
        axes[1, 1].set_title("Monthly Returns")
        axes[1, 1].set_xticks([])

        axes[1, 2].plot(dates, self.hrp_stats["max_w_long"] * 100, color="blue", lw=1, alpha=0.8, label="Max Long Wt %")
        axes[1, 2].plot(dates, self.hrp_stats["max_w_short"] * 100, color="red", lw=1, alpha=0.8, label="Max Short Wt %")
        axes[1, 2].axhline((1/self.n_longs)*100 if self.n_longs > 0 else 0, color="gray", ls="--", lw=1, label="Eq. Wt Base %")
        axes[1, 2].set_title("HRP Concentration (Max Single Position %)")
        axes[1, 2].legend(loc="upper right")

        axes[2, 0].plot(dates, self.components["gross_exposure"], color="darkorange", lw=1)
        axes[2, 0].set_title("Gross Exposure (Dollar Base)")
        axes[2, 0].set_ylim(0, 1.2)

        axes[2, 1].plot(dates, self.results["turnover"], color="purple", lw=1, alpha=0.5)
        axes[2, 1].set_title("Daily Turnover")
        
        if hasattr(self, "scaling_history") and len(self.scaling_history) > 0:
            sh = self.scaling_history
            ax_s = axes[2, 2]

            scalar_cols = ["dd_scalar", "vol_scalar", "ic_scalar", "spread_scalar"]
            colors = {
                "dd_scalar":     "#d62728",  # red
                "vol_scalar":    "#1f77b4",  # blue
                "ic_scalar":     "#2ca02c",  # green
                "spread_scalar": "#ff7f0e",  # orange
            }
            labels = {
                "dd_scalar": "Drawdown",
                "vol_scalar": "Volatility",
                "ic_scalar": "IC",
                "spread_scalar": "Spread",
            }

            # Identify which scalar is binding (the argmin) on each day.
            binding = sh[scalar_cols].idxmin(axis=1)

            # Shade the area under active_scalar with the color of whichever
            # scalar is binding. This is the key visual: the COLOR of the shaded
            # region tells you which scalar is throttling exposure.
            for col in scalar_cols:
                mask = (binding == col).values
                if mask.any():
                    y = sh["active_scalar"].where(mask)
                    ax_s.fill_between(sh.index, 0, y,
                                      color=colors[col], alpha=0.55,
                                      linewidth=0, label=labels[col])

            # Thin reference lines for each individual scalar, drawn UNDER nothing
            # so they remain visible above the shaded region for context.
            for col in scalar_cols:
                ax_s.plot(sh.index, sh[col],
                          color=colors[col], lw=0.8, alpha=0.45)

            # Active scalar as a thin black outline on top of the shading,
            # so the exact exposure level is readable but doesn't dominate.
            ax_s.plot(sh.index, sh["active_scalar"],
                      color="black", lw=1.0, alpha=0.9)

            ax_s.set_ylim(0, 1.05)
            ax_s.set_title("Scaling Coefficients (color = binding constraint)")
            ax_s.legend(loc="lower left", fontsize=7, ncol=2)
        else:
            axes[2, 2].axis("off")
        
        plt.tight_layout()
        plt.show()

    def set_scaling_params(self, 
                           target_vol=0.10, vol_lookback=20, max_vol_leverage=1.0,
                           dd_warning_threshold=-0.15, dd_penalty=0.50, dd_kill_threshold=-0.25,
                           use_ic_scaling=True, ic_lookback=20, ic_threshold=0.02,
                           # --- NEW PARAMS BELOW ---
                           use_spread_scaling=True, spread_lookback=30, spread_floor=0.3):
        """
        Configures dynamic capital scaling based on Volatility, Drawdowns, Model Rank IC, and Prediction Spread.
        """
        self.use_dynamic_scaling = True
        
        # Volatility & Drawdown Params
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.max_vol_leverage = max_vol_leverage
        self.dd_warning_threshold = dd_warning_threshold
        self.dd_penalty = dd_penalty
        self.dd_kill_threshold = dd_kill_threshold
        
        # IC Conviction Params
        self.use_ic_scaling = use_ic_scaling
        self.ic_lookback = ic_lookback
        self.ic_threshold = ic_threshold

        # Spread (Dispersion) Params
        self.use_spread_scaling = use_spread_scaling
        self.spread_lookback = spread_lookback
        self.spread_floor = spread_floor
        
        print(f"Dynamic Scaling Enabled: Target Vol={target_vol*100}%, DD Warn/Kill={dd_warning_threshold*100}%/{dd_kill_threshold*100}%")
        print(f"IC Scaling Enabled: Lookback={ic_lookback}d, Target IC={ic_threshold}")
        print(f"Spread Scaling Enabled: Lookback={spread_lookback}d, Min Exposure Floor={spread_floor*100}%")

    def plot_volatility(self, window=60):
        """
        Plots the chosen volatility metric of the strategy on the left Y-axis 
        and the cumulative returns on the right twin Y-axis.
        """
        if not hasattr(self, "results"):
            raise Exception("Run run() first.")

        dr = self.results["return"]
        dates = self.results.index

        # 1. Calculate Cumulative Returns
        cum_ret = np.cumprod(1 + dr)

        # 2. Setup the Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color1 = 'tab:blue'
        ax1.set_xlabel('Date')
        
        # =======================================================
        # 3. DYNAMIC VOLATILITY DATA SELECTION
        # =======================================================
        if getattr(self, 'volatility_type', 'simple') == "yang_zhang" and getattr(self, 'yz_vol_panel', None) is not None:
            # --- YANG ZHANG VOLATILITY ---
            # The scaler uses the daily cross-sectional median. 
            # Note: The YZ formula already natively calculated a 20-day rolling variance under the hood.
            vol_series = self.yz_vol_panel.median(axis=1).reindex(dates).ffill()
            
            ax1.set_ylabel('Median Yang-Zhang Volatility (Annualized)', color=color1)
            ax1.plot(dates, vol_series, color=color1, linewidth=1.5, alpha=0.8, label='YZ Volatility (20d Base)')
            title_str = "Yang-Zhang Market Volatility vs. Cumulative Returns"
            
        elif getattr(self, 'volatility_type', 'simple') == "ewma" and getattr(self, 'ewma_vol_panel', None) is not None:
            # --- NEW: EWMA PLOTTING ---
            vol_series = self.ewma_vol_panel.median(axis=1).reindex(dates).ffill()
            ax1.set_ylabel('Median EWMA Volatility (Annualized)', color=color1)
            ax1.plot(dates, vol_series, color=color1, linewidth=1.5, alpha=0.8, label='EWMA Volatility (20d Span)')
            title_str = "EWMA Market Volatility vs. Cumulative Returns"

        else:
            # --- SIMPLE VOLATILITY ---
            # Rolling standard deviation of the portfolio's net returns
            vol_series = dr.rolling(window=window).std() * np.sqrt(252)
            
            ax1.set_ylabel(f'Rolling {window}-Day Volatility (Annualized)', color=color1)
            ax1.plot(dates, vol_series, color=color1, linewidth=1.5, alpha=0.8, label=f'Simple {window}d Volatility')
            title_str = "Simple Portfolio Volatility vs. Cumulative Returns"

        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Optional: Draw a dashed line representing the target volatility (if dynamic scaling is on)
        if hasattr(self, 'target_vol') and getattr(self, 'use_dynamic_scaling', False):
            ax1.axhline(self.target_vol, color='gray', linestyle='--', linewidth=1.2, 
                        label=f'Target Vol ({self.target_vol:.1%})')

        # =======================================================
        # 4. PLOT CUMULATIVE RETURNS ON TWIN AXIS
        # =======================================================
        ax2 = ax1.twinx()  
        color2 = 'purple'
        ax2.set_ylabel('Cumulative Return', color=color2)  
        ax2.plot(dates, cum_ret, color=color2, linewidth=2, label='Cumulative Return')
        ax2.tick_params(axis='y', labelcolor=color2)

        # --- Formatting ---
        plt.title(title_str, fontsize=14)
        
        # Combine legends from both axes into one box
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Format left Y-axis as percentages
        import matplotlib.ticker as mtick
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        fig.tight_layout()  
        plt.show()
        return dates, vol_series
        

def optimize_hrp_strategy_old(portfolio, train_start, train_end, n_trials=50):
    """
    Optimizes HRPBacktest parameters over a specific In-Sample time period.
    
    Args:
        portfolio: The loaded Portfolio object.
        train_start (str): Start date for In-Sample training (e.g., '2016-01-01').
        train_end (str): End date for In-Sample training (e.g., '2020-12-31').
        n_trials (int): Number of combinations to test.
        
    Returns:
        dict: The best parameter combination.
    """
    print(f"--- Starting HRP Optimization ({train_start} to {train_end}) ---")
    
    # 1. Isolate the Training Data (In-Sample)
    # We create a dummy portfolio to safely slice data without modifying the original
    class SlicedPortfolio: pass
    train_port = SlicedPortfolio()
    train_port.has_data = True
    
    # Filter the dates
    mask = (pd.to_datetime(portfolio.data['date']) >= pd.to_datetime(train_start)) & \
           (pd.to_datetime(portfolio.data['date']) <= pd.to_datetime(train_end))
    train_port.data = portfolio.data[mask].copy()

    # 2. Define the Optuna Objective
    def objective(trial):
        # Suggest parameter combinations
        n_pos = trial.suggest_int("n_positions", 10, 40, step=5)
        rebalance_days = trial.suggest_int("rebalance_days", 1, 5)
        holding_period = trial.suggest_int("holding_period", 1, 10)
        hrp_lookback = trial.suggest_int("hrp_lookback", 30, 150, step=30)
        buffer_multiplier = trial.suggest_float("buffer_multiplier", 1.0, 3.0, step=0.5)
        smooth_preds = trial.suggest_categorical("smooth_predictions", [True, False])
        
        # Run the backtest silently
        try:
            bt = HRPBacktest(
                portfolio=train_port,
                n_longs=n_pos,
                n_shorts=n_pos,
                holding_period=holding_period,
                rebalance_days=rebalance_days,
                hrp_lookback=hrp_lookback,
                smooth_predictions=smooth_preds,
                buffer_multiplier=buffer_multiplier,
                cost_bps=5.0
            )
            res = bt.run(verbose=False)
        except Exception as e:
            # If the parameters crash the math, instantly fail the trial
            return -999.0
            
        # 3. Calculate Fitness Score
        dr = res['return'].values
        if len(dr) < 50 or np.std(dr) == 0:
            return -999.0
            
        cum = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()
        
        # If the strategy went bankrupt, fail the trial
        if max_dd <= -0.99 or cum[-1] <= 0:
            return -999.0
            
        sharpe = (dr.mean() / dr.std(ddof=1)) * np.sqrt(252)
        
        # THE OBJECTIVE SCORE: Sharpe Ratio penalized by Max Drawdown
        # E.g. Sharpe of 2.0 with a -50% DD -> 2.0 * (1 - 0.50) = Score of 1.0
        # This prevents the "Frankenstein Effect" by weeding out fragile portfolios.
        score = sharpe * (1.0 + max_dd) 
        
        return score

    # 3. Run the Optimizer
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress heavy console spam
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\nOptimization Complete!")
    print(f"Best Objective Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    return study.best_params


def optimize_hrp_strategy(portfolio, train_start, train_end, n_trials=50):
    """
    Optimizes HRPBacktest parameters over a specific In-Sample time period.
    Includes both structural parameters (N, lookbacks, holding) and 
    dynamic scaling parameters (vol target, drawdown thresholds, IC, spread).
    
    Args:
        portfolio: The loaded Portfolio object.
        train_start (str): Start date for In-Sample training (e.g., '2016-01-01').
        train_end (str): End date for In-Sample training (e.g., '2020-12-31').
        n_trials (int): Number of combinations to test.
        
    Returns:
        dict: The best parameter combination.
    """
    print(f"--- Starting HRP Optimization ({train_start} to {train_end}) ---")
    
    # 1. Isolate the Training Data (In-Sample)
    class SlicedPortfolio: pass
    train_port = SlicedPortfolio()
    train_port.has_data = True
    
    mask = (pd.to_datetime(portfolio.data['date']) >= pd.to_datetime(train_start)) & \
           (pd.to_datetime(portfolio.data['date']) <= pd.to_datetime(train_end))
    train_port.data = portfolio.data[mask].copy()

    # 2. Define the Optuna Objective
    def objective(trial):
        # ============================================================
        # STRUCTURAL PARAMETERS
        # ============================================================
        n_pos             = trial.suggest_int("n_positions", 2, 10, step=1)
        rebalance_days    = 1 #trial.suggest_int("rebalance_days", 1, 5)
        holding_period    = 5 #trial.suggest_int("holding_period", 1, 10)
        hrp_lookback      = trial.suggest_int("hrp_lookback", 30, 150, step=30)
        buffer_multiplier = trial.suggest_float("buffer_multiplier", 1.0, 3.0, step=0.2)
        smooth_preds      = trial.suggest_categorical("smooth_predictions", [True, False])
        sizing_method     = trial.suggest_categorical("sizing_method", ["dollar_neutral", "beta_neutral"])
        
        # ============================================================
        # DYNAMIC SCALING PARAMETERS
        # Each scaler has an on/off toggle plus its own settings. When 
        # off, the scaler is pinned at 1.0 and contributes nothing to 
        # the active min. This lets the optimizer decide whether each 
        # scaler is helpful at all, not just how to tune it.
        # ============================================================
        
        # Volatility scaling (always on - it's the baseline risk control)
        target_vol        = trial.suggest_float("target_vol", 0.03, 0.51, step=0.03)
        vol_lookback      = trial.suggest_int("vol_lookback", 5, 60, step=5)
        max_vol_leverage  = 1 #trial.suggest_float("max_vol_leverage", 1.0, 2.0, step=0.25)
        
        # Drawdown scaling (toggleable)
        use_dd_scaling = False #trial.suggest_categorical("use_dd_scaling", [True, False])
        if use_dd_scaling:
            dd_warning_threshold = trial.suggest_float("dd_warning_threshold", -0.4, -0.03, step=0.02)
            dd_kill_threshold    = trial.suggest_float("dd_kill_threshold", -0.50, -0.25, step=0.05)
            dd_penalty           = trial.suggest_float("dd_penalty", 0.1, 0.75, step=0.15)
            # Guard: kill threshold must be strictly worse than warning threshold.
            if dd_kill_threshold >= dd_warning_threshold:
                return -999.0
        else:
            # Effectively disabled: thresholds far enough that they never trigger.
            dd_warning_threshold = -0.99
            dd_kill_threshold    = -0.999
            dd_penalty           = 1.0
        
        # IC scaling (toggleable)
        use_ic_scaling = False #trial.suggest_categorical("use_ic_scaling", [True, False])
        if use_ic_scaling:
            ic_lookback  = trial.suggest_int("ic_lookback", 5, 60, step=5)
            ic_threshold = trial.suggest_float("ic_threshold", 0.01, 0.4, step=0.01)
        else:
            ic_lookback  = 20
            ic_threshold = 0.02
        
        # Spread scaling (toggleable)
        use_spread_scaling = False #trial.suggest_categorical("use_spread_scaling", [True, False])
        if use_spread_scaling:
            spread_lookback = trial.suggest_int("spread_lookback", 10, 60, step=5)
            spread_floor    = trial.suggest_float("spread_floor", 0.1, 0.7, step=0.1)
        else:
            spread_lookback = 30
            spread_floor    = 0.3
        
        # ============================================================
        # RUN THE BACKTEST
        # ============================================================
        try:
            bt = HRPBacktest(
                portfolio=train_port,
                n_longs=n_pos,
                n_shorts=n_pos,
                holding_period=holding_period,
                rebalance_days=rebalance_days,
                hrp_lookback=hrp_lookback,
                smooth_predictions=smooth_preds,
                buffer_multiplier=buffer_multiplier,
                sizing_method=sizing_method,
                cost_bps=5.0,
                volatility_type = "yang_zhang"
            )
            
            # Apply dynamic scaling configuration.
            bt.set_scaling_params(
                target_vol=target_vol,
                vol_lookback=vol_lookback,
                max_vol_leverage=max_vol_leverage,
                dd_warning_threshold=dd_warning_threshold,
                dd_penalty=dd_penalty,
                dd_kill_threshold=dd_kill_threshold,
                use_ic_scaling=use_ic_scaling,
                ic_lookback=ic_lookback,
                ic_threshold=ic_threshold,
                use_spread_scaling=use_spread_scaling,
                spread_lookback=spread_lookback,
                spread_floor=spread_floor,
            )
            
            res = bt.run(verbose=False)
        except Exception as e:
            return -999.0
            
        # ============================================================
        # FITNESS SCORE
        # ============================================================
        dr = res['return'].values
        if len(dr) < 50 or np.std(dr) == 0:
            return -999.0
            
        cum = np.cumprod(1 + dr)
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()
        
        if max_dd <= -0.99 or cum[-1] <= 0:
            return -999.0
            
        sharpe = (dr.mean() / dr.std(ddof=1)) * np.sqrt(252)
        
        # Sharpe penalized by drawdown - prevents fragile high-Sharpe portfolios.
        score = sharpe * (1.0 + max_dd) 
        
        return score

    # 3. Run the Optimizer
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\nOptimization Complete!")
    print(f"Best Objective Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    return study.best_params

def optimize_hrp3(portfolio, train_start, train_end, n_trials=50, 
                          objective_type="Sortino", objective_params=None):
    """
    Optimizes HRPBacktest parameters over a specific In-Sample time period.
    Includes both structural parameters and dynamic scaling parameters.
    
    Args:
        portfolio: The loaded Portfolio object.
        train_start (str): Start date for In-Sample training (e.g., '2016-01-01').
        train_end (str): End date for In-Sample training (e.g., '2020-12-31').
        n_trials (int): Number of combinations to test.
        objective_type (str): "Sortino" or "CAGR".
        objective_params (dict): Constraints for the optimizer.
            - If "Sortino": Must contain 'min_cagr' (e.g., {'min_cagr': 0.10})
            - If "CAGR": Must contain 'max_dd' (e.g., {'max_dd': -0.25})
            *Note: You can pass both keys simultaneously for maximum safety.
        
    Returns:
        dict: The best parameter combination.
    """
    # ============================================================
    # INPUT VALIDATION
    # ============================================================
    if objective_type not in ["Sortino", "CAGR"]:
        raise ValueError(f"Invalid objective_type '{objective_type}'. Must be 'Sortino' or 'CAGR'.")
        
    if objective_params is None or not isinstance(objective_params, dict):
        raise TypeError("objective_params must be a dictionary. e.g., {'min_cagr': 0.10, 'max_dd': -0.25}")
        
    if objective_type == "Sortino" and "min_cagr" not in objective_params:
        raise ValueError("When optimizing for 'Sortino', objective_params must contain a 'min_cagr' limit.")
        
    if objective_type == "CAGR" and "max_dd" not in objective_params:
        raise ValueError("When optimizing for 'CAGR', objective_params must contain a 'max_dd' limit.")

    # Defensive formatting for max_dd (ensures it is negative)
    if "max_dd" in objective_params:
        objective_params["max_dd"] = -abs(objective_params["max_dd"])

    print(f"--- Starting HRP Optimization ({train_start} to {train_end}) ---")
    print(f"Objective: Maximize {objective_type} | Constraints: {objective_params}")
    
    # 1. Isolate the Training Data (In-Sample)
    class SlicedPortfolio: pass
    train_port = SlicedPortfolio()
    train_port.has_data = True
    
    mask = (pd.to_datetime(portfolio.data['date']) >= pd.to_datetime(train_start)) & \
           (pd.to_datetime(portfolio.data['date']) <= pd.to_datetime(train_end))
    train_port.data = portfolio.data[mask].copy()

    # 2. Define the Optuna Objective
    def objective(trial):
        # ============================================================
        # STRUCTURAL PARAMETERS
        # ============================================================
        n_pos             = trial.suggest_int("n_positions", 2, 10, step=1)
        rebalance_days    = 1 #trial.suggest_int("rebalance_days", 1, 5)
        holding_period    = 5 #trial.suggest_int("holding_period", 1, 10)
        hrp_lookback      = trial.suggest_int("hrp_lookback", 30, 150, step=30)
        buffer_multiplier = trial.suggest_float("buffer_multiplier", 1.0, 3.0, step=0.2)
        smooth_preds      = trial.suggest_categorical("smooth_predictions", [True, False])
        sizing_method     = trial.suggest_categorical("sizing_method", ["dollar_neutral", "beta_neutral"])
        
        # ============================================================
        # DYNAMIC SCALING PARAMETERS
        # ============================================================
        target_vol        = trial.suggest_float("target_vol", 0.03, 0.51, step=0.03)
        vol_lookback      = trial.suggest_int("vol_lookback", 5, 60, step=5)
        max_vol_leverage  = 1 #trial.suggest_float("max_vol_leverage", 1.0, 2.0, step=0.25)
        
        use_dd_scaling = False #trial.suggest_categorical("use_dd_scaling", [True, False])
        if use_dd_scaling:
            dd_warning_threshold = trial.suggest_float("dd_warning_threshold", -0.4, -0.03, step=0.02)
            dd_kill_threshold    = trial.suggest_float("dd_kill_threshold", -0.50, -0.25, step=0.05)
            dd_penalty           = trial.suggest_float("dd_penalty", 0.1, 0.75, step=0.15)
            if dd_kill_threshold >= dd_warning_threshold:
                return -999.0
        else:
            dd_warning_threshold = -0.99
            dd_kill_threshold    = -0.999
            dd_penalty           = 1.0
        
        use_ic_scaling = False #trial.suggest_categorical("use_ic_scaling", [True, False])
        if use_ic_scaling:
            ic_lookback  = trial.suggest_int("ic_lookback", 5, 60, step=5)
            ic_threshold = trial.suggest_float("ic_threshold", 0.01, 0.4, step=0.01)
        else:
            ic_lookback  = 20
            ic_threshold = 0.02
        
        use_spread_scaling = False #trial.suggest_categorical("use_spread_scaling", [True, False])
        if use_spread_scaling:
            spread_lookback = trial.suggest_int("spread_lookback", 10, 60, step=5)
            spread_floor    = trial.suggest_float("spread_floor", 0.1, 0.7, step=0.1)
        else:
            spread_lookback = 30
            spread_floor    = 0.3
        
        # ============================================================
        # RUN THE BACKTEST
        # ============================================================
        try:
            bt = HRPBacktest(
                portfolio=train_port,
                n_longs=n_pos,
                n_shorts=n_pos,
                holding_period=holding_period,
                rebalance_days=rebalance_days,
                hrp_lookback=hrp_lookback,
                smooth_predictions=smooth_preds,
                buffer_multiplier=buffer_multiplier,
                sizing_method=sizing_method,
                cost_bps=15,
                volatility_type="yang_zhang"
            )
            
            bt.set_scaling_params(
                target_vol=target_vol,
                vol_lookback=vol_lookback,
                max_vol_leverage=max_vol_leverage,
                dd_warning_threshold=dd_warning_threshold,
                dd_penalty=dd_penalty,
                dd_kill_threshold=dd_kill_threshold,
                use_ic_scaling=use_ic_scaling,
                ic_lookback=ic_lookback,
                ic_threshold=ic_threshold,
                use_spread_scaling=use_spread_scaling,
                spread_lookback=spread_lookback,
                spread_floor=spread_floor,
            )
            
            res = bt.run(verbose=False)
        except Exception as e:
            return -999.0
            
        # ============================================================
        # FITNESS SCORE & CONSTRAINTS
        # ============================================================
        dr = res['return'].values
        if len(dr) < 50 or np.std(dr) == 0:
            return -999.0
            
        cum = np.cumprod(1 + dr)
        if cum[-1] <= 0:
            return -999.0
            
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min()  # This yields a negative number
        
        # Calculate Base Metrics
        years = len(dr) / 252.0
        cagr = (cum[-1] ** (1 / years)) - 1
        
        annualized_return = dr.mean() * 252
        downside_returns = dr[dr < 0]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        else:
            downside_dev = 1e-8
            
        sortino = annualized_return / downside_dev
        
        # --- Apply Constraints based on Objective Type ---
        
        # Guardrail 1: Max Drawdown
        if "max_dd" in objective_params:
            if max_dd < objective_params["max_dd"]:
                return -999.0  # Instant disqualification
                
        # Guardrail 2: Minimum CAGR
        if "min_cagr" in objective_params:
            if cagr < objective_params["min_cagr"]:
                return -999.0  # Instant disqualification
                
        # --- Return Final Score ---
        if objective_type == "Sortino":
            return sortino
        elif objective_type == "CAGR":
            return cagr

    # 3. Run the Optimizer
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\nOptimization Complete!")
    print(f"Best {objective_type} Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    return study.best_params
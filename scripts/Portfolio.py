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
        '''
        wf_folder: the walk forward folder holding the models
        '''
        pass

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

                    new_w = {}
                    for idx, r in longs.iterrows():
                        new_w[r["act_symbol"]] = long_norm.loc[idx]
                    for idx, r in shorts.iterrows():
                        new_w[r["act_symbol"]] = -short_norm.loc[idx]

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
    Signal-weighted long-short backtest with dynamic risk management.
    
    Three independent exposure controls:
      1. Drawdown breaker:  reduces exposure as portfolio draws down
      2. Volatility scaling: reduces exposure when portfolio vol is elevated
      3. Exposure cap:       hard limit on gross exposure
    
    Each produces a multiplier in [0, 1]. Final exposure = product of all three.
    """

    def __init__(self, portfolio, n_longs=20, n_shorts=20, rebalance_days=5, cost_bps=5.0):
        if not portfolio.has_data:
            raise Exception("Portfolio has no data. Run get_model_data() first.")

        self.data = portfolio.data.copy()
        self.n_longs = n_longs
        self.n_shorts = n_shorts
        self.rebalance_days = rebalance_days
        self.cost_bps = cost_bps

        if "close" not in self.data.columns:
            raise Exception("Portfolio data must contain 'close' column.")

        # Compute daily log returns
        self.data = self.data.sort_values(["act_symbol", "date"])
        self.data["daily_log_ret"] = self.data.groupby("act_symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        ).fillna(0)
        self.pnl_col = "daily_log_ret"

        self.data = self.data.sort_values(["date", "pred_return"], ascending=[True, False])
        self.trading_dates = sorted(self.data["date"].unique())
        self._date_groups = {d: g for d, g in self.data.groupby("date")}

        print(f"ManagedBacktest: {len(self.trading_dates)} days, "
              f"{self.data['act_symbol'].nunique()} stocks")

    def run(self,
            # Drawdown breaker
            dd_threshold=-0.05,      # start scaling at 5% drawdown
            dd_full_cut=-0.15,       # zero exposure at 15% drawdown
            # Volatility targeting
            target_vol=0.10,         # annualized target volatility (10%)
            vol_lookback=20,         # days for realized vol estimate
            vol_cap=2.0,             # max vol multiplier (don't lever up beyond 2x)
            # Gross exposure
            max_gross_exposure=2.0,  # hard cap on total |weights|
            dd_lookback = 60
            ):
        """
        Run backtest with dynamic risk management.
        All parameters have sensible defaults — tune to your risk tolerance.
        """
        n_l = self.n_longs
        n_s = self.n_shorts
        rebal_set = set(self.trading_dates[::self.rebalance_days])
        cost_rate = self.cost_bps / 10_000

        weights = {}
        daily_rets = []
        daily_scalars = []
        daily_components = []

        # Track portfolio state
        cum_ret = 1.0
        cum_history = []
        recent_rets = []

        for date in self.trading_dates:
            cost = 0.0

            # ── Update portfolio state ──
            if daily_rets:
                cum_ret *= (1 + daily_rets[-1])
                cum_history.append(cum_ret)
                if len(cum_history) > dd_lookback:
                    trailing_peak = max(cum_history[-dd_lookback:])
                else:
                    trailing_peak = max(cum_history)
            else:
                trailing_peak = 1
            current_dd = (cum_ret - trailing_peak) / trailing_peak

            # ── Compute exposure scalars ──

            # 1. Drawdown breaker: linear scale from 1.0 at dd_threshold to 0.0 at dd_full_cut
            # In run(), change dd_scalar calculation:
            if current_dd <= dd_threshold:
                dd_scalar = max(0.2, (current_dd - dd_full_cut) / (dd_threshold - dd_full_cut))
            else:
                dd_scalar = 1.0

            # 2. Volatility targeting: scale to keep realized vol near target
            if len(recent_rets) >= vol_lookback:
                realized_vol = np.std(recent_rets[-vol_lookback:]) * np.sqrt(252)
                if realized_vol > 0:
                    vol_scalar = min(vol_cap, target_vol / realized_vol)
                else:
                    vol_scalar = 1.0
            else:
                vol_scalar = 1.0

            combined_scalar = dd_scalar * vol_scalar

            # ── Rebalance ──
            if date in rebal_set and date in self._date_groups:
                df = self._date_groups[date]

                if len(df) >= n_l + n_s:
                    sorted_df = df.sort_values("pred_return", ascending=False)
                    longs = sorted_df.head(n_l)
                    shorts = sorted_df.tail(n_s)

                    # Signal-weighted allocation
                    long_scores = longs["pred_zscore"].abs()
                    long_total = long_scores.sum()
                    long_norm = long_scores / long_total if long_total > 0 else pd.Series(1.0 / n_l, index=long_scores.index)

                    short_scores = shorts["pred_zscore"].abs()
                    short_total = short_scores.sum()
                    short_norm = short_scores / short_total if short_total > 0 else pd.Series(1.0 / n_s, index=short_scores.index)

                    new_w = {}
                    for idx, r in longs.iterrows():
                        new_w[r["act_symbol"]] = long_norm.loc[idx] * combined_scalar
                    for idx, r in shorts.iterrows():
                        new_w[r["act_symbol"]] = -short_norm.loc[idx] * combined_scalar

                    # Enforce gross exposure cap
                    gross = sum(abs(w) for w in new_w.values())
                    if gross > max_gross_exposure:
                        ratio = max_gross_exposure / gross
                        new_w = {k: v * ratio for k, v in new_w.items()}

                    # Transaction costs
                    turnover = sum(
                        abs(new_w.get(t, 0) - weights.get(t, 0))
                        for t in set(list(new_w) + list(weights))
                    )
                    cost = turnover * cost_rate
                    weights = new_w

            # ── Portfolio return ──
            if date in self._date_groups and weights:
                rets = self._date_groups[date].set_index("act_symbol")[self.pnl_col]
                port_ret = sum(w * rets.get(t, 0) for t, w in weights.items()) - cost
            else:
                port_ret = 0.0

            daily_rets.append(port_ret)
            recent_rets.append(port_ret)
            daily_scalars.append(combined_scalar)
            daily_components.append({
                "dd_scalar": dd_scalar,
                "vol_scalar": vol_scalar,
                "combined": combined_scalar,
                "drawdown": current_dd,
                "gross_exposure": sum(abs(w) for w in weights.values()),
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
    


    
        


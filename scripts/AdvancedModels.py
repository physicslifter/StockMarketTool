'''
Advanced models built on basic model
'''
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
import re
import math
import random
from skfolio.model_selection import CombinatorialPurgedCV
from clean_analysis import Model

#===================
#Walk forward strategy for model
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def run_walk_forward_analysis(universe_path, target, features, folder_path,
                              target_type="regression", train_years=4, 
                              val_years=1, test_years=1, n_trials=40):
    """
    Performs a rolling Walk-Forward Analysis using the base Model class.
    Saves all models, features, and metrics into a specified directory structure.
    
    Args:
        folder_path: The parent directory where all models and metrics will be saved.
    """
    # Create the parent directory
    os.makedirs(folder_path, exist_ok=True)
    
    # Quick load just to find the global start and end years
    df_temp = pd.read_feather(universe_path)
    global_start_year = df_temp['date'].dt.year.min()
    global_end_year = df_temp['date'].dt.year.max()
    del df_temp

    all_oos_predictions =[]
    metrics_records =[]
    
    current_year = global_start_year
    fold = 1

    # 2. Instantiate a fresh model for this specific fold, passing the fold_folder
    train_start = f"{current_year}-01-01"
    val_start = f"{current_year + train_years}-01-01"
    test_start = f"{current_year + train_years + val_years}-01-01"
    test_end = f"{current_year + train_years + val_years + test_years}-01-01"
    fold_folder = os.path.join(folder_path, f"fold_{test_start[:4]}_{test_end[:4]}")
    model = Model(universe_path, model_folder=fold_folder)

    while True:
        # Define the chronological window boundaries
        train_start = f"{current_year}-01-01"
        val_start = f"{current_year + train_years}-01-01"
        test_start = f"{current_year + train_years + val_years}-01-01"
        test_end = f"{current_year + train_years + val_years + test_years}-01-01"

        # Break the loop if our test period pushes beyond our available data
        if int(test_start[:4]) > global_end_year:
            break

        print("\n" + "="*60)
        print(f"WALK-FORWARD FOLD {fold}")
        print(f"Train : {train_start} to {val_start}")
        print(f"Val   : {val_start} to {test_start}")
        print(f"Test  : {test_start} to {test_end}")
        print("="*60)

        # 1. Create a specific sub-folder for this fold
        fold_folder = os.path.join(folder_path, f"fold_{test_start[:4]}_{test_end[:4]}")
        
        # 2. Instantiate a fresh model for this specific fold, passing the fold_folder
        model = Model(universe_path, model_folder=fold_folder)
        
        # Save=True ensures the targets and features are pickled into the fold folder
        model.add_features(features, save=True)
        model.add_target(target, target_type=target_type, save=True)

        # 3. Split using the date-based method (this automatically saves dates.csv)
        model.split_data_by_dates(train_start, val_start, test_start, test_end)

        # If the test set ends up empty (e.g. data ends mid-year), skip it
        if model.test_df.empty:
            print("Test set is empty. Ending Walk-Forward.")
            break

        # 4. Tune and Train (this automatically calls test_model and saves model.txt)
        model.tune_params(n_trials=n_trials)
        model.train_model()

        # 5. Extract Out-Of-Sample Predictions
        fold_oos_data = model.test_df.copy()
        all_oos_predictions.append(fold_oos_data)
        
        # 6. Extract Metrics for the Distribution Tracking
        fold_metrics = {
            "Fold": fold,
            "Test_Start": test_start,
            "Test_End": test_end
        }
        
        if target_type == "classification":
            preds = fold_oos_data["prob_up"]
            y_true = model.y_test_bin
            fold_metrics["Accuracy"] = accuracy_score(y_true, (preds > 0.5).astype(int))
            fold_metrics["AUC"] = roc_auc_score(y_true, preds)
        else:
            preds = fold_oos_data["pred_return"]
            y_true = model.y_test_bin
            fold_metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, preds))
            fold_metrics["Dir_Accuracy"] = ((y_true > 0) == (preds > 0)).mean()
            
            # Robust IC Metrics
            mean_ic, std_ic, ic_ir, ann_ic_ir, t_stat = model._calculate_robust_ic_metrics()
            fold_metrics["Mean_IC"] = mean_ic
            fold_metrics["Ann_IC_IR"] = ann_ic_ir
            
            # Quantile Sharpe Ratio
            # We run it with plot=False to avoid drawing 10 charts during WFA
            spread_res = model.evaluate_quantile_spread(quantiles=10, plot=False)
            if spread_res is not None:
                _, daily_spread = spread_res
                mean_spread = daily_spread.mean()
                std_spread = daily_spread.std()
                match = re.search(r'_(\d+)', model.target_key)
                N = int(match.group(1)) if match else 1
                if std_spread != 0 and not np.isnan(std_spread):
                    fold_metrics["Sharpe"] = (mean_spread / std_spread) * np.sqrt(252 / N)
                else:
                    fold_metrics["Sharpe"] = np.nan
            else:
                fold_metrics["Sharpe"] = np.nan

        metrics_records.append(fold_metrics)

        # Move the sliding window forward
        current_year += test_years
        fold += 1

    print("\n" + "="*60)
    print(f"WALK-FORWARD ANALYSIS COMPLETE ({fold-1} Folds)")
    
    # 7. Save the Metrics to the Parent Folder
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(folder_path, "walk_forward_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to: {metrics_csv_path}")
    
    # 8. Plot and Save the Metric Distributions
    if target_type == "regression" and "Sharpe" in metrics_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        metrics_df['Sharpe'].plot(kind='bar', ax=axes[0], color='purple', edgecolor='black')
        axes[0].set_title("OOS Sharpe Ratio per Fold")
        axes[0].set_xticklabels(metrics_df['Test_Start'].str[:4], rotation=45)
        
        metrics_df['Mean_IC'].plot(kind='bar', ax=axes[1], color='blue', edgecolor='black')
        axes[1].set_title("Mean Rank IC per Fold")
        axes[1].set_xticklabels(metrics_df['Test_Start'].str[:4], rotation=45)
        
        metrics_df['Ann_IC_IR'].plot(kind='bar', ax=axes[2], color='green', edgecolor='black')
        axes[2].set_title("Annualized IC-IR per Fold")
        axes[2].set_xticklabels(metrics_df['Test_Start'].str[:4], rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(folder_path, "metrics_distribution.png")
        plt.savefig(plot_path)
        print(f"Distribution plot saved to: {plot_path}")
        plt.show()

    # 9. Combine all Out-Of-Sample periods into one continuous timeline
    if len(all_oos_predictions) > 0:
        final_oos_df = pd.concat(all_oos_predictions).sort_values(['date', 'act_symbol'])
        return final_oos_df, metrics_df
    else:
        return pd.DataFrame(), metrics_df


#==================
#CPCV Model
#==================
class CPCVModel(Model):
    def __init__(self, universe_path, model_folder=None, n_folds=6, n_test_folds=2):
        """
        Initializes the CPCV Model, inheriting all core utilities from the base Model.
        
        Args:
            universe_path: Path to the universe feather file.
            model_folder: Directory to save model artifacts.
            n_folds: Total number of chronological blocks to divide the dataset into.
            n_test_folds: How many of those blocks to use as test data in each combination.
        """
        super().__init__(universe_path, model_folder)
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.models =[]          # Will hold the ensemble of LightGBM models
        self.cpcv_results = {}    # Will hold the test-set DataFrames for each split
        self.production_model = None

    def split_data(self):
        """
        Overrides the base split_data. Defines the entire dataset as X and y, 
        and initializes the skfolio CombinatorialPurgedCV object.
        """
        if self.data_generated == False:
            self.generate_targets_and_features()

        if not self.has_features or not self.has_target:
            raise Exception("Features or target missing. Add them before splitting.")

        # 1. Sort chronologically (CRITICAL for CPCV so blocks represent continuous time)
        self.data = self.data.sort_values(['date', 'act_symbol']).reset_index(drop=True)

        self.features =[key for key in self.data.keys() if "F" in key.split("_")]
        self.targets =[key for key in self.data.keys() if "T" in key.split("_")]
        self.target_key = self.targets[0]

        self.X = self.data[self.features]
        self.y = self.data[self.target_key]

        # 2. Extract Purge and Embargo logic
        match = re.search(r'_(\d+)', self.target_key)
        purge_gap_days = int(match.group(1)) if match else 1

        unique_dates = sorted(self.data['date'].unique())
        embargo_days = max(5, int(len(unique_dates) * 0.01))

        # 3. Create a Dummy TimeSeries for skfolio
        # skfolio's CPCV operates on a 1D index of time. Because your data is panel data 
        # (multiple rows per date), we generate the splits on the unique dates, and map 
        # those selected dates back to your rows. This prevents a purge from cutting a day in half.
        self.unique_dates_series = pd.Series(index=unique_dates, data=np.arange(len(unique_dates)))

        self.cv = CombinatorialPurgedCV(
            n_folds=self.n_folds,
            n_test_folds=self.n_test_folds,
            purged_size=purge_gap_days,
            embargo_size=embargo_days
        )

        self.data_split = True
        
        # Calculate how many combinations this will generate ( N choose K )
        n_combinations = math.comb(self.n_folds, self.n_test_folds)
        
        print("\n--- CPCV Initialization Complete ---")
        print(f"Folds: {self.n_folds} | Test Folds: {self.n_test_folds}")
        print(f"Purge: {purge_gap_days} days | Embargo: {embargo_days} days")
        print(f"Total Model Combinations that will be generated: {n_combinations}\n")

    def tune_params(self, n_trials=50):
        """
        Overrides tune_params. Evaluates hyperparameters across MULTIPLE regimes 
        to ensure they generalize well outside of a single specific time period.
        """
        if not self.data_split:
            raise Exception("Data must be split before tuning params")

        print("TUNING PARAMS ACROSS MULTIPLE CPCV REGIMES...")
        float_cols = self.data.select_dtypes(include=['float64']).columns
        self.data[float_cols] = self.data[float_cols].astype('float32')

        # Get all splits from the CV object
        splits = list(self.cv.split(self.unique_dates_series))
        
        # To avoid tuning taking 10 hours, we randomly sample up to 3 combinations 
        # to evaluate each Optuna trial. (The Law of Large Numbers handles the rest).
        eval_splits = random.sample(splits, min(3, len(splits)))

        direction = "maximize" if self.target_type == "classification" else "minimize"

        def objective(trial):
            param = {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "seed": 42
            }
            
            # Dynamic Setup for Classification vs Regression
            if self.target_type == "classification":
                param.update({"objective": "binary", "metric": "auc"})
                pruning_metric = "auc"
            else:
                param.update({"objective": "huber", "metric": "huber", "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True)})
                pruning_metric = "huber"

            scores =[]
            
            # Loop through the sampled CPCV splits
            for train_date_idx, test_date_idx in eval_splits:
                print(test_date_idx)
                # Map selected indices back to actual Dates
                train_dates = self.unique_dates_series.iloc[train_date_idx].index
                test_dates = self.unique_dates_series.iloc[test_date_idx].index

                # Mask the main dataframe
                train_mask = self.data['date'].isin(train_dates)
                test_mask = self.data['date'].isin(test_dates)

                X_tr, y_tr = self.X[train_mask], self.y[train_mask]
                X_te, y_te = self.X[test_mask], self.y[test_mask]

                # Weighting logic
                if self.target_type == "classification":
                    train_weights = np.log1p(np.abs(y_tr) * 100)
                    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=train_weights)
                else:
                    dtrain = lgb.Dataset(X_tr, label=y_tr)

                dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

                gbm = lgb.train(
                    param, dtrain, valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20),
                        optuna.integration.LightGBMPruningCallback(trial, pruning_metric)
                    ]
                )
                
                preds = gbm.predict(X_te)

                if self.target_type == "classification":
                    scores.append(roc_auc_score(y_te, preds))
                else:
                    scores.append(mean_absolute_error(y_te, preds))

            # Return the AVERAGE score across the regimes evaluated in this trial
            return np.mean(scores)

        study = optuna.create_study(direction=direction) 
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.study = study
        self.params_tuned = True

    def train_model(self, perturb_hyperparameters=False):
        """
        Overrides train_model. Trains an ENSEMBLE of models based on every CPCV combination.
        """
        if not self.params_tuned:
            self.tune_params()

        # Perturbing hyperparameters
        if perturb_hyperparameters:
            params = {}
            rng = np.random.default_rng()
            for param, val in self.best_params.items():
                multiplier = 1 + (rng.choice([1, -1]) * rng.uniform(low=0.05, high=0.1))
                new_val = val * multiplier
                if isinstance(val, int) and not isinstance(val, bool):
                    params[param] = int(round(new_val))
                else:
                    params[param] = new_val
        else:
            params = self.best_params.copy()

        splits = list(self.cv.split(self.unique_dates_series))
        
        print(f"\n--- TRAINING ENSEMBLE OF {len(splits)} CPCV MODELS ---")
        self.models =[]

        for i, (train_date_idx, test_date_idx) in enumerate(splits):
            print(f"Training Model {i+1}/{len(splits)}...")
            
            train_dates = self.unique_dates_series.iloc[train_date_idx].index
            train_mask = self.data['date'].isin(train_dates)

            X_tr, y_tr = self.X[train_mask], self.y[train_mask]

            # Weighting logic for the final training
            if self.target_type == "classification":
                train_weights = np.log1p(np.abs(y_tr) * 100)
                dtrain = lgb.Dataset(X_tr, label=y_tr, weight=train_weights)
            else:
                dtrain = lgb.Dataset(X_tr, label=y_tr)

            # Train for fixed rounds (no early stopping on test set to prevent leakage)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=100 
            )
            
            # Save the model and the test indices it is meant to predict on
            self.models.append({
                'split_id': i,
                'model': model,
                'test_date_idx': test_date_idx
            })

    def test_model(self):
        """
        Overrides test_model. Generates Out-Of-Sample predictions for EVERY split.
        Stores them in self.cpcv_results dictionary and prints aggregate ML metrics.
        """
        print(f"\n--- GENERATING PREDICTIONS FOR ALL CPCV PATHS ---")
        self.cpcv_results = {}
        
        auc_scores = []
        acc_scores = []
        rmse_scores =[]

        for entry in self.models:
            split_id = entry['split_id']
            model = entry['model']
            test_date_idx = entry['test_date_idx']

            # Map the test indices back to actual dates
            test_dates = self.unique_dates_series.iloc[test_date_idx].index
            test_mask = self.data['date'].isin(test_dates)

            # Create a copy of the dataframe for this split to hold predictions
            test_df_split = self.data[test_mask].copy()
            X_te = self.X[test_mask]
            y_te = self.y[test_mask]

            # Predict
            preds = model.predict(X_te)
            test_df_split['pred_return'] = preds

            self.cpcv_results[split_id] = test_df_split
            
            # Calculate metrics for this path
            if self.target_type == "classification":
                # Ensure y_te is binary for accuracy/auc scoring if necessary
                auc_scores.append(roc_auc_score(y_te, preds))
                test_preds_class = (preds > 0.5).astype(int)
                acc_scores.append(accuracy_score(y_te, test_preds_class))
            else:
                rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))
                
        # Print aggregated metrics across all regimes
        print("MODEL ASSESSMENT ACROSS ALL REGIMES\n" + "="*35)
        if self.target_type == "classification":
            print(f"Mean CPCV AUC      : {np.mean(auc_scores):.4f}")
            print(f"Worst Path AUC     : {np.min(auc_scores):.4f}")
            print(f"Mean CPCV Accuracy : {np.mean(acc_scores):.2%}")
        else:
            print(f"Mean CPCV RMSE     : {np.mean(rmse_scores):.4f}")
            print(f"Worst Path RMSE    : {np.max(rmse_scores):.4f}")
            
        print("\nDone. Ready for evaluate_quantile_spread()")

    def evaluate_quantile_spread(self, quantiles=10, plot=True):
        """
        Evaluates the strategy's Sharpe Ratio across ALL combinations.
        Calculates the Probability of Backtest Overfitting (PBO).
        """
        if not self.cpcv_results:
            raise Exception("Run test_model() first.")
            
        print(f"\nCPCV DISTRIBUTION ANALYSIS (Top {100/quantiles:.1f}% vs Bottom {100/quantiles:.1f}%)")
        print("=========================================================")

        match = re.search(r'_(\d+)', self.target_key)
        N = int(match.group(1)) if match else 1
        ann_factor = np.sqrt(252 / N)

        split_sharpes = []
        split_means =[]

        # Iterate through every simulated path
        for split_id, df in self.cpcv_results.items():
            
            df['quantile'] = df.groupby('date')['pred_return'].transform(
                lambda x: pd.qcut(x.rank(method='first'), q=quantiles, labels=False) + 1
            )
            
            daily_q_ret = df.groupby(['date', 'quantile'])[self.target_key].mean().unstack().dropna()
            if daily_q_ret.empty:
                continue
                
            daily_spread = daily_q_ret[quantiles] - daily_q_ret[1]
            mean_spread = daily_spread.mean()
            std_spread = daily_spread.std()
            
            if std_spread != 0 and not np.isnan(std_spread):
                sharpe = (mean_spread / std_spread) * ann_factor
                split_sharpes.append(sharpe)
                split_means.append(mean_spread)

        if not split_sharpes:
            print("Not enough data to evaluate quantiles across splits.")
            return

        # Probability of Backtest Overfitting (PBO)
        pbo = sum(s <= 0 for s in split_sharpes) / len(split_sharpes)

        print(f"Total OOS Splits Evaluated : {len(split_sharpes)}")
        print(f"Mean Target Spread         : {np.mean(split_means):.4f}")
        print(f"Average Annualized Sharpe  : {np.mean(split_sharpes):.4f}")
        print(f"Worst Case Sharpe (Min)    : {np.min(split_sharpes):.4f}")
        print(f"Best Case Sharpe (Max)     : {np.max(split_sharpes):.4f}")
        print("-" * 40)
        print(f"Probability of Backtest Overfitting (PBO): {pbo:.2%} <-- (Target: < 5%)")
        print("-" * 40)

        if plot:
            plt.figure(figsize=(10, 5))
            plt.hist(split_sharpes, bins=8, color='purple', edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(split_sharpes), color='red', linestyle='dashed', linewidth=2, label='Mean Sharpe')
            plt.axvline(0, color='black', linewidth=2)
            plt.title('Distribution of OOS Sharpe Ratios (CPCV)')
            plt.xlabel('Annualized Sharpe Ratio')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        return split_sharpes

    def train_production_model(self, save_name="production_model.txt"):
        """
        NEW METHOD: Once CPCV proves the strategy works, train ONE final model 
        on 100% of the historical data to use in live deployment.
        """
        print("\n--- TRAINING FINAL PRODUCTION MODEL ON 100% OF DATA ---")
        if not self.params_tuned:
            raise Exception("Parameters must be tuned before training production model.")

        if self.target_type == "classification":
            train_weights = np.log1p(np.abs(self.y) * 100)
            dtrain = lgb.Dataset(self.X, label=self.y, weight=train_weights)
        else:
            dtrain = lgb.Dataset(self.X, label=self.y)

        self.production_model = lgb.train(
            self.best_params,
            dtrain,
            num_boost_round=150
        )

        if self.has_folder:
            self.production_model.save_model(f"{self.model_folder}/{save_name}")
            print(f"Production model saved to {self.model_folder}/{save_name}")
            
        print("Ready for live trading deployment.")

    def save_cpcv_ic_metrics(self, filename="cpcv_ic_metrics.csv"):
        """
        Calculates the Information Coefficient (IC) metrics for each CPCV split 
        and saves them to a CSV file in the model folder.
        
        Metrics calculated per split:
        - Mean IC (Information Coefficient)
        - IC Standard Deviation
        - Daily IC-IR (Information Ratio)
        - Annual IC-IR
        - T-Statistic
        """
        if not self.cpcv_results:
            raise Exception("Run test_model() first to generate Out-Of-Sample predictions.")
            
        if self.model_folder is None:
            raise Exception("A model_folder must be provided during initialization to save the CSV.")

        print(f"\n--- CALCULATING IC METRICS FOR ALL {len(self.cpcv_results)} SPLITS ---")
        
        # Extract the holding period (N) to calculate the annualized factor
        match = re.search(r'_(\d+)', self.target_key)
        N = int(match.group(1)) if match else 1
        ann_factor = np.sqrt(252 / N)

        metrics_records =[]

        # Iterate over every Out-Of-Sample dataframe generated during test_model()
        for split_id, df in self.cpcv_results.items():
            
            # 1. Calculate the Daily Cross-Sectional Information Coefficient (IC)
            # Using Spearman rank correlation between predictions and actual targets
            daily_ic = df.groupby('date').apply(
                lambda x: x['pred_return'].corr(x[self.target_key], method='spearman') if len(x) > 1 else np.nan
            ).dropna()

            # If a split doesn't have enough valid data, skip or append NaNs
            if daily_ic.empty or len(daily_ic) < 2:
                metrics_records.append({
                    "Split_ID": split_id,
                    "Mean_IC": np.nan,
                    "Std_IC": np.nan,
                    "Daily_IC_IR": np.nan,
                    "Ann_IC_IR": np.nan,
                    "T_Statistic": np.nan
                })
                continue

            # 2. Calculate the specific summary statistics
            T = len(daily_ic)          # Number of days in the test split
            mean_ic = daily_ic.mean()
            std_ic = daily_ic.std()
            
            # Handle potential division by zero
            if std_ic == 0 or np.isnan(std_ic):
                daily_ic_ir = np.nan
                ann_ic_ir = np.nan
                t_stat = np.nan
            else:
                daily_ic_ir = mean_ic / std_ic
                ann_ic_ir = daily_ic_ir * ann_factor
                t_stat = mean_ic / (std_ic / np.sqrt(T))

            # 3. Store the record for this combination
            metrics_records.append({
                "Split_ID": split_id,
                "Mean_IC": mean_ic,
                "Std_IC": std_ic,
                "Daily_IC_IR": daily_ic_ir,
                "Ann_IC_IR": ann_ic_ir,
                "T_Statistic": t_stat
            })

        # 4. Convert to DataFrame and save to CSV
        metrics_df = pd.DataFrame(metrics_records)
        
        # Ensure the directory exists
        import os
        os.makedirs(self.model_folder, exist_ok=True)
        
        save_path = os.path.join(self.model_folder, filename)
        metrics_df.to_csv(save_path, index=False)
        
        # Print summary of what was saved
        print(f"Successfully saved IC metrics to: {save_path}")
        print(f"Average Annualized IC-IR across all splits: {metrics_df['Ann_IC_IR'].mean():.4f}")
        
        return metrics_df

#=================

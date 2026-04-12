#implementing portfolio strategies
import lightgbm as lgb
import pandas as pd
from FeatureEngine import *
import os
import pickle
from clean_analysis import Model

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
        info = pd.read_csv(f"../{portfolio_folder}/info.csv")
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
        '''
        Drops everything from the df except:
            date
            act_symbol
            prediction
            open/close
        '''
        
        keep_cols = ["date", "act_symbol", "pred_return", "pred_zscore"]

        # Keep open/close if they exist
        for col in ["open", "close"]:
            if col in self.model.data.columns:
                keep_cols.append(col)

        # Keep target column if it exists
        target_cols = [k for k in self.model.data.columns if "T" in k.split("_")]
        keep_cols += target_cols

        # Only keep columns that actually exist
        keep_cols = [c for c in keep_cols if c in self.model.data.columns]

        self.model.data = self.model.data[keep_cols]
        self.data = self.model.data
        del self.model

    def save(self, folder):
        '''
        Saves the portfolio into a folder
        '''
        os.makedirs("folder")
        info = {
            "model_folder": self.model_folder
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

    

    


    
        


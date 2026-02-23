'''
Clean file for demonstrating every feature
'''

'''
Order of Features:
    1. Price momentum: show price momentum over past year vs price.
'''
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
import pandas as pd
from FeatureEngine import *
from matplotlib import pyplot as plt
plt.style.use("dark_background")

def generate_side_by_side(df, left_plot_key, right_plot_key):
    #generates a sid-by-side comparison plot
    fig = plt.figure(figsize = (12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex = ax1)
    for ax, key in zip([ax1, ax2], [left_plot_key, right_plot_key]):
        ax.set_xlabel("Date")
        ax.set_ylabel(key)
        ax.plot(df.date, df[key], label = key)
        ax.legend()
    plt.tight_layout()
    plt.tight_layout()
    plt.show()

def plot_feature_v_price(df, feature:FeatureRequest):
    feature_engine = FeatureEngine([feature])
    df = feature_engine.compute(df)
    feature_key = [key for key in df.keys() if "F" in key.split("_")][0]
    generate_side_by_side(df.tail(252), feature_key, "close")

#tests
price_momentum = 1

#logic for the tests
if True in [price_momentum]:
    df = pd.read_feather("../Data/all_ohlcv.feather")
    df["date"] = pd.to_datetime(df["date"])

if price_momentum == True:
    momentum = FeatureRequest("MOM", #make a momentum features
                              params = {"timeperiod": 20}, #look back over last 20 days
                              shift = -1, #lag by 1 day to make it a feature
                              input_type = "log_price", #perform momentum calculation on the log prices
                              alias = "log_mom"
                              )
    plot_feature_v_price(df, momentum)


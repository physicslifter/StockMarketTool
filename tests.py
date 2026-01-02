import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates
from scipy.stats import norm, t
import numpy as np
from pdb import set_trace as st
from sim_strat1 import StockData
plt.style.use('fivethirtyeight')
from Analysis import *


show_all_data = 0
show_daily_changes = 0
test_class = 0
test_analysis_plot = 0
test_analysis_comparison_plot = 0
test_baseline_compare = 0
compare_beverage_stocks = 0
test_show_portfolio_mix = 0
test_portfolio_performance_against_mean = 0
compare_pharma = 1

stock = "F"
if stock in ["SPY"]:
    df = pd.read_csv(f"Data/{stock}.csv")
elif stock in ["F", "PFE", "AMC", "STLA", "GM", "BA", "MSTR", "BNDW", "ROBO", "BB", "AI", "GRAB", "SCHD"]:
    converters = {}
    for key in ["Close/Last", "Open", "High", "Low"]:
        converters[key] = lambda s: float(s.replace('$', ''))
    df = pd.read_csv(f"Data/{stock}.csv", converters = converters)
df = df.iloc[::-1].reset_index(drop = True)
df["log_ret"] = np.log(df["Close/Last"]/df["Open"])

if show_all_data == True:
    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.Date, df.High, label = stock)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_locator(dates.MonthLocator(interval = 12))
    ax.set_title(f"{stock} in the last decade")
    plt.show()

if show_daily_changes == True:
    fig = plt.figure(figsize = (12, 5))
    ax_dailypct = fig.add_subplot(1, 2, 1)
    ax = ax_dailypct.twinx()
    hist_ax = fig.add_subplot(1, 2, 2)
    ret_roll = df.log_ret.rolling(window = 30)
    hist_ax.hist(ret_roll.sum(), bins = "auto", color = "green", label = "Binned Data", density = True)
    ax_dailypct.plot(df.Date, df.High.pct_change(), c = "dodgerblue", label = "Daily pct change")
    ax.plot(df.Date, df.High, label = stock, zorder = 0, c = "red")
    ax.xaxis.set_major_locator(dates.MonthLocator(interval = 20))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax_dailypct.set_ylabel("pct change")
    hist_ax.set_xlabel("Log return")
    ax.legend()
    ax_dailypct.legend()
    #fit dist to data
    mu, std = norm.fit(ret_roll.sum().dropna())
    xmin, xmax = hist_ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    hist_ax.plot(x, norm.pdf(x, mu, std), c = "red", linestyle = "--", label = "Gaussian Fit")
    hist_ax.axvline(x = ret_roll.sum().quantile(0.7), label = "Sell Call", c = "red")
    hist_ax.axvline(x = ret_roll.sum().quantile(0.1), label = "Buy Put", c = "magenta")

    #fit t dist to data 
    df_t, loc, scale = t.fit(ret_roll.sum().dropna())
    print(df_t, loc, scale)
    hist_ax.plot(x, t.pdf(x, df_t, loc, scale), c = "aqua", linestyle = "dashdot", label = "t-dist. fit")
    hist_ax.axvline(t.ppf(0.7, df_t, loc, scale), c = "red", linestyle = "dashdot")
    hist_ax.axvline(t.ppf(0.1, df_t, loc, scale), c = "magenta", linestyle = "dashdot")

    #for normal fit ...
    fit = norm.pdf(x, mu, std)
    hist_ax.axvline(x = mu + 0.524*std, c = "red", linestyle = "--")
    hist_ax.axvline(x = mu - 1.282*std, c = "magenta", linestyle = "--")
    hist_ax.legend()
    ax.set_title(f"{stock} Change over last 10 years")
    hist_ax.set_title("Daily pct changes")

    plt.tight_layout()
    plt.show()

if test_class == True:
    my_data = StockData(df["Date"].values, df["Close/Last"].values, df["Open"].values, df["High"].values, df["Low"].values)
    my_data.add_realized_vol(annualize = 1)
    my_data.calc_return(call_threshold = 0.7, put_threshold = 0.1, window_length = 250, contract_length = 22)
    my_data.plot_return()
    plt.show()

if test_analysis_plot == True:
    my_data = Stock("RIVN")
    ax = my_data.show_data()
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("GM Cumulative log returns")
    plt.tight_layout()
    plt.show()

if test_analysis_comparison_plot == True:
    stocks = ["F", "GM", "STLA"]
    portfolio = Portfolio(stocks)
    ax = portfolio.show_return_comparison()
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("American Automakers Comparative Returns")
    plt.tight_layout()
    plt.show()

if test_baseline_compare == True:
    stock = Stock("RIVN")
    baseline = Stock("STLA")
    print(len(stock.data), len(baseline.data))
    ax = stock.show_data(label = "Return")
    comparative_return = stock.compare_to_baseline(baseline)
    print("=====\n", stock.data.Date.values[0], stock.data.Date.values[-1], "\n", baseline.data.Date.values[0], baseline.data.Date.values[-1], "\n=====\n")
    ax.plot(stock.data.Date, comparative_return.cumsum(), label = "vs SPY", c = "red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend()
    ax.set_title("GM returns vs SPY baseline")
    plt.tight_layout()
    plt.show()

if compare_beverage_stocks == True:
    bev_stocks = ['KO', "PEP", "KDP"]
    portfolio = Portfolio(bev_stocks)
    ax = portfolio.show_return_comparison()
    mix = {
        "KO": 0.333,
        "PEP": 0.333,
        "KDP": 0.333
    }
    portfolio.show_mix(mix = mix, ax = ax, label = "Equally weighted mix")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("Bev companies comparative returns 2015 - 2025")
    plt.tight_layout()
    plt.show()

if test_show_portfolio_mix == True:
    #stocks = ["KDP", "KO", "PEP"]
    stocks = ["F", "GM", "STLA"]
    portfolio = Portfolio(stocks)
    start_date = np.datetime64("2022-06-01", "D")
    end_date = np.datetime64("2025-12-10", "D")
    portfolio.chop_data(start_date, end_date)
    ax = portfolio.show_return_comparison()
    '''mix = {
        "KDP": 0.333,
        "KO": 0.333,
        "PEP": 0.333
    }'''
    
    mix = {
        "F": 0.333,
        "GM": 0.333,
        "STLA": 0.333
    }
    mix_data = portfolio.show_mix(mix = mix, ax = ax, label = "Equally weighted mix")
    st()
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("American Bev Co.'s Comparative Returns")
    plt.tight_layout()
    plt.show()

if test_portfolio_performance_against_mean == True:
    fig = plt.figure(figsize = (8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("US Automakers perf against avg")
    stocks = ["F", "GM", "STLA"]
    portfolio = Portfolio(stocks)
    start_date = np.datetime64("2023-12-01", "D")
    end_date = np.datetime64("2025-12-10", "D")
    portfolio.chop_data(start_date, end_date)
    portfolio.show_performance_wrt_mean(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

if compare_pharma == True:
    fig = plt.figure(figsize = (8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("Pharma perf against avg")
    stocks = ["PFE", "MRK", "RHHBY", "JNJ", "AZNCF", "NVS"]
    portfolio = Portfolio(stocks)
    start_date = np.datetime64("2022-06-01", "D")
    end_date = np.datetime64("2025-12-10", "D")
    portfolio.chop_data(start_date, end_date)
    #portfolio.show_return_comparison(ax = ax)
    mix = {
        "PFE": 0.1667,
        "MRK": 0.1667,
        "RHHBY": 0.1667,
        "JNJ": 0.1667,
        "AZNCF": 0.1667,
        "NVS": 0.1667
    }
    portfolio.show_performance_wrt_mean(ax = ax)
    ax.legend()
    plt.tight_layout()
    plt.show()
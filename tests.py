import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates
from scipy.stats import norm, t
import numpy as np
from pdb import set_trace as st
from sim_strat1 import StockData
plt.style.use('dark_background')
from Analysis import *
from DoltReader import *

show_all_data = 0
show_daily_changes = 0
test_class = 0
test_analysis_plot = 0
test_analysis_comparison_plot = 0
test_baseline_compare = 0
compare_beverage_stocks = 0
test_show_portfolio_mix = 0
test_portfolio_performance_against_mean = 0
compare_pharma = 0
test_read_dolt_data = 0
test_get_earnings_data = 0
test_ROIC = 0
test_operating_margin = 0
test_fundamentals_visualizer = 0 #view fundamentals of diff stocks
test_get_eps_CAGR = 0 #get EPS CAGR data
test_get_dolt_stock_data = 0
test_get_dolt_data = 0
test_add_release_date = 0
test_data_alignment = 0
test_stock_split = 0
test_batch_stock_split = 1 


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
    ax.set_title("Paypal vs competitors")
    stocks = ["PYPL", "ADYEY", "BLOCK", "GPN"]
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
    #ax.set_xlabel("Date")
    #ax.set_ylabel("Return")
    #ax.set_title("Pharma perf against avg")
    #stocks = ["PFE", "MRK", "RHHBY", "JNJ", "AZNCF", "NVS"]
    stocks = ["PFE", "NVS"]
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
    mix = {"PFE":0.5, "NVS":0.5}
    colors = [""]
    portfolio.show_performance_wrt_mean(ax = ax)
    ax.legend(fontsize = "x-large")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

if test_get_earnings_data == True:
    reader = DataReader()
    data = reader.get_earnings_data("PFE", "2023-01-01", "2025-12-01", "Quarter")
    st()

if test_ROIC == True:
    reader = DataReader()
    reader.get_earnings_data("F", "2010-01-01", "2022-03-14", "Quarter")
    ROIC = reader.calc_ROIC()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("ROIC")
    ax.plot(pd.to_datetime(reader.earnings_data["income_statement"]["date"]), ROIC)
    ax.set_title("Quarterly ROIC for F from 2010 to 2022")
    plt.tight_layout()
    plt.show()

if test_operating_margin == True:
    reader = DataReader()
    reader.get_earnings_data("PFE", "2010-01-01", "2025-12-01", "Quarter")
    operating_margin = reader.calc_operating_margin()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Operating Margin pct")
    ax.plot(pd.to_datetime(reader.earnings_data["income_statement"]["date"]), operating_margin)
    ax.set_title("Quarterly Operating Margin for PFE from 2010 to 2022")
    plt.tight_layout()
    plt.show()

if test_fundamentals_visualizer == True:
    stocks = ["XOM", "CVX", "COP", "WMB", "EPD"]
    visualizer = FundamentalsVisualizer(stocks = stocks, 
                                        start_date = "2023-01-01",
                                        end_date = "2025-12-01")
    
if test_get_eps_CAGR == True:
    reader = DataReader()
    reader.get_earnings_data("PFE", "2010-01-01", "2025-12-01", "Quarter")
    eps_CAGR = reader.calc_EPS_CAGR()
    st()

if test_get_dolt_stock_data == True:
    dr = DataReader()
    dr.get_stock_data("F", "2024-01-01", "2025-01-31")
    print(dr.stock_data["ohlcv"].head())
    st()

if test_get_dolt_data == True:
    #same as test performance against mean, but now we're reading data from dolt
    fig = plt.figure(figsize = (8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title("Paypal vs competitors")
    stocks = ["PYPL", "ADYEY", "BLOCK", "GPN"]
    portfolio = Portfolio(stocks, data_type = "dolt")
    start_date = np.datetime64("2023-12-01", "D")
    end_date = np.datetime64("2025-12-10", "D")
    portfolio.chop_data(start_date, end_date)
    portfolio.show_performance_wrt_mean(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

if test_add_release_date == True:
    dr = DataReader()
    dr.get_all_data("F", "2022-01-01", "2025-01-31")
    df = dr.stock_data["ohlcv"]
    st()

if test_data_alignment == True:
    dr = DataReader()
    dr.get_all_data("F", "2022-01-01", "2025-01-31")
    keys_to_show = ["release_date", "date", "FCF_margin"]
    hiding_df_keys = [key for key in dr.stock_data["ohlcv"].keys() if key not in keys_to_show]
    data_to_show = dr.stock_data["ohlcv"].drop(hiding_df_keys, axis = 1)
    fun_data_to_show = pd.DataFrame({"date": dr.earnings_data["income_statement"].date.values, "FCF_margin":dr.earnings_variables["FCF_margin"]})
    data_to_show = data_to_show[data_to_show.date > np.datetime64("2022-06-01")]
    #fun_data_to_show = fun_data_to_show[fun_data_to_show.date > np.datetime64("2022-06-01")]
    print(data_to_show, fun_data_to_show)
    print(data_to_show[data_to_show.date == np.datetime64("2023-05-18")])
    print(data_to_show[data_to_show.date == np.datetime64("2023-05-01")])

if test_stock_split == True:
    dr = DataReader()
    stocks = ["TSLA", "GOOGL", "SHOP", "AMZN"]
    fig = plt.figure(figsize = (8, 8))
    for c, stock in enumerate(stocks):
        ax = fig.add_subplot(2, 2, c + 1)

        #show data without split accounted for
        dr.get_all_data(stock, "2022-01-01", "2023-01-31", stock_splits = False)
        ax.plot(dr.stock_data["ohlcv"].date, dr.stock_data["ohlcv"].open, label = "Splits not accounted for", linewidth = 2)
        
        #show data with split accounted for
        dr.get_all_data(stock, "2022-01-01", "2023-01-31", stock_splits = True)
        ax.plot(dr.stock_data["ohlcv"].date, dr.stock_data["ohlcv"].open, label = "Splits accounted for", linestyle = "--", linewidth = 2)
        
        #show data from NASDAQ (which already accounts for splits)
        N = Stock(stock)
        N.chop_data(start_date = np.datetime64("2022-01-01"), end_date = np.datetime64("2023-01-31"))
        ax.plot(N.data["Date"], N.data["Open"], label = "NASDAQ Data", linestyle = ":", c = "magenta")

        #labeling
        ax.set_xlabel("Date")
        ax.set_ylabel("Open")
        ax.set_title(f"{stock} Split vs Unsplit data")
        ax.legend()
    plt.show()

if test_batch_stock_split == True:
    dr_single = DataReader()
    dr_batch = DataReader()
    
    # These stocks all had major splits in 2022
    stocks = ["TSLA", "GOOGL", "SHOP", "AMZN"]
    start_date = "2022-01-01"
    end_date = "2023-01-31"
    
    # 1. Run the NEW Batch Method ONCE for all stocks
    # This tests the efficiency and the cross-contamination logic
    print(f"Running Batch Retrieval for {stocks}...")
    dr_batch.get_batch_stock_data(stocks_list=stocks, start_date=start_date, end_date=end_date)
    batch_ohlcv = dr_batch.stock_data["ohlcv"]

    fig = plt.figure(figsize=(12, 10))
    
    for c, stock in enumerate(stocks):
        ax = fig.add_subplot(2, 2, c + 1)
        
        # 2. Run the ORIGINAL Single Method for this stock (Baseline 1)
        print(f"Running Single Retrieval for {stock}...")
        dr_single.get_all_data(stock, start_date, end_date, stock_splits=True)
        single_ohlcv = dr_single.stock_data["ohlcv"]
        
        # 3. Get NASDAQ Data (Baseline 2 - External Truth)
        nasdaq_ref = Stock(stock)
        nasdaq_ref.chop_data(start_date=np.datetime64(start_date), end_date=np.datetime64(end_date))
        
        # 4. Extract this specific stock from the Batch result
        this_stock_batch = batch_ohlcv[batch_ohlcv['act_symbol'] == stock].sort_values('date')

        # --- PLOTTING ---
        # CYAN: The original single-stock function
        ax.plot(single_ohlcv.date, single_ohlcv.open, 
                label="Original Single-Stock", linewidth=4, c="cyan", alpha=0.4)
        
        # MAGENTA: The NASDAQ truth (already adjusted)
        ax.plot(nasdaq_ref.data["Date"], nasdaq_ref.data["Open"], 
                label="NASDAQ Reference", linestyle=":", c="magenta", linewidth=2)
        
        # WHITE DASHED: The new Batch result
        ax.plot(this_stock_batch.date, this_stock_batch.open, 
                label="New Batch Method", linestyle="--", c="white", linewidth=1.5)

        # Verification: Check if Single matches Batch exactly
        # We handle potential length mismatches (NaNs) using np.allclose
        match = len(single_ohlcv) == len(this_stock_batch)
        status = "PASSED" if match else "LEN MISMATCH"
        
        # Labeling
        ax.set_title(f"{stock} Split Verification - {status}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Open Price")
        ax.legend(fontsize='small')
        
    plt.suptitle("Batch vs Single-Stock Split Adjustment Verification", fontsize=16)
    plt.tight_layout()
    plt.show()
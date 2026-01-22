'''
The goal of this script is to identify stocks which are well-suited for our model

Goal end result:
    dataframe showing:
        top 2000 stocks for 1st of each month
'''

import pandas as pd
from pdb import set_trace as st
import numpy as np
from datetime import date

has_data = True
save_sample = False

if has_data == False:
    from mysql import connector as cnc
    #1. get all price data for all stocks
    #get connection
    conn = cnc.connect(host = "127.0.0.1", 
                       port = 3306, 
                       user = "root", 
                       password = "", 
                       database = None)
    query = "SELECT * FROM stocks.ohlcv"
    all_data = pd.read_sql(query, conn)
    all_data.to_feather("Data/all_ohlcv.feather")
    
    etf_query = "SELECT * FROM stocks.symbol WHERE is_etf = '1';"
    etf_data = pd.read_sql(etf_query, conn)
    etf_data.to_feather("ETFs.feather")
    #get etf data
else:
    all_data = pd.read_feather("Data/all_ohlcv.feather")
    #get etf data
    etf_data = pd.read_feather("Data/ETFs.feather")

if save_sample == True:
    sample_data = all_data.head(500)
    sample_data.to_csv("sample.csv")

#remove ETFs from the dataset
ETFs = etf_data.act_symbol.unique()
all_data = all_data[~all_data['act_symbol'].isin(ETFs)]

#2. loop through months
start_month = 1
start_year = 2012
end_month = 12
end_year = 2025

year = start_year
month = start_month
at_end = False

months = np.linspace(1, 12, 12)
years = np.linspace(start_year, end_year, end_year - start_year + 1)

#filter out ETFs
#all_data = all_data[~all_data['is_etf']]
months_data = []
years_data = []
top_1000_tickers = []
for year in years:
    year = int(year)
    for month in months:
        month = int(month)
        print(month, year)

        #get data for all stocks from the previous month
        prev_month = month - 1 if month != 1 else 12
        prev_year = year if month != 1 else year - 1
        print(month, year, prev_month, prev_year)
        prev_month_data = all_data.loc[(all_data["date"] < date(year, month, 1)) & (all_data["date"] > date(year - 1, month, 1))]

        #add volume to return
        prev_month_data["dollar_vol"] = prev_month_data["close"]*prev_month_data["volume"]
        #get avg volume over the last month
        top_10 = prev_month_data.groupby('act_symbol')['dollar_vol'].mean().nlargest(10).index.tolist()
        avg_dollar_vol_df = prev_month_data.groupby('act_symbol')['dollar_vol'].mean().reset_index()
        avg_dollar_vol_df = avg_dollar_vol_df.sort_values(by='dollar_vol', ascending=False)
        #top_half = avg_dollar_vol_df.head(int(len(avg_dollar_vol_df)/2))
        #top_half = top_half.reset_index(drop = True)
        top_1000 = avg_dollar_vol_df.head(1000).reset_index(drop = True)
        top_1000_names = top_1000.act_symbol.tolist()
        #break on end condition
        months_data.append(month)
        years_data.append(year)
        top_1000_tickers.append(top_1000_names)
        print("Data added")
        if year == end_year and month == end_month:
            break

universe_data = pd.DataFrame({"month":months_data, "year":years_data, "stocks":top_1000_tickers})
universe_data.to_feather("Data/universe_data.feather")

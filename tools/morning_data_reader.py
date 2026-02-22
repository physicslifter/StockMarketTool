'''
Gets data at market open
'''
from yfinance import download
import pandas as pd
from pdb import set_trace as st

TEST = 1

def get_morning_tickers(tickers):
    data = download(tickers, period = "1d", interval = "1m")["Open"].iloc[-1]
    return data

if TEST == True:
    #all_data = pd.read_feather("Data/all_ohlcv.feather")
    #test_tickers = all_data.act_symbol.unique()[0:1000].tolist()
    top_100_tickers = [
    'NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO',
    'NFLX', 'QCOM', 'INTC', 'MU', 'JPM', 'XOM', 'BAC', 'WFC', 'C', 'V',
    'MA', 'CRM', 'ORCL', 'CSCO', 'ACN', 'ADBE', 'LIN', 'TXN', 'COST', 'PEP',
    'KO', 'WMT', 'PG', 'JNJ', 'UNH', 'LLY']
    data = get_morning_tickers(top_100_tickers)
    st()
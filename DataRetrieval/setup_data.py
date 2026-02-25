'''
Script for setting up data
'''

import pandas as pd
from pdb import set_trace as st
import numpy as np
from datetime import date

has_data = False
save_sample = False


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
etf_data.to_feather("Data/ETFs.feather")

#remove ETFs from the dataset
ETFs = etf_data.act_symbol.unique()
all_data = all_data[~all_data['act_symbol'].isin(ETFs)]
all_data.to_feather("all_ohlcv_no_ETFs.feather")
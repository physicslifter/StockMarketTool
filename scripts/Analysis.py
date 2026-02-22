'''
Classes & methods for performing analysis on stock(s)
'''
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from pdb import set_trace as st
from DoltReader import DataReader

class Strategy:
    def __init__(self, name):
        self.name = name

class Stock:
    '''
    Class for stocks downloaded from NASDAQ
    '''
    def __init__(self, name, data_type="nasdaq_csv"):
        self.name = name
        self.set_data_type(data_type)
        self.get_data()
        self.get_log_returns()

    def set_data_type(self, data_type):
        valid_data_types = ["nasdaq_csv", "dolt"]
        if data_type not in valid_data_types:
            raise Exception(f"{data_type} is not valid must be nasdaq_csv or dolt")
        else:
            self.data_type = data_type

    def get_data(self):
        if self.data_type == "nasdaq_csv":
            try:
                if self.name not in ["SPY"]:
                    converters = {}
                    for key in ["Close/Last", "Open", "High", "Low"]:
                        converters[key] = lambda s: float(s.replace('$', ''))
                    self.data = pd.read_csv(f"Data/{self.name}.csv", converters = converters)
                else:
                    self.data = pd.read_csv(f"Data/{self.name}.csv")
            except:
                raise Exception(f"Data for {self.name} not found")
            self.data["Date"] = pd.to_datetime(self.data["Date"])
            self.data = self.data.iloc[::-1].reset_index(drop = True)
        else:
            now = datetime.now()
            month_string = f"0{now.month}" if len(str(now.month)) == 1 else now.month
            day_string = f"0{now.day}" if len(str(now.day)) == 1 else now.day
            current_date = f"{now.year}-{month_string}-{day_string}"
            dr = DataReader()
            dr.get_stock_data(self.name, "2000-01-01", current_date)
            self.data = dr.stock_data["ohlcv"]
            self.data = self.data.rename(columns = {"date":"Date", "open":"Open", "high": "High", "low": "Low", "close":"Close/Last", "volume": "Volume"})
            self.data["Date"] = pd.to_datetime(self.data["Date"])
            
        
    def chop_data(self, start_date, end_date):
        '''
        start and end dates should be np.datetime64
        '''
        self.data = self.data[self.data["Date"].between(start_date, end_date)]

    def get_log_returns(self):
        self.data["log_ret"] = np.log(self.data["Close/Last"]/self.data["Close/Last"].shift(1))
        
    def show_data(self, ax=None, label=None):
        if type(ax) == type(None):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        label = "log return" if type(label) == type(None) else label
        ax.plot(self.data["Date"], self.data["log_ret"].cumsum(), label = label)
        return ax
    
    def compare_to_baseline(self, stock):
        #ensure date range is the same, otherwise crop
        self.data, stock.data = align_data(self.data, stock.data)
        #remove any dates which are not in both
        common_dates = set(stock.data["Date"]).intersection(self.data["Date"])
        #print(stock.data[stock.data["Date"].isin(common_dates) == False])
        stock.data = stock.data[stock.data["Date"].isin(common_dates)]
        self.data = self.data[self.data["Date"].isin(common_dates)]
        for asset in [self, stock]: #recalc w/ new dates
            asset.get_log_returns()
        merged = self.data.merge(stock.data, on = "Date", suffixes = ("_self", "_stock"), how = "inner")
        return merged["log_ret_self"] - merged["log_ret_stock"]
    
def align_stock_datasets(stock1, stock2):
    '''
    given 2 stocks, this function removes data for any dates which are not shared by both stocks
    '''
    if len(stock1.data["Date"].values) != len(stock2.data["Date"].values):
        stock1.chop_data(start_date = stock2.data["Date"].values[0], end_date = stock2.data["Date"].values[-1]) #first chop the reference
        stock2.chop_data(start_date = stock1.data.Date.values[0], end_date = stock1.data.Date.values[-1]) #now chop data to be in the range of the ref
    else:
        if np.array_equal(stock1.data["Date"].values, stock2.data["Date"].values) == False:
            stock1.chop_data(start_date = stock2.data["Date"].values[0], end_date = stock2.data["Date"].values[-1])
    if np.array_equal(stock1.data["Date"].values, stock2.data["Date"].values) == False:
        raise Exception("Unsolved error in compare_to_baseline")
    return stock1, stock2

def align_data(data1, data2):
    '''
    given 2 datasets from NASDAQ, this function removes any dates which are not shared by both datasets
    '''
    #print(data1.Date.min(), data1.Date.max(), data2.Date.min(), data2.Date.max())
    if len(data1["Date"].values) != len(data2["Date"].values):
        data1 = data1[data1["Date"].between(data2["Date"].values[0], data2["Date"].values[-1])] #chop the 1st dataset to be in the date range of the 2nd dataset
        data2 = data2[data2["Date"].between(data1["Date"].values[0], data1["Date"].values[-1])] #chop the 2nd dataset to be in the date range of the 1st dataset
    if np.array_equal(data1["Date"].values, data2["Date"].values) == False: #if the length is the same but the arrays are not equal
            raise Exception("Arrays still diff in align_data")
    #print(data1.Date.min(), data1.Date.max(), data2.Date.min(), data2.Date.max())
    for df in [data1, data2]:
        df.reset_index(drop = True, inplace = True)
    return data1, data2

        
class Portfolio:
    def __init__(self, stocks:list, data_type:str="nasdaq_csv"):
        self.stocks = {}
        for stock in stocks:
            self.stocks[stock] = Stock(stock, data_type)
        self.data = None
    
    def chop_data(self, start_date, end_date):
        for stock in self.stocks:
            self.stocks[stock].chop_data(start_date, end_date)

    def show_return_comparison(self, stocks=[], ax=None):
        return_ax = True if type(ax) == type(None) else False
        if len(stocks) == 0:
            my_stocks = self.stocks
        else:
            my_stocks = {}
            for stock in stocks:
                my_stocks[stock] = self.stocks[stock]
        for c, stock in enumerate(my_stocks.keys()):
            stock = my_stocks[stock]
            if c == 0:
                if type(ax) == type(None):
                    ax = stock.show_data(label = stock.name)
                else:
                    stock.show_data(label = stock.name, ax = ax)
            else:
                stock.show_data(label = stock.name, ax = ax)
        if return_ax == True:
            return ax
        
    def generate_mix(self, mix):
        '''
        generates a dataset for a certain mix of stocks
        mix is a dictionary with stocks as keys and a decimal value between 0 and 1,
            which corresponds to the weight of the stock in the mix
        '''
        for c, stock_name in enumerate(mix.keys()):
            if stock_name.lower() not in [self.stocks[i].name.lower() for i in self.stocks.keys()]:
                raise Exception(f"{stock_name} part of mix but not found in self.stocks")
            if c == 0:
                data = self.stocks[stock_name].data
            else:
                self.stocks[stock_name].data, data = align_data(self.stocks[stock_name].data, data)
                #sum the dataframes
                dates = data["Date"]
                aligned_data, aligned_stock_data = data.align(self.stocks[stock_name].data)

                print(aligned_data.Date.min(), aligned_data.Date.max(), self.stocks[stock_name].data.Date.min(), self.stocks[stock_name].data.Date.max())
                print(aligned_stock_data.Date.min(), aligned_stock_data.Date.max())
                print(self.stocks[stock_name].data.index.min(), self.stocks[stock_name].data.index.max(), data.index.min(), data.index.max())
                numeric_cols = data.select_dtypes(include = "number").columns
                #print(numeric_cols)
                data = aligned_data[numeric_cols] + aligned_stock_data[numeric_cols]
                data["Date"] = dates
            #print(stock_name, self.stocks[stock_name].data.Date.min(), self.stocks[stock_name].data.Date.max(), data.Date.min(), data.Date.max())
            
        data["log_ret"] = np.log(data["Close/Last"]/data["Close/Last"].shift(1))
        return data

    def get_performance_wrt_mix(self, mix):
        mix_data = self.generate_mix(mix)
        mix_performance = {}
        for stock in self.stocks.keys():
            data, mix_data = align_data(self.stocks[stock].data, mix_data)
            mix_performance[stock] = data["log_ret"].values - mix_data["log_ret"].values
            merged = data.merge(mix_data, on = "Date", suffixes = ("_stock", "_merged"), how = "inner")
            result = merged["log_ret_stock"] - merged["log_ret_merged"]
            mix_performance[stock] = result.cumsum()
        #print(len(self.stocks[stock].data["log_ret"].values), len(mix_data["log_ret"].values))
        return mix_performance, mix_data.Date.values

    def show_mix(self, mix, ax, label = None):
        data = self.generate_mix(mix)
        ax.plot(data["Date"], data["log_ret"].cumsum(), label = label)
        return data

    def show_performance_wrt_mean(self, ax):
        mix = {}
        proportion = 1/len(self.stocks.keys())
        for stock in self.stocks:
            mix[stock] = proportion
        performance_result, dates = self.get_performance_wrt_mix(mix)
        for stock in performance_result.keys():
            print(len(dates), len(performance_result[stock]), stock)
            ax.plot(dates, performance_result[stock], label = stock)
'''
Simulate 1st strategy

Covered collar for F

relevant params:
    1. threshold for point @ which to make covered call
    2. threshold for placing log return
    3. previous window over which to calculate distribution
    4. num_days (# of market days of the contract)


values to calculate
    1. log return over the contract for the last num_days
    2. Distribution for each day
    3. sell call/buy put marks from the distribution
'''
import pandas as pd
import numpy as np
from scipy.stats import t, norm
from datetime import datetime, timedelta
import math
from matplotlib import pyplot as plt
from matplotlib import dates

#functions from github.com/erikhox/blacksholes.py
def d1(price, strike, rf, years, volatility):
    '''returns the d1 of the Black-Scholes formula'''
    return (np.log(price/strike)+years*(rf+np.power(volatility, 2)/2))/(volatility*np.sqrt(years))

def d2(price, strike, rf, years, volatility):
    '''returns the d2 of the Black-Scholes formula'''
    return d1(price, strike, rf, years, volatility) - volatility*np.sqrt(years)

def call_value(price, strike, rf, years, volatility):
    '''returns the call premium cost using the Black-Scholes formula'''
    d1_val = d1(price, strike, rf, years, volatility)
    d2_val = d2(price, strike, rf, years, volatility)
    return price*norm.cdf(d1_val)-strike*np.exp(-rf*years)*norm.cdf(d2_val)

def put_value(price, strike, rf, years, volatility):
    '''returns the put premium cost using the Black-Scholes formula'''
    d1_val = d1(price, strike, rf, years, volatility)
    d2_val = d2(price, strike, rf, years, volatility)
    return strike*np.exp(-rf*years)*norm.cdf(-d2_val)-price*norm.cdf(-d1_val)
#===========


class StockData:
    def __init__(self, date, open, close, high, low):
        self.data = pd.DataFrame({"date":date, "open":open, "close":close, "low":low, "high":high})
        
    def calc_return(self, call_threshold, put_threshold, window_length, contract_length):
        self.calc_log_return(contract_length)
        self.get_calls_puts(window = window_length, put_threshold = put_threshold, call_threshold = call_threshold)

    def plot_return(self):
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        ax.plot(self.data.date, self.data.open*100, label = "Opening price")
        ax.plot(self.data.date, self.stock_vals, label = "Strategy principal val", c = "blue")
        ax2.plot(self.data.date, self.put_profits, label = "Strategy put profits", c = "mediumorchid")
        ax2.plot(self.data.date, self.call_profits, label = "Strategy call profits", c = "green")
        total_strat_val = np.array(self.stock_vals) + np.array(self.call_profits)# + np.array(self.put_profits)
        ax.plot(self.data.date, total_strat_val, label = "Strategy Portfolio", c = "red")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(dates.MonthLocator(interval = 12))
        ax.set_ylabel("$ (USD)")
        ax.legend()

    def calc_log_return(self, contract_length:int):
        self.data["ret"] = np.log(self.data.close/self.data.open)
        self.data["contract_ret"] = self.data.ret.rolling(contract_length).sum()

    def add_realized_vol(self, lookback: int = 30, annualize: bool = False):
        # use daily close-to-close log returns for vol estimate
        #vol needed for Black-Scholes
        cc = np.log(self.data["close"] / self.data["close"].shift(1))
        vol = cc.rolling(lookback).std()
        self.data["sigma"] = vol * np.sqrt(252) if annualize else vol

    def get_calls_puts(self, window, put_threshold, call_threshold):
        #initialize portfolio val
        stock_val = self.data.open[0]*100 #100 shares of the initial price of the stock
        stock_vals = []
        call_profit = 0
        call_profits = []
        put_profit = 0
        put_profits = []
        #iterate through the data
        holding_options = False
        for index, row in self.data.iterrows():
            if index < window:
                stock_val = row["open"]
                put_profit = 0
                call_profit = 0
                prev_open = row.open
            else:
                if index == 0:
                    prev_open = row.open
                    stock_val = row.open*100
                else:
                    stock_val_change = row.open - prev_open
                    prev_open = row.open
            
                current_date = datetime.strptime(row.date, "%m/%d/%Y")
                if holding_options == False: #if not holding options, purchase them
                    expiration_date = current_date + timedelta(days = 30) #buy option to expire in 30 days
                
                    #get call and put prices
                    #1. fit data
                    mindex = index - window
                    window_data = self.data.contract_ret[mindex:index]
                    df_t, loc, scale = t.fit(window_data.dropna())
                    call_log_ret = t.ppf(call_threshold, df_t, loc, scale)
                    call_price = row["open"]*np.exp(call_log_ret)
                    call_price = math.ceil(call_price*2)/2

                    call_val = call_value(price = row["open"], strike = call_price, rf = 0.06, years = 30/365, volatility = row["sigma"])
                    
                    put_log_ret = t.ppf(put_threshold, df_t, loc, scale)
                    put_price = row["open"]*np.exp(put_log_ret)
                    put_price = math.floor(put_price*2)/2

                    put_val = put_value(price = row["open"], strike = put_price, rf = 0.06, years = 30/365, volatility = row["sigma"])

                    call_profit += call_val
                    put_profit -= put_val #subtract profit from buying put
                    holding_options = True
                    print(call_val, call_price, put_val, put_price, row["open"])
                else: #if holding options...
                    if current_date >= expiration_date: #if the date to sell the option
                        current_price = row.close
                        if current_price > call_price:
                            #stock_val = call_price*100
                            stock_val = current_price*100
                            call_profit -= (current_price - call_price)*100 #we have to buy back the stock at the higher price to keep the 100 shares
                        else:
                            if current_price > put_price:
                                stock_val += stock_val_change*100
                            else:
                                stock_val = put_price*100
                                put_profit += (put_price - current_price)*100 #we saved this money with the put
                        holding_options = False
                        print("SOLVED", row.date, holding_options, row.close, stock_val, put_price, call_price, current_price)
                    else:
                        stock_val += stock_val_change*100

            stock_vals.append(stock_val)
            call_profits.append(call_profit)
            put_profits.append(put_profit)
        self.stock_vals = stock_vals
        self.call_profits = call_profits
        self.put_profits = put_profits

                




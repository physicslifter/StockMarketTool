'''
Tools for visualizing and manipulating data from 
the dolt stock market database from post-no-preference
@ https://www.dolthub.com/users/post-no-preference
'''
from mysql import connector as cnc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st

database_names = ["earnings", "options", "stocks"]
report_types = ["quarterly", "yearly"]

class DataReader:
    def __init__(self):
        self.has_earnings_data = False
        self.has_stock_data = False
        self.set_connection()
    
    def set_connection(self):
        self.conn = cnc.connect(host = "127.0.0.1", 
                           port = 3306, 
                           user = "root", 
                           password = "", 
                           database = None)
    #=====
    #Earnings
    def check_earnings_data(self):
        #ensures that earnings data has been retrieved
        if self.has_earnings_data == False:
            raise Exception("Earnings data has not yet been retrieved")
    
    def add_release_date(self, start_date, end_date, stock):
        '''
        Adds release dates for reports on all earnings data and prepares 
        calendar for stock data calculation.
        '''
        # 1. Get earnings calendar
        query = f"SELECT * FROM earnings.earnings_calendar WHERE act_symbol = '{stock}' AND date > '{start_date}' AND date < '{end_date}'"
        calendar_df = pd.read_sql(query, self.conn)
        
        # Ensure it is datetime
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        # 2. Store calendar for later use (needed for stock data merging)
        self.earnings_calendar = calendar_df.sort_values('date')

        # 3. Update Earnings Data (Existing Logic)
        if self.has_earnings_data:
            for table in self.earnings_data.keys():
                # We simply attach the calendar dates, shifting if necessary 
                # (Logic preserved from your original script)
                self.earnings_data[table]["release_date"] = self.earnings_calendar['date'].shift(1)

        # 4. Update Stock Data (The new request)
        # We check if stock data exists yet. If yes, calculate immediately.
        if self.has_stock_data:
            self._compute_days_since()

    def _compute_days_since(self):
        '''
        Internal helper: Merges OHLCV with Calendar to calculate days since release.
        Uses merge_asof for high performance.
        '''
        if not hasattr(self, 'earnings_calendar') or 'ohlcv' not in self.stock_data:
            return

        ohlcv = self.stock_data['ohlcv']
        calendar = self.earnings_calendar

        # 1. Ensure Types and Sort (Required for merge_asof)
        ohlcv['date'] = pd.to_datetime(ohlcv['date'])
        ohlcv = ohlcv.sort_values('date')
        
        # 2. Prepare for Merge
        # We merge the 'date' from calendar onto the 'date' of OHLCV
        # direction='backward' finds the most recent past release date
        merged_df = pd.merge_asof(
            ohlcv,
            calendar[['date']].rename(columns={'date': 'release_date'}),
            left_on='date',
            right_on='release_date',
            direction='backward'
        )

        # 3. Calculate Days Difference
        # If no previous report exists (start of dataset), result is NaN
        days_since = (merged_df['date'] - merged_df['release_date']).dt.days
        
        # 4. Save back to the main dataframe
        # We use the index to ensure alignment in case sort changed order
        self.stock_data['ohlcv']['days_since_release'] = days_since.values

    def get_all_data(self, stock, start_date, end_date):
        self.get_earnings_data(stock, start_date, end_date, report_type = "Quarter")
        self.calc_earnings_variables()
        self.get_stock_data(stock, start_date, end_date)

    def get_earnings_data(self, stock, start_date, end_date, report_type):
        if report_type not in ["Quarter", "Year"]:
            raise Exception("Report type must be quarter or year")
        self.report_type = report_type
        tables = ["balance_sheet_liabilities", "balance_sheet_equity", "balance_sheet_assets", "income_statement", "cash_flow_statement"]
        pandas_data = {}
        for table_name in tables:
                print(table_name)
                query = f"SELECT * FROM earnings.{table_name} WHERE act_symbol = '{stock}' AND date > '{start_date}' AND date < '{end_date}' AND period = '{report_type}';"
                print(query)
                df = pd.read_sql(query, self.conn)
                #df["date"] = pd.to_datetime(df["date"]) #coerce col to be datetime
                pandas_data[table_name] = df
        self.earnings_data = pandas_data
        self.has_earnings_data = True
        self.add_release_date(start_date, end_date, stock)
        return pandas_data
    
    def calc_earnings_variables(self):
        variables = {
            "ROIC": self.calc_ROIC(),
            "operating_margin": self.calc_operating_margin(),
            "FCF_margin": self.calc_FCF_margin(),
            "FCF_yield": self.calc_FCF_yield(),
            "EV/EBITDA": self.calc_EV_EBITDA(),
            "rev_CAGR": self.calc_revenue_CAGR(),
            "eps_CAGR": self.calc_EPS_CAGR()
        }
        return variables
    
    def calc_ROIC(self):
        self.check_earnings_data()
        invested_capital = self.earnings_data["balance_sheet_assets"]["total_assets"] - self.earnings_data["balance_sheet_liabilities"]["total_liabilities"]
        NOPAT = self.earnings_data["income_statement"]["pretax_income"] - self.earnings_data["income_statement"]["income_taxes"] - self.earnings_data["income_statement"]["non_operating_income"]
        return NOPAT/invested_capital
    
    def calc_operating_margin(self):
        self.check_earnings_data()
        operating_income = self.earnings_data["income_statement"].pretax_income - self.earnings_data["income_statement"].non_operating_income
        return operating_income/self.earnings_data["income_statement"].sales
    
    def calc_FCF_margin(self):
        self.check_earnings_data()
        operating_cash = self.earnings_data["cash_flow_statement"].net_cash_from_operating_activities
        capex = self.earnings_data["cash_flow_statement"].property_and_equipment
        revenue = self.earnings_data["income_statement"].sales
        return (operating_cash - capex)/revenue
    
    def calc_FCF_yield(self):
        operating_cash = self.earnings_data["cash_flow_statement"].net_cash_from_operating_activities
        capex = self.earnings_data["cash_flow_statement"].property_and_equipment
        market_cap = self.earnings_data["balance_sheet_equity"].shares_outstanding*self.earnings_data["balance_sheet_equity"].book_value_per_share
        return (operating_cash - capex)/market_cap

    def calc_EV(self):
        self.check_earnings_data()
        market_cap = self.earnings_data["balance_sheet_equity"].shares_outstanding*self.earnings_data["balance_sheet_equity"].book_value_per_share
        #debt = notes payable plus long term debt
        total_debt = self.earnings_data["balance_sheet_liabilities"].long_term_debt + self.earnings_data["balance_sheet_liabilities"].notes_payable
        preferred_stock = self.earnings_data["balance_sheet_equity"].preferred_stock
        minority_interest = self.earnings_data["balance_sheet_liabilities"].minority_interest
        cash_and_equivalents = self.earnings_data["balance_sheet_assets"].cash_and_equivalents
        return market_cap + total_debt + preferred_stock + minority_interest - cash_and_equivalents
    
    def calc_EBITDA(self):
        return self.earnings_data["income_statement"].income_before_depreciation_and_amortization
    
    def calc_EV_EBITDA(self):
        EV = self.calc_EV()
        EBITDA = self.calc_EBITDA()
        return EV/EBITDA
    
    def calc_revenue_CAGR(self):
        revenue = self.earnings_data["income_statement"].sales
        offset_revenue = revenue.shift(1)
        my_exp = 4 if self.report_type == "Quarter" else 1
        return (revenue/offset_revenue)**(my_exp)

    def calc_EPS_CAGR(self):
        eps = self.earnings_data["cash_flow_statement"].diluted_net_eps
        offset_eps = eps.shift(1)
        my_exp = 4 if self.report_type == "Quarter" else 1
        return (eps/offset_eps)**my_exp - 1
    #=====

    #=====
    #Stock data
    def check_stock_data(self):
        if self.has_stock_data == False:
            raise Exception("No stock data retrieved")
        
    def get_stock_data(self, stock, start_date, end_date):
        tables = ["dividend", "ohlcv", "split"]
        pandas_data = {}
        for table_name in tables:
            print(table_name)
            date_key = "date" if table_name == "ohlcv" else "ex_date"
            query = f"SELECT * FROM stocks.{table_name} WHERE act_symbol = '{stock}' AND {date_key} > '{start_date}' AND {date_key} < '{end_date}';"
            df = pd.read_sql(query, self.conn)
            pandas_data[table_name] = df
            
        # add symbol info
        symbol_info_query = f"SELECT * FROM stocks.symbol WHERE act_symbol = '{stock}'"
        symbol_info = pd.read_sql(symbol_info_query, self.conn)
        pandas_data["symbol_info"] = symbol_info
        
        self.stock_data = pandas_data
        self.has_stock_data = True
        
        # --- NEW ADDITION ---
        # If earnings data (and calendar) was loaded first, trigger the calculation now
        if hasattr(self, 'earnings_calendar'):
            self._compute_days_since()

class FundamentalsVisualizer:
    def __init__(self, stocks, start_date, end_date):
        '''
        Stocks should be a list of stocks
        '''
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.get_data()
        self.setup_plot()

    def get_data(self):
        fundamental_vars = {}
        self.dates = {}
        for c, stock in enumerate(self.stocks):
            data = DataReader()
            data.get_earnings_data(stock = stock, start_date = self.start_date, end_date = self.end_date, report_type = "Quarter")
            fundamental_vars[stock] = data.calc_earnings_variables()
            dates = pd.to_datetime(data.earnings_data["income_statement"]["date"])
            self.dates[stock] = dates
        self.fundamental_vars = fundamental_vars

    def setup_plot(self):
        '''
        Setup grid for 7 plots
        '''
        fig = plt.figure(figsize = (14, 8))
        for c, var in enumerate(self.fundamental_vars[self.stocks[0]].keys()):
            print(var)
            ax = fig.add_subplot(2, 4, c + 1)
            ax.set_title(var)
            for stock in self.stocks:
                if c == 0:
                    print(stock, self.dates[stock])
                ax.plot(self.dates[stock], self.fundamental_vars[stock][var], label = stock)
            ax.legend()
        for ax in fig.axes:
            ax.tick_params(axis="x", labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        #fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()


        

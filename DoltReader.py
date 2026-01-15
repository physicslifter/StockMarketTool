'''
Tools for visualizing and manipulating data from 
the dolt stock market database from post-no-preference
@ https://www.dolthub.com/users/post-no-preference
'''
from mysql import connector as cnc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

database_names = ["earnings", "options", "stocks"]
report_types = ["quarterly", "yearly"]

class DataReader:
    def __init__(self):
        self.has_earnings_data = False

    def get_data(self, database_name, stock, start_date:np.datetime64, end_date:np.datetime64, report_type:str, table_names:list = None):
        if database_name.lower() not in database_names:
            raise Exception(f"Invalid dataset name. Must be one of {database_names}")
        if report_type not in ["Quarter", "Year"]:
            raise Exception("Report type must be quarter or year")
        #connect to the dataset
        conn = cnc.connect(host = "127.0.0.1", 
                   port = 3306, 
                   user = "root", 
                   password = "", 
                   database = database_name)
        cursor = conn.cursor(buffered = True)
        cursor.execute("SHOW TABLES;")
        tables = [i[0] for i in cursor.fetchall()]
        cursor.close()
        if type(table_names) == type(None): #if no tables are specified, use all tables
            table_names = tables
        else:
            for name in table_names:
                if name not in tables:
                    raise Exception(f"Table name {name} invalid for database {database_name}. Must be one of {tables}")
        
        start_date_str, end_date_str = str(start_date).split(" ")[0], str(end_date).split(" ")[0] 
        pandas_data = {}
        for table_name in table_names:
            print(table_name)
            query = f"SELECT * FROM {table_name} WHERE act_symbol = '{stock}' AND date > '{start_date}' AND date < '{end_date}' AND period = '{report_type}';"
            print(query)
            df = pd.read_sql(query, conn)
            pandas_data[table_name] = df

        return pandas_data
    
    def check_earnings_data(self):
        #ensures that earnings data has been retrieved
        if self.has_earnings_data == False:
            raise Exception("Earnings data has not yet been retrieved")

    def get_earnings_data(self, stock, start_date, end_date, report_type):
        if report_type not in ["Quarter", "Year"]:
            raise Exception("Report type must be quarter or year")
        self.report_type = report_type
        conn = cnc.connect(host = "127.0.0.1", 
                           port = 3306, 
                           user = "root", 
                           password = "", 
                           database = "earnings")
        cursor = conn.cursor(buffered = True)
        tables = ["balance_sheet_liabilities", "balance_sheet_equity", "balance_sheet_assets", "income_statement", "cash_flow_statement"]
        pandas_data = {}
        for table_name in tables:
                print(table_name)
                query = f"SELECT * FROM {table_name} WHERE act_symbol = '{stock}' AND date > '{start_date}' AND date < '{end_date}' AND period = '{report_type}';"
                print(query)
                df = pd.read_sql(query, conn)
                pandas_data[table_name] = df
        self.earnings_data = pandas_data
        self.has_earnings_data = True
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


        

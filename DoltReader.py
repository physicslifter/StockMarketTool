'''
Tools for visualizing and manipulating data from 
the dolt stock market database from post-no-preference
@ https://www.dolthub.com/users/post-no-preference
'''
from mysql import connector as cnc
import pandas as pd
import numpy as np

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
    

    def get_earnings_data(self, stock, start_date, end_date, report_type):
        if report_type not in ["Quarter", "Year"]:
            raise Exception("Report type must be quarter or year")
        conn = cnc.connect(host = "127.0.0.1", 
                           port = 3306, 
                           user = "root", 
                           password = "", 
                           database = "earnings")
        cursor = conn.cursor(buffered = True)
        tables = ["balance_sheet_liabilities", "balance_sheet_equity", "balance_sheet_assets", "income_statement"]
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
    
    def calc_ROIC(self):
        if self.has_earnings_data == False:
            raise Exception("No earnings data has been retrieved")
        invested_capital = self.earnings_data["balance_sheet_assets"]["total_assets"] - self.earnings_data["balance_sheet_liabilities"]["total_liabilities"]
        NOPAT = self.earnings_data["income_statement"]["pretax_income"] - self.earnings_data["income_statement"]["income_taxes"] - self.earnings_data["income_statement"]["non_operating_income"]
        return NOPAT/invested_capital

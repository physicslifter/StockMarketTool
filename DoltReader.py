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

    def merge_earnings_vars_into_ohlcv(self):
        '''
        Merges calculated earnings variables into the OHLCV stock data.
        Aligns variables to release dates by matching period end dates to the 
        next available calendar date, then merges into stock price history.
        '''
        if not hasattr(self, 'earnings_variables') or 'ohlcv' not in self.stock_data:
            return

        # 1. Prepare Financial Variables with Period End Dates
        vars_df = pd.DataFrame(self.earnings_variables)
        
        # We need the period end date to map to a release date.
        # Assuming income_statement aligns with the calculated variables.
        if 'income_statement' in self.earnings_data:
            vars_df['period_end'] = pd.to_datetime(self.earnings_data['income_statement']['date'])
        else:
            return

        # Filter out rows without a period date and sort
        vars_df = vars_df.dropna(subset=['period_end']).sort_values('period_end')

        # 2. Map Release Dates to Financial Periods using merge_asof
        if hasattr(self, 'earnings_calendar'):
            calendar = self.earnings_calendar.copy()
            calendar['date'] = pd.to_datetime(calendar['date'])
            calendar = calendar.sort_values('date')
            
            # Find the NEXT release date for each period end (direction='forward')
            # matching period_end to the closest future calendar date
            vars_with_dates = pd.merge_asof(
                vars_df,
                calendar[['date']].rename(columns={'date': 'release_date'}),
                left_on='period_end',
                right_on='release_date',
                direction='forward'
            )
        else:
            return

        # 3. Merge Financials into OHLCV
        ohlcv = self.stock_data['ohlcv']
        ohlcv['date'] = pd.to_datetime(ohlcv['date'])
        ohlcv = ohlcv.sort_values('date')
        
        # Prepare right-side (financials) for merge
        vars_with_dates = vars_with_dates.dropna(subset=['release_date'])
        vars_with_dates = vars_with_dates.sort_values('release_date')
        
        # Drop period_end (and duplicate release_date from merge if any) to clean up
        # We only need the payload and the key
        cols_to_drop = ['period_end']
        vars_final = vars_with_dates.drop(columns=cols_to_drop, errors='ignore')
        
        # Merge backward: For every stock date, get the most recent past release
        merged_df = pd.merge_asof(
            ohlcv,
            vars_final,
            left_on='date',
            right_on='release_date',
            direction='backward'
        )
        
        self.stock_data['ohlcv'] = merged_df

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

    def get_all_data(self, stock, start_date, end_date, stock_splits:bool = True):
        self.has_stock_data = False
        self.stock_data = {}
        self.get_earnings_data(stock, start_date, end_date, report_type = "Quarter")
        self.calc_earnings_variables()
        self.get_stock_data(stock, start_date, end_date, stock_splits, calc_all = True)
        self.merge_earnings_vars_into_ohlcv()

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
        self.earnings_variables = variables
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
        
    def adjust_for_splits(self):
        '''
        Adjusts OHLCV prices and volume for stock splits.
        Uses 'to_factor' and 'for_factor'.
        Includes logic to detect and skip 'phantom' duplicate split records 
        where the price did not actually drop (e.g. Record Date vs Ex-Date).
        '''
        if not hasattr(self, 'stock_data'): return
        if "split" not in self.stock_data or "ohlcv" not in self.stock_data: return
        
        splits = self.stock_data["split"]
        ohlcv = self.stock_data["ohlcv"]

        if splits.empty or ohlcv.empty: return

        # Work on copies to prevent index warnings
        splits = splits.copy()
        ohlcv['date'] = pd.to_datetime(ohlcv['date'])
        
        # Sort splits by date to process chronologically
        splits = splits.sort_values('ex_date')

        for _, row in splits.iterrows():
            split_date = pd.to_datetime(row['ex_date'])
            
            # 1. Calculate Ratio from to_factor / for_factor
            try:
                # Use .get() to be safe, default to 0
                tf = float(row.get('to_factor', 0))
                ff = float(row.get('for_factor', 0))
                
                # Calculate ratio (e.g. 20 / 1 = 20.0)
                if tf > 0 and ff > 0:
                    ratio = tf / ff
                else:
                    # Fallback if columns are missing but 'ratio' exists
                    ratio = float(row.get('ratio', 0))
            except:
                continue

            # Skip invalid ratios
            if ratio <= 0 or ratio == 1.0:
                continue

            # 2. VALIDATION: Check if the price actually dropped on this date
            # This filters out the duplicate "May 26" row where price didn't move.
            
            # Get data immediately surrounding the split
            # We look for the last close BEFORE the split and the first close AFTER
            mask_pre = ohlcv['date'] < split_date
            mask_post = ohlcv['date'] >= split_date
            
            # Only validate if we have data on both sides of the date
            if mask_pre.any() and mask_post.any():
                prev_close = ohlcv.loc[mask_pre, 'close'].iloc[-1]
                curr_close = ohlcv.loc[mask_post, 'close'].iloc[0]
                
                # Calculate the observed drop in the market data
                # e.g. Prev $2400 / Curr $120 = 20.0
                observed_drop = prev_close / curr_close
                
                # Compare Observed vs Expected (Allow 25% slack for market volatility)
                # If ratio is 20, we accept an observed drop between 15 and 25.
                error_margin = abs(observed_drop - ratio) / ratio
                
                if error_margin > 0.25:
                    print(f"DEBUG: Skipping split on {split_date.date()} (Ratio {ratio}). "
                          f"Price did not drop (Observed: {observed_drop:.2f}).")
                    continue
                else:
                    print(f"DEBUG: Verified split on {split_date.date()}. "
                          f"Price dropped by factor {observed_drop:.2f}. Adjusting...")

            # 3. Apply Adjustment
            # Divide historical prices by the ratio
            price_cols = ['open', 'high', 'low', 'close', 'prev_close']
            # Case-insensitive match for columns
            for col in ohlcv.columns:
                if col.lower() in price_cols:
                    ohlcv.loc[mask_pre, col] = ohlcv.loc[mask_pre, col] / ratio
            
            # Multiply historical volume by the ratio
            for col in ohlcv.columns:
                if col.lower() == 'volume':
                    ohlcv.loc[mask_pre, col] = ohlcv.loc[mask_pre, col] * ratio

        self.stock_data["ohlcv"] = ohlcv

    def get_stock_data(self, stock, start_date, end_date, account_for_splits:bool=True, calc_all:bool = True):
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

        if account_for_splits == True:
            self.adjust_for_splits()

        ohlcv = self.stock_data["ohlcv"]
        self.stock_data["ohlcv"]["log_ret"] = np.log(ohlcv.close/ohlcv.close.shift(1))
        self.stock_data["ohlcv"]["prev_close"] = ohlcv.close.shift(1)
        self.stock_data["ohlcv"]["prev_ret"] = ohlcv.log_ret.shift(1)
        self.has_stock_data = True
        
        if calc_all == True:
            self.calc_additional_price_variables()
            # If earnings data (and calendar) was loaded first, trigger the calculation now
        if hasattr(self, 'earnings_calendar'):
            self._compute_days_since()

    def calc_additional_price_variables(self):
        if self.has_stock_data == False:
            raise Exception("No stock data exists")
        #returns (1, 5 & 20 day)
        for length in [5, 20]:
            self.stock_data["ohlcv"][f"{length}_day_log_ret"] = self.stock_data["ohlcv"]["log_ret"].rolling(window = length).sum()

        #volatility (20d)
        self.stock_data["ohlcv"]["20d_volatility"] = self.stock_data["ohlcv"]["log_ret"].rolling(window = 20).std()

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


        

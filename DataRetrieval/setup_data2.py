'''
Script for setting up data with efficient Earnings Merge
'''
import pandas as pd
import numpy as np
from mysql import connector as cnc

# Database Connection
conn = cnc.connect(host="127.0.0.1", port=3306, user="root", password="", database=None)

print("1. Fetching OHLCV Data...")
query = "SELECT * FROM stocks.ohlcv"
all_data = pd.read_sql(query, conn)

# --- FILTER ETFS EARLY (Saves RAM/Time) ---
print("2. Removing ETFs...")
etf_query = "SELECT act_symbol FROM stocks.symbol WHERE is_etf = '1';"
etf_data = pd.read_sql(etf_query, conn)
ETFs = etf_data.act_symbol.unique()

# Filter rows where symbol is NOT in the ETF list
all_data = all_data[~all_data['act_symbol'].isin(ETFs)].copy()

# Ensure datetime for merging later
all_data['date'] = pd.to_datetime(all_data['date'])
all_data = all_data.sort_values(['act_symbol', 'date'])

# --- EARNINGS SECTION ---
print("3. Fetching Earnings Data...")

# Fetch necessary tables
inc = pd.read_sql("SELECT * FROM earnings.income_statement", conn)
bal_ast = pd.read_sql("SELECT * FROM earnings.balance_sheet_assets", conn)
bal_liab = pd.read_sql("SELECT * FROM earnings.balance_sheet_liabilities", conn)
calendar = pd.read_sql("SELECT * FROM earnings.earnings_calendar", conn)

# Merge tables to create a "Raw Fundamentals" dataframe
# We join on Symbol, Date (Period End), and Period (Q/Y)
fund_df = pd.merge(inc, bal_ast, on=['act_symbol', 'date', 'period'], suffixes=('', '_ast'))
fund_df = pd.merge(fund_df, bal_liab, on=['act_symbol', 'date', 'period'], suffixes=('', '_liab'))

print("4. Calculating Fundamental Variables...")
# --- VECTORIZED CALCULATIONS ---
# Calculate ROIC, Margins, etc. here so we only merge the RESULTS into OHLCV
# (Merging 100 raw columns into OHLCV will explode your RAM)

fund_df['invested_capital'] = fund_df['total_assets'] - fund_df['total_liabilities']
fund_df['nopat'] = fund_df['pretax_income'] - fund_df['income_taxes'] - fund_df['non_operating_income']

# Handle division by zero safely
fund_df['ROIC'] = np.where(fund_df['invested_capital'] != 0, 
                           fund_df['nopat'] / fund_df['invested_capital'], np.nan)

fund_df['operating_margin'] = (fund_df['pretax_income'] - fund_df['non_operating_income']) / fund_df['sales']

# Select only what we want to merge into stock prices
cols_to_keep = ['act_symbol', 'date', 'ROIC', 'operating_margin'] 
clean_funds = fund_df[cols_to_keep].copy()

# --- DATE ALIGNMENT (CRITICAL) ---
print("5. Aligning Release Dates...")
# We must map the "Period End Date" to the "Public Release Date"
calendar['date'] = pd.to_datetime(calendar['date']) # This is Release Date
clean_funds['date'] = pd.to_datetime(clean_funds['date']) # This is Period End

clean_funds = clean_funds.sort_values('date')
calendar = calendar.sort_values('date')

# Find the NEXT release date for every financial period
aligned_funds = pd.merge_asof(
    clean_funds,
    calendar[['act_symbol', 'date']].rename(columns={'date': 'release_date'}),
    left_on='date',
    right_on='release_date',
    by='act_symbol',
    direction='forward' # Look forward to find when this data was released
)

# Remove data that hasn't been released yet or didn't match
aligned_funds = aligned_funds.dropna(subset=['release_date'])

# --- FINAL MERGE ---
print("6. Merging Fundamentals into OHLCV...")

# Sort for merge_asof
aligned_funds = aligned_funds.sort_values('release_date')
all_data = all_data.sort_values('date')

# Merge Fundamentals into Price History
# direction='backward': For every price date, find the most recent PAST release date
final_df = pd.merge_asof(
    all_data,
    aligned_funds[['act_symbol', 'release_date', 'ROIC', 'operating_margin']],
    left_on='date',
    right_on='release_date',
    by='act_symbol',
    direction='backward'
)

# Calculate days since release (Vectorized)
final_df['days_since_release'] = (final_df['date'] - final_df['release_date']).dt.days

# Fill NaN for days_since (e.g., before the company ever reported earnings)
final_df['days_since_release'] = final_df['days_since_release'].fillna(-1)

print("7. Saving Data...")
# Reset index to make it clean
final_df = final_df.reset_index(drop=True)

# Save
final_df.to_feather("../Data/all_data_with_fundamentals.feather")
print("Done!")
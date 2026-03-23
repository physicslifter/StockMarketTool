'''
Testing fundamental features

test_fundamental_price_match: tests whether all price data is shared in the fundamental features
'''
import pandas as pd
import numpy as np

test_fundamental_price_match = 1

if test_fundamental_price_match == True:
    def validate_price_and_fundamentals(price_df, fund_df, ticker_col='act_symbol', date_col='date'):
        """
        Compares a price dataframe and a fundamental dataframe to find missing tickers, 
        misaligned date ranges, and massive data gaps.
        """
        # 1. Ensure dates are datetime objects
        price_df[date_col] = pd.to_datetime(price_df[date_col])
        fund_df[date_col] = pd.to_datetime(fund_df[date_col])

        # Filter both to 2016-onwards (just in case there's older data)
        price_df = price_df[price_df[date_col] >= '2016-01-01']
        fund_df = fund_df[fund_df[date_col] >= '2016-01-01']

        # ---------------------------------------------------------
        # TEST 1: Completely Missing Tickers
        # ---------------------------------------------------------
        price_tickers = set(price_df[ticker_col].unique())
        fund_tickers = set(fund_df[ticker_col].unique())

        missing_entirely = price_tickers - fund_tickers
        print("="*60)
        print(f"TEST 1: {len(missing_entirely)} tickers in Price Data have ZERO Fundamental Data.")
        if missing_entirely:
            print(f"Examples of missing tickers: {list(missing_entirely)[:10]}")

        # ---------------------------------------------------------
        # TEST 2: Date Range Coverage
        # ---------------------------------------------------------
        # Group by ticker to get the first and last dates in both datasets
        price_ranges = price_df.groupby(ticker_col)[date_col].agg(['min', 'max']).rename(
            columns={'min': 'price_start', 'max': 'price_end'}
        )
        fund_ranges = fund_df.groupby(ticker_col)[date_col].agg(['min', 'max', 'count']).rename(
            columns={'min': 'fund_start', 'max': 'fund_end', 'count': 'fund_reports'}
        )

        # Merge them together (Left join on price data)
        coverage_df = price_ranges.join(fund_ranges, how='left')

        # A ticker is missing early fundamental data if fundamentals start significantly AFTER price data.
        # We allow a 90-day grace period because a newly listed IPO might not file its first report immediately.
        coverage_df['missing_early_data'] = coverage_df['fund_start'] > (coverage_df['price_start'] + pd.Timedelta(days=90))

        # A ticker is missing recent fundamental data if fundamentals end significantly BEFORE price data.
        coverage_df['missing_recent_data'] = coverage_df['fund_end'] < (coverage_df['price_end'] - pd.Timedelta(days=120))

        early_fails = coverage_df[coverage_df['missing_early_data']].index.tolist()
        recent_fails = coverage_df[coverage_df['missing_recent_data']].index.tolist()

        print("\n" + "="*60)
        print(f"TEST 2: Date Boundary Alignment")
        print(f"- {len(early_fails)} tickers have price data that starts BEFORE their fundamental data.")
        print(f"- {len(recent_fails)} tickers have price data that ends AFTER their fundamental data stops.")

        # ---------------------------------------------------------
        # TEST 3: Density (Checking for massive missing gaps)
        # ---------------------------------------------------------
        # Calculate how many years the fundamental data spans for each ticker
        coverage_df['fund_years_spanned'] = (coverage_df['fund_end'] - coverage_df['fund_start']).dt.days / 365.25

        # Calculate reports per year (Expect ~4 for quarterly, ~1 for annual)
        # Avoid division by zero for tickers with less than a year of data
        coverage_df['reports_per_year'] = np.where(
            coverage_df['fund_years_spanned'] > 0.5, 
            coverage_df['fund_reports'] / coverage_df['fund_years_spanned'], 
            np.nan
        )

        # Let's say we expect Quarterly data. Anything under 3 reports/year means heavy missing data.
        # (Adjust this to < 0.8 if your fundamental data is ANNUAL instead of quarterly).
        gap_fails = coverage_df[coverage_df['reports_per_year'] < 2.5].index.tolist()

        print("\n" + "="*60)
        print(f"TEST 3: Internal Gaps / Density")
        print(f"- {len(gap_fails)} tickers seem to have massive gaps (averaging < 2.5 reports per year).")
        print("="*60)

        # Return the dataframe so you can manually inspect the problematic tickers
        return coverage_df
    
    #read in datasets
    price_data = pd.read_feather(f"../Data/Universes/1.feather")
    fundamental_data = pd.read_feather("../Data/all_fundamentals.feather")
    validate_price_and_fundamentals(price_df = price_data, fund_df = fundamental_data)


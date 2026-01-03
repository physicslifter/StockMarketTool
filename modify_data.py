'''
This script manipulates data from https://www.dolthub.com/repositories/post-no-preference/earnings/data/master/income_statement
to be in a single dataset with required variables calculated
'''
import pandas as pd
from pdb import set_trace as st

#Read in the datasets
income_statement = pd.read_parquet("Data/Financials/income_statement.parquet")
cash_flow_statement = pd.read_parquet("Data/Financials/cash_flow_statement.parquet")
balance_sheet_assets = pd.read_parquet("Data/Financials/balance_sheet_assets.parquet")

for dataset in [income_statement, cash_flow_statement, balance_sheet_assets]:
    print(len(dataset))

#1. Calculate ROIC
#1A: calculate NOPAT (Net Operating Profit After Tax)
tax_rate = income_statement.income_taxes/income_statement.pretax_income
income_statement["NOPAT"] = income_statement.income_after_depreciation_and_amortization*(1 - tax_rate)
st()

#2. operating margin

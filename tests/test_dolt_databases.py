"""
Ensures that functionality is working for dolt 

Test logic:
1. Demonstrate data retrieval works for both earnings and stocks
2. Demonstrate stock splits
    If this works then the heart of the logic is working correctly
"""
from DoltReader import *

try:
    print("========\nTesting Dolt Data Functionality...\n========")
    dr = DataReader()
    stocks = ["TSLA", "GOOGL", "SHOP", "AMZN"]
    for stock in stocks:
        dr.get_all_data(stock, "2022-01-01", "2023-01-31", stock_splits = True)
        print(f"{stock} DATA RETRIEVED")
    print("\nSUCCESS: DoltReader can Read the database\n========")
except:
    print("TEST FAILURE: DoltReader could not read from databases")
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("dark_background")

PERFORM_ADJUSTMENT = False

#get dataframes to adjust
splits = pd.read_csv("Data/SPLITS.csv")
ohlcv = pd.read_feather("Data/all_ohlcv.feather")
ohlcv["volume"] = ohlcv["volume"].astype(float)

for df, key in zip([ohlcv, splits], ["date", "ex_date"]):
    df[key] = pd.to_datetime(df[key])

#show unadjusted data for aapl over 2020
AAPL = ohlcv[ohlcv["act_symbol"] == "AAPL"]
AAPL_2020 = AAPL[
    (AAPL.date >= pd.Timestamp(year = 2020, month = 1, day = 1)) & 
    (AAPL.date < pd.Timestamp(year = 2021, month = 1, day = 1))
    ]
plt.plot(AAPL_2020.date, AAPL_2020.close, c = "teal", label = "Not adjusted for split")

if PERFORM_ADJUSTMENT == True:
    #adjust for splits
    from DoltReader import DataReader

    dr = DataReader()
    dr.stock_data = {
        "ohlcv": ohlcv,
        "split": splits
    }
    print("Adjusting...")
    dr.adjust_for_splits_batch()
    ohlcv = dr.stock_data["ohlcv"]

else:
    ohlcv = pd.read_feather("Data/all_ohlcv_w_splits.feather")

AAPL = ohlcv[ohlcv["act_symbol"] == "AAPL"]
AAPL_2020 = AAPL[
    (AAPL.date >= pd.Timestamp(year = 2020, month = 1, day = 1)) & 
    (AAPL.date < pd.Timestamp(year = 2021, month = 1, day = 1))
    ]
plt.plot(AAPL_2020.date, AAPL_2020.close, c = "mediumspringgreen", linestyle = "--", label = "Adjusted")

plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Splits accounted for")
plt.show()

ohlcv.to_feather("Data/all_ohlcv_w_splits.feather")
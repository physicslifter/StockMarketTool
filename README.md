# StockMarketTool
Tool for analyzing stock data and backtesting strategies

## 1. Setup the repository as a python package
In the main folder, run:
```
pip install -e .
```

## 2. Setup dolt sql databases
Databases @: https://www.dolthub.com/repositories/post-no-preference
  
### 2.1. Create a new empty directory to put the databases in 
Somewhere convenient on your computer (doesn't have to be in this repo)
```
mkdir post-no-preference-data
cd post-no-preference-data
```
  
### 2.2. Use dolt to get the databases onto your computer  
```
dolt clone https://www.dolthub.com/repositories/post-no-preference/earnings
dolt clone https://www.dolthub.com/repositories/post-no-preference/stocks
```  
  
### 2.3. Go into each database to make sure it is working properly
```
cd stocks
dolt status

cd ../earnings
dolt status
cd ..
```
The output to each 'dolt status' command should look like:
```
On branch master
Your branch is up to date with 'origin/master'.
nothing to commit, working tree clean
```

### 2.4 Activate the server
in /post-no-preference-data, run
```
dolt sql-server
```
to activate the databases.

### 2.5. Run tests/test_dolt_databases.py to ensure data functionality is working properly.  
```
cd tests
python3 test_dolt_databases.py
```
The output should say 'SUCCESS'
If the output says something different, the python code is not able to read from the databases. Try setting up databases again. Make sure you have activated the databases from the parent folder. If this is not working, please open a new issue in the repository.

## 3. Setup repository with useful data
You can read in data each time you train a model with the DoltReader functions. However, it is often more effecient to save the data locally as a feather file. If you do not specify a dataframe for the universe or model, the classes automatically pull this saved feather file as the universe.
```
cd DataRetrieval
python3 setup_data.py
```
The data will be saved as Data/all_ohlcv.feather.

## 4. Create a Universe  
```
import pandas as pd
from clean_analysis import TopLiquidityFilter, PriceFilter, AdvancedStatsFilter, Universe

df = pd.read_feather("../Data/all_ohlcv.feather") #get data
df["date"] = pd.to_datetime(df["date"]) #coerce data to pd.Timestamp

#define filters for the universe
filters = [
    TopLiquidityFilter(N = 3000),
    PriceFilter(min_price = 5),
    AdvancedStatsFilter(min_history_days = None,
                        max_crash = -0.5,
                        require_uptrend = None,
                        volatility_n = 1000)
]
universe = Universe(master_df = df) #create Universe instance
universe.add_filters(filters) #add filters to universe
dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")] #start/end dates for the universe
universe_data = universe.get_all_universe_data(dates = dates) #get the data
```

## 5. Create Features & target for the model
```
from FeatureEngine import FeatureRequest

volatility = FeatureRequest(name='VOL_ZSCORE', params={'timeperiod': 20}, shift=-1, input_type='raw', alias='vol_z')
liquidity = FeatureRequest(name='SPREAD_AR', params={'timeperiod': 20}, shift=-1, alias='liquidity', transform='rank')
autocorrelation = FeatureRequest(name='AUTOCORR', params={'timeperiod': 20}, shift=-1, alias='auto_corr', input_type='log_ret')
z_score = FeatureRequest(name='ZSCORE', params={'timeperiod': 20}, shift=-1, input_type='log_ret', alias='vol_zscore')

target = FeatureRequest(name='SUM', params={'timeperiod': 5}, shift=1, input_type='log_ret', alias='target_5d', transform='binary')
```

## 6. Train model
```
from clean_analysis import model

model = Model(universe_data)
model.add_features([volatility, liquidity, autocorrelation, z_score])
model.add_target(target)
model.split_data(cutoffs = [0.7, 0.85, 1])
model.train_model(save_name = "clean_model_test")
```


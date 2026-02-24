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

### 1.5. Run tests/test_dolt_databases.py to ensure data functionality is working properly.  
```
cd tests
python3 test_dolt_databases.py
```
The output should say 'SUCCESS'
If the output says something different, the python code is not able to read from the databases. Try setting up databases again. Make sure you have activated the databases from the parent folder. If this is not working, please open a new issue in the repository.

## 3. Create a Universe  
## 4. Create a model from the universe

'''
Tools for visualizing and manipulating data from 
the dolt stock market database from post-no-preference
@ https://www.dolthub.com/users/post-no-preference
'''
from mysql import connector as cnc
import pandas as pd

database_names = ["earnings", "options", "stocks"]
report_types = ["quarterly", "yearly"]

class DataReader:
    def __init__(self):
        pass

    def get_data(database_name, stock, start_date, end_date, report_type, table_names = None):
        if database_name.lower() not in database_names:
            raise Exception(f"Invalid dataset name. Must be one of {database_names}")
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
            for name in table_names, tables:
                if name not in tables:
                    raise Exception(f"Table name {name} invalid for database {database_name}. Must be one of {tables}")
        
        query = "SELECT * FROM"
        query += ""

    

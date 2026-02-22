import talib
from talib import abstract

def get_timeperiod_functions():
    # Get a list of all available TA-Lib function names
    all_functions = talib.get_functions()
    
    timeperiod_funcs = []

    print(f"{'Function':<15} | {'Default Timeperiod'}")
    print("-" * 35)

    for func_name in all_functions:
        try:
            # Use the abstract interface to get metadata about the function
            func_info = abstract.Function(func_name).info
            
            # Check if 'timeperiod' exists in the parameters dictionary
            parameters = func_info.get('parameters', {})
            
            if 'timeperiod' in parameters:
                # Get the default value for display purposes
                default_val = parameters['timeperiod']
                timeperiod_funcs.append(func_name)
                print(f"{func_name:<15} | {default_val}")
                
        except Exception as e:
            # Some functions might not load via abstract depending on version
            continue

    return timeperiod_funcs

if __name__ == "__main__":
    print("Searching for TA-Lib functions with 'timeperiod' argument...\n")
    results = get_timeperiod_functions()
    print(f"\nFound {len(results)} functions.")
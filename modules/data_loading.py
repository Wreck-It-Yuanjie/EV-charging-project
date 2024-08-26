
import pandas as pd

## Make a class

def load_data(filepath):
    """
    Loads the EV charging data from a CSV file.
    
    Parameters:
        filepath (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    EVdf0 = pd.read_csv(filepath, low_memory=False)
    return EVdf0

if __name__ == "__main__":
    EVdf0 = load_data("EVChargingData2010_2020.csv")
    ## Assert
    

import os
import pandas as pd

def load_data(filepath):
    """
    Loads the EV charging data from a CSV file.
    
    Parameters:
        filepath (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    data = pd.read_csv(filepath, low_memory=False)
    return data

def test_load_data():
    mock_csv_path = os.path.join('data', 'EVChargingData2010_2020.csv')
    df = load_data(mock_csv_path)
    assert list(df.columns) == ['Station Name', 'MAC Address', 'Org Name', 'Start Date',
       'Start Time Zone', 'End Date', 'End Time Zone',
       'Transaction Date (Pacific Time)', 'Total Duration (hh:mm:ss)',
       'Charging Time (hh:mm:ss)', 'Energy (kWh)', 'GHG Savings (kg)',
       'Gasoline Savings (gallons)', 'Port Type', 'Port Number', 'Plug Type',
       'EVSE ID', 'Address 1', 'City', 'State/Province', 'Postal Code',
       'Country', 'Latitude', 'Longitude', 'Currency', 'Fee', 'Ended By',
       'Plug In Event Id', 'Driver Postal Code', 'User ID', 'County',
       'System S/N', 'Model Number'] ## Checking columns
    
    print("All tests passed")

# Test
if __name__ == '__main__':
    test_load_data()
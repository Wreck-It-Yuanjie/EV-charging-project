import os
import pandas as pd
import numpy as np
import math

def convert_duration_to_time_window(duration_str, time_window='1H'):
    """
    Convert charging duration from hh:mm:ss format to the specified time window unit.
    
    Parameters:
        duration_str (str): The charging duration in hh:mm:ss format.
        time_window (str): The time window for conversion. Can be '1H', '1D', '1W', or '1M'.
        
    Returns:
        float: The total hours converted to the specified time window unit.
    """
    # Convert seconds to total hours
    total_hours = pd.Timedelta(duration_str).total_seconds() / 3600
    
    # Convert based on the specified time window
    if time_window == '1H':
        return total_hours
    elif time_window == '1D':
        return total_hours / 24
    elif time_window == '1W':
        return total_hours / (24 * 7)
    elif time_window == '1M':
        return total_hours / (24 * 30)
    else:
        raise ValueError("Unsupported time window. Choose from '1H', '1D', '1W', '1M'.")

def aggregate_ev_charging(data, time_window = '1D'):
    """
    Aggregate energy consumption, total charging duration, and count EV charging events
    within specified time windows.
    
    Parameters:
        data (pd.DataFrame): The EV charging data.
        time_window (str): The time window for aggregation. Can be '1H', '1D', '1W', or '1M'.
    
    Returns:
        pd.DataFrame: Aggregated data with total energy, total charging duration, and count of events.
    """
    dataframe = data.copy()
    
    dataframe['Start Date'] = pd.to_datetime(dataframe['Start Date'], errors='coerce')
    dataframe = dataframe.dropna(subset=['Start Date'])
    
    dataframe.set_index('Start Date', inplace=True)
    
    # Convert Charging Time to time hours
    dataframe[f'Charging Time ({time_window})'] = dataframe['Charging Time (hh:mm:ss)'].apply(convert_duration_to_time_window, time_window=time_window)
    
    # Resample and aggregate data
    aggregated_data = dataframe.groupby('Station Name').resample(time_window).agg({
        'Energy (kWh)': 'sum',
        f'Charging Time ({time_window})': 'sum',
        'User ID': 'count'  # Count of events
    })
    
    aggregated_data.rename(columns={
        'Energy (kWh)': 'Total Energy (kWh)',
        f'Charging Time ({time_window})': f'Total Charging Duration ({time_window})',
        'User ID': 'Event Count'
    }, inplace=True)

    aggregated_data = aggregated_data.reset_index()
    aggregated_data.set_index('Start Date', inplace=True)
    
    return aggregated_data

def test_aggregate_ev_charging():
    """
    Test the aggregate_ev_charging function to ensure it correctly aggregates the EV charging data.
    """

    result = aggregate_ev_charging(df, time_window='1D')
    result.to_csv('./data/aggregated_D.csv', index=True)
    
    # Test: Check the shape of the result (should be 3 rows)
    assert result.shape[1] == 4, f"Expected 4 columns, got {result.shape[1]}"
    print("All tests passed for aggregate_ev_charging! Dataset saved.")

if __name__ == "__main__":
    # Path to the dataset in the data folder
    data_file_path = os.path.join('data', 'cleaned_df.csv') 

    # Load the clean dataset
    df = pd.read_csv(data_file_path, low_memory=False)

    test_aggregate_ev_charging()
    
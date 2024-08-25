
import pandas as pd
import numpy as np
import math
from gluonts.time_feature.holiday import SpecialDateFeatureSet, CHRISTMAS_DAY, CHRISTMAS_EVE
from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.mx import DeepAREstimator, Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName
from gluonts.mx.trainer.callback import TrainingHistory
from sklearn.preprocessing import LabelEncoder

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

    # aggregated_data.set_index('Start Date', inplace=True)
    
    return aggregated_data.reset_index()

def multiple_time_series(data, target, fields):
    """
    Prepare multiple time series data for forecasting by encoding categorical variables.
    
    Parameters:
        data (pd.DataFrame): The EV charging data.
        target (str): The target variable for forecasting.
        fields (list): The list of features to include.
        
    Returns:
        pd.DataFrame: DataFrame formatted for time series forecasting.
    """
    label_encoder = LabelEncoder()
    testData = data.loc[:, fields]
    testData = testData.rename(columns = {"Total Energy (kWh)": "target"})
    testData['item_id'] = label_encoder.fit_transform(testData['Station Name'])
    station_name_map = testData[['Station Name', 'item_id']]
    station_name_map = station_name_map.reset_index().drop(columns = ["Start Date"]).drop_duplicates().reset_index().drop(columns = ["index"])
    testData.drop(columns = ['Station Name'], inplace = True)
    return testData, station_name_map

def handle_missing_data(testData, freq, target_column):
    """
    Handle missing data in the dataset by forward-filling and interpolating where necessary.
    
    Parameters:
        testData (pd.DataFrame): The time series dataset.
        freq (str): The frequency of the data.
        target_column (str): The name of the target column.
        
    Returns:
        pd.DataFrame: DataFrame with missing data handled.
    """

    ds = PandasDataset.from_long_dataframe(testData, target=target_column, item_id='item_id', freq=freq)
    
    max_end = max(testData.groupby("item_id").apply(lambda _df: _df.index[-1]))
    dfs_dict = {}
    for item_id, gdf in testData.groupby("item_id"):
        new_index = pd.date_range(gdf.index[0], end=max_end, freq=freq)
        dfs_dict[item_id] = gdf.reindex(new_index).drop("item_id", axis=1)
        dfs_dict[item_id][np.isnan(dfs_dict[item_id])] = 0


def train_test_split(ds, train_val_test_split):

    train_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[0])
    validation_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[1])
    prediction_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[2])

    return train_length, validation_length, prediction_length


def add_multiple_features(EVdata, freq):
    """
    Add temporal and additional features for EV charging data.
    
    Parameters:
        EVdata (pd.DataFrame): The EV charging data.
        freq (str): Frequency string like '1H', '1D', etc.
    
    Returns:
        Tuple: Contains multiple arrays for features like day of week, day of month, etc.
    """
    
    start_date = EVdata.index.min()
    end_date = EVdata.index.max()
    
    date_indices = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    day_of_week_variable = np.array([day_of_week(date_indices) for item in ds])
    day_of_month_variable = np.array([day_of_month(date_indices) for item in ds])
    day_of_year_variable = np.array([day_of_year(date_indices) for item in ds])
    week_of_year_variable = np.array([week_of_year(date_indices) for item in ds])
    month_of_year_variable = np.array([month_of_year(date_indices) for item in ds])
    
    return day_of_week_variable, day_of_month_variable, day_of_year_variable, week_of_year_variable, month_of_year_variable

# Example usage:
# df_aggregated = aggregate_ev_charging(EVdf0, time_window='1H')
# multiple_ts = multiple_time_series(df_aggregated, target="Total Energy (1H)", fields=["Station Name", "Event Count", "Total Energy (1H)", "Total Charging Duration (1H)"])
# missing_data_handled = handle_missing_data(multiple_ts, freq="1H", target_column="Total Energy (1H)")
# features = add_multiple_features(missing_data_handled, freq="1H")

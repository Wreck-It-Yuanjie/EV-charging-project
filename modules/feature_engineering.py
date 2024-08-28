
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from pandas import Period
from gluonts.time_feature import (
    day_of_week,
    day_of_month,
    day_of_year,
    week_of_year,
    month_of_year, 
)
from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.field_names import FieldName
from sklearn.preprocessing import LabelEncoder
from gluonts.dataset.common import TrainDatasets, MetaData
from pathlib import Path

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
    return dfs_dict

def generate_counts_duration(dfs_dict):
    """
    Generate two features: counts and duration, and ds
    
    Parameters:
        dfs_dict
        
    Returns:
        counts and duration
    """
    ds = PandasDataset(dfs_dict, target="target")
    counts = np.array([dfs_dict[item].loc[:, "Event Count"].to_numpy() for item in dfs_dict])
    charge_duration = np.array([dfs_dict[item].loc[:, "Total Charging Duration (1D)"].to_numpy() for item in dfs_dict])
    return ds, counts, charge_duration

def train_test_split(ds, train_val_test_split):

    train_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[0])
    validation_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[1])
    prediction_length = math.floor(next(iter(ds))["target"].shape[0]*train_val_test_split[2])

    return train_length, validation_length, prediction_length

def get_min_max_date(testData):
    start_date = testData.index.min()
    end_date = testData.index.max()
    return start_date, end_date

def add_multiple_features(ds, start_date, end_date, freq):
    """
    Add temporal and additional features for EV charging data.
    
    Parameters:
        EVdata (pd.DataFrame): The EV charging data.
        freq (str): Frequency string like '1H', '1D', etc.
    
    Returns:
        Tuple: Contains multiple arrays for features like day of week, day of month, etc.
    """
    
    date_indices = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    day_of_week_variable = np.array([day_of_week(date_indices) for item in ds])
    day_of_month_variable = np.array([day_of_month(date_indices) for item in ds])
    day_of_year_variable = np.array([day_of_year(date_indices) for item in ds])
    week_of_year_variable = np.array([week_of_year(date_indices) for item in ds])
    month_of_year_variable = np.array([month_of_year(date_indices) for item in ds])
    
    return day_of_week_variable, day_of_month_variable, day_of_year_variable, week_of_year_variable, month_of_year_variable

def pad_time_series(ds, counts, charge_duration, start_date, freq):
    """
    Pads the time series data to the specified maximum length.
    
    Parameters:
        target_list (list): List of target arrays to be padded.
        max_length (int): The maximum length to pad the arrays to.
        
    Returns:
        np.array: Padded and stacked array of time series.
    """
    max_length = max(len(item['target']) for item in ds)
    padded_targets = [np.pad(arr['target'], (max_length - len(arr['target']), 0), mode='constant') for arr in ds]
    padded_targets_stack = np.vstack(padded_targets)

    max_length_counts = max(len(item) for item in counts)
    padded_counts = [np.pad(arr, (max_length_counts - len(arr), 0), mode='constant') for arr in counts]
    padded_counts_stack = np.vstack(padded_counts)

    max_length_duration = max(len(item) for item in charge_duration)
    padded_duration = [np.pad(arr, (max_length_duration - len(arr), 0), mode='constant') for arr in charge_duration]
    padded_duration_stack = np.vstack(padded_duration)

    start_date_period = pd.Period(start_date, freq=freq)
    start_stack = [start_date_period for _ in range(len(padded_targets_stack))]

    return padded_targets_stack, padded_counts_stack, padded_duration_stack, start_stack

def create_datasets(padded_targets_stack, start_stack, padded_counts_stack, 
                    padded_duration_stack,
                    day_of_week_variable, month_of_year_variable, 
                    train_length, prediction_length, freq):
    """
    Creates training, validation, and test datasets for time series forecasting.
    
    Parameters:
        padded_targets_stack (np.array): Array of target values.
        start_stack (list): List of start dates.
        padded_counts_stack (np.array): Array of padded event counts.
        padded_duration_stack (np.array): Array of padded charging durations.
        day_of_week_variable (np.array): Day of week features.
        month_of_year_variable (np.array): Month of year features.
        train_length (int): Length of training data.
        prediction_length (int): Length of prediction data.
        freq (str): Frequency of the data.
        
    Returns:
        Tuple: Training, validation, and test datasets.
    """
    train_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: [counts, duration, dayofweek, monthofyear],
            }
            for (target, start, counts, duration, dayofweek, monthofyear) in zip(
                padded_targets_stack[:, :train_length],
                start_stack,
                padded_counts_stack[:, :train_length],
                padded_duration_stack[:, :train_length],
                day_of_week_variable[:, :train_length],
                month_of_year_variable[:, :train_length]
            )
        ],
        freq=freq,
    )

    val_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: [counts, duration, dayofweek, monthofyear],
            }
            for (target, start, counts, duration, dayofweek, monthofyear) in zip(
                padded_targets_stack[:, :-prediction_length],
                start_stack,
                padded_counts_stack[:, :-prediction_length],
                padded_duration_stack[:, :-prediction_length],
                day_of_week_variable[:, :-prediction_length],
                month_of_year_variable[:, :-prediction_length]
            )
        ],
        freq=freq,
    )

    test_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: [counts, duration, dayofweek, monthofyear],
            }
            for (target, start, counts, duration, dayofweek, monthofyear) in zip(
                padded_targets_stack,
                start_stack,
                padded_counts_stack,
                padded_duration_stack,
                day_of_week_variable,
                month_of_year_variable
            )
        ],
        freq=freq,
    )


    return train_ds, val_ds, test_ds

def get_metadata(freq, prediction_length):
    metadata = MetaData(
        freq = freq,
        prediction_length = prediction_length
    )
    return metadata

def save_dataset(train_ds, val_ds, test_ds, metadata):
    """
    Saves the training, validation, and test datasets to JSON format.

    Parameters:
    - train_ds: The training dataset.
    - val_ds: The validation dataset.
    - test_ds: The test dataset.
    - metadata: The metadata for the datasets.
    - path_str: The base directory where datasets should be saved.
    """

    def save_to_json(dataset, file_path):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):  # Handle numpy integers
                return int(obj)
            if isinstance(obj, np.float64):  # Handle numpy floats
                return float(obj)
            if isinstance(obj, Period):  # Convert Period to string
                return str(obj)
            return obj
        
        data = list(dataset)
        with open(file_path, 'w') as f:
            for entry in data:
                serializable_entry = {k: convert_to_serializable(v) for k, v in entry.items()}
                json.dump(serializable_entry, f)
                f.write('\n')

    # Define the data directory relative to the current working directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    save_to_json(train_ds, os.path.join(data_dir, "train.json"))
    save_to_json(test_ds, os.path.join(data_dir, "test.json"))
    save_to_json(val_ds, os.path.join(data_dir, "val.json"))
    
    # Save metadata
    with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
        json.dump(metadata.__dict__, f)

def load_dataset():
    """
    Loads the training, validation, and test datasets from JSON format.

    Parameters:
    - path_str: The base directory where datasets are saved.

    Returns:
    - Tuple: Loaded training, validation, and test datasets.
    """

    # Define the data directory relative to the current working directory
    data_dir = "./data"

    def load_from_json(file_path):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        # Convert back any necessary fields here
        for entry in data:
            if "start" in entry:
                entry["start"] = Period(entry["start"])  # Convert start field back to Period
        return ListDataset(data, freq=metadata["freq"])

    # Load metadata
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    train_ds = load_from_json(os.path.join(data_dir, "train.json"))
    test_ds = load_from_json(os.path.join(data_dir, "test.json"))
    val_ds = load_from_json(os.path.join(data_dir, "val.json"))

    return train_ds, val_ds, test_ds, metadata

def visualize_train_val_test_data(train_ds, val_ds, test_ds):
    """
    Visualizes the training, validation, and test datasets.
    
    Parameters:
        train_ds (ListDataset): The training dataset.
        val_ds (ListDataset): The validation dataset.
        test_ds (ListDataset): The test dataset.
    """
    train_entry = next(iter(train_ds))
    train_series = to_pandas(train_entry)
    
    val_entry = next(iter(val_ds))
    val_series = to_pandas(val_entry)
    
    test_entry = next(iter(test_ds))
    test_series = to_pandas(test_entry)
    
    plt.figure(figsize=(10, 6))
    
    test_series.plot(color='grey')
    plt.axvline(train_series.index[-1], color="red")  # end of train dataset
    plt.axvline(val_series.index[-1], color="blue")  # end of train dataset
    plt.grid(which="both")
    plt.legend(["test series", "end of train series", "end of val series"], loc="upper left")
    plt.show()

    plt.savefig('./assets/train_val_test_data_vis.jpg')


def test_feature_engineering():
    freq = '1D'
    target = "Energy (kWh)"
    # fields = ["counts", "Energy (kWh)"]
    fields = ["Station Name", "Event Count", "Total Energy (kWh)", f"Total Charging Duration ({freq})"]

    testData, station_name_map = multiple_time_series(df, target, fields)

    dfs_dict = handle_missing_data(testData, freq, "target")

    ds, counts, charge_duration = generate_counts_duration(dfs_dict)

    train_length, validation_length, prediction_length = train_test_split(ds, train_val_test_split = [0.7, 0.2, 0.1])
    
    start_date, end_date = get_min_max_date(testData)
    
    day_of_week_variable, day_of_month_variable, day_of_year_variable, week_of_year_variable, month_of_year_variable = add_multiple_features(ds, start_date, end_date, freq)

    padded_targets_stack, padded_counts_stack, padded_duration_stack, start_stack = pad_time_series(ds, counts, charge_duration, start_date, freq)

    train_ds, val_ds, test_ds = create_datasets(padded_targets_stack, start_stack, padded_counts_stack, 
                    padded_duration_stack,
                    day_of_week_variable, month_of_year_variable, 
                    train_length, prediction_length, freq)

    metadata = get_metadata(freq, prediction_length)
    save_dataset(train_ds, val_ds, test_ds, metadata)

    train_ds, val_ds, test_ds, metadata = load_dataset()
    print(train_ds)
    
    visualize_train_val_test_data(train_ds, val_ds, test_ds)
    print("Passed all the tests!")

if __name__ == "__main__":

    # Path to the dataset in the data folder
    data_file_path = os.path.join('data', 'aggregated_D.csv') 

    # Load the aggregated dataset
    df = pd.read_csv(data_file_path, low_memory=False)

    df.set_index('Start Date', inplace=True)


    test_feature_engineering()

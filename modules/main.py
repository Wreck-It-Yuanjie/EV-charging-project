import os
import pandas as pd
import data_loading
import data_preprocessing
import feature_engineering

def main():
    target = "Energy (kWh)"
    freq = '1D'
    fields = ["Station Name", "Event Count", "Total Energy (kWh)", f"Total Charging Duration ({freq})"]

    ## Step 1: load and prepare data
    EVdf0 = data_loading.load_data("EVChargingData2010_2020.csv")
    EVdf0 = data_preprocessing.preprocess_data(EVdf0)

    # df_aggregated = feature_engineering.aggregate_ev_charging(EVdf0, time_window=freq)
    # df_aggregated.set_index('Start Date', inplace=True)
    print(df_aggregated)
    # multiple_ts, station_name_map = feature_engineering.multiple_time_series(df_aggregated, target, fields)
    # print(multiple_ts)
    # missing_data_handled = feature_engineering.handle_missing_data(multiple_ts, freq="1H", target_column="Total Energy (1H)")
    # features = feature_engineering.add_multiple_features(missing_data_handled, freq="1H")

    ## Step 2: feature engineering

    ## Step 3: define model architecture
    
    ## Step 4: train initial model


    ## Step 5: Fine tune the model

    ## Step 6: Retrain with best hyperparameters

    ## Step 7: evaluate and visualize results

# Conditional to allow script execution
if __name__ == "__main__":
    main()

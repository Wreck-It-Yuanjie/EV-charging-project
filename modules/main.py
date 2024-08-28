import os
import pandas as pd
import data_loading
import data_cleaning
import feature_engineering
import models
import model_training
import fine_tuning

def main():
    target = "Energy (kWh)"
    freq = '1D'
    fields = ["Station Name", "Event Count", "Total Energy (kWh)", f"Total Charging Duration ({freq})"]

    ## Step 1: load and prepare data
    data = data_loading.load_data("EVChargingData2010_2020.csv")
    data_clean = data_cleaning.clean_data(EVdf0)

    data_agg = aggregate_ev_charging(df, time_window='1D')

    ## Step 2: feature engineering
    freq = '1D'
    target = "Energy (kWh)"
    # fields = ["counts", "Energy (kWh)"]
    fields = ["Station Name", "Event Count", "Total Energy (kWh)", f"Total Charging Duration ({freq})"]

    testData, station_name_map = multiple_time_series(data_agg, target, fields)

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

    ## step 4: model training
    estimators = [simple_feed_forward_estimator, deepar, gp_estimator, transformer]
    models = [select_estimator(estimator, freq, metadata.prediction_length) for estimator in estimators]

    model_metrics = {}

    ## name of sample stations you'd like to plot
    station_samples = ["WEBSTER 1", "MPL 5", "RINCONADA LIB 1", "BRYANT 4"]
    samples_index = map_station_name_index(station_samples)

    predictor, agg_metrics, item_metrics, forecast_it, ts_it = multiple_models(train_ds, val_ds, test_ds, models, samples_index)
    
    weights = {
        'MSE': 0.4,
        'MASE': 0.3,
        'RMSE': 0.2,
        'MSIS': 0.1
    }
    best_model = find_best_model(weights, models, agg_metrics)

    save_best_model(best_model)
    
    ## Step 5: Fine tune the model
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(
        modelTuningObjective(
            val_ds, validation_length, freq
        ),
        n_trials=1,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    print(time.time() - start_time)

    ## Step 6: Retrain with best hyperparameters
    final_predictor = final_estimator.train(training_data = train_ds, validation_data = val_ds)

    ## Step 7: evaluate and visualize results
    final_predictor.serialize(Path("/tmp/"))

# Conditional to allow script execution
if __name__ == "__main__":
    main()

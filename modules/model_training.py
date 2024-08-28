from feature_engineering import load_dataset
from models import {
    create_simple_feed_forward_estimator, 
    create_mqrnn_estimator, 
    create_deepar_estimator, 
    create_deepvar_estimator, 
    create_lstnet_estimator,
    create_tft_estimator,
    create_transformer_estimator,
    create_deepfactor_estimator,
    create_gp_estimator
}
from pathlib import Path
from gluonts.model.predictor import Predictor


def select_estimator(estimator, freq, prediction_length):
    ## transformer_estimator: transformer
    ## tft_estimator: temporal fusion transformer
    ## lstnet_estimator: LSTnet
    ## deepVAR_estimator: DeepVAR
    ## deepAR_estimator: DeepAR
    ## simple_feed_forward_estimator: MLP
    ## deepFactor_estimator: deep factor
    ## gp_estimator: gaussian process
    ## mqcnn_estimator: MQ CNN
    ## rnn_estimator: RNN
    if(estimator == "simple_feed_forward_estimator"):
        return create_simple_feed_forward_estimator(prediction_length)
    if(estimator == "transformer_estimator"):
        return create_transformer_estimator(freq, prediction_length)
    if(estimator == "gp_estimator"):
        return create_gp_estimator(freq, prediction_length)
    if(estimator == "deepAR_estimator"):
        return create_deepar_estimator(freq, prediction_length)
    if(estimator == "deepFactor_estimator"):
        return create_deepfactor_estimator(freq, prediction_length)
    if(estimator == "tft_estimator"):
        return create_tft_estimator(freq, prediction_length)
    if(estimator == "deepVAR_estimator"):
        return create_deepvar_estimator(freq, prediction_length)

def train_and_predict(train_ds, val_ds, test_ds, estimator):
    """
    Trains a model using the provided estimator and makes predictions.
    
    Parameters:
        train_ds (ListDataset): The training dataset.
        val_ds (ListDataset): The validation dataset.
        test_ds (ListDataset): The test dataset.
        estimator: The GluonTS estimator to use for training.
        
    Returns:
        Predictor: The trained predictor.
    """
    predictor = estimator.train(training_data=train_ds, validation_data=val_ds)
    forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor)
    return predictor, list(forecast_it), list(ts_it)

def evaluate_model(forecast_it, ts_it):
    """
    Evaluates the model's performance.
    
    Parameters:
        forecast_it: Iterator of forecast results.
        ts_it: Iterator of ground truth time series.
        
    Returns:
        dict: Evaluation metrics.
    """
    evaluator = Evaluator(quantiles=[0.5, 0.05, 0.95])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it)
    return agg_metrics, item_metrics

def replace_negatives_with_zero(data):
    return [max(0, x) for x in data]

def multiple_models(train_ds, val_ds, test_ds, estimators, samples):
    predictors_all = []
    agg_metrics_all = []
    item_metrics_all = []
    forecast_it_all = []
    ts_it_all = []
    for estimator in estimators:
        
        ## model training
        predictor, forecast_it, ts_it = train_and_predict(train_ds, val_ds, test_ds, estimator, negative_control = True)

        ## loss curve
        loss_data_framework = get_loss_curve(history)
        export_loss_curve(loss_data_framework, estimator)
        plot_loss_curve(loss_data_framework, estimator)

        ## evluation metrics
        print_forecast_basic_info(forecast_it, ts_it)
        export_original_time_series_no_missing_values(ts_it)
        export_predicted_values(ts_it, forecast_it, estimator)
        plot_orginal_prediction(forecast_it, ts_it, estimator, samples)
        agg_metrics, item_metrics = get_evaluation_metrics(forecast_it, ts_it)
        export_eval_metrics(agg_metrics, item_metrics, estimator)

        predictors_all.append(predictor)
        agg_metrics_all.append(agg_metrics)
        item_metrics_all.append(item_metrics)
        forecast_it_all.append(forecast_it)
        ts_it_all.append(ts_it)

        return predictors_all, agg_metrics_all, item_metrics_all, forecast_it_all, ts_it_all
    
def get_loss_curve(history):
    loss_data_framework = pd.DataFrame({
        "training_loss": history.loss_history,
        "val_loss": history.validation_loss_history
    })
    loss_data_framework.head()
    return loss_data_framework

def export_loss_curve(loss_data_framework, estimator):
    # Ensure the results directory exists
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    loss_curve_name = os.path.join(results_dir,"loss_curve_model_{}.csv".format(get_model_name(estimator)))
    loss_data_framework.to_csv(loss_curve_name)

def plot_loss_curve(loss_data_framework, estimator):
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    # Clear the current figure
    plt.clf()

    loss_data_framework["epochs"] = range(1, len(loss_data_framework) + 1)

    plt.plot(loss_data_framework["epochs"], loss_data_framework["training_loss"], 'o-', label='Training Loss', color='blue', linestyle='--')
    plt.plot(loss_data_framework["epochs"], loss_data_framework["val_loss"], 's-', label='Validation Loss', color='red', linestyle='-.')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.savefig(os.path.join(results_dir, f'{get_model_name(estimator)}_train_val_loss.jpg'))

def print_forecast_basic_info(forecast_it, ts_it):
    forecast_entry = forecast_it[0]
    ts_entry = ts_it[0]

    print(f"Number of sample paths: {forecast_entry.num_samples}")
    print(f"Dimension of samples: {forecast_entry.samples.shape}")
    print(f"Start date of the forecast window: {forecast_entry.start_date}")
    print(f"Frequency of the time series: {forecast_entry.freq}")

def export_original_time_series_no_missing_values(ts_it):
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    ts_entry = ts_it[0]
    
    time_series_original = pd.DataFrame(ts_entry)
    time_series_original.columns = ['true_value']
    time_series_original.to_csv(os.path.join(results_dir,"time_series_original_no_missing_values.csv"))

def export_predicted_values(ts_it, forecast_it, estimator):
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    forecast_entry = forecast_it[0]
    ts_entry = ts_it[0]
    
    predicted_values = os.path.join(results_dir,"predicted_values_model_{}.csv".format(get_model_name(estimator)))
    predicted_values_100_series = pd.DataFrame(forecast_entry.samples.T)
    predicted_values_100_series.to_csv(predicted_values)

def plot_orginal_prediction(forecast_it, ts_it, estimator, samples):
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    # Clear the current figure
    plt.figure(figsize=[10, 6])
    
    for i in samples:
        plt.clf()
        forecast_entry = forecast_it[i]
        ts_entry = ts_it[i]

        plt.plot(ts_entry[-800:].to_timestamp())
        forecast_entry.plot(show_label=True)
        plt.legend()
        plt.savefig(os.path.join(results_dir,"testing_{}{}.png".format(get_model_name(estimator), i)))

def get_evaluation_metrics(forecast_it, ts_it):

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it)

    return agg_metrics, item_metrics

def export_eval_metrics(agg_metrics, item_metrics, estimator):
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    ## agg_metrics
    print("Evaluation metrics: {}".format(get_model_name(estimator)))
    agg_metrics_data_frame = pd.DataFrame([agg_metrics])

    agg_metrics_csv_name = os.path.join(results_dir,"agg_metrics_{}.csv".format(get_model_name(estimator)))

    item_metrics_csv_name = os.path.join(results_dir,"item_metrics_{}.csv".format(get_model_name(estimator)))

    agg_metrics_data_frame.to_csv(agg_metrics_csv_name)
    item_metrics.to_csv(item_metrics_csv_name)

def map_station_name_index(station_samples):
    station_name_index = station_name_map[station_name_map["Station Name"].isin(station_samples)]
    samples_index = station_name_index['item_id'].tolist()
    return samples_index

def find_best_model(weights, estimators, agg_metrics):
    """
    Finds the best model based on the combined score of weighted metrics.

    Parameters:
    - weights (dict): Dictionary of metric weights.
    - estimators (list): List of estimator objects.
    - agg_metrics (list of dicts): List where each entry corresponds to the metrics for an estimator.

    Returns:
    - str: The name of the best model.
    """
    agg_metrics_multiple_models = {}

    for index, value in enumerate(estimators):
        combined_score = sum(weights[metric] * agg_metrics[index][metric] for metric in weights)
        agg_metrics_multiple_models[get_model_name(value)] = combined_score

    best_model = min(agg_metrics_multiple_models, key=agg_metrics_multiple_models.get)
    return best_model

def save_best_model(final_predictor):
    # save the trained model in tmp/
    final_predictor.serialize(Path("/results/"))


def test_modeling_training():
    ## Available estmators: 

    # estimators = [simple_feed_forward_estimator, \
    #               deepAR_estimator, \
    #               deepVAR_estimator, \
    #               lstNetEstimator, \
    #               transformer_estimator, \
    #               mqcnn_estimator]
    train_ds, val_ds, test_ds, metadata = load_dataset()

    estimators = [simple_feed_forward_estimator]
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

if __name__ == "__main__":
    test_modeling_training()
    
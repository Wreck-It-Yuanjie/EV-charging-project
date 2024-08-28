from .feature_engineering import some_feature_engineering_function
from .models import create_simple_feed_forward_estimator, create_tft_estimator, create_transformer_estimator
from .time_window_aggregation import aggregate_data
from .model_training import train_and_predict
from .time_window_aggregation import aggregate_ev_charging, convert_duration_to_time_window

def run_analysis(aggregation_method, estimator_type, data):
    # Choose aggregation method
    if aggregation_method == '1H':
        aggregated_data = aggregate_ev_charging(df, time_window='1H')
    elif aggregation_method == '1D':
        aggregated_data = aggregate_ev_charging(df, time_window='1D')
    elif aggregation_method == '1W':
        aggregated_data = aggregate_ev_charging(df, time_window='1W')
    elif aggregation_method == '1M':
        aggregated_data = aggregate_ev_charging(df, time_window='1M')
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Choose estimator
    if estimator_type == 'simple_feed_forward':
        estimator = create_simple_feed_forward_estimator()
    elif estimator_type == 'tft':
        estimator = create_tft_estimator()
    elif estimator_type == 'transformer':
        estimator = create_transformer_estimator()
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

    # Perform feature engineering and training
    features = some_feature_engineering_function(aggregated_data)
    train_and_predict(features, estimator)
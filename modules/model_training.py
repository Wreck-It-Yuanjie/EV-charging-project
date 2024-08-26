

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

# Model Definitions
def get_simple_feed_forward_estimator(freq, prediction_length):
    return SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=prediction_length,
        context_length=100,
        freq=freq,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_deepar_estimator(freq, prediction_length):
    return DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_layers=2,
        dropout_rate=0.1,
        use_feat_dynamic_real=True,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_transformer_estimator(freq, prediction_length):
    return TransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        use_feat_dynamic_real=True,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_temporal_fusion_transformer_estimator(freq, prediction_length):
    return TemporalFusionTransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        hidden_dim=20,
        num_heads=8,
        dropout_rate=0.1,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_lstnet_estimator(freq, prediction_length):
    return LSTNetEstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_series=46,
        ar_window=100,
        channels=16,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_deepfactor_estimator(freq, prediction_length):
    return DeepFactorEstimator(
        freq=freq,
        prediction_length=prediction_length,
        use_feat_dynamic_real=True,
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_gaussian_process_estimator(freq, prediction_length):
    return GaussianProcessEstimator(
        freq=freq,
        prediction_length=prediction_length,
        cardinality=[1],
        trainer=Trainer(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            num_batches_per_epoch=50
        )
    )

def get_naive_predictor(freq, prediction_length):
    return Naive2Predictor(
        freq=freq,
        prediction_length=prediction_length,
        season_length=100
    )

# Example usage:
# target_padded = pad_time_series(target_list, max_length=3444)
# train_ds, val_ds, test_ds = create_datasets(target_padded, start_stack, counts_stack, duration_stack, day_of_week_variable, month_of_year_variable, train_length=2337, prediction_length=2307, freq='1D')
# visualize_train_val_test_data(train_ds, val_ds, test_ds)
# estimator = get_simple_feed_forward_estimator(freq="1D", prediction_length=2337)
# predictor, forecast_it, ts_it = train_and_predict(train_ds, val_ds, test_ds, estimator)
# agg_metrics, item_metrics = evaluate_model(forecast_it, ts_it)

import optuna
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx import Trainer
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.temporal_fusion_transformer import TemporalFusionTransformerEstimator
from gluonts.mx.model.lstnet import LSTNetEstimator
from gluonts.mx.model.deepfactor import DeepFactorEstimator
from gluonts.mx.model.gp_forecaster import GaussianProcessEstimator


# Utility function to convert dataset entry to DataFrame
def datanentry_to_dataframe(entry):
    df = pd.DataFrame(
        entry["target"],
        columns=[entry.get("item_id")],
        index=pd.period_range(
            start=entry["start"], periods=len(entry["target"]), freq=entry["start"].freq
        ),
    )
    return df

# Objective function class for model tuning
class ModelTuningObjective:
    def __init__(self, dataset, prediction_length, freq, metric_type="mean_wQuantileLoss"):
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.freq = freq
        self.metric_type = metric_type
        self.train, test_template = split(dataset, offset=-self.prediction_length)
        validation = test_template.generate_instances(
            prediction_length=self.prediction_length
        )
        self.validation_input = [entry[0] for entry in validation]
        self.validation_label = [
            datanentry_to_dataframe(entry[1]) for entry in validation
        ]

    # Function to suggest hyperparameters
    def get_params(self, trial) -> dict:
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "hidden_size": trial.suggest_int("hidden_size", 10, 50),
            "num_hidden_dimensions": trial.suggest_int("num_hidden_dimensions", 20, 80),
            "context_length": trial.suggest_int("context_length", 20, 140),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.001, 0.1),
            "num_batches_per_epoch": trial.suggest_int("num_batches_per_epoch", 20, 120),
            "dropout_rate": trial.suggest_uniform("dropout_rate", 0.0, 0.3),
            "num_layers_2": trial.suggest_int("num_layers", 2, 8),
            "num_heads": trial.suggest_int("num_heads", 2, 16),
        }

    # Function to train and evaluate the model based on suggested hyperparameters
    def __call__(self, trial):
        params = self.get_params(trial)
        estimator = self._get_estimator(params)
        predictor = estimator.train(self.train, cache_data=True)
        forecast_it = predictor.predict(self.validation_input)
        forecasts = list(forecast_it)
        
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(
            self.validation_label, forecasts, num_series=len(self.dataset)
        )
        
        return agg_metrics[self.metric_type]

    # Helper function to get the estimator based on model type
    def _get_estimator(self, params):
        # This is where the best_model logic can be updated to dynamically switch between models
        best_model = "transformer_estimator"
        
        if best_model == "simple_feed_forward_estimator":
            return SimpleFeedForwardEstimator(
                num_hidden_dimensions=[params["num_hidden_dimensions"]],
                prediction_length=self.prediction_length,
                batch_normalization=False,
                mean_scaling=True,
                context_length=100,
                freq=self.freq,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100
                ),
            )
        elif best_model == "gp_estimator":
            return GaussianProcessEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                cardinality=1,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=params["learning_rate"],
                    num_batches_per_epoch=params["num_batches_per_epoch"]
                ),
            )
        elif best_model == "transformer_estimator":
            return TransformerEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                use_feat_dynamic_real=True,
                context_length=100,
                dropout_rate=params["learning_rate"],
                num_heads=params["num_heads"],
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100
                ),
            )
        elif best_model == "deepFactor_estimator":
            return DeepFactorEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                use_feat_dynamic_real=True,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=20,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100
                ),
            )
        elif best_model == "tft_estimator":
            return TemporalFusionTransformerEstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                hidden_dim=20,
                num_heads=params["num_heads"],
                dropout_rate=0.1,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=params["learning_rate"],
                    num_batches_per_epoch=params["num_batches_per_epoch"]
                ),
            )
        elif best_model == "deepVAR_estimator":
            return DeepVAREstimator(
                freq=self.freq,
                prediction_length=self.prediction_length,
                num_layers=params["num_layers"],
                dropout_rate=0.1,
                use_feat_dynamic_real=True,
                target_dim=1,
                trainer=Trainer(
                    ctx="cpu",
                    epochs=1,
                    learning_rate=params["learning_rate"],
                    num_batches_per_epoch=params["num_batches_per_epoch"]
                ),
            )


# Function to run Optuna study
def run_optuna_study(val_ds, validation_length, freq, n_trials=1):
    best_model = "transformer_estimator"
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(
        ModelTuningObjective(
            val_ds, validation_length, freq
        ),
        n_trials=n_trials,
    )
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    print(time.time() - start_time)
    
    return trial

# Function to retrain the model with best hyperparameters
def retrain_with_best_hyperparameters(train_ds, val_ds, test_ds, trial, freq, validation_length):
    final_estimator = TransformerEstimator(
        freq=freq,
        prediction_length=validation_length,
        use_feat_dynamic_real=True,
        context_length=100,
        dropout_rate=trial.params["learning_rate"],
        num_heads=trial.params["num_heads"],
        trainer=Trainer(
            ctx="cpu",
            epochs=5,
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

    final_predictor = final_estimator.train(training_data=train_ds, validation_data=val_ds)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=final_predictor,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    plt.figure(figsize=(10, 6))
    plt.plot(tss[0][:800].to_timestamp())
    forecasts[0].plot(show_label=True)
    plt.legend()

    return final_predictor, forecasts, tss


# Example Usage:
# trial = run_optuna_study(val_ds, validation_length, freq, n_trials=10)
# final_predictor, forecasts, tss = retrain_with_best_hyperparameters(train_ds, val_ds, test_ds, trial, freq, validation_length)
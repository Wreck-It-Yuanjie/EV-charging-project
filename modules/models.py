from gluonts.model.predictor import Predictor
from gluonts.mx import SimpleFeedForwardEstimator, MQRNNEstimator, DeepAREstimator, DeepVAREstimator, LSTNetEstimator, TemporalFusionTransformerEstimator, TransformerEstimator, DeepFactorEstimator, GaussianProcessEstimator
from gluonts.mx.trainer import Trainer
from gluonts.ext.naive_2 import Naive2Predictor

# Simple Feed Forward Estimator
def create_simple_feed_forward_estimator(prediction_length):
    return SimpleFeedForwardEstimator(
        num_hidden_dimensions=[50],  # default: 50
        prediction_length=prediction_length,
        batch_normalization=False,  # default: false
        mean_scaling=True,  # default: true
        context_length=100,
        trainer=Trainer(
            ctx="cpu",
            epochs=2,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# MQRNN Estimator
def create_mqrnn_estimator(freq, prediction_length):
    return MQRNNEstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer=Trainer(
            ctx="cpu",
            epochs=10,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# DeepAR Estimator
def create_deepar_estimator(freq, prediction_length):
    return DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_layers=2,  # number of LSTM layers
        use_feat_dynamic_real=True,
        dropout_rate=0.1,
        trainer=Trainer(
            ctx="cpu",
            epochs=2,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# DeepVAR Estimator
def create_deepvar_estimator(freq, prediction_length):
    return DeepVAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        num_layers=2,  # number of LSTM layers
        dropout_rate=0.1,
        use_feat_dynamic_real=True,
        target_dim=1,
        trainer=Trainer(
            ctx="cpu",
            epochs=2,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# LSTNet Estimator
def create_lstnet_estimator(freq, prediction_length):
    return LSTNetEstimator(
        num_series=46,
        prediction_length=prediction_length,
        ar_window=10,
        channels=1,
        context_length=20,
        kernel_size=2,
        skip_size=1,
        dropout_rate=0.1,
        trainer=Trainer(
            ctx="cpu",
            epochs=2,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# Temporal Fusion Transformer Estimator
def create_tft_estimator(freq, prediction_length):
    return TemporalFusionTransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        hidden_dim=20,
        num_heads=8,
        dropout_rate=0.1,
        trainer=Trainer(
            ctx="cpu",
            epochs=2,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# Transformer Estimator
def create_transformer_estimator(freq, prediction_length):
    return TransformerEstimator(
        freq=freq,
        prediction_length=prediction_length,
        use_feat_dynamic_real=True,
        context_length=100,
        num_heads=8,
        dropout_rate=0.1,
        trainer=Trainer(
            ctx="cpu",
            epochs=20,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# DeepFactor Estimator
def create_deepfactor_estimator(freq, prediction_length):
    return DeepFactorEstimator(
        freq=freq,
        prediction_length=prediction_length,
        use_feat_dynamic_real=True,
        trainer=Trainer(
            ctx="cpu",
            epochs=20,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# Gaussian Process Estimator
def create_gp_estimator(freq, prediction_length):
    return GaussianProcessEstimator(
        freq=freq,
        prediction_length=prediction_length,
        cardinality=1,
        trainer=Trainer(
            ctx="cpu",
            epochs=20,
            callbacks=[TrainingHistory()],
            learning_rate=1e-3,
            num_batches_per_epoch=100
        ),
    )

# Naive Predictor
def create_naive_predictor(prediction_length):
    return Naive2Predictor(
        prediction_length=prediction_length,
        season_length=100
    )

def test_models():
    pass


if __name__ == "__main__":
    test_models()
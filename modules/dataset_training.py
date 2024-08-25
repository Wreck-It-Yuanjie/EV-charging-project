
import numpy as np
import matplotlib.pyplot as plt
from gluonts.model.predictor import Predictor
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.time_feature.field_names import FieldName
from gluonts.mx import Trainer
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deepar import DeepVAR
from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.temporal_fusion_transformer import TemporalFusionTransformerEstimator
from gluonts.mx.model.lstnet import LSTNetEstimator
from gluonts.mx.model.deepfactor import DeepFactorEstimator
from gluonts.mx.model.gp_forecaster import GaussianProcessEstimator
from gluonts.ext.naive_2 import Naive2Predictor

def pad_time_series(target_list, max_length):
    """
    Pads the time series data to the specified maximum length.
    
    Parameters:
        target_list (list): List of target arrays to be padded.
        max_length (int): The maximum length to pad the arrays to.
        
    Returns:
        np.array: Padded and stacked array of time series.
    """
    padded_targets = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in target_list]
    return np.vstack(padded_targets)

def create_datasets(target, start_stack, padded_counts_stack, padded_duration_stack,
                    day_of_week_variable, month_of_year_variable, train_length, prediction_length, freq):
    """
    Creates training, validation, and test datasets for time series forecasting.
    
    Parameters:
        target (np.array): Array of target values.
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
        [{
            FieldName.TARGET: target[:, :train_length],
            FieldName.START: start_stack,
            FieldName.FEAT_DYNAMIC_REAL: [padded_counts_stack[:, :train_length], padded_duration_stack[:, :train_length],
                                          day_of_week_variable[:, :train_length], month_of_year_variable[:, :train_length]],
        }],
        freq=freq
    )

    val_ds = ListDataset(
        [{
            FieldName.TARGET: target[:, train_length:],
            FieldName.START: start_stack,
            FieldName.FEAT_DYNAMIC_REAL: [padded_counts_stack[:, -prediction_length:], padded_duration_stack[:, -prediction_length:],
                                          day_of_week_variable[:, -prediction_length:], month_of_year_variable[:, -prediction_length:]],
        }],
        freq=freq
    )

    test_ds = ListDataset(
        [{
            FieldName.TARGET: target[:, train_length:],
            FieldName.START: start_stack,
            FieldName.FEAT_DYNAMIC_REAL: [padded_counts_stack[:, -prediction_length:], padded_duration_stack[:, -prediction_length:],
                                          day_of_week_variable[:, -prediction_length:], month_of_year_variable[:, -prediction_length:]],
        }],
        freq=freq
    )

    return train_ds, val_ds, test_ds


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

    plt.savefig('train_val_test_data_vis.jpg')

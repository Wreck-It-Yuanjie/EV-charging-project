# Easy EV charging demand prediction

## Overview

This package provides tools and methods for EV charging demand prediction, including time series data aggregation, feature engineering, and model training using various estimators. Users can easily select different aggregation methods and estimators to build and fine-tune their models.

## Features

- **Data Cleaning**: Clean and preprocess raw data for time series analysis.
- **Time Window Aggregation**: Aggregate data over specified time windows for better analysis.
- **Feature Engineering**: Generate features from raw data to improve model performance.
- **Model Training**: Train models using different estimators (e.g., Simple Feed Forward, TFT, Transformer).
- **Fine Tuning**: Fine-tune model hyperparameters to achieve optimal performance.
- **Utility Functions**: Various utility functions to support the data science workflow.

## Installation

To install the package, you can use the following command:

\`\`\`bash
pip install git+https://github.com/yourusername/yourpackagename.git
\`\`\`

Or if you have a \`setup.py\`:

\`\`\`bash
python setup.py install
\`\`\`

## Usage

### 1. Run Analysis

To run the analysis with a chosen aggregation method and estimator, use the following:

\`\`\`python
from your_package_name import run_analysis

data = load_your_data_function() # Replace with actual data loading function
best_model = run_analysis(aggregation_method='method1', estimator_type='simple_feed_forward', data=data)
print(f"The best model is: {best_model}")
\`\`\`

### 2. Aggregation Methods

- **Method 1**: Description of method 1.
- **Method 2**: Description of method 2.

### 3. Estimators

- **Simple Feed Forward Estimator**: A basic neural network model for time series forecasting.
- **TFT Estimator**: Temporal Fusion Transformer for handling time series with varying temporal patterns.
- **Transformer Estimator**: Transformer model for advanced sequence modeling.

### 4. Fine Tuning

You can fine-tune the estimators using the \`fine_tuning.py\` module:

\`\`\`python
from your_package_name.fine_tuning import tune_hyperparameters

best_params = tune_hyperparameters(estimator)
print(f"Best hyperparameters: {best_params}")
\`\`\`

## Directory Structure

\`\`\`plaintext
your_package_name/
│
├── your_package_name/
│ ├── **init**.py # Package initialization
│ ├── feature_engineering.py # Feature engineering module
│ ├── models.py # Model creation functions
│ ├── model_training.py # Model training and evaluation
│ ├── data_loading.py # Data loading utilities
│ ├── data_cleaning.py # Data cleaning functions
│ ├── time_window_aggregation.py # Time window aggregation methods
│ ├── fine_tuning.py # Fine-tuning of model hyperparameters
│ ├── utilities.py # Utility functions
│
├── tests/
│ ├── **init**.py # Tests initialization
│ ├── test_feature_engineering.py # Unit tests for feature engineering
│ ├── test_models.py # Unit tests for models
│
├── assets/
│ ├── any_data_or_assets_files # Data or assets used by the package
│
├── setup.py # Setup script for installing the package
├── README.md # Package overview and usage
└── MANIFEST.in # Package manifest file (optional)
\`\`\`

## Contributing

If you wish to contribute to the package, please fork the repository and submit a pull request. Make sure to write tests for your code.

## License

This package is licensed under the MIT License. See \`LICENSE\` for more details.

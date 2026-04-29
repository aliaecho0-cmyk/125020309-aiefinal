# Time Series Forecast

A Streamlit web application for time series forecasting using ARIMA(1,1,1) and Exponential Smoothing (ETS) with optimized ensemble weights.

## Features

- **ARIMA + ETS Ensemble**: Combines ARIMA(1,1,1) and Exponential Smoothing models
- **Auto-Optimized Weights**: Automatically finds the best weight for the ensemble using validation RMSE
- **One-Step-Ahead Forecast**: Predict the next value in your time series
- **Rolling Backtest**: Evaluate model performance on a hold-out test set
- **Excel Export**: Download forecast results as Excel files

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Input File Format

Upload an Excel file (.xlsx) containing:
- A column named `y` with your time series values
- (Optional) A column named `date` for date labels

## How It Works

1. **Weight Optimization**: The app uses 80% of your training data to find the optimal ARIMA weight by testing values from 0.0 to 1.0
2. **One-Step Forecast**: Uses all available data to predict the next value using the optimized ensemble
3. **Rolling Backtest**: Performs rolling one-step forecasts on the test set (last 20% of data)

## Example

Sample data files are available for testing the application.

## License

MIT License

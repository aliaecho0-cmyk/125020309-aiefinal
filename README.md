# Time Series Forecast WebApp

A Streamlit web application for time series forecasting using ARIMA(1,1,1) and Exponential Smoothing (ETS) with auto-optimized ensemble weights.

## Live Demo

[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## Features

- **Auto-Optimized Ensemble**: Automatically finds the optimal weight combination between ARIMA(1,1,1) and ETS using validation RMSE
- **One-Step-Ahead Forecast**: Predicts the next value in your time series immediately after upload
- **Rolling Backtest**: Performs walk-forward validation with 80% training / 20% test split
- **Excel Export**: Downloads backtest forecasts as an Excel file

## Model Details

| Model | Specification |
|-------|---------------|
| ARIMA | order=(1,1,1) |
| ETS | ExponentialSmoothing, trend=None, seasonal=None |
| Ensemble | Final = w × ARIMA + (1-w) × ETS, where w is auto-optimized |

Weight optimization is performed on the training set only (no data leakage). The algorithm tests w = 0.0 to 1.0 in 0.1 increments and selects the weight that minimizes RMSE on a validation subset.

## Input File Format

Upload an Excel file (.xlsx) with the following structure:

| Column | Required | Description |
|--------|----------|-------------|
| y | Yes | Time series values (numeric, oldest to newest, no missing values) |
| date | No | Date values for display and output alignment |

### Example

| date | y |
|------|-----|
| 2020-01-01 | 1.23 |
| 2020-01-02 | 1.25 |
| ... | ... |
| 2021-12-31 | 3.87 |

## Output

### Part 1: One-Step Forecast
Displays the predicted next value `y_{N+1}` on the web interface.

### Part 2: Rolling Backtest Export
Downloads an Excel file with:

| Column | Description |
|--------|-------------|
| date | Test period dates (if input contains date column) |
| y | Forecast values for each test point (0.2 × N rows) |

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

bash
pip install -r requirements.txt
Run the app:

bash
streamlit run app.py
Open your browser to http://localhost:8501

Requirements
Python 3.10 or higher

Dependencies listed in requirements.txt

How It Works
Upload: User uploads an Excel file with column y

Validation: Checks for missing values and correct column names

Weight Optimization: Uses the first 80% of data to find optimal ARIMA weight

One-Step Forecast: Predicts next value using full dataset with optimized weight

Rolling Backtest: Iteratively forecasts each test point using only past data

Export: Provides downloadable Excel file with test period forecasts

Project Structure
text
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # This file

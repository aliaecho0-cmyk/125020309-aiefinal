import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io

st.set_page_config(page_title="Time Series Forecast", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Fira+Code:wght@400&display=swap');
    
    .stApp { background-color: #0A0A0A; color: #FFFFFF; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .stMainBlockContainer { max-width: 900px; padding: 2rem; }
    
    .header-line { height: 1px; background-color: #00FFAA; margin: 0.5rem 0 2rem 0; }
    
    .page-title { font-size: 1.5rem; font-weight: 400; letter-spacing: -0.02em; color: #FFFFFF; margin-bottom: 0.25rem; }
    .page-subtitle { font-size: 0.85rem; color: #888888; margin-bottom: 0; }
    
    .upload-section { border: 1px dashed #2A2A2A; border-radius: 4px; padding: 2rem; margin: 1.5rem 0; text-align: center; transition: border-color 0.2s; }
    .upload-section:hover { border-color: #00FFAA; }
    .upload-label { color: #888888; font-size: 0.9rem; }
    
    .section-title { font-size: 0.9rem; font-weight: 500; color: #FFFFFF; letter-spacing: -0.01em; margin: 2rem 0 1rem 0; }
    .section-divider { height: 1px; background-color: #2A2A2A; margin: 2rem 0; }
    
    .forecast-card { background-color: #111111; border-left: 3px solid #00FFAA; padding: 1.5rem; border-radius: 2px; margin: 1rem 0; }
    .forecast-label { font-size: 0.8rem; color: #888888; margin-bottom: 0.5rem; }
    .forecast-value { font-family: 'Fira Code', monospace; font-size: 2rem; color: #00FFAA; }
    
    .download-btn { background-color: transparent; border: 1px solid #00FFAA; color: #00FFAA; padding: 0.75rem 1.5rem; border-radius: 2px; font-size: 0.85rem; cursor: pointer; transition: background-color 0.2s; }
    .download-btn:hover { background-color: #111111; }
    
    .error-text { color: #FF4444; font-size: 0.8rem; margin-top: 1rem; }
    
    .stDataFrame { background-color: #111111; border: 1px solid #2A2A2A; border-radius: 2px; }
    .stDataFrame td { color: #FFFFFF; font-family: 'Fira Code', monospace; font-size: 0.85rem; }
    .stDataFrame th { color: #888888; font-weight: 400; }
    
    .stMetric { background-color: #111111; padding: 1rem; border-radius: 2px; }
    div[data-testid="stMetricValue"] { font-family: 'Fira Code', monospace; color: #00FFAA; font-size: 1.5rem; }
    div[data-testid="stMetricLabel"] { color: #888888; font-size: 0.75rem; }
    
    .stFileUploader > div { background-color: transparent; }
    .stFileUploader label { color: #888888; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">Time Series Forecast</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">ARIMA(1,1,1) + ETS | Weighted Average</p>', unsafe_allow_html=True)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

def fit_arima(data):
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    model = ARIMA(data, order=(1, 1, 1))
    return model.fit()

def fit_ets(data):
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    model = ExponentialSmoothing(data, trend=None, seasonal=None, initialization_method='estimated')
    return model.fit()

def one_step_forecast(data):
    arima_model = fit_arima(data)
    arima_pred = float(arima_model.forecast(steps=1).values[0])
    
    ets_model = fit_ets(data)
    ets_pred = float(ets_model.forecast(steps=1).values[0])
    
    final_pred = (arima_pred + ets_pred) / 2
    return final_pred, arima_pred, ets_pred

def rolling_backtest(data, dates=None):
    n = len(data)
    train_size = int(n * 0.8)
    
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()
    
    predictions = []
    working_data = train_data.copy()
    
    for i in range(len(test_data)):
        arima_model = fit_arima(working_data)
        arima_pred = float(arima_model.forecast(steps=1).values[0])
        
        ets_model = fit_ets(working_data)
        ets_pred = float(ets_model.forecast(steps=1).values[0])
        
        final_pred = (arima_pred + ets_pred) / 2
        predictions.append(final_pred)
        
        working_data = np.append(working_data, final_pred)
    
    return predictions, train_size, test_data

st.markdown('<p class="upload-label">Upload Excel File</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload", type=["xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        if 'y' not in df.columns:
            st.markdown('<p class="error-text">Error: File must contain a column named "y"</p>', unsafe_allow_html=True)
        else:
            y_data = df['y'].values
            has_date = 'date' in df.columns
            
            if pd.isna(y_data).any():
                st.markdown('<p class="error-text">Column "y" contains missing values. Please provide complete data.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="section-title">One-Step-Ahead Forecast</p>', unsafe_allow_html=True)
                forecast_val, arima_val, ets_val = one_step_forecast(y_data)
                
                st.markdown(f'''
                <div class="forecast-card">
                    <p class="forecast-label">FORECAST VALUE</p>
                    <p class="forecast-value">{forecast_val:.4f}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ARIMA(1,1,1)", f"{arima_val:.4f}")
                with col2:
                    st.metric("ETS", f"{ets_val:.4f}")
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                st.markdown('<p class="section-title">Rolling Backtest Results</p>', unsafe_allow_html=True)
                
                predictions, train_size, test_data = rolling_backtest(y_data)
                
                output_df = pd.DataFrame({'y': predictions})
                
                if has_date:
                    test_dates = df['date'].values[train_size:]
                    output_df.insert(0, 'date', test_dates)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='Forecasts')
                buffer.seek(0)
                
                st.download_button(
                    label="Download Forecasts (Excel)",
                    data=buffer,
                    file_name="forecasts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary"
                )
                
                st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                st.dataframe(output_df, width=600)
                
    except Exception as e:
        st.markdown(f'<p class="error-text">Error processing file: {str(e)}</p>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Forecast", layout="centered")

# 直接用 st.title 和 st.caption 代替自定义 HTML，确保显示完整
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 移除 block 容器的额外 padding */
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* 标题样式 - 使用更可靠的选择器 */
    .custom-title {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1A1A2E !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.02em !important;
        text-align: left !important;
        line-height: 1.2 !important;
    }
    
    .custom-subtitle {
        font-size: 0.9rem !important;
        color: #666666 !important;
        margin-bottom: 0 !important;
        text-align: left !important;
    }
    
    .header-line {
        height: 3px;
        width: 60px;
        background: linear-gradient(90deg, #3B82F6, #10B981);
        margin: 1rem 0 2rem 0;
        border-radius: 3px;
    }
    
    .stFileUploader {
        margin: 1rem 0;
    }
    
    .stFileUploader > div:first-child {
        border: 1px dashed #D1D5DB;
        border-radius: 12px;
        background-color: #FAFAFA;
        transition: all 0.2s;
    }
    
    .stFileUploader > div:first-child:hover {
        border-color: #3B82F6;
        background-color: #F5F9FF;
    }
    
    .upload-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.01em;
    }
    
    .section-divider {
        height: 1px;
        background-color: #E5E7EB;
        margin: 2rem 0;
    }
    
    .forecast-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .forecast-label {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        color: rgba(255,255,255,0.7);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .forecast-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1F2937;
        font-family: monospace;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: white !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
    }
    
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame th {
        background-color: #F9FAFB;
        font-weight: 600;
        color: #374151;
    }
    
    .stDataFrame td {
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    .error-text {
        font-size: 0.85rem;
        color: #EF4444;
        background-color: #FEF2F2;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-text {
        font-size: 0.8rem;
        color: #10B981;
        background-color: #ECFDF5;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
    }
    
    .weight-card {
        background-color: #F0F9FF;
        border-left: 3px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .weight-label {
        font-size: 0.75rem;
        color: #3B82F6;
        font-weight: 500;
    }
    
    .weight-value {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A8A;
        font-family: monospace;
    }
    
    /* 修复 st.caption 样式 */
    .stCaption {
        color: #6B7280 !important;
    }
</style>
""", unsafe_allow_html=True)

# 使用 st.markdown 但保证标题完整显示
st.markdown('<div class="custom-title">Time Series Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">ARIMA(1,1,1) + Exponential Smoothing | Auto-Optimized Weights</div>', unsafe_allow_html=True)
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

def find_best_weight(train_data):
    """在训练集内部找最佳权重 (ARIMA权重)"""
    n = len(train_data)
    inner_train_size = int(n * 0.8)
    
    inner_train = train_data[:inner_train_size].copy()
    inner_val = train_data[inner_train_size:].copy()
    
    best_weight = 0.5
    best_rmse = float('inf')
    results = []
    
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        preds = []
        history = inner_train.copy()
        
        for actual in inner_val:
            arima_model = fit_arima(history)
            arima_pred = float(arima_model.forecast(steps=1).values[0])
            
            ets_model = fit_ets(history)
            ets_pred = float(ets_model.forecast(steps=1).values[0])
            
            pred = w * arima_pred + (1 - w) * ets_pred
            preds.append(pred)
            history = np.append(history, actual)
        
        rmse = np.sqrt(np.mean((np.array(preds) - inner_val) ** 2))
        results.append((w, rmse))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w
    
    return best_weight, best_rmse, results

def one_step_forecast(data, weight):
    arima_model = fit_arima(data)
    arima_pred = float(arima_model.forecast(steps=1).values[0])
    
    ets_model = fit_ets(data)
    ets_pred = float(ets_model.forecast(steps=1).values[0])
    
    final_pred = weight * arima_pred + (1 - weight) * ets_pred
    return final_pred, arima_pred, ets_pred

def rolling_backtest(data, weight):
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
        
        final_pred = weight * arima_pred + (1 - weight) * ets_pred
        predictions.append(final_pred)
        
        working_data = np.append(working_data, final_pred)
    
    return predictions, train_size, test_data

st.markdown('<p class="upload-label">Upload Excel file (.xlsx)</p>', unsafe_allow_html=True)
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
                # Step 1: Find best weight using training set only
                st.markdown('<p class="section-title">Weight Optimization</p>', unsafe_allow_html=True)
                
                with st.spinner('Optimizing ensemble weights...'):
                    n = len(y_data)
                    train_for_opt = y_data[:int(n * 0.8)].copy()
                    best_weight, best_rmse, results = find_best_weight(train_for_opt)
                
                # 显示结果用 st.metric 更可靠
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimized ARIMA Weight", f"{best_weight:.2f}")
                with col2:
                    st.metric("Validation RMSE", f"{best_rmse:.6f}")
                
                with st.expander("View all weights comparison"):
                    results_df = pd.DataFrame(results, columns=['ARIMA Weight', 'RMSE'])
                    st.dataframe(results_df, use_container_width=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Part 1: One-step forecast with optimized weight
                st.markdown('<p class="section-title">One-Step-Ahead Forecast</p>', unsafe_allow_html=True)
                forecast_val, arima_val, ets_val = one_step_forecast(y_data, best_weight)
                
                st.markdown(f'''
                <div class="forecast-card">
                    <div class="forecast-label">PREDICTED NEXT VALUE (y_{len(y_data)+1})</div>
                    <div class="forecast-value">{forecast_val:.6f}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">ARIMA(1,1,1)</div>
                        <div class="metric-value">{arima_val:.6f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">ETS</div>
                        <div class="metric-value">{ets_val:.6f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.caption(f"Ensemble: {best_weight:.2f} × ARIMA + {1-best_weight:.2f} × ETS")
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Part 2: Rolling backtest with optimized weight
                st.markdown('<p class="section-title">Rolling Backtest Results</p>', unsafe_allow_html=True)
                
                with st.spinner('Computing rolling forecasts...'):
                    predictions, train_size, test_data = rolling_backtest(y_data, best_weight)
                
                n_test = len(test_data)
                st.info(f"Test size: {n_test} observations ({int(0.2*len(y_data))} total)")
                
                # Prepare output dataframe
                output_df = pd.DataFrame({'y': predictions})
                
                if has_date:
                    test_dates = df['date'].values[train_size:]
                    output_df.insert(0, 'date', test_dates)
                
                # Export to Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name='Forecasts')
                buffer.seek(0)
                
                st.download_button(
                    label="Download Forecasts (Excel)",
                    data=buffer,
                    file_name="forecasts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                st.caption("Preview (first 5 rows)")
                st.dataframe(output_df.head(5), use_container_width=True)
                
    except Exception as e:
        st.markdown(f'<p class="error-text">Error: {str(e)}</p>', unsafe_allow_html=True)
else:
    st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)
    st.info("Upload an Excel file with a column named 'y'")

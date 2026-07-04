import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="GreenGrid Optimizer", layout="wide")

st.title("⚡ GreenGrid Optimizer")
st.markdown("Renewable Energy Generation Forecasting for the German Electricity Grid")

# Sidebar
st.sidebar.header("Configuration")
region = st.sidebar.selectbox("Region", ["DE"])
hours_ahead = st.sidebar.slider("Forecast Horizon (hours)", min_value=1, max_value=72, value=24)

# Fetch Metrics
st.sidebar.markdown("---")
st.sidebar.subheader("Production Model Metrics")
try:
    metrics_res = requests.get(f"{API_BASE}/metrics").json()
    model_info_res = requests.get(f"{API_BASE}/model/info").json()
    
    st.sidebar.write(f"**Model:** {model_info_res['model_type']}")
    
    # Assuming XGBoost is the best one based on our pipeline
    best_model_metrics = None
    for k, v in metrics_res.items():
        if "XGBoost" in k:
            best_model_metrics = v
            break
            
    if best_model_metrics:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("MAE (MW)", f"{best_model_metrics.get('MAE', 0):.0f}")
        col2.metric("R²", f"{best_model_metrics.get('R2', 0):.3f}")
        col1.metric("RMSE", f"{best_model_metrics.get('RMSE', 0):.0f}")
        col2.metric("MAPE %", f"{best_model_metrics.get('MAPE', 0):.1f}")
except Exception as e:
    st.sidebar.warning("Could not fetch metrics from API. Ensure backend is running.")

# Main Dashboard
st.header("Forecast vs Actuals")

try:
    # Fetch historical
    hist_res = requests.get(f"{API_BASE}/historical?region={region}&limit=72").json()
    df_hist = pd.DataFrame.from_dict(hist_res, orient="index")
    df_hist.index = pd.to_datetime(df_hist.index)
    
    # Fetch forecast
    forecast_res = requests.get(f"{API_BASE}/forecast?region={region}&hours_ahead={hours_ahead}").json()
    df_forecast = pd.DataFrame(forecast_res)
    df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
    df_forecast.set_index('timestamp', inplace=True)
    
    # Combine for plot
    # Just wind onshore for demo
    fig = px.line(title="Wind Onshore Generation (MW)")
    fig.add_scatter(x=df_hist.index, y=df_hist['wind_onshore'], mode='lines', name='Actual')
    fig.add_scatter(x=df_forecast.index, y=df_forecast['wind_onshore_mw'], mode='lines', name='Forecast', line=dict(dash='dash'))
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Failed to fetch data: {e}")

st.markdown("---")
st.header("Forecast Explainability (SHAP)")

try:
    explain_res = requests.get(f"{API_BASE}/forecast/explain").json()
    st.write(f"**Base Value:** {explain_res['base_value']:.2f} MW")
    
    contribs = explain_res['feature_contributions']
    df_shap = pd.DataFrame(list(contribs.items()), columns=['Feature', 'Contribution (MW)'])
    df_shap = df_shap.sort_values(by='Contribution (MW)', key=abs, ascending=True)
    
    fig2 = px.bar(df_shap, x='Contribution (MW)', y='Feature', orientation='h', 
                  title="Top 10 Feature Contributions for Latest Forecast",
                  color='Contribution (MW)', color_continuous_scale=px.colors.diverging.RdBu)
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning("Could not fetch SHAP explanations.")

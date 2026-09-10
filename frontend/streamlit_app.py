import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# In docker-compose, the frontend container reaches the API by service name (api:8000),
# not localhost. Defaults to localhost for running the frontend directly on the host.
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

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

    for target, model_type in model_info_res['models'].items():
        st.sidebar.write(f"**{target}:** {model_type}")

    explain_target = st.sidebar.selectbox("Explain target", list(model_info_res['models'].keys()))

    # Assuming XGBoost is the best one based on our pipeline
    best_model_metrics = None
    for k, v in metrics_res.items():
        if k.startswith(f"XGBoost_{explain_target}"):
            best_model_metrics = v
            break

    if best_model_metrics:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("MAE (MW)", f"{best_model_metrics.get('MAE', 0):.0f}")
        col2.metric("R²", f"{best_model_metrics.get('R2', 0):.3f}")
        col1.metric("RMSE", f"{best_model_metrics.get('RMSE', 0):.0f}")
        col2.metric("MAPE %", f"{best_model_metrics.get('MAPE', 0):.1f}")
except Exception as e:
    explain_target = "wind_onshore"
    st.sidebar.warning("Could not fetch metrics from API. Ensure backend is running.")

# Main Dashboard
st.header("Forecast vs Actuals")

hist_res = {}
forecast_res = []
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
    
    # One chart per modeled generation source (wind_onshore, wind_offshore, solar)
    sources = [
        ("wind_onshore", "wind_onshore_mw", "Wind Onshore Generation (MW)"),
        ("wind_offshore", "wind_offshore_mw", "Wind Offshore Generation (MW)"),
        ("solar", "solar_mw", "Solar Generation (MW)"),
    ]
    cols = st.columns(3)
    for (hist_col, forecast_col, title), col in zip(sources, cols):
        with col:
            fig = px.line(title=title)
            fig.add_scatter(x=df_hist.index, y=df_hist[hist_col], mode='lines', name='Actual')
            fig.add_scatter(x=df_forecast.index, y=df_forecast[forecast_col], mode='lines', name='Forecast', line=dict(dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Failed to fetch data: {e}")

st.markdown("---")
st.header(f"Forecast Explainability (SHAP) — {explain_target}")

explain_res = {}
try:
    explain_res = requests.get(f"{API_BASE}/forecast/explain", params={"target": explain_target}).json()
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

st.markdown("---")
st.header("💬 Ask about the forecast")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input('Ask e.g. "Why is wind onshore low tomorrow morning?"')
if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    dashboard_context = {
        "region": region,
        "forecast_horizon_hours": hours_ahead,
        "explain_target": explain_target,
        "forecast": forecast_res,
        "recent_actuals": hist_res,
        "shap_explanation": explain_res,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_res = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "message": user_question,
                        "history": st.session_state.chat_history[:-1],
                        "context": dashboard_context,
                    },
                )
                chat_res.raise_for_status()
                reply = chat_res.json()["reply"]
            except requests.exceptions.HTTPError:
                detail = chat_res.json().get("detail", "Chat request failed.")
                reply = f"⚠️ {detail}"
            except Exception as e:
                reply = f"⚠️ Could not reach the chat endpoint: {e}"
        st.write(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

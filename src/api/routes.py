from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import json
import sqlite3
import shap
from src.api.schemas import ForecastResponse, ModelInfoResponse, ExplanationResponse
from src.utils.config import settings
from src.features.feature_engineering import compute_predictor_features, GENERATION_COLUMNS
from src.ingestion.weather_client import WeatherClient

# How much history to seed the recursive forecast with. Needs to comfortably cover
# the longest lag/rolling window (168h) so those features aren't computed from a
# truncated, low-variance window.
FORECAST_HISTORY_HOURS = 24 * 21
MAX_FORECAST_HOURS_AHEAD = 72

router = APIRouter()

# In-memory loaded model (lazy load)
_model = None

def get_model():
    global _model
    if _model is None:
        model_path = settings.PRODUCTION_MODEL_DIR / "model.pkl"
        if model_path.exists():
            _model = joblib.load(model_path)
    return _model

@router.get("/")
def read_root():
    return {"name": "GreenGrid Optimizer API", "version": "1.0"}

@router.get("/health")
def healthcheck():
    return {"status": "healthy"}

@router.get("/forecast", response_model=List[ForecastResponse])
def get_forecast(region: str = "DE", hours_ahead: int = 24):
    """
    Recursive multi-step forecast: the model predicts wind_onshore at t+1 from
    features known at t (current weather + lags/rolling stats). To forecast further
    than 1h out, each step's prediction is fed back in as the "current" generation
    value for the next step, and real Open-Meteo *forecast* weather (not historical)
    is used for the weather features of each future hour.

    Known v1 limitation: only wind_onshore is modeled. wind_offshore and solar are
    held at their last observed value for the weather-feature lags/rolling stats
    (they are not forecast themselves).
    """
    if not 1 <= hours_ahead <= MAX_FORECAST_HOURS_AHEAD:
        raise HTTPException(status_code=400, detail=f"hours_ahead must be between 1 and {MAX_FORECAST_HOURS_AHEAD}")

    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="Feature dataset not available")

    df = pd.read_parquet(features_path)
    raw_cols = [c for c in GENERATION_COLUMNS + WeatherClient.HOURLY_VARIABLES + ["is_generation_gap", "has_long_gap"] if c in df.columns]
    history = df[raw_cols].tail(FORECAST_HISTORY_HOURS).copy()
    history.index = pd.to_datetime(history.index)
    if history.empty:
        raise HTTPException(status_code=503, detail="Not enough historical data to build a forecast")

    last_known_ts = history.index.max()

    # Weather forecast is only needed for the *intermediate* future hours (t+1 .. t+hours_ahead-1),
    # since those feed the feature row used to predict the next step. The final step's own weather
    # isn't needed because there's no step after it.
    weather_future = None
    if hours_ahead > 1:
        forecast_start = last_known_ts + timedelta(hours=1)
        forecast_end = last_known_ts + timedelta(hours=hours_ahead - 1)
        weather_client = WeatherClient()
        weather_future = weather_client.fetch_forecast(forecast_start.date(), forecast_end.date())
        if weather_future is None or weather_future.empty:
            raise HTTPException(status_code=503, detail="Could not fetch weather forecast from Open-Meteo")

        expected_hours = pd.date_range(forecast_start, forecast_end, freq="h", tz="UTC")
        missing = expected_hours.difference(weather_future.index)
        if len(missing) > 0:
            raise HTTPException(
                status_code=503,
                detail=f"Weather forecast does not cover the requested window ({len(missing)} hour(s) missing). "
                       "Historical data may be stale — re-run the ingestion pipeline."
            )

    expected_feature_cols = getattr(model, "feature_names_in_", None)
    results = []
    current_ts = last_known_ts

    for step in range(hours_ahead):
        featured = compute_predictor_features(history)
        X = featured.iloc[[-1]].drop(columns=[c for c in featured.columns if c.startswith('target_')], errors='ignore')
        if expected_feature_cols is not None:
            X = X.reindex(columns=list(expected_feature_cols))

        pred = float(model.predict(X)[0])
        predict_ts = current_ts + timedelta(hours=1)
        results.append(ForecastResponse(timestamp=predict_ts, wind_onshore_mw=pred))

        if step < hours_ahead - 1:
            last_actual = history.iloc[-1]
            new_row = pd.Series(index=history.columns, dtype="float64", name=predict_ts)
            for col in WeatherClient.HOURLY_VARIABLES:
                new_row[col] = weather_future.loc[predict_ts, col]
            new_row['wind_onshore'] = pred
            # wind_offshore/solar aren't modeled yet (see docstring) — persist last known value.
            new_row['wind_offshore'] = last_actual['wind_offshore']
            new_row['solar'] = last_actual['solar']
            if 'is_generation_gap' in history.columns:
                new_row['is_generation_gap'] = 0
            if 'has_long_gap' in history.columns:
                new_row['has_long_gap'] = 0

            history = pd.concat([history, new_row.to_frame().T])
            history.index = pd.to_datetime(history.index)
            current_ts = predict_ts

    return results

@router.get("/forecast/explain", response_model=ExplanationResponse)
def explain_forecast(timestamp: str = None):
    # For demo, explain the very last prediction we can make
    model = get_model()
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not model or not features_path.exists():
        raise HTTPException(status_code=503, detail="Model or data not available")
        
    df = pd.read_parquet(features_path)
    latest_row = df.tail(1)
    drop_cols = [c for c in latest_row.columns if c.startswith('target_')]
    X = latest_row.drop(columns=drop_cols)
    
    # Calculate SHAP for this row
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    except Exception:
        # Fallback for HistGradientBoosting
        background = shap.sample(X, 10)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X)
        
    contributions = dict(zip(X.columns, shap_values.values[0]))
    base_val = float(shap_values.base_values[0])
    
    # Take top 10 features
    top_10 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    return ExplanationResponse(
        timestamp=datetime.now(),
        base_value=base_val,
        feature_contributions=dict(top_10)
    )

@router.get("/historical")
def get_historical(region: str = "DE", limit: int = 168):
    features_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="Historical data not available")
    df = pd.read_parquet(features_path).tail(limit)
    # Convert datetime index to string format for JSON
    df.index = df.index.astype(str)
    return df[['wind_onshore', 'wind_offshore', 'solar']].to_dict(orient="index")

@router.get("/metrics")
def get_metrics():
    db_path = settings.MLRUNS_DIR / "mlflow.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Metrics DB not found")
        
    conn = sqlite3.connect(str(db_path))
    query = """
    SELECT runs.name, m.key, m.value 
    FROM runs 
    JOIN metrics m ON runs.run_uuid = m.run_uuid
    WHERE runs.status = 'FINISHED'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Format nicely
    result = {}
    for run in df['name'].unique():
        run_data = df[df['name'] == run]
        result[run] = dict(zip(run_data['key'], run_data['value']))
    return result

@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    features_used = []
    if features_path.exists():
        df = pd.read_parquet(features_path)
        features_used = [c for c in df.columns if not c.startswith('target_')]
        
    return ModelInfoResponse(
        model_type=type(model).__name__,
        training_date=str(datetime.now().date()), # Ideally fetched from mlflow
        features_used=features_used[:20] # Return first 20 just for brevity
    )

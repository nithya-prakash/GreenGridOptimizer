from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import json
import sqlite3
import shap
import anthropic
from src.api.schemas import ForecastResponse, ModelInfoResponse, ExplanationResponse, ChatRequest, ChatResponse
from src.utils.config import settings
from src.features.feature_engineering import compute_predictor_features, GENERATION_COLUMNS
from src.ingestion.weather_client import WeatherClient

# Forecast history/horizon and the chat model are non-secret pipeline parameters —
# see configs/pipeline.yaml, loaded into settings.FORECAST_HISTORY_HOURS /
# settings.MAX_FORECAST_HOURS_AHEAD / settings.CHAT_MODEL.

CHAT_SYSTEM_PROMPT = """You are the assistant embedded in the GreenGrid Optimizer dashboard, a \
renewable energy generation forecasting tool for the German electricity grid (wind onshore, wind \
offshore, and solar). Answer the user's question using ONLY the dashboard data provided below \
(recent actuals, the current forecast, and SHAP feature contributions explaining one target's \
latest prediction). Reference specific numbers and timestamps. When asked why generation is high \
or low, ground the explanation in the SHAP contributions. If the data doesn't answer the question, \
say so rather than guessing. Keep answers to 2-4 sentences.

Dashboard data (JSON):
{context_json}"""

TARGETS = GENERATION_COLUMNS  # ['wind_onshore', 'wind_offshore', 'solar']

router = APIRouter()

# In-memory loaded models, keyed by target (lazy load)
_models: Dict[str, Any] = {}

def get_models() -> Dict[str, Any]:
    """Loads and caches the production model for each modeled target."""
    for target in TARGETS:
        if target not in _models:
            model_path = settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl"
            if model_path.exists():
                _models[target] = joblib.load(model_path)
    return _models

@router.get("/")
def read_root():
    return {"name": "GreenGrid Optimizer API", "version": "1.0"}

@router.get("/health")
def healthcheck():
    return {"status": "healthy"}

@router.get("/forecast", response_model=List[ForecastResponse])
def get_forecast(region: str = "DE", hours_ahead: int = 24):
    """
    Recursive multi-step forecast: each model predicts its target at t+1 from
    features known at t (current weather + lags/rolling stats of wind_onshore,
    wind_offshore, solar and weather). To forecast further than 1h out, each step's
    three predictions are fed back in as the "current" generation values for the
    next step, and real Open-Meteo *forecast* weather (not historical) is used for
    the weather features of each future hour.
    """
    if not 1 <= hours_ahead <= settings.MAX_FORECAST_HOURS_AHEAD:
        raise HTTPException(status_code=400, detail=f"hours_ahead must be between 1 and {settings.MAX_FORECAST_HOURS_AHEAD}")

    models = get_models()
    missing_models = [t for t in TARGETS if t not in models]
    if missing_models:
        raise HTTPException(status_code=503, detail=f"Model(s) not loaded for: {', '.join(missing_models)}")

    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="Feature dataset not available")

    df = pd.read_parquet(features_path)
    raw_cols = [c for c in GENERATION_COLUMNS + WeatherClient.HOURLY_VARIABLES + ["is_generation_gap", "has_long_gap"] if c in df.columns]
    history = df[raw_cols].tail(settings.FORECAST_HISTORY_HOURS).copy()
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

    expected_feature_cols = {t: getattr(m, "feature_names_in_", None) for t, m in models.items()}
    results = []
    current_ts = last_known_ts

    for step in range(hours_ahead):
        featured = compute_predictor_features(history)
        X = featured.iloc[[-1]].drop(columns=[c for c in featured.columns if c.startswith('target_')], errors='ignore')

        preds = {}
        for target in TARGETS:
            X_target = X
            if expected_feature_cols[target] is not None:
                X_target = X.reindex(columns=list(expected_feature_cols[target]))
            preds[target] = float(models[target].predict(X_target)[0])

        predict_ts = current_ts + timedelta(hours=1)
        results.append(ForecastResponse(
            timestamp=predict_ts,
            wind_onshore_mw=preds['wind_onshore'],
            wind_offshore_mw=preds['wind_offshore'],
            solar_mw=preds['solar']
        ))

        if step < hours_ahead - 1:
            # Built as a dict -> single-row DataFrame (not a uniformly-typed Series) so pandas
            # infers a dtype per column. is_generation_gap/has_long_gap are genuinely bool in
            # the historical data; concatenating a float64-cast copy of them onto history would
            # silently upcast the column to `object`, which XGBoost's predict() then rejects.
            new_row_data = {col: float(weather_future.loc[predict_ts, col]) for col in WeatherClient.HOURLY_VARIABLES}
            new_row_data.update({target: float(preds[target]) for target in TARGETS})
            if 'is_generation_gap' in history.columns:
                new_row_data['is_generation_gap'] = False
            if 'has_long_gap' in history.columns:
                new_row_data['has_long_gap'] = False
            new_row_df = pd.DataFrame([new_row_data], index=[predict_ts])[history.columns]

            history = pd.concat([history, new_row_df])
            history.index = pd.to_datetime(history.index)
            current_ts = predict_ts

    return results

@router.get("/forecast/explain", response_model=ExplanationResponse)
def explain_forecast(target: str = "wind_onshore", timestamp: str = None):
    # For demo, explain the very last prediction we can make
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of: {', '.join(TARGETS)}")

    models = get_models()
    model = models.get(target)
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
        target=target,
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
    models = get_models()
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    features_used = []
    if features_path.exists():
        df = pd.read_parquet(features_path)
        features_used = [c for c in df.columns if not c.startswith('target_')]

    return ModelInfoResponse(
        models={target: type(model).__name__ for target, model in models.items()},
        training_date=str(datetime.now().date()), # Ideally fetched from mlflow
        features_used=features_used[:20] # Return first 20 just for brevity
    )

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Answers questions about the forecast currently shown on the dashboard. The caller
    (the Streamlit frontend) passes the forecast/historical/SHAP data it already has as
    `context` rather than this endpoint recomputing a forecast per chat message.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Chat is not configured: ANTHROPIC_API_KEY is not set.")

    context_json = json.dumps(request.context, indent=2, default=str)[:12000]
    system_prompt = CHAT_SYSTEM_PROMPT.format(context_json=context_json)
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.message})

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=settings.CHAT_MODEL,
            max_tokens=4096,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=503, detail="Chat is not configured: invalid ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Claude API. Try again shortly.")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the Claude API.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    reply = next((block.text for block in response.content if block.type == "text"), "")
    return ChatResponse(reply=reply)

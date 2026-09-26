from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from threading import Lock
import time
import pandas as pd
import joblib
import json
import sqlite3
import shap
import anthropic
from src.api.schemas import ForecastResponse, ModelInfoResponse, ExplanationResponse, ChatRequest, ChatResponse
from src.api.security import require_api_token, chat_rate_limit, forecast_rate_limit, read_rate_limit
from src.utils.config import settings
from src.features.feature_engineering import GENERATION_COLUMNS
from src.ingestion.weather_client import WeatherClient
from src.models.recursive import recursive_forecast, GAP_FLAG_COLUMNS

# Forecast history/horizon, chat model and chat limits are non-secret pipeline
# parameters — see configs/pipeline.yaml.

CHAT_SYSTEM_PROMPT = """You are the assistant embedded in the GreenGrid Optimizer dashboard, a \
renewable energy generation forecasting tool for the German electricity grid (wind onshore, wind \
offshore, and solar). Answer the user's question using ONLY the dashboard data provided below \
(recent actuals, the current forecast, and SHAP feature contributions explaining one target's \
prediction). Reference specific numbers and timestamps. When asked why generation is high \
or low, ground the explanation in the SHAP contributions. If the data doesn't answer the question, \
say so rather than guessing. Keep answers to 2-4 sentences.

Dashboard data (JSON):
{context_json}"""

TARGETS = GENERATION_COLUMNS  # ['wind_onshore', 'wind_offshore', 'solar']
SUPPORTED_REGIONS = {"DE"}

router = APIRouter()

# In-memory model cache: name -> (file mtime, model). Reloaded when the file
# changes, so models retrained by the refresh job are picked up without a restart.
_models: Dict[str, Any] = {}
_cache: Dict[str, tuple] = {}

def _load(name: str) -> Optional[Any]:
    path = settings.PRODUCTION_MODEL_DIR / f"model_{name}.pkl"
    if not path.exists():
        _cache.pop(name, None)
        return None
    mtime = path.stat().st_mtime
    if name not in _cache or _cache[name][0] != mtime:
        _cache[name] = (mtime, joblib.load(path))
    return _cache[name][1]

def get_models() -> Dict[str, Any]:
    """The 1h-ahead production model for each target that has been trained."""
    _models.clear()
    for target in TARGETS:
        model = _load(target)
        if model is not None:
            _models[target] = model
    return _models

def get_solar_direct_model() -> Optional[Any]:
    """The direct multi-horizon solar model (leads 2..72h), if trained."""
    return _load("solar_direct")

def reset_model_cache():
    _cache.clear()
    _models.clear()
    _forecast_cache.clear()


# The full-horizon forecast is computed once and served (sliced) to every request
# until it expires or the data/models change. Smaller horizons are exact prefixes
# of the full one: the recursion is deterministic given the same inputs.
_forecast_cache: Dict[str, Any] = {}
_forecast_lock = Lock()


def _check_region(region: str):
    # Only Germany-wide data is ingested; accepting other values and silently
    # returning German numbers would be misleading.
    if region not in SUPPORTED_REGIONS:
        raise HTTPException(status_code=400, detail=f"region must be one of: {', '.join(sorted(SUPPORTED_REGIONS))}")


@router.get("/")
def read_root():
    return {"name": "GreenGrid Optimizer API", "version": "1.0"}

@router.get("/health")
def healthcheck():
    return {"status": "healthy"}

@router.get("/data/status", dependencies=[Depends(read_rate_limit)])
def data_status():
    """How current the served data is. Forecasts start from the last published hour."""
    master_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    if not master_path.exists():
        raise HTTPException(status_code=503, detail="Historical dataset not available")
    last_ts = pd.read_parquet(master_path, columns=["wind_onshore"]).index.max()
    age_hours = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 3600
    return {
        "last_data_timestamp": last_ts.isoformat(),
        "age_hours": round(age_hours, 1),
        "stale": age_hours > settings.STALE_AFTER_HOURS,
        "stale_after_hours": settings.STALE_AFTER_HOURS,
    }

def _compute_full_forecast(models: Dict[str, Any], solar_direct: Any, master_path) -> pd.DataFrame:
    df = pd.read_parquet(master_path)
    raw_cols = [c for c in GENERATION_COLUMNS + WeatherClient.WEATHER_COLUMNS + GAP_FLAG_COLUMNS if c in df.columns]
    history = df[raw_cols].dropna(subset=GENERATION_COLUMNS).tail(settings.FORECAST_HISTORY_HOURS).copy()
    history.index = pd.to_datetime(history.index)
    if history.empty:
        raise HTTPException(status_code=503, detail="Not enough historical data to build a forecast")

    horizon = settings.MAX_FORECAST_HOURS_AHEAD
    last_known_ts = history.index.max()

    # Every step needs the forecast weather for the hour it predicts (the *_next_1h features).
    forecast_start = last_known_ts + timedelta(hours=1)
    forecast_end = last_known_ts + timedelta(hours=horizon)
    weather_future = WeatherClient().fetch_forecast(forecast_start.date(), forecast_end.date())
    if weather_future is None or weather_future.empty:
        raise HTTPException(status_code=503, detail="Could not fetch weather forecast from Open-Meteo")

    expected_hours = pd.date_range(forecast_start, forecast_end, freq="h", tz="UTC")
    missing = expected_hours.difference(weather_future.dropna().index)
    if len(missing) > 0:
        raise HTTPException(
            status_code=503,
            detail=f"Weather forecast does not cover the forecast window ({len(missing)} hour(s) missing). "
                   "Historical data may be stale — check the refresh job."
        )
    return recursive_forecast(models, history, weather_future, horizon, solar_direct_model=solar_direct)


@router.get("/forecast", response_model=List[ForecastResponse], dependencies=[Depends(forecast_rate_limit)])
def get_forecast(region: str = "DE", hours_ahead: int = 24):
    """
    Multi-step forecast from the last hour in the dataset (see
    src/models/recursive.py — the same function the backtest evaluates): wind is
    recursive, solar beyond 1h comes from the direct multi-horizon model. Uses
    live Open-Meteo *forecast* weather for every future hour. Cached for
    FORECAST_CACHE_MINUTES, and recomputed as soon as the data or models change.
    """
    _check_region(region)
    if not 1 <= hours_ahead <= settings.MAX_FORECAST_HOURS_AHEAD:
        raise HTTPException(status_code=400, detail=f"hours_ahead must be between 1 and {settings.MAX_FORECAST_HOURS_AHEAD}")

    models = get_models()
    missing_models = [t for t in TARGETS if t not in models]
    solar_direct = get_solar_direct_model()
    if solar_direct is None:
        missing_models.append("solar_direct")
    if missing_models:
        raise HTTPException(status_code=503, detail=f"Model(s) not loaded for: {', '.join(missing_models)}")

    master_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    if not master_path.exists():
        raise HTTPException(status_code=503, detail="Historical dataset not available")

    key = (master_path.stat().st_mtime, tuple(sorted((name, entry[0]) for name, entry in _cache.items())))
    with _forecast_lock:  # one computation at a time; concurrent requests wait and reuse it
        fresh = (_forecast_cache.get("key") == key
                 and time.monotonic() - _forecast_cache.get("at", 0) < settings.FORECAST_CACHE_MINUTES * 60)
        if not fresh:
            _forecast_cache.update(key=key, at=time.monotonic(),
                                   forecast=_compute_full_forecast(models, solar_direct, master_path))
        forecast = _forecast_cache["forecast"].head(hours_ahead)

    return [
        ForecastResponse(
            timestamp=ts,
            wind_onshore_mw=row["wind_onshore"],
            wind_offshore_mw=row["wind_offshore"],
            solar_mw=row["solar"],
        )
        for ts, row in forecast.iterrows()
    ]

@router.get("/forecast/explain", response_model=ExplanationResponse, dependencies=[Depends(read_rate_limit)])
def explain_forecast(target: str = "wind_onshore", timestamp: Optional[str] = None):
    """
    SHAP contributions for one 1h-ahead prediction from the feature dataset: the
    row at `timestamp` (ISO-8601, UTC), or the most recent row if omitted. The
    response says which row was explained and which hour it predicts.
    """
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of: {', '.join(TARGETS)}")

    models = get_models()
    model = models.get(target)
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not model or not features_path.exists():
        raise HTTPException(status_code=503, detail="Model or data not available")

    df = pd.read_parquet(features_path)
    if timestamp is None:
        row = df.tail(1)
    else:
        try:
            ts = pd.Timestamp(timestamp)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="timestamp must be an ISO-8601 datetime")
        if ts not in df.index:
            raise HTTPException(
                status_code=404,
                detail=f"No feature row at {ts}. Available range: {df.index.min()} to {df.index.max()} (hourly)."
            )
        row = df.loc[[ts]]

    X = row.drop(columns=[c for c in row.columns if c.startswith('target_')])
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        X = X.reindex(columns=list(expected))

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    except Exception:
        # Model-agnostic fallback for non-tree models.
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)

    contributions = dict(zip(X.columns, shap_values.values[0]))
    top_10 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    row_ts = pd.Timestamp(X.index[0])

    return ExplanationResponse(
        target=target,
        timestamp=row_ts,
        prediction_for=row_ts + timedelta(hours=1),
        predicted_mw=float(model.predict(X)[0]),
        base_value=float(shap_values.base_values[0]),
        feature_contributions={k: float(v) for k, v in top_10},
    )

@router.get("/historical", dependencies=[Depends(read_rate_limit)])
def get_historical(region: str = "DE", limit: int = Query(168, ge=1, le=24 * 365)):
    _check_region(region)
    features_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="Historical data not available")
    df = pd.read_parquet(features_path).tail(limit)
    # Convert datetime index to string format for JSON
    df.index = df.index.astype(str)
    return df[['wind_onshore', 'wind_offshore', 'solar']].to_dict(orient="index")

@router.get("/metrics", dependencies=[Depends(read_rate_limit)])
def get_metrics():
    """Single-holdout (last week) 1h-ahead metrics logged by trainer.py to MLflow."""
    db_path = settings.MLRUNS_DIR / "mlflow.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Metrics DB not found")

    conn = sqlite3.connect(str(db_path))
    query = """
    SELECT runs.name, m.key, m.value
    FROM runs
    JOIN metrics m ON runs.run_uuid = m.run_uuid
    WHERE runs.status = 'FINISHED'
    ORDER BY runs.start_time
    """
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    # Later runs with the same name overwrite earlier ones, so each name shows its latest run.
    result = {}
    for run in df['name'].unique():
        run_data = df[df['name'] == run]
        result[run] = dict(zip(run_data['key'], run_data['value']))
    return result

@router.get("/backtest", dependencies=[Depends(read_rate_limit)])
def get_backtest():
    """Walk-forward backtest summary (1h-ahead and recursive error by horizon)."""
    path = settings.MODELS_DIR / "backtest_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Backtest results not found. Run src/evaluation/backtest.py.")
    with open(path) as f:
        return json.load(f)

@router.get("/model/info", response_model=ModelInfoResponse, dependencies=[Depends(read_rate_limit)])
def get_model_info():
    models = get_models()
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    metadata_path = settings.PRODUCTION_MODEL_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            training_date = json.load(f).get("trained_at")
        source = "metadata"
    else:
        # Older artifacts without metadata: the model file's mtime is when it was saved.
        mtimes = [(settings.PRODUCTION_MODEL_DIR / f"model_{t}.pkl").stat().st_mtime for t in models]
        training_date = datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()
        source = "model_file_mtime"

    first_model = next(iter(models.values()))
    features = list(getattr(first_model, "feature_names_in_", []))

    model_types = {target: type(model).__name__ for target, model in models.items()}
    solar_direct = get_solar_direct_model()
    if solar_direct is not None:
        model_types["solar_direct"] = f"{type(solar_direct).__name__} (leads 2-{settings.MAX_FORECAST_HOURS_AHEAD}h)"

    return ModelInfoResponse(
        models=model_types,
        training_date=training_date,
        training_date_source=source,
        n_features=len(features),
        features_used=features[:20],
    )

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_token), Depends(chat_rate_limit)])
def chat(request: ChatRequest):
    """
    Answers questions about the forecast currently shown on the dashboard. The caller
    (the Streamlit frontend) passes the forecast/historical/SHAP data it already has as
    `context` rather than this endpoint recomputing a forecast per chat message.
    """
    if not settings.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Chat is not configured: ANTHROPIC_API_KEY is not set.")

    # Compact, deterministic serialization. Oversized context is rejected rather than
    # cut mid-JSON, which would hand the model malformed/partial data.
    context_json = json.dumps(request.context, separators=(",", ":"), sort_keys=True, default=str)
    if len(context_json) > settings.CHAT_MAX_CONTEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Chat context is {len(context_json)} characters; the limit is {settings.CHAT_MAX_CONTEXT_CHARS}."
        )
    system_prompt = CHAT_SYSTEM_PROMPT.format(context_json=context_json)
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.message})

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        # No prompt caching: the system prompt embeds per-request dashboard data, so a
        # cached prefix would never be reused.
        response = client.messages.create(
            model=settings.CHAT_MODEL,
            max_tokens=4096,
            system=system_prompt,
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

    if response.stop_reason == "refusal":
        return ChatResponse(reply="The model declined to answer this question.")
    reply = next((block.text for block in response.content if block.type == "text"), "")
    return ChatResponse(reply=reply)

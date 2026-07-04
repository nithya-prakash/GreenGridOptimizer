from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import joblib
import json
import sqlite3
import shap
from src.api.schemas import ForecastResponse, ModelInfoResponse, ExplanationResponse
from src.utils.config import settings
from src.features.feature_engineering import create_features
from src.ingestion.weather_client import WeatherClient

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
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # To forecast t+1 to t+hours_ahead, we need weather forecast for the next `hours_ahead` hours.
    # In a real app, we'd fetch Open-Meteo *forecast* API.
    # For this v1 demo, we'll return a simulated forecast based on the last row of our features dataset.
    # Or actually, we can just grab the last N rows of features.parquet as a mock "upcoming" features if Open-Meteo forecast isn't built.
    # Let's mock it using the last N hours from the dataset.
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        raise HTTPException(status_code=503, detail="Feature dataset not available")
        
    df = pd.read_parquet(features_path)
    # Take the last `hours_ahead` rows
    recent_features = df.tail(hours_ahead)
    
    drop_cols = [c for c in recent_features.columns if c.startswith('target_')]
    X = recent_features.drop(columns=drop_cols)
    
    preds = model.predict(X)
    
    results = []
    # Mocking future timestamps
    start_time = datetime.now()
    for i, pred in enumerate(preds):
        results.append(ForecastResponse(
            timestamp=start_time + timedelta(hours=i+1),
            wind_onshore_mw=float(pred)
        ))
        
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

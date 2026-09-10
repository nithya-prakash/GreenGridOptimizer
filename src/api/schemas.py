from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ForecastRequest(BaseModel):
    region: str = "DE"
    hours_ahead: int = 24

class ForecastResponse(BaseModel):
    timestamp: datetime
    wind_onshore_mw: float
    wind_offshore_mw: float
    solar_mw: float

class HistoricalRequest(BaseModel):
    region: str = "DE"
    start_date: str
    end_date: str

class ModelInfoResponse(BaseModel):
    models: Dict[str, str]  # target -> model type, e.g. {"wind_onshore": "XGBRegressor"}
    training_date: Optional[str]
    features_used: List[str]

class ExplanationResponse(BaseModel):
    target: str
    timestamp: datetime
    base_value: float
    feature_contributions: Dict[str, float]

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    context: Dict[str, Any] = {}  # forecast/historical/SHAP data the dashboard currently shows

class ChatResponse(BaseModel):
    reply: str

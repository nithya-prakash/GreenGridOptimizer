from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
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
    training_date: Optional[str]  # from models/production/metadata.json, else the model file's mtime
    training_date_source: str  # "metadata" | "model_file_mtime"
    n_features: int
    features_used: List[str]  # first 20, for brevity

class ExplanationResponse(BaseModel):
    target: str
    timestamp: datetime  # time of the feature row being explained (the "now" of that prediction)
    prediction_for: datetime  # the hour being predicted (timestamp + 1h)
    predicted_mw: float
    base_value: float
    feature_contributions: Dict[str, float]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)
    context: Dict[str, Any] = Field(default_factory=dict)  # forecast/historical/SHAP data the dashboard currently shows

class ChatResponse(BaseModel):
    reply: str

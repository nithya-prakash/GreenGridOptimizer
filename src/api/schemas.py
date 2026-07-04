from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ForecastRequest(BaseModel):
    region: str = "DE"
    hours_ahead: int = 24

class ForecastResponse(BaseModel):
    timestamp: datetime
    wind_onshore_mw: float
    # We can expand to others later

class HistoricalRequest(BaseModel):
    region: str = "DE"
    start_date: str
    end_date: str

class ModelInfoResponse(BaseModel):
    model_type: str
    training_date: Optional[str]
    features_used: List[str]

class ExplanationResponse(BaseModel):
    timestamp: datetime
    base_value: float
    feature_contributions: Dict[str, float]

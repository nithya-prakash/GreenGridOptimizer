import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class Config(BaseModel):
    # API Keys
    ENTSOE_API_KEY: str = os.getenv("ENTSOE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Optional shared secret for the credit-spending /chat endpoint (sent as the
    # X-API-Key header). Unset = no auth, which is only appropriate on localhost.
    API_AUTH_TOKEN: str = os.getenv("API_AUTH_TOKEN", "")
    # Comma-separated origins allowed by CORS. The Streamlit frontend calls the API
    # server-side, so browsers never need cross-origin access by default.
    CORS_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()]

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    PRODUCTION_MODEL_DIR: Path = MODELS_DIR / "production"
    MLRUNS_DIR: Path = BASE_DIR / "mlruns"
    CONFIGS_DIR: Path = BASE_DIR / "configs"

    # Non-secret pipeline parameters — defaults here, overridden by
    # configs/pipeline.yaml when present. See that file for what each means.
    TEST_SIZE_HOURS: int = 168
    BACKTEST_FOLDS: int = 4
    BACKTEST_ORIGIN_STEP_HOURS: int = 7
    FORECAST_HISTORY_HOURS: int = 504
    MAX_FORECAST_HOURS_AHEAD: int = 72
    CHAT_MODEL: str = "claude-opus-5"
    CHAT_RATE_LIMIT_PER_MINUTE: int = 10
    READ_RATE_LIMIT_PER_MINUTE: int = 120
    FORECAST_RATE_LIMIT_PER_MINUTE: int = 20
    FORECAST_CACHE_MINUTES: float = 30
    REFRESH_INTERVAL_HOURS: float = 6
    RETRAIN_AFTER_DAYS: float = 7
    STALE_AFTER_HOURS: float = 36
    REFRESH_HISTORY_DAYS: int = 120
    CHAT_MAX_CONTEXT_CHARS: int = 40000
    # Default weather sites if pipeline.yaml has none (a single central point).
    WEATHER_LAND_SITES: List[Dict[str, Any]] = [{"name": "central_de", "lat": 51.2, "lon": 10.4, "weight": 1.0}]
    WEATHER_OFFSHORE_SITES: List[Dict[str, Any]] = [{"name": "north_sea", "lat": 54.3, "lon": 6.5, "weight": 1.0}]

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_pipeline_config()

    def _load_pipeline_config(self):
        config_path = self.CONFIGS_DIR / "pipeline.yaml"
        if not config_path.exists():
            return
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        evaluation = data.get("evaluation", {})
        forecast = data.get("forecast", {})
        chat = data.get("chat", {})
        weather = data.get("weather", {})
        refresh = data.get("refresh", {})
        api = data.get("api", {})
        self.TEST_SIZE_HOURS = evaluation.get("test_size_hours", self.TEST_SIZE_HOURS)
        self.BACKTEST_FOLDS = evaluation.get("backtest_folds", self.BACKTEST_FOLDS)
        self.BACKTEST_ORIGIN_STEP_HOURS = evaluation.get("backtest_origin_step_hours", self.BACKTEST_ORIGIN_STEP_HOURS)
        self.FORECAST_HISTORY_HOURS = forecast.get("history_hours", self.FORECAST_HISTORY_HOURS)
        self.MAX_FORECAST_HOURS_AHEAD = forecast.get("max_hours_ahead", self.MAX_FORECAST_HOURS_AHEAD)
        self.CHAT_MODEL = chat.get("model", self.CHAT_MODEL)
        self.CHAT_RATE_LIMIT_PER_MINUTE = chat.get("rate_limit_per_minute", self.CHAT_RATE_LIMIT_PER_MINUTE)
        self.CHAT_MAX_CONTEXT_CHARS = chat.get("max_context_chars", self.CHAT_MAX_CONTEXT_CHARS)
        self.READ_RATE_LIMIT_PER_MINUTE = api.get("read_rate_limit_per_minute", self.READ_RATE_LIMIT_PER_MINUTE)
        self.FORECAST_RATE_LIMIT_PER_MINUTE = api.get("forecast_rate_limit_per_minute", self.FORECAST_RATE_LIMIT_PER_MINUTE)
        self.FORECAST_CACHE_MINUTES = api.get("forecast_cache_minutes", self.FORECAST_CACHE_MINUTES)
        self.REFRESH_INTERVAL_HOURS = refresh.get("interval_hours", self.REFRESH_INTERVAL_HOURS)
        self.RETRAIN_AFTER_DAYS = refresh.get("retrain_after_days", self.RETRAIN_AFTER_DAYS)
        self.STALE_AFTER_HOURS = refresh.get("stale_after_hours", self.STALE_AFTER_HOURS)
        self.REFRESH_HISTORY_DAYS = refresh.get("history_days", self.REFRESH_HISTORY_DAYS)
        self.WEATHER_LAND_SITES = weather.get("land_sites", self.WEATHER_LAND_SITES)
        self.WEATHER_OFFSHORE_SITES = weather.get("offshore_sites", self.WEATHER_OFFSHORE_SITES)

settings = Config()

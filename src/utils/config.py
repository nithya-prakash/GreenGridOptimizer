import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class Config(BaseModel):
    # API Keys
    ENTSOE_API_KEY: str = os.getenv("ENTSOE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Region config for Open-Meteo
    REGION_LAT: float = float(os.getenv("REGION_LAT", "52.52"))
    REGION_LON: float = float(os.getenv("REGION_LON", "13.40"))

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
    FORECAST_HISTORY_HOURS: int = 504
    MAX_FORECAST_HOURS_AHEAD: int = 72
    CHAT_MODEL: str = "claude-opus-5"

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
        self.TEST_SIZE_HOURS = evaluation.get("test_size_hours", self.TEST_SIZE_HOURS)
        self.BACKTEST_FOLDS = evaluation.get("backtest_folds", self.BACKTEST_FOLDS)
        self.FORECAST_HISTORY_HOURS = forecast.get("history_hours", self.FORECAST_HISTORY_HOURS)
        self.MAX_FORECAST_HOURS_AHEAD = forecast.get("max_hours_ahead", self.MAX_FORECAST_HOURS_AHEAD)
        self.CHAT_MODEL = chat.get("model", self.CHAT_MODEL)

settings = Config()

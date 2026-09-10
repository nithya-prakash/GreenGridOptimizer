import os
from pathlib import Path
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
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

settings = Config()

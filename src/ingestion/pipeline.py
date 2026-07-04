import pandas as pd
from datetime import date, timedelta
from src.utils.logger import log
from src.utils.config import settings
from src.ingestion.weather_client import WeatherClient
from src.ingestion.entsoe_client import ENTSOEClient
from src.ingestion.smard_client import SMARDClient
from src.preprocessing.clean import clean_and_join
from src.preprocessing.validation import validate_master_dataset

def run_ingestion_pipeline(start_date: date, end_date: date):
    log.info(f"Starting ingestion pipeline for period: {start_date} to {end_date}")
    
    # 1. Fetch Weather Data
    weather_client = WeatherClient()
    weather_df = weather_client.fetch_historical(start_date, end_date)
    if weather_df is None or weather_df.empty:
        log.error("Failed to fetch weather data. Pipeline aborted.")
        return
        
    # 2. Fetch Generation Data (Try ENTSO-E, fallback to SMARD)
    generation_df = None
    entsoe_client = ENTSOEClient(settings.ENTSOE_API_KEY)
    generation_df = entsoe_client.fetch_historical(start_date, end_date)
    
    if generation_df is None or generation_df.empty:
        log.info("Falling back to SMARD client for generation data...")
        smard_client = SMARDClient()
        generation_df = smard_client.fetch_historical(start_date, end_date)
        
    if generation_df is None or generation_df.empty:
        log.error("Failed to fetch generation data from all sources. Pipeline aborted.")
        return
        
    # 3. Clean and Join
    master_df = clean_and_join(weather_df, generation_df)
    
    # 4. Validate
    is_valid = validate_master_dataset(master_df)
    if not is_valid:
        log.error("Dataset validation failed. Saving to RAW_DATA_DIR for inspection.")
        master_df.to_parquet(settings.RAW_DATA_DIR / "failed_validation_dataset.parquet")
        return
        
    # 5. Save to Processed Data
    output_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    master_df.to_parquet(output_path)
    log.info(f"Pipeline completed successfully. Dataset saved to {output_path}")
    
    # Print a summary to verify
    print("\n--- Pipeline Verification ---")
    print(f"Shape: {master_df.shape}")
    print("\nColumns:")
    for col in master_df.columns:
        print(f" - {col}")
    print("\nMissing Values:")
    print(master_df.isna().sum())
    print(f"\nDate Range: {master_df.index.min()} to {master_df.index.max()}")
    print("\nSample (first 3 rows):")
    print(master_df.head(3))
    print("-----------------------------\n")

if __name__ == "__main__":
    # By default, let's fetch the last 30 days of data for the first run
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)
    run_ingestion_pipeline(start, end)

import os
import pandas as pd
from datetime import date, timedelta
from src.utils.logger import log
from src.utils.config import settings
from src.ingestion.weather_client import WeatherClient
from src.ingestion.entsoe_client import ENTSOEClient
from src.ingestion.smard_client import SMARDClient
from src.preprocessing.clean import clean_and_join
from src.preprocessing.validation import validate_master_dataset

# Lead-day weather kept for the backtest (see WeatherClient.fetch_previous_runs):
# a 72h horizon needs forecasts issued up to 3 days earlier.
PREVIOUS_RUN_DAYS = [1, 2, 3]


def atomic_to_parquet(df: pd.DataFrame, path):
    """Write-then-rename, so a reader (the API) never sees a half-written file while
    the refresh job is rewriting it."""
    tmp = path.with_name(path.stem + ".tmp.parquet")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def fetch_lead_day_weather(start_date: date, end_date: date) -> bool:
    client = WeatherClient()
    frames = {}
    for days in PREVIOUS_RUN_DAYS:
        df = client.fetch_previous_runs(start_date, end_date, days)
        if df is None or df.empty:
            log.warning(f"No previous-run (day {days}) weather; the backtest will fall back to short-lead weather.")
            return False
        frames[days] = df.ffill(limit=3)
    combined = pd.concat([df.add_suffix(f"__d{days}") for days, df in frames.items()], axis=1)
    atomic_to_parquet(combined, settings.PROCESSED_DATA_DIR / "weather_previous_runs.parquet")
    log.info(f"Saved lead-day weather ({len(combined)} rows, days {PREVIOUS_RUN_DAYS}).")
    return True


def run_ingestion_pipeline(start_date: date, end_date: date, verbose: bool = True) -> bool:
    """Fetches, cleans, validates and saves the master dataset. Returns True on
    success; on any failure the previously saved dataset is left untouched."""
    log.info(f"Starting ingestion pipeline for period: {start_date} to {end_date}")

    # 1. Fetch Weather Data
    weather_df = WeatherClient().fetch_historical(start_date, end_date)
    if weather_df is None or weather_df.empty:
        log.error("Failed to fetch weather data. Pipeline aborted.")
        return False

    # 2. Fetch Generation Data (Try ENTSO-E, fallback to SMARD)
    generation_df = ENTSOEClient(settings.ENTSOE_API_KEY).fetch_historical(start_date, end_date)
    if generation_df is None or generation_df.empty:
        log.info("Falling back to SMARD client for generation data...")
        generation_df = SMARDClient().fetch_historical(start_date, end_date)

    if generation_df is None or generation_df.empty:
        log.error("Failed to fetch generation data from all sources. Pipeline aborted.")
        return False

    # 3. Clean and Join
    master_df = clean_and_join(weather_df, generation_df)

    # Keep only the span where generation has actually been published. The weather
    # APIs return the whole requested day and SMARD lags real time by some hours;
    # those trailing weather-only rows would otherwise be forward-filled into fake
    # "actuals" and used as the forecast origin.
    published = generation_df.dropna(how="any").index
    master_df = master_df.loc[published.min():published.max()]

    # 4. Validate
    if not validate_master_dataset(master_df):
        log.error("Dataset validation failed. Saving to RAW_DATA_DIR for inspection.")
        master_df.to_parquet(settings.RAW_DATA_DIR / "failed_validation_dataset.parquet")
        return False

    # 5. Save to Processed Data
    output_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    atomic_to_parquet(master_df, output_path)
    log.info(f"Pipeline completed successfully. Dataset saved to {output_path} "
             f"({master_df.index.min()} to {master_df.index.max()})")

    fetch_lead_day_weather(start_date, end_date)

    if verbose:
        print("\n--- Pipeline Verification ---")
        print(f"Shape: {master_df.shape}")
        print("\nMissing Values:")
        print(master_df.isna().sum())
        print(f"\nDate Range: {master_df.index.min()} to {master_df.index.max()}")
        print("-----------------------------\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch generation + weather data and build the master dataset.")
    parser.add_argument("--days", type=int, default=120,
                        help="Days of history ending today (default 120: enough for the 4-fold backtest).")
    args = parser.parse_args()
    # Through today: generation is trimmed to what SMARD has published so far.
    end = date.today()
    start = end - timedelta(days=args.days)
    run_ingestion_pipeline(start, end)

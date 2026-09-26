import pandas as pd
import holidays
from src.utils.logger import log
from src.utils.config import settings

GENERATION_COLUMNS = ['wind_onshore', 'wind_offshore', 'solar']
LAG_ROLL_COLUMNS = [
    'wind_onshore', 'wind_offshore', 'solar',
    'wind_speed_10m', 'wind_speed_100m', 'temperature_2m',
    'cloud_cover', 'shortwave_radiation', 'offshore_wind_speed_100m'
]
# Weather at the *target* hour (t+1). The model predicts generation at t+1, and at
# serving time the forecast weather for t+1 is known, so it is a legitimate (and
# the most informative) input — not leakage. In training these come from the
# archived historical *forecast* weather, matching what serving sees.
NEXT_HOUR_WEATHER_COLUMNS = [
    'wind_speed_10m', 'wind_speed_100m', 'temperature_2m', 'cloud_cover',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'offshore_wind_speed_10m', 'offshore_wind_speed_100m'
]
LAGS = [1, 2, 3, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 48, 168]
# Rows of history needed so the last row's lag/rolling features are complete.
REQUIRED_HISTORY_HOURS = max(max(LAGS), max(ROLLING_WINDOWS)) + 1


def compute_predictor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates calendar, lag, rolling, and interaction features (no target column,
    no NaN dropping). Shared by training (create_features) and live inference, so
    a forecast request builds features the exact same way the model was trained on.

    The *_next_1h columns read the following row's weather, so the row for time t
    is only complete once the weather for t+1 has been appended to `df`.
    """
    # Columns are collected in a dict and joined once: inserting ~250 columns one at
    # a time fragments the frame, which is slow in the per-step recursive forecast.
    new = {}

    # 1. Calendar Features
    new['hour'] = df.index.hour
    new['day'] = df.index.day
    new['dayofweek'] = df.index.dayofweek
    new['month'] = df.index.month
    new['quarter'] = df.index.quarter
    new['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # German holidays
    de_holidays = holidays.DE(years=df.index.year.unique().tolist())
    new['is_holiday'] = df.index.map(lambda d: d.date() in de_holidays).astype(int)

    # 2. Lag Features (apply to both generation and weather variables)
    for col in LAG_ROLL_COLUMNS:
        for lag in LAGS:
            new[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # 3. Rolling Features
    for col in LAG_ROLL_COLUMNS:
        for window in ROLLING_WINDOWS:
            # The current value at time t is known (it's what we're rolling over to
            # predict t+1), so including it in the window is not leakage.
            rolling_window = df[col].rolling(window=window, min_periods=1)
            new[f'{col}_roll_mean_{window}'] = rolling_window.mean()
            new[f'{col}_roll_std_{window}'] = rolling_window.std()
            new[f'{col}_roll_min_{window}'] = rolling_window.min()
            new[f'{col}_roll_max_{window}'] = rolling_window.max()

    # 4. Target-hour (t+1) weather
    for col in NEXT_HOUR_WEATHER_COLUMNS:
        new[f'{col}_next_1h'] = df[col].shift(-1)

    # 5. Interaction Features
    new['wind_speed_100m_x_radiation'] = df['wind_speed_100m'] * df['shortwave_radiation']
    new['temp_x_cloud_cover'] = df['temperature_2m'] * df['cloud_cover']

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time series, calendar, weather, lag, and rolling features, plus the
    t+1h target columns, for training. Drops rows left with NaNs from lags/target.
    """
    log.info("Starting feature engineering...")
    df = compute_predictor_features(df)

    # Target Variable (t+1h): we want to forecast the generation for the next hour
    log.info("Generating target variables (t+1)...")
    df = pd.concat([df, pd.DataFrame(
        {f'target_{target}': df[target].shift(-1) for target in GENERATION_COLUMNS}, index=df.index
    )], axis=1)

    # Drop rows with NaNs caused by lags (max lag is 168, so we lose 168 rows at the
    # start) and the very last row where the target is NaN (nothing to shift into it).
    log.info("Dropping initial rows with NaNs due to lags...")
    df.dropna(subset=[f'wind_onshore_lag_{max(LAGS)}'], inplace=True)
    df.dropna(subset=['target_wind_onshore'], inplace=True)
    df.dropna(subset=[f'{c}_next_1h' for c in NEXT_HOUR_WEATHER_COLUMNS], inplace=True)

    log.info(f"Feature engineering completed. Final shape: {df.shape}")
    return df

def run_feature_engineering():
    input_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    output_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    
    if not input_path.exists():
        log.error(f"Input file {input_path} does not exist. Run ingestion pipeline first.")
        return
        
    df = pd.read_parquet(input_path)
    features_df = create_features(df)
    
    # Save the full feature matrix
    from src.ingestion.pipeline import atomic_to_parquet  # local import: avoids a cycle at import time
    atomic_to_parquet(features_df, output_path)
    log.info(f"Saved engineered features to {output_path}")
    
    # Validation
    assert not features_df['target_wind_onshore'].isna().any(), "Target contains NaNs!"
    assert not features_df['wind_onshore_lag_1'].isna().any(), "Lag features contain NaNs!"
    
    # Check leakage (target should not be perfectly correlated with current time's generation)
    log.info("Leakage check (correlation of t with t+1):")
    corr = features_df[['wind_onshore', 'target_wind_onshore']].corr().iloc[0, 1]
    log.info(f"wind_onshore correlation: {corr:.3f}")
    assert corr < 0.99, "Potential data leakage detected!"

if __name__ == "__main__":
    run_feature_engineering()

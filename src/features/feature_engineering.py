import pandas as pd
import numpy as np
import holidays
from typing import Tuple
from src.utils.logger import log
from src.utils.config import settings

GENERATION_COLUMNS = ['wind_onshore', 'wind_offshore', 'solar']
LAG_ROLL_COLUMNS = [
    'wind_onshore', 'wind_offshore', 'solar',
    'wind_speed_10m', 'wind_speed_100m', 'temperature_2m',
    'cloud_cover', 'shortwave_radiation'
]
LAGS = [1, 2, 3, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 48, 168]


def compute_predictor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates calendar, lag, rolling, and interaction features (no target column,
    no NaN dropping). Shared by training (create_features) and live inference, so
    a forecast request builds features the exact same way the model was trained on.
    """
    df = df.copy()

    # 1. Calendar Features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # German holidays
    de_holidays = holidays.DE(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda d: d.date() in de_holidays).astype(int)

    # 2. Lag Features (apply to both generation and weather variables)
    for col in LAG_ROLL_COLUMNS:
        for lag in LAGS:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # 3. Rolling Features
    for col in LAG_ROLL_COLUMNS:
        for window in ROLLING_WINDOWS:
            # The current value at time t is known (it's what we're rolling over to
            # predict t+1), so including it in the window is not leakage.
            rolling_window = df[col].rolling(window=window, min_periods=1)
            df[f'{col}_roll_mean_{window}'] = rolling_window.mean()
            df[f'{col}_roll_std_{window}'] = rolling_window.std()
            df[f'{col}_roll_min_{window}'] = rolling_window.min()
            df[f'{col}_roll_max_{window}'] = rolling_window.max()

    # 4. Interaction Features
    df['wind_speed_100m_x_radiation'] = df['wind_speed_100m'] * df['shortwave_radiation']
    df['temp_x_cloud_cover'] = df['temperature_2m'] * df['cloud_cover']

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time series, calendar, weather, lag, and rolling features, plus the
    t+1h target columns, for training. Drops rows left with NaNs from lags/target.
    """
    log.info("Starting feature engineering...")
    df = compute_predictor_features(df)

    # Target Variable (t+1h): we want to forecast the generation for the next hour
    log.info("Generating target variables (t+1)...")
    for target in GENERATION_COLUMNS:
        df[f'target_{target}'] = df[target].shift(-1)

    # Drop rows with NaNs caused by lags (max lag is 168, so we lose 168 rows at the
    # start) and the very last row where the target is NaN (nothing to shift into it).
    log.info("Dropping initial rows with NaNs due to lags...")
    df.dropna(subset=[f'wind_onshore_lag_{max(LAGS)}'], inplace=True)
    df.dropna(subset=['target_wind_onshore'], inplace=True)

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
    features_df.to_parquet(output_path)
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

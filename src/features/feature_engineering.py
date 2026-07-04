import pandas as pd
import numpy as np
import holidays
from typing import Tuple
from src.utils.logger import log
from src.utils.config import settings

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time series, calendar, weather, lag, and rolling features.
    """
    log.info("Starting feature engineering...")
    df = df.copy()
    
    # 1. Calendar Features
    log.info("Generating calendar features...")
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # German holidays
    de_holidays = holidays.DE(years=df.index.year.unique().tolist())
    df['is_holiday'] = df.index.map(lambda d: d.date() in de_holidays).astype(int)
    
    # 2. Target Variable (t+1h)
    # We want to forecast the generation for the next hour
    log.info("Generating target variables (t+1)...")
    for target in ['wind_onshore', 'wind_offshore', 'solar']:
        df[f'target_{target}'] = df[target].shift(-1)
        
    # 3. Lag Features
    log.info("Generating lag features...")
    lags = [1, 2, 3, 24, 48, 72, 168]
    # Apply lags to both generation and weather variables
    feature_cols = [
        'wind_onshore', 'wind_offshore', 'solar',
        'wind_speed_10m', 'wind_speed_100m', 'temperature_2m', 
        'cloud_cover', 'shortwave_radiation'
    ]
    
    for col in feature_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # 4. Rolling Features
    log.info("Generating rolling window features...")
    windows = [24, 48, 168]
    for col in feature_cols:
        for window in windows:
            # We must use closed='left' or shift(1) to avoid data leakage
            # meaning the rolling window should not include the current timestamp's value
            # wait, if the current timestamp IS known (e.g. current weather forecast), it's fine.
            # But generation at current timestamp might not be known if we are forecasting it.
            # However, our target is t+1, so at time t, we know generation at time t.
            # Therefore, using the current value in rolling is fine for predicting t+1.
            rolling_window = df[col].rolling(window=window, min_periods=1)
            df[f'{col}_roll_mean_{window}'] = rolling_window.mean()
            df[f'{col}_roll_std_{window}'] = rolling_window.std()
            df[f'{col}_roll_min_{window}'] = rolling_window.min()
            df[f'{col}_roll_max_{window}'] = rolling_window.max()
            
    # 5. Interaction Features
    log.info("Generating interaction features...")
    df['wind_speed_100m_x_radiation'] = df['wind_speed_100m'] * df['shortwave_radiation']
    df['temp_x_cloud_cover'] = df['temperature_2m'] * df['cloud_cover']
    
    # Drop rows with NaNs caused by lags and shifts (except at the very end where target is NaN)
    # Wait, the max lag is 168. We will lose 168 rows at the start.
    # The target shift(-1) will cause the last row to be NaN.
    log.info("Dropping initial rows with NaNs due to lags...")
    df.dropna(subset=[f'wind_onshore_lag_{max(lags)}'], inplace=True)
    
    # Also drop the very last row where target is NaN
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

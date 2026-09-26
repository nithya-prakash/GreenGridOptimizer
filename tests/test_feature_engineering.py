import pandas as pd
from src.features.feature_engineering import create_features

def test_create_features():
    df = pd.DataFrame({
        'wind_onshore': [10.0] * 200, 
        'wind_offshore': [10.0] * 200, 
        'solar': [10.0] * 200,
        'wind_speed_10m': [1.0] * 200, 
        'wind_speed_100m': [1.0] * 200, 
        'wind_direction_10m': [1.0] * 200,
        'wind_direction_100m': [1.0] * 200, 
        'temperature_2m': [1.0] * 200, 
        'cloud_cover': [1.0] * 200,
        'shortwave_radiation': [1.0] * 200, 
        'direct_radiation': [1.0] * 200, 
        'diffuse_radiation': [1.0] * 200,
        'precipitation': [0.0] * 200, 
        'offshore_wind_speed_10m': [1.0] * 200,
        'offshore_wind_speed_100m': [1.0] * 200,
        'offshore_wind_direction_100m': [1.0] * 200,
        'is_generation_gap': [False] * 200, 
        'has_long_gap': [False] * 200
    })
    df.index = pd.date_range("2026-01-01", periods=200, freq="h")
    
    features_df = create_features(df)
    
    # Check that rows with NaNs (first 168 rows + last row) are dropped
    assert len(features_df) == 200 - 168 - 1
    
    # Check that target exists
    assert 'target_wind_onshore' in features_df.columns
    assert 'wind_onshore_lag_168' in features_df.columns
    assert 'wind_speed_100m_x_radiation' in features_df.columns
    assert 'offshore_wind_speed_100m_lag_24' in features_df.columns
    assert 'shortwave_radiation_next_1h' in features_df.columns
    
    # No NaNs in the returned dataframe
    assert not features_df.isna().any().any()


def test_next_hour_weather_features_read_the_following_row():
    idx = pd.date_range("2026-01-01", periods=200, freq="h")
    df = pd.DataFrame({c: [1.0] * 200 for c in [
        'wind_onshore', 'wind_offshore', 'solar', 'wind_speed_10m', 'wind_speed_100m',
        'wind_direction_10m', 'wind_direction_100m', 'temperature_2m', 'cloud_cover',
        'direct_radiation', 'diffuse_radiation', 'precipitation',
        'offshore_wind_speed_10m', 'offshore_wind_speed_100m', 'offshore_wind_direction_100m',
    ]}, index=idx)
    df['shortwave_radiation'] = range(200)
    df['is_generation_gap'] = False
    df['has_long_gap'] = False

    features_df = create_features(df)

    # Row t carries the radiation of t+1 (the hour its target is for).
    assert (features_df['shortwave_radiation_next_1h'] == features_df['shortwave_radiation'] + 1).all()

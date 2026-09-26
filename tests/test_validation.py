import numpy as np
import pandas as pd
from src.preprocessing.validation import validate_master_dataset

def test_validation_missing_columns():
    df = pd.DataFrame({
        'wind_onshore': [10.0, 15.0],
        'solar': [5.0, 0.0]
    })
    df.index = pd.date_range("2026-01-01", periods=2, freq="h")
    assert not validate_master_dataset(df)

def test_validation_wrong_index():
    df = pd.DataFrame({
        'wind_onshore': [10.0], 'wind_offshore': [10.0], 'solar': [10.0],
        'wind_speed_10m': [1.0], 'wind_speed_100m': [1.0], 'wind_direction_10m': [1.0],
        'wind_direction_100m': [1.0], 'temperature_2m': [1.0], 'cloud_cover': [1.0],
        'shortwave_radiation': [1.0], 'direct_radiation': [1.0], 'diffuse_radiation': [1.0],
        'precipitation': [0.0], 'is_generation_gap': [False], 'has_long_gap': [False]
    })
    assert not validate_master_dataset(df)

def test_validation_success():
    df = pd.DataFrame({
        'wind_onshore': [10.0], 'wind_offshore': [10.0], 'solar': [10.0],
        'wind_speed_10m': [1.0], 'wind_speed_100m': [1.0], 'wind_direction_10m': [1.0],
        'wind_direction_100m': [1.0], 'temperature_2m': [1.0], 'cloud_cover': [1.0],
        'shortwave_radiation': [1.0], 'direct_radiation': [1.0], 'diffuse_radiation': [1.0],
        'precipitation': [0.0], 'offshore_wind_speed_10m': [1.0], 'offshore_wind_speed_100m': [1.0],
        'offshore_wind_direction_100m': [1.0], 'is_generation_gap': [False], 'has_long_gap': [False]
    })
    df.index = pd.date_range("2026-01-01", periods=1, freq="h")
    assert validate_master_dataset(df)


def _valid_frame(n, solar):
    idx = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    cols = ['wind_onshore', 'wind_offshore', 'wind_speed_10m', 'wind_speed_100m', 'wind_direction_10m',
            'wind_direction_100m', 'temperature_2m', 'cloud_cover', 'shortwave_radiation', 'direct_radiation',
            'diffuse_radiation', 'precipitation', 'offshore_wind_speed_10m', 'offshore_wind_speed_100m',
            'offshore_wind_direction_100m']
    df = pd.DataFrame({c: 1.0 for c in cols}, index=idx)
    df['solar'] = solar(idx.hour)
    df['is_generation_gap'] = False
    df['has_long_gap'] = False
    return df


def test_validation_accepts_solar_with_a_day_night_cycle():
    df = _valid_frame(96, lambda h: np.clip(np.sin((h - 6) / 12 * np.pi), 0, None) * 30000)
    assert validate_master_dataset(df)


def test_validation_rejects_solar_that_generates_at_night():
    # e.g. a mis-mapped source (pumped storage/wind) stored under "solar"
    df = _valid_frame(96, lambda h: np.full(len(h), 5000.0))
    assert not validate_master_dataset(df)

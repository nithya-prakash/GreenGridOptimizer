import pytest
import pandas as pd
import numpy as np
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
        'precipitation': [0.0], 'is_generation_gap': [False], 'has_long_gap': [False]
    })
    df.index = pd.date_range("2026-01-01", periods=1, freq="h")
    assert validate_master_dataset(df)

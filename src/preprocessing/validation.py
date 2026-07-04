import pandas as pd
from src.utils.logger import log

def validate_master_dataset(df: pd.DataFrame) -> bool:
    """
    Checks schema, types, and value ranges.
    Returns True if valid, False otherwise.
    """
    log.info("Validating master dataset...")
    
    expected_cols = [
        'wind_onshore', 'wind_offshore', 'solar',
        'wind_speed_10m', 'wind_speed_100m', 'wind_direction_10m',
        'wind_direction_100m', 'temperature_2m', 'cloud_cover',
        'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
        'precipitation', 'is_generation_gap', 'has_long_gap'
    ]
    
    # Check columns
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        log.error(f"Validation failed: Missing columns {missing_cols}")
        return False
        
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        log.error("Validation failed: Index is not a DatetimeIndex.")
        return False
        
    # Check value ranges (no negative power generation, usually)
    # Sometimes generation can be slightly negative due to station consumption, but usually >= 0.
    # We will just warn for negative generation.
    for col in ['wind_onshore', 'wind_offshore', 'solar']:
        if (df[col] < -100).any(): # allowing small negative values
            log.warning(f"Found significant negative generation values in {col}.")
            
    # Check for too many missing values
    missing_pct = df.isna().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 10:
            log.warning(f"Column {col} has {pct:.2f}% missing values.")
            
    log.info("Validation passed successfully.")
    return True

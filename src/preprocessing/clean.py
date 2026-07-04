import pandas as pd
from src.utils.logger import log

def clean_and_join(weather_df: pd.DataFrame, generation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins weather and generation data on timestamp.
    Forward-fills short gaps (<= 3 hours) and flags longer gaps.
    """
    log.info("Joining weather and generation datasets...")
    
    # Ensure indices are aligned (both should be datetime UTC)
    # Perform outer join
    df = generation_df.join(weather_df, how='outer')
    
    # Sort by time just in case
    df = df.sort_index()
    
    log.info(f"Dataset size after join: {len(df)}")
    
    # Identify gaps before filling
    # A gap in generation data
    df['is_generation_gap'] = df['wind_onshore'].isna() | df['wind_offshore'].isna() | df['solar'].isna()
    
    # Forward-fill short gaps (limit=3)
    cols_to_fill = list(generation_df.columns) + list(weather_df.columns)
    
    log.info("Forward-filling short gaps (up to 3 hours)...")
    df[cols_to_fill] = df[cols_to_fill].ffill(limit=3)
    
    # Flag remaining long gaps
    df['has_long_gap'] = df[cols_to_fill].isna().any(axis=1)
    
    if df['has_long_gap'].sum() > 0:
        log.warning(f"Found {df['has_long_gap'].sum()} rows with gaps longer than 3 hours after forward-filling.")
        
    return df

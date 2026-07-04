import requests
import pandas as pd
from typing import Optional
from datetime import date, timedelta
from src.utils.logger import log
from src.utils.config import settings

class WeatherClient:
    """Client to fetch historical weather data from Open-Meteo."""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, lat: float = settings.REGION_LAT, lon: float = settings.REGION_LON):
        self.lat = lat
        self.lon = lon
        
    def fetch_historical(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetches hourly weather data for the specified date range.
        Uses UTC timezone to align with standard power market data.
        """
        log.info(f"Fetching weather data from {start_date} to {end_date} for lat:{self.lat}, lon:{self.lon}")
        
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "wind_direction_100m",
                "temperature_2m",
                "cloud_cover",
                "shortwave_radiation",
                "direct_radiation",
                "diffuse_radiation",
                "precipitation"
            ],
            "timezone": "UTC"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                log.error("No hourly data found in Open-Meteo response.")
                return None
                
            df = pd.DataFrame(data["hourly"])
            # Convert time to proper datetime in UTC
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
            df.set_index('time', inplace=True)
            
            log.info(f"Successfully fetched {len(df)} rows of weather data.")
            return df
            
        except requests.RequestException as e:
            log.error(f"Failed to fetch data from Open-Meteo: {e}")
            return None

import requests
import pandas as pd
from typing import Optional
from datetime import date, timedelta
from src.utils.logger import log
from src.utils.config import settings

class WeatherClient:
    """Client to fetch historical and forecast weather data from Open-Meteo."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    HOURLY_VARIABLES = [
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
    ]

    def __init__(self, lat: float = settings.REGION_LAT, lon: float = settings.REGION_LON):
        self.lat = lat
        self.lon = lon

    def _fetch(self, base_url: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": self.HOURLY_VARIABLES,
            "timezone": "UTC"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "hourly" not in data:
                log.error("No hourly data found in Open-Meteo response.")
                return None

            df = pd.DataFrame(data["hourly"])
            # Convert time to proper datetime in UTC
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
            df.set_index('time', inplace=True)

            return df

        except requests.RequestException as e:
            log.error(f"Failed to fetch data from Open-Meteo ({base_url}): {e}")
            return None

    def fetch_historical(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetches hourly *observed* weather data for the specified date range from the
        archive API. Uses UTC timezone to align with standard power market data.
        """
        log.info(f"Fetching historical weather data from {start_date} to {end_date} for lat:{self.lat}, lon:{self.lon}")
        df = self._fetch(self.ARCHIVE_URL, start_date, end_date)
        if df is not None:
            log.info(f"Successfully fetched {len(df)} rows of historical weather data.")
        return df

    def fetch_forecast(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetches hourly *forecast* weather data for the specified date range from the
        forecast API. Open-Meteo's free forecast endpoint covers roughly the next 16
        days, so start_date/end_date must fall within that window.
        """
        log.info(f"Fetching weather forecast from {start_date} to {end_date} for lat:{self.lat}, lon:{self.lon}")
        df = self._fetch(self.FORECAST_URL, start_date, end_date)
        if df is not None:
            log.info(f"Successfully fetched {len(df)} rows of forecast weather data.")
        return df

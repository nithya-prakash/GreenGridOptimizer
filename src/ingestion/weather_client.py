import requests
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import date, timedelta
from src.utils.logger import log
from src.utils.config import settings

REQUEST_TIMEOUT_SECONDS = 60


class WeatherClient:
    """
    Fetches hourly weather from Open-Meteo, aggregated over several sites.

    Training and serving both use *forecast* weather, not observations:
    - fetch_historical() reads the Historical Forecast API (archived NWP model
      output), so the model is trained on the same kind of (imperfect) weather
      inputs it sees at serving time.
    - fetch_forecast() reads the live Forecast API.
    Using the reanalysis/observation archive for training would teach the model
    on "perfect" weather and then serve it noisier forecast weather.

    Output columns: HOURLY_VARIABLES (weighted average over the land sites in
    configs/pipeline.yaml) plus OFFSHORE_VARIABLES (weighted average over the
    offshore wind-farm sites).
    """

    HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
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
    # Fetched at the offshore sites and stored with an `offshore_` prefix.
    # Open-Meteo reports these as the mean/sum over the *preceding* hour (the 10:00
    # value covers 09:00-10:00), while SMARD labels an hour by its start (10:00 =
    # 10:00-11:00). They are relabelled to interval-start so both sources describe
    # the same hour; otherwise every radiation value was an hour behind the solar
    # output it explains (corr with solar 0.93 misaligned vs 0.98 aligned).
    # Instantaneous variables (wind, temperature, cloud cover) are left as-is.
    PRECEDING_HOUR_VARIABLES = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "precipitation"]
    OFFSHORE_SOURCE_VARIABLES = ["wind_speed_10m", "wind_speed_100m", "wind_direction_100m"]
    OFFSHORE_VARIABLES = [f"offshore_{v}" for v in OFFSHORE_SOURCE_VARIABLES]
    WEATHER_COLUMNS = HOURLY_VARIABLES + OFFSHORE_VARIABLES

    def __init__(self, land_sites: List[Dict[str, Any]] = None, offshore_sites: List[Dict[str, Any]] = None):
        self.land_sites = land_sites or settings.WEATHER_LAND_SITES
        self.offshore_sites = offshore_sites or settings.WEATHER_OFFSHORE_SITES

    def _fetch_sites(self, base_url: str, sites: List[Dict[str, Any]], variables: List[str],
                     start_date: date, end_date: date, suffix: str = "") -> Optional[List[pd.DataFrame]]:
        """One request for all sites (Open-Meteo accepts comma-separated coordinates).
        `suffix` selects a variant of each variable (e.g. "_previous_day2"); the
        returned columns are named without it."""
        params = {
            "latitude": ",".join(str(s["lat"]) for s in sites),
            "longitude": ",".join(str(s["lon"]) for s in sites),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [f"{v}{suffix}" for v in variables],
            "timezone": "UTC"
        }
        try:
            response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            log.error(f"Failed to fetch data from Open-Meteo ({base_url}): {e}")
            return None

        # A single location comes back as an object, several as a list.
        payloads = data if isinstance(data, list) else [data]
        if len(payloads) != len(sites) or any("hourly" not in p for p in payloads):
            log.error("Unexpected Open-Meteo response: missing hourly data for one or more sites.")
            return None

        frames = []
        for payload in payloads:
            df = pd.DataFrame(payload["hourly"])
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
            df = df.set_index("time")[[f"{v}{suffix}" for v in variables]].astype(float)
            frames.append(df.set_axis(variables, axis=1))
        return frames

    @staticmethod
    def _weighted_average(frames: List[pd.DataFrame], weights: List[float]) -> pd.DataFrame:
        """Weighted mean across sites, ignoring a site's missing values. Wind
        directions are averaged as unit vectors (the plain mean of 350° and 10° is
        180°, which is exactly wrong)."""
        w = np.asarray(weights, dtype=float)
        index = frames[0].index
        out = {}
        for col in frames[0].columns:
            values = np.stack([f[col].reindex(index).to_numpy() for f in frames])  # (sites, hours)
            mask = ~np.isnan(values)
            site_w = w[:, None] * mask
            total_w = site_w.sum(axis=0)
            total_w[total_w == 0] = np.nan
            if "direction" in col:
                rad = np.deg2rad(np.nan_to_num(values))
                sin = (np.sin(rad) * site_w).sum(axis=0) / total_w
                cos = (np.cos(rad) * site_w).sum(axis=0) / total_w
                out[col] = np.rad2deg(np.arctan2(sin, cos)) % 360
            else:
                out[col] = (np.nan_to_num(values) * site_w).sum(axis=0) / total_w
        return pd.DataFrame(out, index=index)

    def _fetch(self, base_url: str, start_date: date, end_date: date, suffix: str = "") -> Optional[pd.DataFrame]:
        # One extra day so the last hour of end_date still has its (relabelled)
        # preceding-hour values.
        fetch_end = end_date + timedelta(days=1)
        land = self._fetch_sites(base_url, self.land_sites, self.HOURLY_VARIABLES, start_date, fetch_end, suffix)
        offshore = self._fetch_sites(base_url, self.offshore_sites, self.OFFSHORE_SOURCE_VARIABLES, start_date, fetch_end, suffix)
        if land is None or offshore is None:
            return None

        land_df = self._weighted_average(land, [s.get("weight", 1.0) for s in self.land_sites])
        offshore_df = self._weighted_average(offshore, [s.get("weight", 1.0) for s in self.offshore_sites])
        offshore_df = offshore_df.add_prefix("offshore_")
        df = land_df.join(offshore_df, how="outer")[self.WEATHER_COLUMNS]
        df = self.to_interval_start(df)
        last_hour = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
        return df.loc[:last_hour]

    @classmethod
    def to_interval_start(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Relabels preceding-hour variables so the row for hour T holds the value
        for T..T+1 (Open-Meteo's T+1 label)."""
        df = df.copy()
        cols = [c for c in cls.PRECEDING_HOUR_VARIABLES if c in df.columns]
        df[cols] = df[cols].shift(-1, freq="h").reindex(df.index)
        return df

    def fetch_historical(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Archived hourly *forecast* weather (Historical Forecast API, available from 2022)."""
        log.info(f"Fetching historical forecast weather from {start_date} to {end_date} "
                 f"for {len(self.land_sites)} land + {len(self.offshore_sites)} offshore sites")
        df = self._fetch(self.HISTORICAL_FORECAST_URL, start_date, end_date)
        if df is not None:
            log.info(f"Successfully fetched {len(df)} rows of historical weather data.")
        return df

    def fetch_previous_runs(self, start_date: date, end_date: date, days: int) -> Optional[pd.DataFrame]:
        """
        What the forecast issued `days` days earlier said about each hour (Open-Meteo
        Previous Runs API, `*_previous_dayN`: lead time of roughly 24*N .. 24*N+23h).
        Used by the backtest so a 48-72h forecast is evaluated with 48-72h-old
        weather forecasts, not the short-lead forecasts it was trained on.
        """
        log.info(f"Fetching weather forecasts issued {days} day(s) earlier, {start_date} to {end_date}")
        return self._fetch(self.PREVIOUS_RUNS_URL, start_date, end_date, suffix=f"_previous_day{days}")

    def fetch_forecast(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Live hourly forecast weather. Open-Meteo's free forecast endpoint covers
        roughly the past ~3 months to the next 16 days, so start_date/end_date must
        fall within that window.
        """
        log.info(f"Fetching weather forecast from {start_date} to {end_date}")
        df = self._fetch(self.FORECAST_URL, start_date, end_date)
        if df is not None:
            log.info(f"Successfully fetched {len(df)} rows of forecast weather data.")
        return df

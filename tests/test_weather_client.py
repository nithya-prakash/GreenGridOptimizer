import numpy as np
import pandas as pd

from src.ingestion.weather_client import WeatherClient


def _frame(**cols):
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame(cols, index=idx)


def test_weighted_average_of_scalar_variables():
    a = _frame(wind_speed_100m=[10.0, 10.0])
    b = _frame(wind_speed_100m=[20.0, 20.0])
    out = WeatherClient._weighted_average([a, b], [3.0, 1.0])
    assert np.allclose(out["wind_speed_100m"], 12.5)


def test_wind_direction_is_averaged_on_the_circle():
    # The arithmetic mean of 350° and 10° is 180° (due south) — exactly wrong.
    a = _frame(wind_direction_100m=[350.0, 90.0])
    b = _frame(wind_direction_100m=[10.0, 90.0])
    out = WeatherClient._weighted_average([a, b], [1.0, 1.0])
    assert np.isclose(min(out["wind_direction_100m"].iloc[0], 360 - out["wind_direction_100m"].iloc[0]), 0.0, atol=1e-6)
    assert np.isclose(out["wind_direction_100m"].iloc[1], 90.0)


def test_missing_site_value_is_ignored_not_zeroed():
    a = _frame(temperature_2m=[np.nan, 10.0])
    b = _frame(temperature_2m=[20.0, 20.0])
    out = WeatherClient._weighted_average([a, b], [1.0, 1.0])
    assert np.isclose(out["temperature_2m"].iloc[0], 20.0)
    assert np.isclose(out["temperature_2m"].iloc[1], 15.0)


def test_fetch_combines_land_and_offshore_sites(monkeypatch):
    land_sites = [{"lat": 1, "lon": 1, "weight": 1.0}, {"lat": 2, "lon": 2, "weight": 1.0}]
    offshore_sites = [{"lat": 3, "lon": 3, "weight": 1.0}]
    requested = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, params, timeout):
        requested.append(params)
        n_sites = len(params["latitude"].split(","))
        hourly = {"time": ["2026-01-01T00:00", "2026-01-01T01:00"]}
        hourly.update({v: [5.0, 6.0] for v in params["hourly"]})
        payloads = [{"hourly": hourly} for _ in range(n_sites)]
        # Open-Meteo returns an object for one location and a list for several.
        return FakeResponse(payloads if n_sites > 1 else payloads[0])

    monkeypatch.setattr("src.ingestion.weather_client.requests.get", fake_get)
    client = WeatherClient(land_sites=land_sites, offshore_sites=offshore_sites)

    df = client.fetch_forecast(pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-01").date())

    assert list(df.columns) == WeatherClient.WEATHER_COLUMNS
    assert len(requested) == 2  # one request for all land sites, one for offshore
    # One extra day is requested so the last hour of end_date can be relabelled...
    assert requested[0]["end_date"] == "2026-01-02"
    # ...and the result is trimmed back to the requested range.
    assert df.index.max() <= pd.Timestamp("2026-01-01 23:00", tz="UTC")
    assert requested[0]["latitude"] == "1,2"
    assert np.allclose(df["offshore_wind_speed_100m"], [5.0, 6.0])


def test_preceding_hour_variables_are_relabelled_to_interval_start():
    idx = pd.date_range("2026-06-01 09:00", periods=3, freq="h", tz="UTC")
    # Open-Meteo: the 10:00 radiation value is the 09:00-10:00 mean.
    df = pd.DataFrame({"shortwave_radiation": [100.0, 200.0, 300.0], "wind_speed_100m": [5.0, 6.0, 7.0]}, index=idx)

    out = WeatherClient.to_interval_start(df)

    # Row 09:00 (SMARD: 09:00-10:00) must hold Open-Meteo's 10:00 value.
    assert out.loc[idx[0], "shortwave_radiation"] == 200.0
    assert out.loc[idx[1], "shortwave_radiation"] == 300.0
    assert np.isnan(out.loc[idx[2], "shortwave_radiation"])
    # Instantaneous variables are untouched.
    assert list(out["wind_speed_100m"]) == [5.0, 6.0, 7.0]

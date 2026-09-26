import os
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.utils.config import settings
from src.ingestion.weather_client import WeatherClient


def _weather_from(frame):
    """A fetch_forecast stand-in that serves the given frame's weather columns."""
    def fetch(self, start_date, end_date):
        return frame[WeatherClient.WEATHER_COLUMNS]
    return fetch


def test_forecast_fetches_weather_for_every_predicted_hour(synthetic_history, monkeypatch):
    routes = synthetic_history
    calls = []

    def fetch(self, start_date, end_date):
        calls.append((start_date, end_date))
        return routes.synthetic_future[WeatherClient.WEATHER_COLUMNS]
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", fetch)

    results = routes.get_forecast(hours_ahead=1)

    assert len(results) == 1
    # Even t+1 needs the forecast weather *for* t+1 (the *_next_1h features).
    assert len(calls) == 1


def test_forecast_multi_step_is_real_recursive_forecast(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", _weather_from(routes.synthetic_future))

    results = routes.get_forecast(hours_ahead=24)

    assert len(results) == 24
    timestamps = [r.timestamp for r in results]
    assert timestamps == sorted(timestamps)
    assert all((timestamps[i + 1] - timestamps[i]).total_seconds() == 3600 for i in range(23))
    assert all(np.isfinite(r.wind_onshore_mw) for r in results)
    assert all(np.isfinite(r.wind_offshore_mw) for r in results)
    assert all(np.isfinite(r.solar_mw) for r in results)
    # Predictions must vary across the day (a real forecast), not all be identical
    # (which would indicate the endpoint fell back to a static/mocked value).
    assert len(set(round(r.wind_onshore_mw, 3) for r in results)) > 1
    assert len(set(round(r.solar_mw, 3) for r in results)) > 1


def test_forecast_rejects_out_of_range_horizon(synthetic_history):
    routes = synthetic_history
    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=200)
    assert exc_info.value.status_code == 400


def test_forecast_rejects_unsupported_region(synthetic_history):
    routes = synthetic_history
    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(region="FR", hours_ahead=1)
    assert exc_info.value.status_code == 400


def test_forecast_missing_weather_raises_503(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", lambda self, start_date, end_date: None)

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=5)
    assert exc_info.value.status_code == 503


def test_forecast_partial_weather_coverage_raises_503(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", _weather_from(routes.synthetic_future.iloc[:10]))

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=24)
    assert exc_info.value.status_code == 503
    assert "missing" in exc_info.value.detail


def test_forecast_requires_all_target_models(synthetic_history):
    routes = synthetic_history
    (settings.PRODUCTION_MODEL_DIR / "model_solar.pkl").unlink()

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=1)
    assert exc_info.value.status_code == 503
    assert "solar" in exc_info.value.detail


def test_explain_forecast_defaults_to_latest_feature_row(synthetic_history):
    routes = synthetic_history
    result = routes.explain_forecast()
    features = pd.read_parquet(settings.PROCESSED_DATA_DIR / "features.parquet")

    assert result.target == "wind_onshore"
    assert len(result.feature_contributions) == 10
    # The response names the row it actually explained, not the request time.
    assert pd.Timestamp(result.timestamp) == features.index.max()
    assert pd.Timestamp(result.prediction_for) == features.index.max() + pd.Timedelta(hours=1)


def test_explain_forecast_honours_timestamp(synthetic_history):
    routes = synthetic_history
    features = pd.read_parquet(settings.PROCESSED_DATA_DIR / "features.parquet")
    ts = features.index[10]

    result = routes.explain_forecast(target="solar", timestamp=ts.isoformat())

    assert pd.Timestamp(result.timestamp) == ts


def test_explain_forecast_unknown_timestamp_is_404(synthetic_history):
    routes = synthetic_history
    with pytest.raises(HTTPException) as exc_info:
        routes.explain_forecast(timestamp="2001-01-01T00:00:00")
    assert exc_info.value.status_code == 404


def test_explain_forecast_rejects_unknown_target(synthetic_history):
    routes = synthetic_history
    with pytest.raises(HTTPException) as exc_info:
        routes.explain_forecast(target="nuclear")
    assert exc_info.value.status_code == 400


def test_model_info_lists_all_targets(synthetic_history):
    routes = synthetic_history
    info = routes.get_model_info()
    assert set(info.models.keys()) == {"wind_onshore", "wind_offshore", "solar", "solar_direct"}
    # No metadata.json in this fixture: falls back to the model file's real mtime,
    # and says so — never "today's date".
    assert info.training_date_source == "model_file_mtime"
    assert info.n_features > 20


def test_model_info_reads_training_date_from_metadata(synthetic_history):
    routes = synthetic_history
    (settings.PRODUCTION_MODEL_DIR / "metadata.json").write_text('{"trained_at": "2026-01-02T03:04:05+00:00"}')

    info = routes.get_model_info()

    assert info.training_date == "2026-01-02T03:04:05+00:00"
    assert info.training_date_source == "metadata"


def test_forecast_requires_direct_solar_model(synthetic_history):
    routes = synthetic_history
    (settings.PRODUCTION_MODEL_DIR / "model_solar_direct.pkl").unlink()

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=1)
    assert exc_info.value.status_code == 503
    assert "solar_direct" in exc_info.value.detail


def test_model_info_lists_direct_solar_model(synthetic_history):
    routes = synthetic_history
    assert "solar_direct" in routes.get_model_info().models


def test_models_reload_when_retrained(synthetic_history):
    routes = synthetic_history
    first = routes.get_models()["solar"]
    path = settings.PRODUCTION_MODEL_DIR / "model_solar.pkl"
    joblib.dump(first, path)
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))

    assert routes.get_models()["solar"] is not first  # new file -> reloaded


def test_data_status_reports_age(synthetic_history):
    routes = synthetic_history
    status = routes.data_status()
    # The synthetic data ends in August 2026, well past the staleness threshold.
    assert status["stale"] is True
    assert status["last_data_timestamp"].startswith("2026-08-17")


def test_forecast_is_cached_and_sliced_for_smaller_horizons(synthetic_history, monkeypatch):
    routes = synthetic_history
    calls = []

    def fetch(self, start_date, end_date):
        calls.append(1)
        return routes.synthetic_future[WeatherClient.WEATHER_COLUMNS]
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", fetch)

    full = routes.get_forecast(hours_ahead=72)
    short = routes.get_forecast(hours_ahead=6)

    assert len(calls) == 1  # computed once, reused
    assert [r.model_dump() for r in short] == [r.model_dump() for r in full[:6]]


def test_forecast_cache_is_invalidated_when_data_changes(synthetic_history, monkeypatch):
    routes = synthetic_history
    calls = []

    def fetch(self, start_date, end_date):
        calls.append(1)
        return routes.synthetic_future[WeatherClient.WEATHER_COLUMNS]
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", fetch)

    routes.get_forecast(hours_ahead=24)
    master = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    os.utime(master, (master.stat().st_atime, master.stat().st_mtime + 10))  # the refresh job rewrote it
    routes.get_forecast(hours_ahead=24)

    assert len(calls) == 2

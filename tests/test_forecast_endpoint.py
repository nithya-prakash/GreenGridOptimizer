import numpy as np
import pandas as pd
import pytest
import joblib
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import settings
from src.features.feature_engineering import create_features


@pytest.fixture
def synthetic_history(tmp_path, monkeypatch):
    """Points the app's data/model dirs at tmp_path and seeds a trained model plus
    400h of synthetic feature history, so /forecast can run without live infra."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "PRODUCTION_MODEL_DIR", tmp_path / "models" / "production")
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    n = 400
    idx = pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    hours = np.arange(n)
    wind_speed = 5 + 3 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 0.3, n)
    raw = pd.DataFrame({
        "wind_onshore": 100 + 40 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 2, n),
        "wind_offshore": 50 + rng.normal(0, 2, n),
        "solar": np.clip(80 * np.sin((hours % 24 - 6) / 12 * np.pi), 0, None),
        "wind_speed_10m": wind_speed,
        "wind_speed_100m": wind_speed * 1.3,
        "wind_direction_10m": rng.uniform(0, 360, n),
        "wind_direction_100m": rng.uniform(0, 360, n),
        "temperature_2m": 15 + 5 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 1, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "shortwave_radiation": np.clip(400 * np.sin((hours % 24 - 6) / 12 * np.pi), 0, None),
        "direct_radiation": np.clip(300 * np.sin((hours % 24 - 6) / 12 * np.pi), 0, None),
        "diffuse_radiation": np.clip(100 * np.sin((hours % 24 - 6) / 12 * np.pi), 0, None),
        "precipitation": rng.uniform(0, 1, n),
        # bool, matching clean_and_join's real output (not int) — a previous version of this
        # fixture used int and missed a bug where concatenating a predicted row onto these
        # bool columns silently upcast them to `object`, which XGBoost's predict() rejects.
        "is_generation_gap": False,
        "has_long_gap": False,
    }, index=idx)

    features_df = create_features(raw)
    features_df.to_parquet(settings.PROCESSED_DATA_DIR / "features.parquet")

    drop_cols = [c for c in features_df.columns if c.startswith("target_")]
    X_train = features_df.drop(columns=drop_cols)
    for target in ["wind_onshore", "wind_offshore", "solar"]:
        model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)
        model.fit(X_train, features_df[f"target_{target}"])
        joblib.dump(model, settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl")

    import src.api.routes as routes
    routes._models.clear()  # reset the lazily-loaded model cache between tests
    return routes


def _fake_forecast_weather(start_date, end_date):
    days = (end_date - start_date).days + 1
    fidx = pd.date_range(start_date, periods=days * 24, freq="h", tz="UTC")
    fh = np.arange(len(fidx))
    return pd.DataFrame({
        "wind_speed_10m": 5 + 3 * np.sin(fh / 24 * 2 * np.pi),
        "wind_speed_100m": (5 + 3 * np.sin(fh / 24 * 2 * np.pi)) * 1.3,
        "wind_direction_10m": 180.0,
        "wind_direction_100m": 180.0,
        "temperature_2m": 15.0,
        "cloud_cover": 50.0,
        "shortwave_radiation": np.clip(400 * np.sin((fh % 24 - 6) / 12 * np.pi), 0, None),
        "direct_radiation": 100.0,
        "diffuse_radiation": 50.0,
        "precipitation": 0.0,
    }, index=fidx)


def test_forecast_single_hour_skips_weather_fetch(synthetic_history, monkeypatch):
    routes = synthetic_history
    calls = []
    monkeypatch.setattr(
        routes.WeatherClient, "fetch_forecast",
        lambda self, start_date, end_date: calls.append(1) or _fake_forecast_weather(start_date, end_date)
    )

    results = routes.get_forecast(hours_ahead=1)

    assert len(results) == 1
    assert calls == []  # t+1 only needs features already known at t; no forecast weather required


def test_forecast_multi_step_is_real_recursive_forecast(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(
        routes.WeatherClient, "fetch_forecast",
        lambda self, start_date, end_date: _fake_forecast_weather(start_date, end_date)
    )

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


def test_forecast_missing_weather_coverage_raises_503(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", lambda self, start_date, end_date: None)

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=5)
    assert exc_info.value.status_code == 503


def test_forecast_requires_all_target_models(synthetic_history, monkeypatch):
    routes = synthetic_history
    (settings.PRODUCTION_MODEL_DIR / "model_solar.pkl").unlink()
    routes._models.clear()

    with pytest.raises(HTTPException) as exc_info:
        routes.get_forecast(hours_ahead=1)
    assert exc_info.value.status_code == 503
    assert "solar" in exc_info.value.detail


def test_explain_forecast_defaults_to_wind_onshore(synthetic_history):
    routes = synthetic_history
    result = routes.explain_forecast()
    assert result.target == "wind_onshore"
    assert len(result.feature_contributions) == 10


def test_explain_forecast_rejects_unknown_target(synthetic_history):
    routes = synthetic_history
    with pytest.raises(HTTPException) as exc_info:
        routes.explain_forecast(target="nuclear")
    assert exc_info.value.status_code == 400


def test_model_info_lists_all_targets(synthetic_history):
    routes = synthetic_history
    info = routes.get_model_info()
    assert set(info.models.keys()) == {"wind_onshore", "wind_offshore", "solar"}

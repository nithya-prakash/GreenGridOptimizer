import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import settings
from src.features.feature_engineering import create_features


def synthetic_raw(n: int = 400, start: str = "2026-08-01") -> pd.DataFrame:
    """Raw hourly frame shaped like clean_and_join's real output (same columns, dtypes)."""
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    hours = np.arange(n)
    daylight = np.clip(np.sin((hours % 24 - 6) / 12 * np.pi), 0, None)
    wind_speed = 5 + 3 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 0.3, n)
    offshore_speed = 8 + 4 * np.sin(hours / 36 * 2 * np.pi) + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "wind_onshore": 100 + 40 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 2, n),
        "wind_offshore": 50 + 3 * offshore_speed + rng.normal(0, 2, n),
        "solar": 80 * daylight,
        "wind_speed_10m": wind_speed,
        "wind_speed_100m": wind_speed * 1.3,
        "wind_direction_10m": rng.uniform(0, 360, n),
        "wind_direction_100m": rng.uniform(0, 360, n),
        "temperature_2m": 15 + 5 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 1, n),
        "cloud_cover": rng.uniform(0, 100, n),
        "shortwave_radiation": 400 * daylight,
        "direct_radiation": 300 * daylight,
        "diffuse_radiation": 100 * daylight,
        "precipitation": rng.uniform(0, 1, n),
        "offshore_wind_speed_10m": offshore_speed / 1.2,
        "offshore_wind_speed_100m": offshore_speed,
        "offshore_wind_direction_100m": rng.uniform(0, 360, n),
        # bool, matching clean_and_join's real output (not int) — a previous version of this
        # fixture used int and missed a bug where concatenating a predicted row onto these
        # bool columns silently upcast them to `object`, which XGBoost's predict() rejects.
        "is_generation_gap": False,
        "has_long_gap": False,
    }, index=idx)


@pytest.fixture
def synthetic_history(tmp_path, monkeypatch):
    """Points the app's data/model dirs at tmp_path and seeds trained models plus the
    master/features datasets, so the API can run without live infra. The last 72h of
    the raw data are held back from the stored datasets and exposed as the "future"
    (actuals + weather) for tests that need them."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "PRODUCTION_MODEL_DIR", tmp_path / "models" / "production")
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    full = synthetic_raw(472)
    raw, future = full.iloc[:400], full.iloc[400:]
    raw.to_parquet(settings.PROCESSED_DATA_DIR / "master_dataset.parquet")
    features_df = create_features(raw)
    features_df.to_parquet(settings.PROCESSED_DATA_DIR / "features.parquet")

    drop_cols = [c for c in features_df.columns if c.startswith("target_")]
    X_train = features_df.drop(columns=drop_cols)
    for target in ["wind_onshore", "wind_offshore", "solar"]:
        model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0)
        model.fit(X_train, features_df[f"target_{target}"])
        joblib.dump(model, settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl")

    from src.models.solar_direct import train_solar_direct
    last_train_target = features_df.index.max() + pd.Timedelta(hours=1)
    joblib.dump(train_solar_direct(raw, last_train_target, horizon=72, origin_step=6),
                settings.PRODUCTION_MODEL_DIR / "model_solar_direct.pkl")

    import src.api.routes as routes
    routes.reset_model_cache()  # tmp_path differs per test
    routes.synthetic_future = future
    routes.synthetic_raw_full = full
    return routes

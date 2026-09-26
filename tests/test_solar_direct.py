import numpy as np
import pandas as pd

from src.models.solar_direct import build_direct_frame, train_solar_direct, predict_solar_direct, DIRECT_FEATURES
from src.ingestion.weather_client import WeatherClient
from tests.conftest import synthetic_raw


def test_direct_frame_uses_only_data_up_to_origin_except_weather():
    raw = synthetic_raw(200)
    origin = raw.index[150]
    # Poison everything after the origin except the weather columns.
    poisoned = raw.copy()
    poisoned.loc[poisoned.index > origin, ["solar", "wind_onshore", "wind_offshore"]] = 1e9

    clean = build_direct_frame(raw, raw, pd.DatetimeIndex([origin]), horizon=48)
    dirty = build_direct_frame(poisoned, poisoned, pd.DatetimeIndex([origin]), horizon=48)

    pd.testing.assert_frame_equal(clean, dirty)
    assert list(clean.columns) == DIRECT_FEATURES
    assert list(clean["lead"]) == list(range(1, 49))


def test_seasonal_naive_feature_is_same_hour_on_last_observed_day():
    raw = synthetic_raw(200)
    origin = raw.index[150]
    frame = build_direct_frame(raw, raw, pd.DatetimeIndex([origin]), horizon=30)

    targets = frame.index.get_level_values("target_ts")
    naive_ts = targets - pd.to_timedelta(np.ceil(frame["lead"].to_numpy() / 24) * 24, unit="h")
    assert (naive_ts <= origin).all()
    assert np.allclose(frame["solar_seasonal_naive"], raw["solar"].reindex(naive_ts))


def test_training_never_sees_targets_after_cutoff():
    raw = synthetic_raw(400)
    cutoff = raw.index[300]
    a = train_solar_direct(raw, cutoff, horizon=24, origin_step=6)
    changed = raw.copy()
    changed.loc[changed.index > cutoff, "solar"] = 1e9
    b = train_solar_direct(changed, cutoff, horizon=24, origin_step=6)

    X = build_direct_frame(raw, raw, pd.DatetimeIndex([raw.index[200]]), horizon=24)
    assert np.allclose(a.predict(X), b.predict(X))


def test_prediction_is_non_negative_and_covers_horizon():
    raw = synthetic_raw(472)
    model = train_solar_direct(raw.iloc[:400], raw.index[399], horizon=72, origin_step=6)
    history, future = raw.iloc[:400], raw.iloc[400:][WeatherClient.WEATHER_COLUMNS]

    preds = predict_solar_direct(model, history, future, horizon=72)

    assert len(preds) == 72
    assert preds.index[0] == history.index[-1] + pd.Timedelta(hours=1)
    assert (preds >= 0).all()

import pandas as pd

import numpy as np

from src.evaluation.backtest import _fold_bounds, _forecast_origins, horizon_errors, baseline_forecasts


def test_fold_bounds_are_contiguous_non_overlapping_and_expanding():
    n_rows, n_folds, test_size = 1511, 4, 168
    bounds = list(_fold_bounds(n_rows, n_folds, test_size))

    assert len(bounds) == n_folds
    for train_end, test_end in bounds:
        assert test_end - train_end == test_size

    # Each fold's train window strictly grows, and test windows are back-to-back.
    for (prev_train_end, prev_test_end), (train_end, test_end) in zip(bounds, bounds[1:]):
        assert train_end == prev_test_end
        assert train_end > prev_train_end

    # Last fold's test window ends exactly at the end of the dataset.
    assert bounds[-1][1] == n_rows


def test_forecast_origins_need_a_full_horizon_of_actuals():
    idx = pd.date_range("2026-01-01", periods=168, freq="h", tz="UTC")
    raw_end = idx[-1]  # the test window is the end of the data

    origins = _forecast_origins(idx, raw_end, step=24, horizon=72)

    assert origins[0] == idx[0]
    assert all(o + pd.Timedelta(hours=72) <= raw_end for o in origins)
    assert len(origins) == 4  # 0, 24, 48, 72h; 96h+ would run past the data


def test_horizon_errors_are_indexed_by_lead_time():
    origin = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=3, freq="h")
    forecast = pd.DataFrame({"wind_onshore": [10.0, 20.0, 30.0], "wind_offshore": 0.0, "solar": 0.0}, index=idx)
    actuals = pd.DataFrame({"wind_onshore": [12.0, 15.0, 30.0], "wind_offshore": 1.0, "solar": 0.0}, index=idx)

    err = horizon_errors(forecast, actuals, origin)

    assert list(err.index) == [1, 2, 3]
    assert list(err["wind_onshore"]) == [2.0, 5.0, 0.0]
    assert (err["wind_offshore"] == 1.0).all()


def test_baselines_only_use_data_up_to_the_origin():
    idx = pd.date_range("2026-01-01", periods=24 * 5, freq="h", tz="UTC")
    actuals = pd.DataFrame({c: np.arange(len(idx), dtype=float) for c in ["wind_onshore", "wind_offshore", "solar"]}, index=idx)
    origin = idx[48]

    baselines = baseline_forecasts(actuals, origin, horizon=30)

    assert (baselines["persistence"]["solar"] == 48.0).all()
    seasonal = baselines["seasonal_naive"]["solar"]
    # lead 1..24 -> one day earlier, lead 25..30 -> two days earlier; never after the origin
    assert seasonal.iloc[0] == 48 + 1 - 24
    assert seasonal.iloc[24] == 48 + 25 - 48
    assert (seasonal <= 48).all()

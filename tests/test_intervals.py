import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.intervals import build_intervals, conformal_halfwidth, halfwidth_for_lead
from src.utils.config import settings
from tests.test_forecast_endpoint import _weather_from


def test_halfwidth_is_the_corrected_order_statistic():
    assert conformal_halfwidth(np.arange(1, 100), alpha=0.1) == 90.0
    assert conformal_halfwidth([-5, 5, 5, -5], alpha=0.1) == 5.0
    with pytest.raises(ValueError):
        conformal_halfwidth([])


def _errors(rng, n_per_fold=400, folds=3):
    frames = []
    for f in range(folds):
        lead = np.tile(np.arange(1, 73), n_per_fold // 72 + 1)[:n_per_fold]
        # Error grows with lead time, so wider horizons must get wider intervals.
        err = np.abs(rng.normal(0, 1, n_per_fold)) * (10 + lead)
        frames.append(pd.DataFrame({"solar": err, "fold": f}, index=lead))
    return pd.concat(frames)


def test_intervals_widen_with_lead_time_and_hold_out_coverage():
    errors = _errors(np.random.default_rng(0), n_per_fold=7200)
    out = build_intervals(errors, ["solar"], [(1, 6), (7, 24), (25, 72)], horizon=72)
    hw = [out["targets"]["solar"][k]["halfwidth_mw"] for k in ("1-6", "7-24", "25-72")]
    assert hw == sorted(hw) and hw[0] < hw[-1]
    assert out["held_out_fold"] == 2 and out["calibration_folds_for_coverage"] == [0, 1]
    for b in out["targets"]["solar"].values():
        assert b["held_out_coverage"] == pytest.approx(0.9, abs=0.04)


def test_halfwidth_for_lead_lookup():
    intervals = {"targets": {"solar": {"1": {"lo": 1, "hi": 1, "halfwidth_mw": 5.0},
                                       "2-6": {"lo": 2, "hi": 6, "halfwidth_mw": 9.0}}}}
    assert halfwidth_for_lead(intervals, "solar", 1) == 5.0
    assert halfwidth_for_lead(intervals, "solar", 4) == 9.0
    assert halfwidth_for_lead(intervals, "solar", 40) is None
    assert halfwidth_for_lead(intervals, "wind_onshore", 1) is None


def test_forecast_includes_clipped_intervals_when_available(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", _weather_from(routes.synthetic_future))
    buckets = {t: {"all": {"lo": 1, "hi": 72, "halfwidth_mw": 1e9}} for t in ("wind_onshore", "wind_offshore", "solar")}
    (settings.MODELS_DIR / "forecast_intervals.json").write_text(json.dumps({"alpha": 0.1, "targets": buckets}))

    r = routes.get_forecast(hours_ahead=3)[0]
    assert r.interval_confidence == pytest.approx(0.9)
    assert r.solar_lower_mw == 0.0  # clipped: generation cannot be negative
    assert r.solar_upper_mw == pytest.approx(r.solar_mw + 1e9)


def test_forecast_has_no_interval_fields_without_the_file(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", _weather_from(routes.synthetic_future))
    r = routes.get_forecast(hours_ahead=2)[0]
    assert r.interval_confidence is None and r.solar_lower_mw is None


def test_dispatch_endpoint_reduces_deviation(synthetic_history, monkeypatch):
    routes = synthetic_history
    monkeypatch.setattr(routes.WeatherClient, "fetch_forecast", _weather_from(routes.synthetic_future))
    out = routes.get_dispatch(hours_ahead=24, power_mw=500.0, capacity_mwh=2000.0, round_trip_efficiency=0.9)
    assert len(out["schedule"]) == 24
    assert out["deviation_mwh_after"] <= out["deviation_mwh_before"]
    assert out["schedule"][0]["soc_mwh"] >= 0

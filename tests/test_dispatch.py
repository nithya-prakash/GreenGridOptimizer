import numpy as np
import pandas as pd
import pytest

from src.optimization.dispatch import Battery, optimize_dispatch


def _series(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC"), dtype=float)


def test_a_flat_forecast_needs_no_battery_action():
    out = optimize_dispatch(_series([100] * 12), Battery(power_mw=50, capacity_mwh=100))
    assert out["deviation_mwh_before"] == 0
    assert out["deviation_mwh_after"] == pytest.approx(0, abs=1e-6)


def test_battery_reduces_deviation_from_the_block():
    gen = _series([0, 0, 0, 0, 200, 200, 200, 200])
    out = optimize_dispatch(gen, Battery(power_mw=100, capacity_mwh=400, round_trip_efficiency=1.0, initial_soc_fraction=1.0))
    # Perfect battery, enough capacity: the block (mean 100) is met exactly.
    assert out["deviation_mwh_after"] == pytest.approx(0, abs=1e-6)
    assert out["deviation_reduction_pct"] == pytest.approx(100.0)


def test_schedule_respects_power_and_soc_limits():
    rng = np.random.default_rng(1)
    gen = _series(rng.uniform(0, 500, 48))
    bat = Battery(power_mw=80, capacity_mwh=200)
    s = optimize_dispatch(gen, bat)["schedule"]
    assert (s["charge_mw"] <= 80 + 1e-6).all() and (s["discharge_mw"] <= 80 + 1e-6).all()
    assert (s["soc_mwh"] >= -1e-6).all() and (s["soc_mwh"] <= 200 + 1e-6).all()
    assert (s["delivered_mw"] - (s["generation_mw"] - s["charge_mw"] + s["discharge_mw"])).abs().max() < 1e-6


def test_losses_mean_a_lossy_battery_helps_less_than_a_perfect_one():
    gen = _series([0, 0, 0, 0, 200, 200, 200, 200])
    perfect = optimize_dispatch(gen, Battery(100, 400, round_trip_efficiency=1.0))
    lossy = optimize_dispatch(gen, Battery(100, 400, round_trip_efficiency=0.8))
    assert lossy["deviation_mwh_after"] > perfect["deviation_mwh_after"]
    assert lossy["energy_lost_mwh"] > 0


def test_battery_ends_at_least_as_full_as_it_started():
    gen = _series([0, 0, 0, 0, 200, 200, 200, 200])
    bat = Battery(power_mw=100, capacity_mwh=400, initial_soc_fraction=0.5)
    s = optimize_dispatch(gen, bat)["schedule"]
    assert s["soc_mwh"].iloc[-1] >= 200 - 1e-6


def test_tiny_battery_gives_a_small_improvement():
    gen = _series([0, 200] * 12)
    out = optimize_dispatch(gen, Battery(power_mw=10, capacity_mwh=10))
    assert 0 < out["deviation_reduction_pct"] < 20


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        Battery(power_mw=0, capacity_mwh=10)
    with pytest.raises(ValueError):
        optimize_dispatch(_series([]), Battery(10, 10))

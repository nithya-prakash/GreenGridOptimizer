from datetime import timedelta
from typing import Dict, Any, Optional
import pandas as pd

from src.features.feature_engineering import (
    compute_predictor_features, GENERATION_COLUMNS, REQUIRED_HISTORY_HOURS,
)
from src.models.solar_direct import predict_solar_direct

GAP_FLAG_COLUMNS = ["is_generation_gap", "has_long_gap"]
# First lead (hours) at which solar comes from the direct model rather than the 1h model.
SOLAR_DIRECT_FROM_LEAD = 2
# Deep night: solar is currently below this (MW) and no radiation is forecast for
# the target hour. On 655 such hours in the data, actual solar never exceeded
# ~10 MW, but the 1h model sometimes predicted hundreds of MW there.
NIGHT_SOLAR_THRESHOLD_MW = 10.0


def recursive_forecast(models: Dict[str, Any], history: pd.DataFrame,
                       future_weather: pd.DataFrame, hours_ahead: int,
                       solar_direct_model: Optional[Any] = None) -> pd.DataFrame:
    """
    Recursive multi-step forecast shared by the /forecast endpoint and the
    backtest, so what gets evaluated is exactly what gets served.

    Each model predicts its target at t+1 from features known at t plus the
    forecast weather for t+1. To go further than 1h, each step's three
    predictions are written back as that hour's "actual" generation for the
    next step.

    If `solar_direct_model` is given, solar for leads 2..hours_ahead comes from the
    direct multi-horizon model (src/models/solar_direct.py) instead of recursion —
    recursive solar compounded its own errors and lost to a seasonal-naive
    baseline beyond ~6h. Lead 1 still uses the 1h model. The direct values are
    written back into history, so the wind models' solar lag features see them.

    history: raw hourly frame (generation + weather columns + gap flags) whose
             last row is the forecast origin.
    future_weather: weather columns covering origin+1h .. origin+hours_ahead.
    Returns a frame indexed by predicted timestamp with one column per target.
    """
    expected_cols = {t: getattr(m, "feature_names_in_", None) for t, m in models.items()}
    weather_cols = [c for c in history.columns if c not in GENERATION_COLUMNS and c not in GAP_FLAG_COLUMNS]
    history = history.copy()
    origin = history.index.max()
    predictions = []
    solar_direct = (
        predict_solar_direct(solar_direct_model, history, future_weather, hours_ahead)
        if solar_direct_model is not None and hours_ahead >= SOLAR_DIRECT_FROM_LEAD else None
    )

    for step in range(hours_ahead):
        current_ts = origin + timedelta(hours=step)
        predict_ts = current_ts + timedelta(hours=1)

        # Append the target hour with its forecast weather (generation unknown), so
        # current_ts's *_next_1h features are filled in. Built as a dict -> DataFrame
        # so pandas infers per-column dtypes: the gap flags are genuinely bool, and a
        # float-cast copy of them would upcast the column to `object`, which XGBoost's
        # predict() rejects.
        next_row = {col: float(future_weather.loc[predict_ts, col]) for col in weather_cols}
        next_row.update({t: float("nan") for t in GENERATION_COLUMNS})
        next_row.update({flag: False for flag in GAP_FLAG_COLUMNS if flag in history.columns})
        history = pd.concat([history, pd.DataFrame([next_row], index=[predict_ts])[history.columns]])

        # Only the trailing window affects current_ts's lag/rolling features.
        featured = compute_predictor_features(history.tail(REQUIRED_HISTORY_HOURS + 1))
        X = featured.loc[[current_ts]]

        preds = {}
        for target in GENERATION_COLUMNS:
            cols = expected_cols[target]
            X_target = X.reindex(columns=list(cols)) if cols is not None else X
            # Generation can't be negative; tree ensembles can dip slightly below 0 near
            # zero output (night-time solar, calm-wind hours).
            preds[target] = max(0.0, float(models[target].predict(X_target)[0]))
        if solar_direct is not None and step + 1 >= SOLAR_DIRECT_FROM_LEAD:
            preds["solar"] = float(solar_direct.loc[predict_ts])
        elif (history.loc[current_ts, "solar"] < NIGHT_SOLAR_THRESHOLD_MW
              and future_weather.loc[predict_ts, "shortwave_radiation"] == 0):
            preds["solar"] = 0.0

        for target, value in preds.items():
            history.loc[predict_ts, target] = value
        predictions.append({"timestamp": predict_ts, **preds})

    return pd.DataFrame(predictions).set_index("timestamp")

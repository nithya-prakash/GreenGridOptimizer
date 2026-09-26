"""
Direct multi-horizon solar model.

The recursive setup feeds the model's own solar predictions back in as lags, so
solar errors compound and, beyond ~6h, it lost to a "same hour yesterday"
baseline. Solar is driven almost entirely by radiation at the target hour, which
is known from the weather forecast, so this model predicts solar at origin+lead
*directly* — one pooled model over all lead times, with `lead` as a feature:

- forecast weather at the target hour (radiation components, cloud, temperature)
- calendar position of the target hour
- the seasonal-naive value (same hour on the last fully observed day) and the
  radiation at that time, so the model can rescale a known day by today's sun
- solar at the origin and the max over the 24h before it (current capacity level)

Nothing after the origin is used except the weather forecast.
"""
from datetime import timedelta
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.utils.logger import log

TARGET_WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover", "temperature_2m"]
DIRECT_FEATURES = (
    ["lead", "target_hour", "target_dayofyear"]
    + [f"{c}_target" for c in TARGET_WEATHER]
    + ["solar_seasonal_naive", "shortwave_radiation_seasonal_naive", "solar_origin", "solar_max_24h"]
)
# Fixed hyperparameters: the pooled (origin, lead) rows overlap heavily in time,
# so a naive TimeSeriesSplit search over them would leak across folds.
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, random_state=42, objective="reg:squarederror")


def build_direct_frame(raw: pd.DataFrame, weather: pd.DataFrame,
                       origins: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """One row per (origin, lead) for lead 1..horizon. `raw` supplies observed
    generation/radiation up to each origin; `weather` supplies forecast weather at
    the target hours (it may be `raw` itself when building training data)."""
    leads = np.arange(1, horizon + 1)
    lead = np.tile(leads, len(origins))
    origin_idx = origins.repeat(len(leads))  # keeps the timezone
    target_idx = origin_idx + pd.to_timedelta(lead, unit="h")
    naive_idx = target_idx - pd.to_timedelta(np.ceil(lead / 24) * 24, unit="h")
    solar_max_24h = raw["solar"].rolling(24, min_periods=1).max()

    frame = pd.DataFrame({
        "lead": lead,
        "target_hour": target_idx.hour,
        "target_dayofyear": target_idx.dayofyear,
        **{f"{c}_target": weather[c].reindex(target_idx).to_numpy() for c in TARGET_WEATHER},
        "solar_seasonal_naive": raw["solar"].reindex(naive_idx).to_numpy(),
        "shortwave_radiation_seasonal_naive": raw["shortwave_radiation"].reindex(naive_idx).to_numpy(),
        "solar_origin": raw["solar"].reindex(origin_idx).to_numpy(),
        "solar_max_24h": solar_max_24h.reindex(origin_idx).to_numpy(),
    }, index=pd.MultiIndex.from_arrays([origin_idx, target_idx], names=["origin", "target_ts"]))
    return frame[DIRECT_FEATURES]


def train_solar_direct(raw: pd.DataFrame, last_target_ts: pd.Timestamp, horizon: int,
                       origin_step: int = 2) -> XGBRegressor:
    """Trains on origins every `origin_step` hours whose whole horizon of targets is
    at or before `last_target_ts` (so a backtest fold never trains on its test
    window). Needs 3 days of history before the first origin."""
    first_origin = raw.index.min() + timedelta(hours=72)
    last_origin = last_target_ts - timedelta(hours=horizon)
    origins = raw.index[(raw.index >= first_origin) & (raw.index <= last_origin)][::origin_step]
    if len(origins) == 0:
        raise ValueError("Not enough history to train the direct solar model.")

    X = build_direct_frame(raw, raw, origins, horizon)
    y = raw["solar"].reindex(X.index.get_level_values("target_ts")).to_numpy()
    keep = ~np.isnan(y) & X.notna().all(axis=1).to_numpy()
    log.info(f"Training direct solar model on {keep.sum()} (origin, lead) rows from {len(origins)} origins")
    model = XGBRegressor(**PARAMS)
    model.fit(X[keep], y[keep])
    return model


def predict_solar_direct(model: XGBRegressor, history: pd.DataFrame,
                         future_weather: pd.DataFrame, horizon: int) -> pd.Series:
    """Solar forecast for origin+1..origin+horizon (origin = last row of history),
    clipped at 0. Returns a Series indexed by target timestamp."""
    origin = history.index.max()
    X = build_direct_frame(history, future_weather, pd.DatetimeIndex([origin]), horizon)
    preds = np.clip(model.predict(X), 0, None)
    return pd.Series(preds, index=X.index.get_level_values("target_ts"), name="solar")

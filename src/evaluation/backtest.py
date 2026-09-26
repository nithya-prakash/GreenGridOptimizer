import json
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.logger import log
from src.utils.config import settings
from src.evaluation.metrics import calculate_metrics
from src.features.feature_engineering import GENERATION_COLUMNS
from src.models.xgboost_model import XGBoostForecaster
from src.models.recursive import recursive_forecast, GAP_FLAG_COLUMNS
from src.models.solar_direct import train_solar_direct, predict_solar_direct
from src.ingestion.weather_client import WeatherClient

# Lead-time buckets (hours ahead, inclusive) for reporting recursive-forecast error.
HORIZON_BUCKETS = [(1, 1), (2, 6), (7, 24), (25, 48), (49, 72)]


def _fold_bounds(n_rows: int, n_folds: int, test_size: int):
    """Expanding-window fold boundaries: each fold's test window is the `test_size`
    hours immediately after the previous fold's, so later folds train on strictly
    more history. Returns (train_end, test_end) index pairs."""
    first_test_start = n_rows - n_folds * test_size
    for fold in range(n_folds):
        train_end = first_test_start + fold * test_size
        test_end = train_end + test_size
        yield train_end, test_end


def _bucket_label(lo: int, hi: int) -> str:
    return f"{lo}h" if lo == hi else f"{lo}-{hi}h"


def _forecast_origins(test_index: pd.DatetimeIndex, raw_end: pd.Timestamp, step: int, horizon: int):
    """Origins every `step` hours through the test window, keeping only those whose
    full `horizon` of actuals exists in the data."""
    return [ts for ts in test_index[::step] if ts + timedelta(hours=horizon) <= raw_end]


def lead_matched_weather(raw: pd.DataFrame, previous_runs: pd.DataFrame, origin: pd.Timestamp,
                         horizon: int) -> pd.DataFrame:
    """Future weather as it was forecast *at the origin*: lead L hours comes from the
    run issued ~L//24 days before the target hour (0 = the short-lead archive the
    models train on, 1-3 = Open-Meteo's previous-run forecasts). Without this, a
    72h-ahead forecast would be evaluated with near-perfect short-lead weather."""
    index = pd.date_range(origin + timedelta(hours=1), periods=horizon, freq="h")
    weather_cols = WeatherClient.WEATHER_COLUMNS
    out = raw.loc[index, weather_cols].copy()
    for i, ts in enumerate(index):
        days = (i + 1) // 24
        if days > 0:
            out.loc[ts, weather_cols] = previous_runs.loc[ts, [f"{c}__d{days}" for c in weather_cols]].to_numpy()
    return out


def baseline_forecasts(actuals: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> dict:
    """Naive reference forecasts, using only data up to `origin`:
    - persistence: every future hour = the value at the origin.
    - seasonal_naive: same hour of day on the last fully observed day
      (target hour minus ceil(lead/24) days).
    A model that doesn't beat these at a given lead time isn't adding value there."""
    index = pd.date_range(origin + timedelta(hours=1), periods=horizon, freq="h")
    persistence = pd.DataFrame([actuals.loc[origin, GENERATION_COLUMNS]] * horizon, index=index)
    leads = np.arange(1, horizon + 1)
    source_ts = index - pd.to_timedelta(np.ceil(leads / 24) * 24, unit="h")
    seasonal = actuals.loc[source_ts, GENERATION_COLUMNS].set_axis(index)
    return {"persistence": persistence, "seasonal_naive": seasonal}


def horizon_errors(forecast: pd.DataFrame, actuals: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Absolute error per target per lead time (hours ahead) for one forecast run."""
    lead = ((forecast.index - origin) / pd.Timedelta(hours=1)).astype(int)
    err = (forecast[GENERATION_COLUMNS] - actuals.loc[forecast.index, GENERATION_COLUMNS]).abs()
    err.index = lead
    return err


def run_backtest(n_folds: int = settings.BACKTEST_FOLDS, test_size: int = settings.TEST_SIZE_HOURS,
                 horizon: int = settings.MAX_FORECAST_HOURS_AHEAD,
                 origin_step: int = settings.BACKTEST_ORIGIN_STEP_HOURS):
    """
    Walk-forward (rolling-origin) backtest over `n_folds` expanding-window folds.

    Per fold, for each target, XGBoost hyperparameters are *re-tuned* on that fold's
    training window only (tuning once on the full dataset and reusing the params
    would let the tuning see the later folds' test weeks).

    Two things are reported per fold:
    - one_step: the 1h-ahead error on every hour of the test window, with the true
      lags. This is what the model is trained for, and it flatters a 72h forecast.
    - recursive: the actual served forecast (src/models/recursive.py, the same code
      /forecast runs) started from an origin every `origin_step` hours in the test
      window and run out to `horizon` hours, with errors grouped by lead time.
      Weather for the future hours comes from the archived historical forecasts
      (short-lead model runs), so long-lead weather error is still somewhat
      understated compared to a real 72h-ahead weather forecast.
    """
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    raw_path = settings.PROCESSED_DATA_DIR / "master_dataset.parquet"
    if not features_path.exists() or not raw_path.exists():
        log.error("Features/master dataset not found. Run the ingestion + feature engineering pipeline first.")
        return None

    df = pd.read_parquet(features_path)
    raw = pd.read_parquet(raw_path)
    raw_cols = [c for c in GENERATION_COLUMNS + WeatherClient.WEATHER_COLUMNS + GAP_FLAG_COLUMNS if c in raw.columns]
    raw = raw[raw_cols]

    previous_runs_path = settings.PROCESSED_DATA_DIR / "weather_previous_runs.parquet"
    previous_runs = pd.read_parquet(previous_runs_path) if previous_runs_path.exists() else None
    if previous_runs is None:
        log.warning("No lead-day weather (weather_previous_runs.parquet): future hours use short-lead "
                    "weather, which understates long-lead error. Re-run the ingestion pipeline.")

    n_rows = len(df)
    first_fold_train_size = n_rows - n_folds * test_size
    min_first_fold_train_size = test_size * 2  # at least 2 weeks of history to train the first fold
    if first_fold_train_size < min_first_fold_train_size:
        log.error(
            f"Not enough data for {n_folds} folds of {test_size}h each: only {first_fold_train_size} rows "
            f"would be left to train the first fold (need >= {min_first_fold_train_size}). "
            f"Reduce n_folds/test_size or ingest more history."
        )
        return None

    drop_cols = [c for c in df.columns if c.startswith('target_')]
    one_step = {t: [] for t in GENERATION_COLUMNS}
    recursive_errors = []  # one frame per forecast run: index = lead hours, cols = targets
    baseline_errors = {"persistence": [], "seasonal_naive": []}
    direct_solar_errors = []  # the direct solar model's own error at every lead (incl. 1h)
    fold_meta = []

    for fold, (train_end, test_end) in enumerate(_fold_bounds(n_rows, n_folds, test_size)):
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train, X_test = train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols)

        models = {}
        for target in GENERATION_COLUMNS:
            forecaster = XGBoostForecaster()
            forecaster.fit(X_train, train_df[f'target_{target}'])
            models[target] = forecaster.model
            metrics = calculate_metrics(test_df[f'target_{target}'].values, forecaster.predict(X_test))
            one_step[target].append({**metrics, "fold": fold, "best_params": forecaster.best_params})
            log.info(f"[{target}] fold {fold} one-step: MAE={metrics['MAE']:.1f} R2={metrics['R2']:.3f}")

        # The last target the 1h models trained on is train_df's last row + 1h; the
        # direct solar model gets exactly the same cut-off.
        last_train_target = train_df.index.max() + timedelta(hours=1)
        solar_direct = train_solar_direct(raw.loc[:last_train_target], last_train_target, horizon)

        origins = _forecast_origins(test_df.index, raw.index.max(), origin_step, horizon)
        if previous_runs is not None:
            origins = [o for o in origins
                       if previous_runs.reindex(pd.date_range(o + timedelta(hours=24), periods=horizon - 23, freq="h")).notna().all().all()]
        for origin in origins:
            history = raw.loc[:origin].tail(settings.FORECAST_HISTORY_HOURS)
            future = (lead_matched_weather(raw, previous_runs, origin, horizon) if previous_runs is not None
                      else raw.loc[origin + timedelta(hours=1): origin + timedelta(hours=horizon)])
            forecast = recursive_forecast(models, history, future, horizon, solar_direct_model=solar_direct)
            direct = predict_solar_direct(solar_direct, history, future, horizon).to_frame()
            for col in ("wind_onshore", "wind_offshore"):
                direct[col] = float("nan")
            direct_err = horizon_errors(direct, raw, origin)
            direct_err["fold"] = fold
            direct_solar_errors.append(direct_err)
            errors = horizon_errors(forecast, raw, origin)
            errors["fold"] = fold
            recursive_errors.append(errors)
            for name, baseline in baseline_forecasts(raw, origin, horizon).items():
                baseline_errors[name].append(horizon_errors(baseline, raw, origin))

        fold_meta.append({
            "fold": fold,
            "train_size": len(train_df),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
            "recursive_origins": len(origins),
        })
        log.info(f"Fold {fold}: {len(origins)} recursive {horizon}h forecast runs evaluated.")

    all_errors = pd.concat(recursive_errors) if recursive_errors else pd.DataFrame()
    all_baseline_errors = {k: pd.concat(v) for k, v in baseline_errors.items() if v}
    summary = {
        "folds": fold_meta, "horizon_hours": horizon, "origin_step_hours": origin_step,
        "future_weather": ("lead-matched: forecasts issued ~lead//24 days before each target hour (Open-Meteo previous runs)"
                           if previous_runs is not None else "short-lead archived forecasts (optimistic at long leads)"),
        "targets": {},
    }
    if direct_solar_errors:
        d = pd.concat(direct_solar_errors)
        summary["solar_direct_model_mae_by_horizon"] = {
            _bucket_label(lo, min(hi, horizon)): float(d[(d.index >= lo) & (d.index <= min(hi, horizon))]["solar"].mean())
            for lo, hi in HORIZON_BUCKETS if lo <= horizon
        }
    for target in GENERATION_COLUMNS:
        folds = one_step[target]
        by_horizon = {}
        for lo, hi in HORIZON_BUCKETS:
            if all_errors.empty or lo > horizon:
                continue
            bucket = all_errors[(all_errors.index >= lo) & (all_errors.index <= min(hi, horizon))]
            per_fold = bucket.groupby("fold")[target].mean()
            entry = {
                "MAE_mean": float(bucket[target].mean()),
                "MAE_std_across_folds": float(per_fold.std(ddof=0)),
                "n_errors": int(len(bucket)),
            }
            for name, errs in all_baseline_errors.items():
                b = errs[(errs.index >= lo) & (errs.index <= min(hi, horizon))]
                entry[f"{name}_MAE"] = float(b[target].mean())
            by_horizon[_bucket_label(lo, min(hi, horizon))] = entry
        summary["targets"][target] = {
            "one_step": {
                "folds": folds,
                "MAE_mean": float(np.mean([r["MAE"] for r in folds])), "MAE_std": float(np.std([r["MAE"] for r in folds])),
                "RMSE_mean": float(np.mean([r["RMSE"] for r in folds])), "RMSE_std": float(np.std([r["RMSE"] for r in folds])),
                "R2_mean": float(np.mean([r["R2"] for r in folds])), "R2_std": float(np.std([r["R2"] for r in folds])),
            },
            "recursive_mae_by_horizon": by_horizon,
            "mean_actual_mw": float(raw[target].mean()),
        }
        log.info(f"{target} MAE by horizon (model / persistence / seasonal-naive): " +
                 ", ".join(f"{k}={v['MAE_mean']:.0f}/{v['persistence_MAE']:.0f}/{v['seasonal_naive_MAE']:.0f}"
                           for k, v in by_horizon.items()))
        _plot_target(target, folds, by_horizon)

    output_path = settings.MODELS_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Backtest summary saved to {output_path}")
    return summary


def _plot_target(target: str, fold_results: list, by_horizon: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    folds = [r["fold"] for r in fold_results]
    maes = [r["MAE"] for r in fold_results]
    ax1.bar(folds, maes, color="#4C72B0")
    ax1.axhline(np.mean(maes), color="r", linestyle="--", label=f"mean={np.mean(maes):.1f}")
    ax1.set_title(f"{target}: 1h-ahead MAE per fold")
    ax1.set_xlabel("Fold (earlier -> later in time)")
    ax1.set_ylabel("MAE (MW)")
    ax1.legend()

    labels = list(by_horizon.keys())
    values = [by_horizon[k]["MAE_mean"] for k in labels]
    errs = [by_horizon[k]["MAE_std_across_folds"] for k in labels]
    x = np.arange(len(labels))
    ax2.bar(x - 0.25, values, 0.25, yerr=errs, color="#DD8452", capsize=4, label="model (recursive)")
    ax2.bar(x, [by_horizon[k]["persistence_MAE"] for k in labels], 0.25, color="#8C8C8C", label="persistence")
    ax2.bar(x + 0.25, [by_horizon[k]["seasonal_naive_MAE"] for k in labels], 0.25, color="#C4C4C4", label="seasonal naive (24h)")
    ax2.set_xticks(x, labels)
    ax2.legend()
    ax2.set_title(f"{target}: forecast MAE by lead time")
    ax2.set_xlabel("Hours ahead")
    ax2.set_ylabel("MAE (MW)")

    plt.tight_layout()
    plt.savefig(settings.MODELS_DIR / f"backtest_{target}.png")
    plt.close()


if __name__ == "__main__":
    run_backtest()

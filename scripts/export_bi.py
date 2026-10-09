"""Star-schema CSV export for Power BI / Tableau.

    python -m scripts.export_bi        # writes bi/*.csv

Tables: dim_target, fact_generation_hourly, fact_backtest_fold,
fact_backtest_horizon, fact_model_runs. Everything comes from the processed
dataset, models/backtest_results.json and the MLflow tracking DB; nothing is
simulated. MAPE is deliberately not exported: solar is ~0 at night, which makes
MAPE meaningless (the raw backtest reports values in the thousands of percent).
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

OUT = Path("bi")
TARGETS = {"wind_onshore": "Wind onshore", "wind_offshore": "Wind offshore", "solar": "Solar"}
WEATHER = ["wind_speed_100m", "temperature_2m", "cloud_cover", "shortwave_radiation", "offshore_wind_speed_10m"]


def backtest_tables(bt: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = {f["fold"]: f for f in bt["folds"]}
    folds, horizons, dims = [], [], []
    for target, t in bt["targets"].items():
        dims.append({"target": target, "label": TARGETS.get(target, target), "mean_actual_mw": round(t["mean_actual_mw"], 1)})
        for f in t["one_step"]["folds"]:
            w = windows.get(f["fold"], {})
            p = f.get("best_params", {})
            folds.append({"target": target, "fold": f["fold"], "mae_mw": f["MAE"], "rmse_mw": f["RMSE"], "r2": f["R2"],
                          "train_size_hours": w.get("train_size"), "test_start": w.get("test_start"), "test_end": w.get("test_end"),
                          "n_estimators": p.get("n_estimators"), "max_depth": p.get("max_depth"), "learning_rate": p.get("learning_rate")})
        for bucket, h in t["recursive_mae_by_horizon"].items():
            horizons.append({"target": target, "horizon": bucket, "mae_mw": h["MAE_mean"], "n_errors": h["n_errors"],
                             "persistence_mae_mw": h["persistence_MAE"], "seasonal_naive_mae_mw": h["seasonal_naive_MAE"],
                             "skill_vs_persistence": 1 - h["MAE_mean"] / h["persistence_MAE"],
                             "skill_vs_seasonal_naive": 1 - h["MAE_mean"] / h["seasonal_naive_MAE"]})
    return pd.DataFrame(dims), pd.DataFrame(folds), pd.DataFrame(horizons)


def model_runs(db: Path) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql("SELECT r.name AS run_name, r.start_time, m.key AS metric, m.value FROM runs r "
                         "JOIN metrics m ON r.run_uuid = m.run_uuid WHERE r.status = 'FINISHED'", conn)
    df = df[df.metric != "MAPE"]
    # Latest run per name, as the API's /metrics endpoint does.
    df = df.sort_values("start_time").drop_duplicates(["run_name", "metric"], keep="last")
    target = df.run_name.str.extract(r"(wind_onshore|wind_offshore|solar)", expand=False)
    df["target"] = target
    df["model"] = [n.replace(f"_{t}", "") if isinstance(t, str) else n for n, t in zip(df.run_name, target)]
    df["start_time"] = pd.to_datetime(df.start_time, unit="ms", utc=True)
    return df[["run_name", "model", "target", "metric", "value", "start_time"]]


def export() -> dict:
    OUT.mkdir(exist_ok=True)
    master = pd.read_parquet("data/processed/master_dataset.parquet")
    gen = master[[*TARGETS, *[c for c in WEATHER if c in master]]].reset_index().rename(columns={master.index.name or "index": "timestamp_utc"})
    dim, folds, horizons = backtest_tables(json.loads(Path("models/backtest_results.json").read_text()))
    tables = {"dim_target": dim, "fact_generation_hourly": gen, "fact_backtest_fold": folds,
              "fact_backtest_horizon": horizons, "fact_model_runs": model_runs(Path("mlruns/mlflow.db"))}
    for name, df in tables.items():
        df.round({c: 4 for c in df.select_dtypes("number")}).to_csv(OUT / f"{name}.csv", index=False)
    return {n: len(d) for n, d in tables.items()}


if __name__ == "__main__":
    print(export())

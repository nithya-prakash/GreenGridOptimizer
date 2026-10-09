import pandas as pd

from scripts.export_bi import backtest_tables, model_runs


def _bt():
    fold = {"MAE": 100.0, "RMSE": 150.0, "R2": 0.9, "fold": 0, "best_params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}}
    h = {"MAE_mean": 50.0, "n_errors": 10, "persistence_MAE": 100.0, "seasonal_naive_MAE": 200.0}
    return {"folds": [{"fold": 0, "train_size": 99, "test_start": "a", "test_end": "b"}],
            "targets": {"solar": {"mean_actual_mw": 7.5, "one_step": {"folds": [fold]}, "recursive_mae_by_horizon": {"1h": h}}}}


def test_backtest_tables_flatten_and_compute_skill():
    dim, folds, horizons = backtest_tables(_bt())
    assert dim.loc[0, "label"] == "Solar"
    assert folds.loc[0, ["mae_mw", "train_size_hours", "n_estimators"]].tolist() == [100.0, 99, 300]
    assert horizons.loc[0, "skill_vs_persistence"] == 0.5 and horizons.loc[0, "skill_vs_seasonal_naive"] == 0.75
    assert "mape" not in " ".join(folds.columns).lower()


def test_model_runs_drops_mape_keeps_latest_and_parses_target(tmp_path):
    import sqlite3
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as c:
        c.execute("create table runs (run_uuid text, name text, status text, start_time int)")
        c.execute("create table metrics (run_uuid text, key text, value real)")
        c.executemany("insert into runs values (?,?,?,?)", [("a", "XGBoost_solar", "FINISHED", 1), ("b", "XGBoost_solar", "FINISHED", 2)])
        c.executemany("insert into metrics values (?,?,?)", [("a", "MAE", 9.0), ("b", "MAE", 5.0), ("b", "MAPE", 1e4)])
    df = model_runs(db)
    assert df.to_dict("records")[0]["value"] == 5.0 and len(df) == 1
    assert df.loc[df.index[0], "target"] == "solar" and df.loc[df.index[0], "model"] == "XGBoost"
    assert isinstance(df.start_time.iloc[0], pd.Timestamp)

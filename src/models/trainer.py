import json
import os
from datetime import datetime, timezone
from sklearn.base import clone
import pandas as pd
import mlflow
import joblib
from src.utils.logger import log
from src.utils.config import settings
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.solar_direct import train_solar_direct, DIRECT_FEATURES
from src.evaluation.metrics import calculate_metrics
from src.evaluation.plots import plot_predictions, plot_residuals

def _atomic_dump(obj, path):
    """The API may be loading models while the refresh job retrains them."""
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def train_and_evaluate():
    log.info("Starting model training and evaluation pipeline...")
    
    # Load dataset
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        log.error("Features dataset not found.")
        return
        
    df = pd.read_parquet(features_path)
    
    # Train/Test Split (last `test_size_hours` for test — see configs/pipeline.yaml)
    test_size = settings.TEST_SIZE_HOURS
    if len(df) <= test_size:
        log.error("Not enough data for test split.")
        return
        
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    log.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(f"sqlite:///{settings.MLRUNS_DIR}/mlflow.db")
    mlflow.set_experiment("GreenGrid_Forecasting")
    
    target_cols = ['wind_onshore', 'wind_offshore', 'solar']
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "holdout_start": str(test_df.index.min()),
        "holdout_end": str(test_df.index.max()),
        "holdout_note": "1h-ahead metrics on the last holdout week; see backtest_results.json for multi-step error by horizon",
        "served_models_trained_through": str(df.index.max()),
        "served_models_note": "model selection and holdout metrics use the train split; the saved models are then refit on all data (incl. the holdout week) with the selected hyperparameters",
        "targets": {},
    }

    # Prepare feature list for XGBoost (drop target columns and current timestamp generation)
    # Target columns start with target_
    drop_cols = [c for c in df.columns if c.startswith('target_')]
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    for target in target_cols:
        best_model = None
        best_mae = float('inf')
        best_model_name = ""
        y_train = train_df[f'target_{target}']
        y_test = test_df[f'target_{target}']

        # --- Prophet (Base) ---
        with mlflow.start_run(run_name=f"Prophet_Base_{target}"):
            model = ProphetForecaster(use_regressors=False)
            model.fit(train_df, f'target_{target}')
            preds = model.predict(test_df)
            
            metrics = calculate_metrics(y_test.values, preds)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "Prophet")
            mlflow.log_param("use_regressors", False)
            
            plot_predictions(y_test, preds, "Prophet_Base", target, settings.MODELS_DIR)
            plot_residuals(y_test, preds, "Prophet_Base", target, settings.MODELS_DIR)
            log.info(f"Prophet Base metrics: {metrics}")
            
        # --- Prophet (Regressors) ---
        with mlflow.start_run(run_name=f"Prophet_Regressors_{target}"):
            model = ProphetForecaster(use_regressors=True)
            model.fit(train_df, f'target_{target}')
            preds = model.predict(test_df)
            
            metrics = calculate_metrics(y_test.values, preds)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "Prophet")
            mlflow.log_param("use_regressors", True)
            
            plot_predictions(y_test, preds, "Prophet_Reg", target, settings.MODELS_DIR)
            plot_residuals(y_test, preds, "Prophet_Reg", target, settings.MODELS_DIR)
            log.info(f"Prophet Reg metrics: {metrics}")
            
        # --- XGBoost ---
        with mlflow.start_run(run_name=f"XGBoost_{target}"):
            model = XGBoostForecaster()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            metrics = calculate_metrics(y_test.values, preds)
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.best_params)
            mlflow.log_param("model_type", "XGBoost")
            
            plot_predictions(y_test, preds, "XGBoost", target, settings.MODELS_DIR)
            plot_residuals(y_test, preds, "XGBoost", target, settings.MODELS_DIR)
            log.info(f"XGBoost metrics: {metrics}")

            # Only XGBoost is saved to production: it's the only model here that takes
            # the full engineered feature matrix, which is what the /forecast endpoint's
            # recursive step-by-step prediction is built around. Prophet is trained and
            # evaluated (logged/plotted above) for comparison but not served.
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_model = model.model
                best_model_name = "XGBoost"
                best_metrics = metrics
                best_params = model.best_params

        # Save the best model for this target, refit on ALL data (train + holdout week)
        # with the hyperparameters selected above: the holdout week has done its job
        # (evaluation), and the served model shouldn't be missing the latest week.
        if best_model is not None:
            final_model = clone(best_model)
            final_model.fit(df.drop(columns=drop_cols), df[f'target_{target}'])
            model_path = settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl"
            _atomic_dump(final_model, model_path)
            log.info(f"Saved {best_model_name} for {target} (holdout MAE {best_mae:.1f}), refit on all {len(df)} rows")
            metadata["targets"][target] = {
                "model": best_model_name,
                "model_class": type(best_model).__name__,
                "holdout_metrics_1h_ahead": {k: float(v) for k, v in best_metrics.items()},
                "best_params": best_params,
            }

    # Direct multi-horizon solar model used by /forecast for solar, trained on all
    # data (its out-of-sample accuracy is measured by the walk-forward backtest).
    raw = pd.read_parquet(settings.PROCESSED_DATA_DIR / "master_dataset.parquet")
    solar_direct = train_solar_direct(raw, raw.index.max(), settings.MAX_FORECAST_HOURS_AHEAD)
    _atomic_dump(solar_direct, settings.PRODUCTION_MODEL_DIR / "model_solar_direct.pkl")
    metadata["targets"]["solar_direct"] = {
        "model": "XGBoost (direct multi-horizon)",
        "model_class": type(solar_direct).__name__,
        "used_for": "solar in /forecast (see recursive.py for which leads)",
        "features": DIRECT_FEATURES,
    }
    log.info("Saved direct multi-horizon solar model.")

    tmp = settings.PRODUCTION_MODEL_DIR / "metadata.json.tmp"
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    os.replace(tmp, settings.PRODUCTION_MODEL_DIR / "metadata.json")
    log.info("Saved production model metadata.")

if __name__ == "__main__":
    train_and_evaluate()

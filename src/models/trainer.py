import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path
from src.utils.logger import log
from src.utils.config import settings
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.evaluation.metrics import calculate_metrics
from src.evaluation.plots import plot_predictions, plot_residuals

def train_and_evaluate():
    log.info("Starting model training and evaluation pipeline...")
    
    # Load dataset
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        log.error("Features dataset not found.")
        return
        
    df = pd.read_parquet(features_path)
    
    # Train/Test Split (last 7 days for test, assuming hourly data = 168 rows)
    test_size = 168
    if len(df) <= test_size:
        log.error("Not enough data for test split.")
        return
        
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    log.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(f"sqlite:///{settings.MLRUNS_DIR}/mlflow.db")
    mlflow.set_experiment("GreenGrid_Forecasting")
    
    target_cols = ['wind_onshore'] # For v1, let's just focus on wind_onshore to keep it manageable
    # Can expand to wind_offshore and solar
    
    # Prepare feature list for XGBoost (drop target columns and current timestamp generation)
    # Target columns start with target_
    drop_cols = [c for c in df.columns if c.startswith('target_')]
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    
    best_overall_model = None
    best_overall_mae = float('inf')
    best_model_name = ""
    
    for target in target_cols:
        y_train = train_df[f'target_{target}']
        y_test = test_df[f'target_{target}']
        y_test_actual = test_df[target].shift(-1).fillna(method='ffill') # actually y_test is exactly what we need
        
        # --- Prophet (Base) ---
        with mlflow.start_run(run_name=f"Prophet_Base_{target}"):
            model = ProphetForecaster(use_regressors=False)
            model.fit(train_df, target)
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
            model.fit(train_df, target)
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
            
            if metrics["MAE"] < best_overall_mae:
                best_overall_mae = metrics["MAE"]
                best_overall_model = model.model
                best_model_name = "XGBoost"
                
    # Save the best model
    if best_overall_model is not None:
        model_path = settings.PRODUCTION_MODEL_DIR / "model.pkl"
        joblib.dump(best_overall_model, model_path)
        log.info(f"Saved best model ({best_model_name}) to {model_path} with MAE: {best_overall_mae}")
        
if __name__ == "__main__":
    train_and_evaluate()

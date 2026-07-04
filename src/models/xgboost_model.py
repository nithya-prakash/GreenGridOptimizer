import pandas as pd
from src.utils.logger import log

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    log.warning(f"Failed to import xgboost (likely missing libomp on Mac). Falling back to HistGradientBoostingRegressor: {e}")
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    HAS_XGB = False

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

class XGBoostForecaster:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.has_xgb = HAS_XGB
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        log.info(f"Training XGBoost model (using actual xgboost: {self.has_xgb})...")
        
        # Define hyperparameter search space
        if self.has_xgb:
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
        else:
            param_dist = {
                'max_iter': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_features': [0.8, 1.0]
            }
            xgb = XGBRegressor(random_state=42)
        
        # TimeSeriesSplit for CV
        # Using 3 splits to save time during RandomizedSearchCV
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=5, # Keep it small for fast training in v1
            scoring='neg_mean_absolute_error',
            cv=tscv,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        log.info(f"XGBoost training complete. Best params: {self.best_params}")
        
    def predict(self, X_test: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X_test)

import pandas as pd
from prophet import Prophet
from src.utils.logger import log

class ProphetForecaster:
    def __init__(self, use_regressors: bool = False):
        self.use_regressors = use_regressors
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        self.regressors = [
            'wind_speed_10m', 'wind_speed_100m', 'temperature_2m',
            'cloud_cover', 'shortwave_radiation'
        ]
        if self.use_regressors:
            for reg in self.regressors:
                self.model.add_regressor(reg)
                
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare dataframe for Prophet (ds and y columns)."""
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index.tz_localize(None) # Prophet doesn't like timezone aware datetime
        prophet_df['y'] = df[target_col].values
        
        if self.use_regressors:
            for reg in self.regressors:
                prophet_df[reg] = df[reg].values
                
        return prophet_df
        
    def fit(self, X_train: pd.DataFrame, target_col: str):
        log.info(f"Training Prophet model (regressors={self.use_regressors}) on {target_col}")
        train_df = self._prepare_data(X_train, target_col)
        self.model.fit(train_df)
        
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        # We don't need a target column for prediction data frame, just regressors if any
        test_df = pd.DataFrame()
        test_df['ds'] = X_test.index.tz_localize(None)
        if self.use_regressors:
            for reg in self.regressors:
                test_df[reg] = X_test[reg].values
                
        forecast = self.model.predict(test_df)
        return forecast['yhat'].values

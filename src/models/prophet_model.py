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
        # Weather at the target hour (t+1), matching what XGBoost gets.
        self.regressors = [
            'wind_speed_10m_next_1h', 'wind_speed_100m_next_1h', 'temperature_2m_next_1h',
            'cloud_cover_next_1h', 'shortwave_radiation_next_1h', 'offshore_wind_speed_100m_next_1h'
        ]
        if self.use_regressors:
            for reg in self.regressors:
                self.model.add_regressor(reg)
                
    @staticmethod
    def _target_hours(df: pd.DataFrame) -> pd.Series:
        # Each feature row at t predicts generation at t+1, so Prophet's ds must be
        # t+1 (it models y as a function of ds). Using ds=t while scoring against the
        # t+1 target compared Prophet against XGBoost off by one hour.
        return (df.index + pd.Timedelta(hours=1)).tz_localize(None)  # Prophet doesn't accept tz-aware ds

    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare dataframe for Prophet (ds and y columns). target_col is the
        t+1 target column, e.g. 'target_wind_onshore'."""
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = self._target_hours(df)
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
        test_df['ds'] = self._target_hours(X_test)
        if self.use_regressors:
            for reg in self.regressors:
                test_df[reg] = X_test[reg].values
                
        forecast = self.model.predict(test_df)
        return forecast['yhat'].values

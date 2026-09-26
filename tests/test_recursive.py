import numpy as np
import pandas as pd

from src.features.feature_engineering import create_features
from src.models.recursive import recursive_forecast
from src.ingestion.weather_client import WeatherClient
from tests.conftest import synthetic_raw


def test_first_step_matches_training_features_exactly(synthetic_history):
    """Train/serve parity: the recursive forecaster's 1h-ahead prediction must equal
    the model's prediction on the row create_features() builds for training, i.e.
    serving computes identical features (including the *_next_1h weather) from the
    trailing window it uses."""
    routes = synthetic_history
    models = routes.get_models()
    full = routes.synthetic_raw_full
    origin = full.index[399]

    forecast = recursive_forecast(models, full.loc[:origin], full.loc[origin:].iloc[1:], hours_ahead=1)

    assert origin.hour not in range(0, 6), "parity check needs a daytime origin (night rule overrides solar)"
    training_row = create_features(full).loc[[origin]]
    X = training_row.drop(columns=[c for c in training_row.columns if c.startswith("target_")])
    for target, model in models.items():
        expected = model.predict(X.reindex(columns=list(model.feature_names_in_)))[0]
        assert np.isclose(forecast.iloc[0][target], max(0.0, expected))


def test_recursive_forecast_feeds_predictions_back(synthetic_history):
    routes = synthetic_history
    models = routes.get_models()
    raw = synthetic_raw(472)
    history, future = raw.iloc[:400], raw.iloc[400:][WeatherClient.WEATHER_COLUMNS]

    forecast = recursive_forecast(models, history, future, hours_ahead=48)

    assert len(forecast) == 48
    assert forecast.index[0] == history.index[-1] + pd.Timedelta(hours=1)
    assert forecast.notna().all().all()
    assert (forecast >= 0).all().all()


def test_first_hour_solar_is_zero_in_deep_night(synthetic_history):
    routes = synthetic_history
    models = routes.get_models()
    raw = synthetic_raw(472)
    # Pick an origin in the middle of the night (synthetic solar is 0 from 18h to 6h).
    origin = next(ts for ts in raw.index[300:] if ts.hour == 1)
    history = raw.loc[:origin]
    future = raw.loc[origin:].iloc[1:][WeatherClient.WEATHER_COLUMNS]
    assert history["solar"].iloc[-1] == 0 and future["shortwave_radiation"].iloc[0] == 0

    forecast = recursive_forecast(models, history, future, hours_ahead=1)

    assert forecast["solar"].iloc[0] == 0.0

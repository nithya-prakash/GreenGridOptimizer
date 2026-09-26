import numpy as np
import pandas as pd

from src.evaluation.backtest import lead_matched_weather
from src.ingestion.weather_client import WeatherClient
from tests.conftest import synthetic_raw


def test_lead_matched_weather_uses_older_forecasts_for_longer_leads():
    raw = synthetic_raw(300)
    cols = WeatherClient.WEATHER_COLUMNS
    # Previous-run forecasts tagged with their issue day so we can see which was used.
    previous = pd.concat(
        [pd.DataFrame(float(d), index=raw.index, columns=[f"{c}__d{d}" for c in cols]) for d in (1, 2, 3)],
        axis=1,
    )
    origin = raw.index[100]

    out = lead_matched_weather(raw, previous, origin, horizon=72)

    lead = np.arange(1, 73)
    expected_day = lead // 24
    for i, day in enumerate(expected_day):
        ts = out.index[i]
        if day == 0:
            assert out.loc[ts, "shortwave_radiation"] == raw.loc[ts, "shortwave_radiation"]
        else:
            assert (out.loc[ts, cols] == day).all()
    assert out.index[0] == origin + pd.Timedelta(hours=1) and len(out) == 72

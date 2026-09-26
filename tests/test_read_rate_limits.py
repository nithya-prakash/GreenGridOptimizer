from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.config import settings


def test_forecast_is_rate_limited_per_client(monkeypatch):
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "FORECAST_RATE_LIMIT_PER_MINUTE", 2)
    client = TestClient(app)

    # hours_ahead=500 is rejected (400) by the handler, but only after the limiter ran.
    codes = [client.get("/forecast?hours_ahead=500").status_code for _ in range(3)]
    assert codes == [400, 400, 429]


def test_read_endpoints_share_a_limit_separate_from_forecast(monkeypatch):
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "READ_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(settings, "FORECAST_RATE_LIMIT_PER_MINUTE", 100)
    client = TestClient(app)

    codes = [client.get("/historical?limit=0").status_code for _ in range(2)]
    codes.append(client.get("/backtest").status_code)
    assert codes == [422, 422, 429]
    assert client.get("/forecast?hours_ahead=500").status_code == 400  # separate bucket
    assert client.get("/health").status_code == 200  # never limited


def test_trusted_caller_is_not_limited(monkeypatch):
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "s3cret")
    monkeypatch.setattr(settings, "FORECAST_RATE_LIMIT_PER_MINUTE", 1)
    client = TestClient(app)

    headers = {"X-API-Key": "s3cret"}
    codes = [client.get("/forecast?hours_ahead=500", headers=headers).status_code for _ in range(3)]
    assert codes == [400, 400, 400]
    # A wrong token gets the anonymous limit.
    wrong = [client.get("/forecast?hours_ahead=500", headers={"X-API-Key": "nope"}).status_code for _ in range(2)]
    assert wrong == [400, 429]

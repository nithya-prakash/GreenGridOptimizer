"""Load test:  locust -f loadtest/locustfile.py --host http://localhost:8000

Env: API_AUTH_TOKEN (optional) is sent as the trusted-caller token, which exempts
the read endpoints from the per-IP rate limit so the model and cache path is
measured rather than the limiter. Without it, 429s are counted as expected
throttling, not failures. /forecast is cached (computed once per cache window), so
its latency is mostly the cached path; the first call after a data/model change is slower.
"""
import os
import random

from locust import HttpUser, between, task

TOKEN = os.environ.get("API_AUTH_TOKEN", "")
HEADERS = {"X-API-Key": TOKEN} if TOKEN else {}


class Dashboard(HttpUser):
    wait_time = between(0.3, 1.5)

    def _get(self, path, name=None, **params):
        with self.client.get(path, params=params, headers=HEADERS, name=name or path, catch_response=True) as r:
            if r.status_code == 429:
                r.success()

    @task(5)
    def forecast(self):
        self._get("/forecast", name="/forecast", hours_ahead=random.choice([24, 48, 72]))

    @task(3)
    def historical(self):
        self._get("/historical", limit=168)

    @task(2)
    def explain(self):
        self._get("/forecast/explain")

    @task(2)
    def model_info(self):
        self._get("/model/info")

    @task(1)
    def backtest(self):
        self._get("/backtest")

    @task(1)
    def metrics(self):
        self._get("/metrics")

    @task(1)
    def health(self):
        self._get("/health")

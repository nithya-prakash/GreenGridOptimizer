"""
Keeps the served data and models current: re-ingests every
REFRESH_INTERVAL_HOURS and retrains once the models are RETRAIN_AFTER_DAYS old.
Run as the `updater` compose service (`python -m src.jobs.refresh`), or once with
`--once` (e.g. from cron).

A failed run leaves the previous dataset/models in place (the pipeline only
replaces files after validation, and writes them atomically), so the API keeps
serving the last good state and reports its age via /data/status.
"""
import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone

from src.utils.config import settings
from src.utils.logger import log
from src.ingestion.pipeline import run_ingestion_pipeline
from src.features.feature_engineering import run_feature_engineering


def models_age_days():
    path = settings.PRODUCTION_MODEL_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        trained_at = datetime.fromisoformat(json.load(f)["trained_at"])
    return (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400


def refresh_once() -> bool:
    end = date.today()
    start = end - timedelta(days=settings.REFRESH_HISTORY_DAYS)
    if not run_ingestion_pipeline(start, end, verbose=False):
        log.error("Refresh: ingestion failed; keeping the previous dataset.")
        return False
    run_feature_engineering()

    age = models_age_days()
    if age is None or age >= settings.RETRAIN_AFTER_DAYS:
        log.info(f"Refresh: models are {'missing' if age is None else f'{age:.1f} days old'}; retraining.")
        from src.models.trainer import train_and_evaluate  # heavy import (prophet), only when needed
        train_and_evaluate()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single refresh and exit.")
    args = parser.parse_args()
    if args.once:
        raise SystemExit(0 if refresh_once() else 1)
    while True:
        try:
            refresh_once()
        except Exception:
            log.exception("Refresh failed; will retry next interval.")
        log.info(f"Next refresh in {settings.REFRESH_INTERVAL_HOURS}h.")
        time.sleep(settings.REFRESH_INTERVAL_HOURS * 3600)

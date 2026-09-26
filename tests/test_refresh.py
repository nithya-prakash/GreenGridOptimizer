import json
from datetime import datetime, timedelta, timezone

import src.jobs.refresh as refresh
from src.utils.config import settings


def _point_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PRODUCTION_MODEL_DIR", tmp_path)


def test_models_age_days(tmp_path, monkeypatch):
    _point_models_dir(tmp_path, monkeypatch)
    assert refresh.models_age_days() is None
    trained = datetime.now(timezone.utc) - timedelta(days=3)
    (tmp_path / "metadata.json").write_text(json.dumps({"trained_at": trained.isoformat()}))
    assert 2.9 < refresh.models_age_days() < 3.1


def test_failed_ingestion_keeps_previous_state(tmp_path, monkeypatch):
    _point_models_dir(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(refresh, "run_ingestion_pipeline", lambda *a, **k: False)
    monkeypatch.setattr(refresh, "run_feature_engineering", lambda: calls.append("features"))

    assert refresh.refresh_once() is False
    assert calls == []  # nothing downstream runs on a failed ingestion


def test_fresh_models_are_not_retrained(tmp_path, monkeypatch):
    _point_models_dir(tmp_path, monkeypatch)
    (tmp_path / "metadata.json").write_text(json.dumps({"trained_at": datetime.now(timezone.utc).isoformat()}))
    monkeypatch.setattr(refresh, "run_ingestion_pipeline", lambda *a, **k: True)
    monkeypatch.setattr(refresh, "run_feature_engineering", lambda: None)
    import src.models.trainer as trainer
    monkeypatch.setattr(trainer, "train_and_evaluate", lambda: (_ for _ in ()).throw(AssertionError("retrained")))

    assert refresh.refresh_once() is True

import yaml

from src.utils.config import Config, settings


def test_load_pipeline_config_overrides_defaults(tmp_path):
    """Constructs a real Config pointed at a temp pipeline.yaml with values that
    deliberately differ from Config's hardcoded fallback defaults — proving the
    YAML is actually read and applied, not just that the defaults happen to match
    what's in the shipped configs/pipeline.yaml."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "pipeline.yaml").write_text(yaml.dump({
        "evaluation": {"test_size_hours": 999, "backtest_folds": 7},
        "forecast": {"history_hours": 111, "max_hours_ahead": 3},
        "chat": {"model": "some-other-model"},
    }))

    cfg = Config(
        CONFIGS_DIR=configs_dir,
        DATA_DIR=tmp_path / "data",
        RAW_DATA_DIR=tmp_path / "data" / "raw",
        PROCESSED_DATA_DIR=tmp_path / "data" / "processed",
        MODELS_DIR=tmp_path / "models",
        PRODUCTION_MODEL_DIR=tmp_path / "models" / "production",
        MLRUNS_DIR=tmp_path / "mlruns",
    )

    assert cfg.TEST_SIZE_HOURS == 999
    assert cfg.BACKTEST_FOLDS == 7
    assert cfg.FORECAST_HISTORY_HOURS == 111
    assert cfg.MAX_FORECAST_HOURS_AHEAD == 3
    assert cfg.CHAT_MODEL == "some-other-model"


def test_missing_pipeline_yaml_keeps_defaults(tmp_path):
    cfg = Config(
        CONFIGS_DIR=tmp_path / "nonexistent",
        DATA_DIR=tmp_path / "data",
        RAW_DATA_DIR=tmp_path / "data" / "raw",
        PROCESSED_DATA_DIR=tmp_path / "data" / "processed",
        MODELS_DIR=tmp_path / "models",
        PRODUCTION_MODEL_DIR=tmp_path / "models" / "production",
        MLRUNS_DIR=tmp_path / "mlruns",
    )
    assert cfg.TEST_SIZE_HOURS == 168
    assert cfg.CHAT_MODEL == "claude-opus-5"


def test_shipped_pipeline_yaml_matches_running_settings():
    # Regression check against accidental edits to the real configs/pipeline.yaml
    # that don't match what the rest of the codebase (trainer.py, backtest.py,
    # routes.py) was verified against.
    assert settings.TEST_SIZE_HOURS == 168
    assert settings.BACKTEST_FOLDS == 4
    assert settings.FORECAST_HISTORY_HOURS == 504
    assert settings.MAX_FORECAST_HOURS_AHEAD == 72
    assert settings.CHAT_MODEL == "claude-opus-5"


def test_shipped_pipeline_yaml_defines_multi_site_weather():
    assert len(settings.WEATHER_LAND_SITES) > 1
    assert len(settings.WEATHER_OFFSHORE_SITES) >= 1
    assert all({"lat", "lon"} <= set(site) for site in settings.WEATHER_LAND_SITES + settings.WEATHER_OFFSHORE_SITES)


def test_shipped_pipeline_yaml_defines_refresh_schedule():
    assert settings.REFRESH_INTERVAL_HOURS > 0
    assert settings.STALE_AFTER_HOURS > settings.REFRESH_INTERVAL_HOURS

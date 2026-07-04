"""Testes da configuração (Pydantic Settings + params.yaml)."""

from recsys.config import ROOT_DIR, Settings, load_params


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.data_dir == ROOT_DIR / "data"
    assert settings.mlflow_tracking_uri == ""


def test_settings_override_by_env(monkeypatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    settings = Settings(_env_file=None)
    assert settings.random_seed == 7
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


def test_params_yaml_has_pipeline_sections() -> None:
    params = load_params()
    assert set(params) >= {"featurize", "train", "evaluate"}
    assert params["train"]["negatives_per_positive"] >= 1

"""Testes da configuração Pydantic Settings."""

from recsys.config import ROOT_DIR, Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.embedding_dim == 64
    assert settings.data_dir == ROOT_DIR / "data"


def test_settings_override_by_env(monkeypatch) -> None:
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("LEARNING_RATE", "0.01")
    settings = Settings(_env_file=None)
    assert settings.random_seed == 7
    assert settings.learning_rate == 0.01

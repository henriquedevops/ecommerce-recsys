"""Configurações do projeto: ambiente (Pydantic Settings + ``.env``) e params (DVC).

Divisão deliberada de responsabilidades:
- ``Settings``/`.env` → configuração de **ambiente** (paths, URI do MLflow, seed).
- ``params.yaml``     → **hiperparâmetros de experimento**, rastreados pelo DVC
  (mudá-los invalida os stages do pipeline no próximo ``dvc repro``).
"""

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuração de ambiente: paths, MLflow e seed global."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"
    mlflow_tracking_uri: str = ""  # vazio = file store local (./mlruns)
    random_seed: int = 42


def load_params() -> dict:
    """Lê o ``params.yaml`` da raiz — hiperparâmetros rastreados pelo DVC."""
    return yaml.safe_load((ROOT_DIR / "params.yaml").read_text(encoding="utf-8"))


settings = Settings()

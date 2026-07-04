"""Configurações do projeto via Pydantic Settings + ``.env``.

Todos os valores têm defaults funcionais; qualquer variável definida no
``.env`` (ou no ambiente do processo) sobrescreve o default correspondente.
Centralizar a seed aqui garante o requisito de "seeds fixados".
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuração central: paths, MLflow, seed e hiperparâmetros."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"
    mlflow_tracking_uri: str = "http://localhost:5000"
    random_seed: int = 42
    embedding_dim: int = 64
    batch_size: int = 256
    learning_rate: float = 0.001


settings = Settings()

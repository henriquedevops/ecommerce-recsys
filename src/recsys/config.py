"""Configuração base: paths do projeto e seed global.

Na Etapa 2 este módulo evolui para Pydantic Settings + ``.env``.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
RANDOM_SEED = 42

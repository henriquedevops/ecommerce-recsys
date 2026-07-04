"""Valida o ambiente: imports críticos, versões e configuração carregada.

Uso: ``uv run python scripts/validate_env.py`` — retorna 0 se está tudo OK.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys

# módulo importável -> nome do pacote no PyPI (nem sempre coincidem)
REQUIRED = {
    "torch": "torch",
    "sklearn": "scikit-learn",
    "mlflow": "mlflow",
    "dvc": "dvc",
    "pandas": "pandas",
    "numpy": "numpy",
    "pydantic_settings": "pydantic-settings",
}


def find_missing_modules() -> list[str]:
    """Retorna os módulos obrigatórios que não estão instalados."""
    return [module for module in REQUIRED if importlib.util.find_spec(module) is None]


def print_versions() -> None:
    """Imprime a versão instalada de cada dependência obrigatória."""
    for package in REQUIRED.values():
        print(f"  {package}=={importlib.metadata.version(package)}")


def check_settings() -> None:
    """Instancia o Settings para garantir que config e .env carregam."""
    from recsys.config import settings

    print(f"  seed={settings.random_seed}  mlflow={settings.mlflow_tracking_uri}")


def main() -> int:
    """Roda todas as checagens de ambiente."""
    missing = find_missing_modules()
    if missing:
        print(f"[ERRO] Faltam modulos: {missing}. Rode: uv sync")
        return 1
    print(f"[OK] Python {sys.version.split()[0]}")
    print_versions()
    check_settings()
    print("[OK] Ambiente validado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

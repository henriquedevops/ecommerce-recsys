"""Registro do modelo no MLflow Model Registry (Staging -> Production).

Executável como ``python -m recsys.models.register``. Usa o run apontado
pelo checkpoint ``models/recsys.pt`` (gravado pelo stage ``train`` do DVC),
registra o artefato ``model`` como nova versão de ``ecommerce-recsys`` e a
promove seguindo o fluxo exigido: **None -> Staging -> Production**.

Requer um tracking store com suporte a Registry (ex.: sqlite) — o file
store puro (``./mlruns``) não suporta o Model Registry.
"""

from __future__ import annotations

import mlflow
import torch
from mlflow.tracking import MlflowClient

from recsys.config import settings

MODEL_NAME = "ecommerce-recsys"


def checkpoint_run_id() -> str:
    """Lê o run MLflow que produziu o checkpoint atual do pipeline."""
    checkpoint = torch.load(settings.models_dir / "recsys.pt", weights_only=True)
    run_id = checkpoint.get("mlflow_run_id")
    if not run_id:
        msg = "checkpoint sem mlflow_run_id — rode o stage train primeiro"
        raise ValueError(msg)
    return run_id


def promote(client: MlflowClient, version: str) -> None:
    """Promove a versão pelo fluxo Staging -> Production (arquiva anteriores)."""
    for stage in ("Staging", "Production"):
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        print(f"[OK] register: versao {version} -> {stage}")


def main() -> None:
    """Entrypoint: registra o modelo do checkpoint e promove a Production."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id = checkpoint_run_id()
    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    promote(MlflowClient(), result.version)
    print(f"[OK] register: {MODEL_NAME} v{result.version} em Production (run {run_id})")


if __name__ == "__main__":
    main()

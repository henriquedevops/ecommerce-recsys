"""Stage DVC ``evaluate``: métricas de ranking no conjunto de teste."""

from __future__ import annotations

import json

import mlflow
import pandas as pd
import torch

from recsys.config import load_params, settings
from recsys.evaluation.metrics import (
    heldout_ranks,
    ranking_metrics,
    seen_items_by_user,
)
from recsys.models.mlp import RecSysMLP


def load_model(checkpoint: dict) -> RecSysMLP:
    """Reconstrói o RecSysMLP a partir do checkpoint salvo pelo treino."""
    params = checkpoint["params"]
    model = RecSysMLP(
        checkpoint["n_users"],
        checkpoint["n_items"],
        embedding_dim=params["embedding_dim"],
        hidden_dims=tuple(params["hidden_dims"]),
        dropout=params["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def evaluate() -> dict[str, float]:
    """Calcula as métricas de teste e grava ``reports/metrics.json``.

    No teste, a máscara de itens vistos inclui train E val — o modelo não
    deve ser premiado por recomendar algo que o usuário já consumiu.
    """
    top_k = load_params()["evaluate"]["top_k"]
    checkpoint = torch.load(settings.models_dir / "recsys.pt", weights_only=True)
    model = load_model(checkpoint)

    processed = settings.data_dir / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    ranks = heldout_ranks(
        model, test, seen_items_by_user(train, val), checkpoint["n_items"]
    )
    metrics = ranking_metrics(ranks, k=top_k)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _log_to_mlflow(metrics, checkpoint.get("mlflow_run_id"))
    return metrics


def _log_to_mlflow(metrics: dict[str, float], run_id: str | None) -> None:
    """Anexa as métricas de teste ao MESMO run MLflow criado pelo treino."""
    if not run_id:
        return
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})


def main() -> None:
    """Entrypoint do stage: ``models/recsys.pt`` -> ``reports/metrics.json``."""
    metrics = evaluate()
    formatted = {name: round(value, 4) for name, value in metrics.items()}
    print(f"[OK] evaluate: {formatted}")


if __name__ == "__main__":
    main()

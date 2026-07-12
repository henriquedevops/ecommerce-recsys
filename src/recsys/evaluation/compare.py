"""Comparação MLP vs. baselines com 4 métricas de ranking (Top-K).

Executável como ``python -m recsys.evaluation.compare``: avalia popularidade,
item-KNN e o RecSysMLP treinado no MESMO protocolo (leave-one-out, ranking no
catálogo inteiro, máscara de itens vistos), grava ``reports/comparison.json``
e loga cada baseline como um run no MLflow.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

import mlflow
import numpy as np
import pandas as pd
import torch

import recsys.models.baselines  # noqa: F401  # registra os baselines na Factory
from recsys.config import load_params, settings
from recsys.evaluation.evaluate import load_model
from recsys.evaluation.metrics import (
    heldout_ranks,
    ranking_metrics,
    seen_items_by_user,
)
from recsys.models.factory import ModelFactory

if TYPE_CHECKING:
    from recsys.models.base import Model

EXPERIMENT_NAME = "ecommerce-recsys"


class CatalogScorer(Protocol):
    """Contrato mínimo para ranquear o catálogo inteiro de um usuário."""

    def catalog_scores(self, user_idx: int) -> np.ndarray:
        """Scores de todos os itens para o usuário."""
        ...


def heldout_ranks_baseline(
    model: CatalogScorer,
    heldout: pd.DataFrame,
    seen: dict[int, set[int]],
) -> np.ndarray:
    """Rank do item held-out por usuário para modelos com ``catalog_scores``.

    Mesmo protocolo de :func:`~recsys.evaluation.metrics.heldout_ranks`:
    máscara dos itens já vistos e contagem de scores acima do held-out.
    """
    users = heldout["user_idx"].to_numpy(np.int64)
    items = heldout["item_idx"].to_numpy(np.int64)
    ranks = np.empty(len(users), dtype=np.int64)
    for row, (user, item) in enumerate(zip(users, items, strict=True)):
        scores = model.catalog_scores(int(user)).copy()
        target = scores[int(item)]
        scores[list(seen.get(int(user), ()))] = -np.inf
        ranks[row] = int((scores > target).sum())
    return ranks


def _load_eval_data() -> dict:
    """Carrega splits processados, meta e a máscara de itens vistos."""
    processed = settings.data_dir / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    return {
        "train": train,
        "test": pd.read_parquet(processed / "test.parquet"),
        "seen": seen_items_by_user(train, val),
        "n_users": meta["n_users"],
        "n_items": meta["n_items"],
    }


def _fit_baseline(name: str, data: dict) -> Model:
    """Cria o baseline pela Factory e ajusta nas interações de treino."""
    kwargs = {"n_items": data["n_items"]}
    if name == "item_knn":
        kwargs["n_users"] = data["n_users"]
    return ModelFactory.create(name, **kwargs).fit(data["train"])


def _mlp_metrics(data: dict, top_k: int) -> dict[str, float]:
    """Avalia o checkpoint do RecSysMLP no mesmo protocolo dos baselines."""
    checkpoint = torch.load(settings.models_dir / "recsys.pt", weights_only=True)
    model = load_model(checkpoint)
    ranks = heldout_ranks(model, data["test"], data["seen"], data["n_items"])
    return ranking_metrics(ranks, k=top_k)


def _log_baseline_run(name: str, metrics: dict[str, float], top_k: int) -> None:
    """Registra o baseline como run MLflow (params + métricas de teste)."""
    with mlflow.start_run(run_name=f"baseline_{name}"):
        mlflow.log_params({"model": name, "top_k": top_k})
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})


def _format_table(comparison: dict[str, dict[str, float]]) -> str:
    """Tabela markdown modelo x métricas, pronta para README e Model Card."""
    metric_names = list(next(iter(comparison.values())))
    header = "| modelo | " + " | ".join(metric_names) + " |"
    sep = "|---" * (len(metric_names) + 1) + "|"
    rows = [
        "| " + name + " | " + " | ".join(f"{m[k]:.4f}" for k in metric_names) + " |"
        for name, m in comparison.items()
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    """Entrypoint: avalia baselines + MLP e grava ``reports/comparison.json``."""
    top_k = load_params()["evaluate"]["top_k"]
    data = _load_eval_data()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    comparison: dict[str, dict[str, float]] = {}
    for name in ("popularity", "item_knn"):
        baseline = _fit_baseline(name, data)
        ranks = heldout_ranks_baseline(baseline, data["test"], data["seen"])
        comparison[name] = ranking_metrics(ranks, k=top_k)
        _log_baseline_run(name, comparison[name], top_k)
    comparison["mlp"] = _mlp_metrics(data, top_k)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "comparison.json"
    out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(_format_table(comparison))
    print(f"[OK] compare: {out}")


if __name__ == "__main__":
    main()

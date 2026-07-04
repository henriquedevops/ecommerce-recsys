"""Stage DVC ``train``: treina o RecSysMLP com early stopping + MLflow."""

from __future__ import annotations

import json
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from recsys.config import load_params, settings
from recsys.evaluation.metrics import (
    heldout_ranks,
    ranking_metrics,
    seen_items_by_user,
)
from recsys.models.mlp import RecSysMLP

EXPERIMENT_NAME = "ecommerce-recsys"


def seed_everything(seed: int) -> None:
    """Fixa as seeds de random, numpy e torch (requisito de reprodutibilidade)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    """Interrompe o treino após ``patience`` épocas sem melhora na validação."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_value = float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = 0
        self.stale = 0

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Registra a época; retorna ``True`` quando o treino deve parar."""
        if value > self.best_value:
            self.best_value = value
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= self.patience


def sample_negatives(
    pos_users: np.ndarray,
    n_items: int,
    ratio: int,
    rng: np.random.Generator,
    seen: dict[int, set[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra ``ratio`` negativos por positivo, re-sorteando colisões.

    Sorteia itens uniformemente do catálogo; os que o usuário já viu são
    re-sorteados (2 rodadas — colisão residual < 0,1%, aceitável e rápido).
    """
    neg_users = np.repeat(pos_users, ratio)
    neg_items = rng.integers(0, n_items, size=neg_users.size)
    for _ in range(2):
        collisions = np.fromiter(
            (
                item in seen[user]
                for user, item in zip(neg_users, neg_items, strict=True)
            ),
            dtype=bool,
            count=neg_users.size,
        )
        if not collisions.any():
            break
        neg_items[collisions] = rng.integers(0, n_items, size=int(collisions.sum()))
    return neg_users, neg_items


def epoch_batches(data: dict, params: dict, rng: np.random.Generator):
    """Gera batches ``(users, items, labels)`` com negativos novos por época."""
    neg_users, neg_items = sample_negatives(
        data["pos_users"],
        data["n_items"],
        params["negatives_per_positive"],
        rng,
        data["seen"],
    )
    users = np.concatenate([data["pos_users"], neg_users])
    items = np.concatenate([data["pos_items"], neg_items])
    labels = np.concatenate([np.ones(data["pos_users"].size), np.zeros(neg_users.size)])
    order = rng.permutation(users.size)
    for start in range(0, users.size, params["batch_size"]):
        idx = order[start : start + params["batch_size"]]
        yield (
            torch.as_tensor(users[idx]),
            torch.as_tensor(items[idx]),
            torch.as_tensor(labels[idx], dtype=torch.float32),
        )


def train_one_epoch(model: nn.Module, optimizer, loss_fn, batches) -> float:
    """Roda uma época de treino; retorna a loss média ponderada."""
    model.train()
    total, count = 0.0, 0
    for users, items, labels in batches:
        optimizer.zero_grad()
        loss = loss_fn(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * labels.numel()
        count += labels.numel()
    return total / count


def fit(model: RecSysMLP, data: dict, params: dict) -> EarlyStopping:
    """Loop de treino com early stopping sobre o NDCG@10 de validação."""
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(settings.random_seed)
    stopper = EarlyStopping(params["patience"])
    for epoch in range(1, params["epochs"] + 1):
        loss = train_one_epoch(
            model, optimizer, loss_fn, epoch_batches(data, params, rng)
        )
        ranks = heldout_ranks(model, data["val"], data["seen"], data["n_items"])
        val_ndcg = ranking_metrics(ranks, k=10)["ndcg_at_10"]
        mlflow.log_metrics({"train_loss": loss, "val_ndcg_at_10": val_ndcg}, step=epoch)
        print(f"[epoch {epoch}] loss={loss:.4f} val_ndcg@10={val_ndcg:.4f}")
        if stopper.step(val_ndcg, model, epoch):
            print(f"[early stop] sem melhora ha {stopper.stale} epocas")
            break
    model.load_state_dict(stopper.best_state)
    return stopper


def _load_training_data() -> dict:
    """Carrega os parquets processados e monta as estruturas de treino."""
    processed = settings.data_dir / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    return {
        "pos_users": train["user_idx"].to_numpy(np.int64),
        "pos_items": train["item_idx"].to_numpy(np.int64),
        "val": pd.read_parquet(processed / "val.parquet"),
        "seen": seen_items_by_user(train),
        "n_users": meta["n_users"],
        "n_items": meta["n_items"],
    }


def _save_checkpoint(model: RecSysMLP, data: dict, params: dict, run_id: str) -> None:
    """Salva pesos + metadados em ``models/recsys.pt`` (lido pelo evaluate)."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": data["n_users"],
            "n_items": data["n_items"],
            "params": params,
            "mlflow_run_id": run_id,
        },
        settings.models_dir / "recsys.pt",
    )


def main() -> None:
    """Entrypoint do stage: ``processed/*`` -> ``models/recsys.pt`` + run MLflow."""
    params = load_params()["train"]
    seed_everything(settings.random_seed)
    data = _load_training_data()
    model = RecSysMLP(
        data["n_users"],
        data["n_items"],
        embedding_dim=params["embedding_dim"],
        hidden_dims=tuple(params["hidden_dims"]),
        dropout=params["dropout"],
    )
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="mlp_embedding") as run:
        mlflow.log_params({**params, "random_seed": settings.random_seed})
        stopper = fit(model, data, params)
        mlflow.log_metric("best_val_ndcg_at_10", stopper.best_value)
        mlflow.pytorch.log_model(model, "model")
        _save_checkpoint(model, data, params, run.info.run_id)
    print(
        f"[OK] train: melhor epoca={stopper.best_epoch} "
        f"val_ndcg@10={stopper.best_value:.4f} run_id={run.info.run_id}"
    )


if __name__ == "__main__":
    main()

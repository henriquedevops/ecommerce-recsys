"""Tuning do RecSysMLP: >= 3 runs MLflow variando hiperparâmetros.

Executável como ``python -m recsys.models.tune``. Cada configuração treina
com early stopping e vira um run próprio no MLflow (params + métricas por
época + melhor NDCG@10 de validação). Ao final, imprime o ranking das
configurações — a vencedora deve ser promovida ao ``params.yaml`` para o
``dvc repro`` gerar o modelo final versionado.
"""

from __future__ import annotations

import mlflow

from recsys.config import load_params, settings
from recsys.models.mlp import RecSysMLP
from recsys.models.train import (
    EXPERIMENT_NAME,
    _load_training_data,
    fit,
    seed_everything,
)

# Grade deliberadamente pequena (CPU): varia capacidade (embedding/hidden),
# regularização (dropout) e learning rate, com folga de épocas para o early
# stopping decidir — o treino baseline parou cedo demais (melhor época = 2).
SEARCH_SPACE: list[dict] = [
    {
        "embedding_dim": 64,
        "hidden_dims": [128, 64],
        "dropout": 0.3,
        "learning_rate": 0.001,
    },
    {
        "embedding_dim": 128,
        "hidden_dims": [256, 128],
        "dropout": 0.3,
        "learning_rate": 0.001,
    },
    {
        "embedding_dim": 64,
        "hidden_dims": [128, 64],
        "dropout": 0.2,
        "learning_rate": 0.0005,
    },
]
TUNE_OVERRIDES = {"epochs": 20, "patience": 3}


def run_config(config: dict, data: dict) -> float:
    """Treina uma configuração num run MLflow; retorna o melhor NDCG@10 val."""
    params = {**load_params()["train"], **TUNE_OVERRIDES, **config}
    seed_everything(settings.random_seed)
    model = RecSysMLP(
        data["n_users"],
        data["n_items"],
        embedding_dim=params["embedding_dim"],
        hidden_dims=tuple(params["hidden_dims"]),
        dropout=params["dropout"],
    )
    run_name = (
        f"tune_emb{params['embedding_dim']}"
        f"_lr{params['learning_rate']}_do{params['dropout']}"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "random_seed": settings.random_seed})
        stopper = fit(model, data, params)
        mlflow.log_metric("best_val_ndcg_at_10", stopper.best_value)
        mlflow.log_metric("best_epoch", stopper.best_epoch)
    return stopper.best_value


def main() -> None:
    """Entrypoint: roda a grade e imprime o ranking por NDCG@10 de validação."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = _load_training_data()

    results = []
    for config in SEARCH_SPACE:
        print(f"[tune] config={config}")
        results.append((run_config(config, data), config))

    results.sort(key=lambda pair: pair[0], reverse=True)
    for position, (value, config) in enumerate(results, start=1):
        print(f"[rank {position}] val_ndcg@10={value:.4f} <- {config}")
    print("[OK] tune: promover a config vencedora ao params.yaml + dvc repro")


if __name__ == "__main__":
    main()

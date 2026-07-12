"""API mínima de inferência (FastAPI) — bônus de deploy em nuvem.

Serve o checkpoint ``models/recsys.pt`` (embutido na imagem Docker, estágio
``serve``) com dois endpoints:

- ``GET /health``               → status e metadados do modelo carregado.
- ``GET /recommend/{user_idx}`` → Top-K itens para um usuário do treino.

Limitação deliberada (documentada no Model Card): a API pontua o catálogo
inteiro sem máscara de itens já vistos — o histórico fica no pipeline de
dados, não no checkpoint. Usuários fora do treino (cold-start) recebem 404.
"""

from __future__ import annotations

import torch
from fastapi import FastAPI, HTTPException, Query

from recsys.config import settings
from recsys.evaluation.evaluate import load_model

app = FastAPI(title="ecommerce-recsys", version="1.0.0")
state: dict = {}


@app.on_event("startup")
def load_checkpoint() -> None:
    """Carrega o checkpoint uma única vez, no boot do processo."""
    checkpoint = torch.load(settings.models_dir / "recsys.pt", weights_only=True)
    model = load_model(checkpoint)
    model.eval()
    state.update(
        model=model,
        n_users=checkpoint["n_users"],
        n_items=checkpoint["n_items"],
        run_id=checkpoint.get("mlflow_run_id", ""),
    )


@app.get("/health")
def health() -> dict:
    """Status do serviço e do modelo carregado."""
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "n_users": state.get("n_users"),
        "n_items": state.get("n_items"),
        "mlflow_run_id": state.get("run_id"),
    }


@app.get("/recommend/{user_idx}")
def recommend(user_idx: int, k: int = Query(default=10, ge=1, le=100)) -> dict:
    """Top-K itens (``item_idx``) para um usuário conhecido do treino."""
    if not 0 <= user_idx < state["n_users"]:
        raise HTTPException(status_code=404, detail="user_idx fora do treino")
    with torch.no_grad():
        users = torch.full((state["n_items"],), user_idx, dtype=torch.int64)
        scores = state["model"](users, torch.arange(state["n_items"]))
    top = torch.topk(scores, k)
    return {
        "user_idx": user_idx,
        "items": top.indices.tolist(),
        "scores": [round(float(s), 4) for s in top.values],
    }

"""Métricas de ranking Top-N para o protocolo leave-one-out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def seen_items_by_user(*frames: pd.DataFrame) -> dict[int, set[int]]:
    """Itens já consumidos por usuário (p/ mascarar ranking e sampling).

    Args:
        *frames: DataFrames com colunas ``user_idx`` e ``item_idx``.

    Returns:
        Dict ``user_idx -> {item_idx, ...}`` agregando todos os frames.
    """
    seen: dict[int, set[int]] = {}
    for frame in frames:
        users = frame["user_idx"].to_numpy()
        items = frame["item_idx"].to_numpy()
        for user, item in zip(users, items, strict=True):
            seen.setdefault(int(user), set()).add(int(item))
    return seen


def heldout_ranks(
    model: nn.Module,
    heldout: pd.DataFrame,
    seen: dict[int, set[int]],
    n_items: int,
    batch_users: int = 256,
) -> np.ndarray:
    """Rank (0-based) do item held-out de cada usuário no catálogo inteiro.

    Para cada usuário: pontua TODOS os itens, masca os já vistos (não faz
    sentido recomendar de novo) e conta quantos superam o score do held-out.
    Rank 0 = o modelo colocaria o item verdadeiro em 1º lugar.

    Args:
        model: Recomendador treinado (modo eval é ativado aqui).
        heldout: DataFrame com o par held-out de cada usuário.
        seen: Itens a mascarar por usuário (histórico de treino).
        n_items: Tamanho do catálogo.
        batch_users: Usuários pontuados por lote (controle de memória).

    Returns:
        Array com o rank do item held-out de cada usuário.
    """
    model.eval()
    users = heldout["user_idx"].to_numpy(np.int64)
    items = heldout["item_idx"].to_numpy(np.int64)
    all_items = torch.arange(n_items)
    ranks = np.empty(len(users), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(users), batch_users):
            batch_u = users[start : start + batch_users]
            batch_i = items[start : start + batch_users]
            scores = _score_catalog(model, batch_u, all_items)
            ranks[start : start + len(batch_u)] = _ranks_in_batch(
                scores, batch_u, batch_i, seen
            )
    return ranks


def ranking_metrics(ranks: np.ndarray, k: int) -> dict[str, float]:
    """Precision@K, Recall@K, NDCG@K e MAP a partir dos ranks held-out.

    Com exatamente 1 item relevante por usuário (protocolo leave-one-out):
    Recall@K equivale ao Hit Rate@K, Precision@K = hits/K e MAP = MRR.

    Args:
        ranks: Rank (0-based) do item held-out de cada usuário.
        k: Tamanho da lista de recomendação (Top-K).

    Returns:
        Dict com as quatro métricas, prontas para o ``metrics.json``.
    """
    hits = ranks < k
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 2.0), 0.0)
    return {
        f"precision_at_{k}": float(hits.mean() / k),
        f"recall_at_{k}": float(hits.mean()),
        f"ndcg_at_{k}": float(ndcg.mean()),
        "map": float((1.0 / (ranks + 1.0)).mean()),
    }


def _score_catalog(
    model: nn.Module, batch_users: np.ndarray, all_items: torch.Tensor
) -> torch.Tensor:
    """Scores de todos os itens para cada usuário do lote: shape [B, n_items]."""
    users_t = torch.as_tensor(batch_users).repeat_interleave(all_items.numel())
    items_t = all_items.repeat(len(batch_users))
    return model(users_t, items_t).view(len(batch_users), -1)


def _ranks_in_batch(
    scores: torch.Tensor,
    batch_users: np.ndarray,
    batch_items: np.ndarray,
    seen: dict[int, set[int]],
) -> np.ndarray:
    """Aplica a máscara de itens vistos e calcula o rank do held-out por linha."""
    out = np.empty(len(batch_users), dtype=np.int64)
    for row, (user, item) in enumerate(zip(batch_users, batch_items, strict=True)):
        user_scores = scores[row]
        target = user_scores[int(item)].clone()
        seen_items = list(seen.get(int(user), ()))
        user_scores[seen_items] = float("-inf")
        out[row] = int((user_scores > target).sum())
    return out

"""Arquitetura neural: embeddings de user/item + MLP -> score de afinidade."""

from __future__ import annotations

import torch
from torch import nn


class RecSysMLP(nn.Module):
    """Recomendador neural: ``concat(emb_user, emb_item) -> MLP -> logit``.

    Os embeddings aprendem uma representação densa de cada usuário e item;
    o MLP captura interações não-lineares entre as duas representações.
    A saída é um logit — usar ``BCEWithLogitsLoss`` no treino; aplicar
    sigmoid quando precisar de probabilidade.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp = _build_mlp(embedding_dim * 2, hidden_dims, dropout)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Retorna o logit de afinidade para cada par ``(user, item)``."""
        features = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.mlp(features).squeeze(-1)


def _build_mlp(
    in_dim: int, hidden_dims: tuple[int, ...], dropout: float
) -> nn.Sequential:
    """Monta o MLP com ReLU + Dropout entre as camadas ocultas."""
    layers: list[nn.Module] = []
    for hidden in hidden_dims:
        layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)

"""Baselines clássicos de recomendação: popularidade e item-KNN (cosseno).

Servem de piso de comparação para o RecSysMLP (requisito: >= 4 métricas).
Ambos entram no catálogo da :class:`~recsys.models.factory.ModelFactory`,
exercitando o Open/Closed: nenhum código existente mudou para acomodá-los.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from recsys.models.base import Model
from recsys.models.factory import ModelFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


@ModelFactory.register("popularity")
class PopularityRecommender(Model):
    """Recomenda os itens mais frequentes do treino, iguais para todo usuário.

    É o piso mínimo de qualquer recomendador: um modelo que não supera a
    popularidade não aprendeu nada além do viés global do catálogo.
    """

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.scores_ = np.zeros(n_items, dtype=np.float64)

    def fit(self, interactions: pd.DataFrame) -> PopularityRecommender:
        """Conta a frequência de cada item nas interações de treino."""
        items = interactions["item_idx"].to_numpy()
        self.scores_ = np.bincount(items, minlength=self.n_items).astype(np.float64)
        return self

    def predict(self, users: Sequence[int], items: Sequence[int]) -> list[float]:
        """Score = popularidade do item (o usuário é ignorado)."""
        return [float(self.scores_[item]) for item in items]

    def recommend(self, user_idx: int, n: int = 10) -> list[int]:
        """Top-N global por popularidade (idêntico para qualquer usuário)."""
        return np.argsort(-self.scores_)[:n].tolist()

    def catalog_scores(self, user_idx: int) -> np.ndarray:
        """Scores do catálogo inteiro — o mesmo vetor para todo usuário."""
        return self.scores_


@ModelFactory.register("item_knn")
class ItemKNNRecommender(Model):
    """KNN item-item: similaridade de cosseno sobre co-ocorrência de usuários.

    Cada item vira um vetor esparso de usuários (L2-normalizado). O score de
    um item candidato é a soma dos cossenos com os itens do histórico do
    usuário — como cosseno é linear no perfil somado, o cálculo inteiro do
    catálogo se reduz a um produto matriz-vetor esparso.
    """

    def __init__(self, n_users: int, n_items: int) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.item_vecs_: csr_matrix | None = None
        self.user_items_: csr_matrix | None = None

    def fit(self, interactions: pd.DataFrame) -> ItemKNNRecommender:
        """Monta a matriz user-item binária e normaliza os vetores de item."""
        users = interactions["user_idx"].to_numpy()
        items = interactions["item_idx"].to_numpy()
        ones = np.ones(len(users), dtype=np.float32)
        self.user_items_ = csr_matrix(
            (ones, (users, items)), shape=(self.n_users, self.n_items)
        )
        self.item_vecs_ = normalize(self.user_items_.T.tocsr())
        return self

    def predict(self, users: Sequence[int], items: Sequence[int]) -> list[float]:
        """Score de cada par (user, item) via perfil somado do usuário."""
        return [
            float(self.catalog_scores(user)[item])
            for user, item in zip(users, items, strict=True)
        ]

    def recommend(self, user_idx: int, n: int = 10) -> list[int]:
        """Top-N por similaridade, excluindo itens já vistos pelo usuário."""
        scores = self.catalog_scores(user_idx).copy()
        scores[self.user_items_[user_idx].indices] = -np.inf
        ranked = np.argsort(-scores)
        return ranked[np.isfinite(scores[ranked])][:n].tolist()

    def catalog_scores(self, user_idx: int) -> np.ndarray:
        """Scores do catálogo inteiro para um usuário (matvec esparso)."""
        history = self.user_items_[user_idx].indices
        profile = np.asarray(self.item_vecs_[history].sum(axis=0)).ravel()
        return self.item_vecs_ @ profile

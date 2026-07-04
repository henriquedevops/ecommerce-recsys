"""Interface base dos modelos de recomendação (Dependency Inversion)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


class Model(ABC):
    """Contrato mínimo de um recomendador user-item.

    Qualquer implementação concreta (MLP neural, baselines de popularidade
    ou KNN) deve respeitar esta interface, permitindo que treino e avaliação
    dependam apenas da abstração — nunca de uma classe concreta.
    """

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> Model:
        """Treina o modelo a partir das interações user-item.

        Args:
            interactions: DataFrame com colunas ``user_idx``, ``item_idx``
                e ``label`` (feedback implícito 0/1).

        Returns:
            O próprio modelo treinado, para permitir encadeamento.
        """

    @abstractmethod
    def predict(self, users: Sequence[int], items: Sequence[int]) -> list[float]:
        """Pontua a afinidade de pares (user, item) alinhados por posição.

        Args:
            users: Índices contíguos de usuários.
            items: Índices contíguos de itens, pareados com ``users``.

        Returns:
            Score de afinidade previsto para cada par.
        """

    @abstractmethod
    def recommend(self, user_idx: int, n: int = 10) -> list[int]:
        """Retorna os N itens mais relevantes para um usuário (Top-N).

        Args:
            user_idx: Índice contíguo do usuário.
            n: Quantidade de itens a recomendar.

        Returns:
            Lista de ``item_idx`` ordenada por score decrescente.
        """

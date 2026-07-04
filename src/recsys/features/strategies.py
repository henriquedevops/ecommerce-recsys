"""Padrão Strategy para pré-processamento das interações user-item."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Preprocessor(ABC):
    """Contrato de transformação das interações user-item.

    Strategies concretas são intercambiáveis: o pipeline depende apenas
    desta interface, nunca de uma implementação específica.
    """

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> Preprocessor:
        """Aprende os parâmetros da transformação (quando houver).

        Args:
            interactions: DataFrame bruto de interações user-item.

        Returns:
            O próprio preprocessor ajustado.
        """

    @abstractmethod
    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação às interações.

        Args:
            interactions: DataFrame bruto de interações user-item.

        Returns:
            DataFrame transformado.
        """

    def fit_transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Ajusta e aplica a transformação em sequência.

        Args:
            interactions: DataFrame bruto de interações user-item.

        Returns:
            DataFrame transformado.
        """
        return self.fit(interactions).transform(interactions)

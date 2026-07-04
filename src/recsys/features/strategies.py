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


class ImplicitEncoder(Preprocessor):
    """Converte ratings explícitos (1-5) em feedback implícito binário.

    Formulação do projeto (implicit/Top-N): ratings >= ``threshold`` viram
    interação positiva (``label = 1.0``); os demais são descartados. A
    amostragem de negativos acontece no estágio de feature engineering.
    """

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    def fit(self, interactions: pd.DataFrame) -> ImplicitEncoder:
        """Sem parâmetros a aprender; retorna o próprio encoder."""
        return self

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filtra positivos (``rating >= threshold``) e cria a coluna ``label``.

        Args:
            interactions: DataFrame com a coluna ``rating`` explícita.

        Returns:
            DataFrame apenas com interações positivas e ``label = 1.0``,
            sem a coluna ``rating``.
        """
        positives = interactions[interactions["rating"] >= self.threshold].copy()
        positives["label"] = 1.0
        return positives.drop(columns=["rating"])

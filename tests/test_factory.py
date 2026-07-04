"""Testes do padrão Factory de modelos."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from recsys.models.base import Model
from recsys.models.factory import ModelFactory


@ModelFactory.register("dummy")
class DummyModel(Model):
    """Modelo trivial para exercitar o registro na Factory."""

    def fit(self, interactions: object) -> DummyModel:
        """Não aprende nada; retorna o próprio modelo."""
        return self

    def predict(self, users: Sequence[int], items: Sequence[int]) -> list[float]:
        """Retorna score neutro para cada par."""
        return [0.0] * len(users)

    def recommend(self, user_idx: int, n: int = 10) -> list[int]:
        """Recomenda os N primeiros itens, sem critério."""
        return list(range(n))


def test_create_returns_registered_model() -> None:
    model = ModelFactory.create("dummy")
    assert isinstance(model, DummyModel)


def test_create_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        ModelFactory.create("inexistente")


def test_available_lists_registered_names() -> None:
    assert "dummy" in ModelFactory.available()

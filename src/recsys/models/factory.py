"""Padrão Factory para criação de modelos de recomendação (Open/Closed)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recsys.models.base import Model

ModelBuilder = Callable[..., Model]


class ModelFactory:
    """Cria modelos por nome, sem expor classes concretas ao chamador.

    Novos modelos entram no catálogo via :meth:`register`, sem alterar
    código existente (princípio Open/Closed).
    """

    _registry: dict[str, ModelBuilder] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ModelBuilder], ModelBuilder]:
        """Retorna um decorator que registra um builder de modelo.

        Args:
            name: Identificador do modelo (ex.: ``"mlp"``, ``"popularity"``).

        Returns:
            Decorator que registra e devolve o builder inalterado.
        """

        def decorator(builder: ModelBuilder) -> ModelBuilder:
            cls._registry[name] = builder
            return builder

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Model:
        """Instancia um modelo registrado.

        Args:
            name: Nome usado no registro.
            **kwargs: Hiperparâmetros repassados ao builder.

        Returns:
            Instância concreta de :class:`Model`.

        Raises:
            ValueError: Se o nome não estiver registrado.
        """
        if name not in cls._registry:
            available = sorted(cls._registry)
            msg = f"Modelo desconhecido: {name!r}. Disponíveis: {available}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Lista os nomes de modelos registrados, em ordem alfabética."""
        return sorted(cls._registry)

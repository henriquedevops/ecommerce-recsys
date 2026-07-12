"""Testes dos baselines de popularidade e item-KNN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.models.baselines import ItemKNNRecommender, PopularityRecommender
from recsys.models.factory import ModelFactory


@pytest.fixture
def interactions() -> pd.DataFrame:
    """4 usuários, 4 itens: item 0 é o mais popular; 0 e 1 co-ocorrem."""
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2, 3],
            "item_idx": [0, 1, 0, 1, 0, 2, 3],
            "label": [1] * 7,
        }
    )


def test_popularity_ranks_most_frequent_first(interactions: pd.DataFrame) -> None:
    model = PopularityRecommender(n_items=4).fit(interactions)
    assert model.recommend(user_idx=0, n=2) == [0, 1]
    assert model.predict([0, 1], [0, 3]) == [3.0, 1.0]


def test_popularity_catalog_scores_are_user_independent(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender(n_items=4).fit(interactions)
    assert np.array_equal(model.catalog_scores(0), model.catalog_scores(3))


def test_item_knn_scores_cooccurring_item_higher(interactions: pd.DataFrame) -> None:
    model = ItemKNNRecommender(n_users=4, n_items=4).fit(interactions)
    scores = model.catalog_scores(user_idx=2)  # histórico: itens 0 e 2
    assert scores[1] > scores[3]  # item 1 co-ocorre com 0; item 3 é isolado


def test_item_knn_recommend_excludes_seen_items(interactions: pd.DataFrame) -> None:
    model = ItemKNNRecommender(n_users=4, n_items=4).fit(interactions)
    assert 0 not in model.recommend(user_idx=0, n=4)
    assert 1 not in model.recommend(user_idx=0, n=4)


def test_baselines_are_registered_in_factory() -> None:
    available = ModelFactory.available()
    assert "popularity" in available
    assert "item_knn" in available

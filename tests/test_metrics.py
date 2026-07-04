"""Testes das métricas de ranking (protocolo leave-one-out)."""

import numpy as np
import pandas as pd
import pytest

from recsys.evaluation.metrics import ranking_metrics, seen_items_by_user


def test_ranking_metrics_with_known_ranks() -> None:
    ranks = np.array([0, 9, 10])  # dois hits no top-10, um fora
    metrics = ranking_metrics(ranks, k=10)
    assert metrics["recall_at_10"] == pytest.approx(2 / 3)
    assert metrics["precision_at_10"] == pytest.approx(2 / 3 / 10)
    expected_ndcg = (1.0 + 1.0 / np.log2(11)) / 3
    assert metrics["ndcg_at_10"] == pytest.approx(expected_ndcg)
    assert metrics["map"] == pytest.approx((1 / 1 + 1 / 10 + 1 / 11) / 3)


def test_perfect_ranks_give_maximum_metrics() -> None:
    metrics = ranking_metrics(np.zeros(5, dtype=np.int64), k=10)
    assert metrics["recall_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["map"] == 1.0


def test_seen_items_by_user_merges_frames() -> None:
    train = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [10, 11, 12]})
    val = pd.DataFrame({"user_idx": [0], "item_idx": [13]})
    seen = seen_items_by_user(train, val)
    assert seen[0] == {10, 11, 13}
    assert seen[1] == {12}

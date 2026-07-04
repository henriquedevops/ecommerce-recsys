"""Smoke tests da arquitetura RecSysMLP."""

import torch

from recsys.models.mlp import RecSysMLP


def _tiny_model() -> RecSysMLP:
    return RecSysMLP(
        n_users=10, n_items=20, embedding_dim=8, hidden_dims=(16,), dropout=0.0
    )


def test_forward_returns_one_score_per_pair() -> None:
    model = _tiny_model()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    assert model(users, items).shape == (3,)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = _tiny_model()
    model.eval()
    users, items = torch.tensor([1, 2]), torch.tensor([7, 9])
    first = model(users, items)
    second = model(users, items)
    assert torch.equal(first, second)

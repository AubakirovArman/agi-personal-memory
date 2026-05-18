import torch

from agim.model.wal_dual_helpers import combine_positive_keys, ridge_constrained_key


def test_ridge_constrained_key_reduces_protected_component():
    key = torch.tensor([1.0, 1.0])
    protected = [torch.tensor([1.0, 0.0])]

    constrained = ridge_constrained_key(key / key.norm(), protected, strength=10.0)

    assert torch.dot(constrained, protected[0]) < 0.2
    assert torch.dot(constrained, torch.tensor([0.0, 1.0])) > 0.9


def test_combine_positive_keys_supports_ridge_constraint_mode():
    primary = torch.tensor([1.0, 0.0])
    positive = torch.tensor([1.0, 1.0])
    protected = [torch.tensor([1.0, 0.0])]

    constrained = combine_positive_keys(
        primary,
        [positive],
        weight=1.0,
        protected_basis=protected,
        projection_strength=10.0,
        constraint_mode="ridge",
    )

    assert torch.dot(constrained, protected[0]) < 0.3

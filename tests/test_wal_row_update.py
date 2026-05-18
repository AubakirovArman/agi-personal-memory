import torch

from agim.model.wal_row_update import add_row_delta


def test_add_row_delta_can_skip_wal_reencoding():
    weight = torch.zeros(3, 2)

    add_row_delta(
        weight,
        row_idx=1,
        delta=torch.tensor([1.0, -2.0]),
        atoms=torch.empty(0),
        lmax=1,
        wal_encode=False,
    )

    assert torch.equal(weight[0], torch.zeros(2))
    assert torch.equal(weight[1], torch.tensor([1.0, -2.0]))
    assert torch.equal(weight[2], torch.zeros(2))

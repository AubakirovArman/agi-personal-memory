import torch

from agim.model.sparse_overlay import RuntimeSparseOverlay


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def test_runtime_sparse_overlay_adjusts_lm_logits_without_mutating_weights():
    model = _TinyModel()
    hidden = torch.tensor([[[1.0, 2.0, 3.0]]])
    before_weight = model.lm_head.weight.detach().clone()
    baseline = model.lm_head(hidden)
    overlay = RuntimeSparseOverlay(model)
    overlay.add_lm_delta(2, torch.tensor([0.5, 0.0, 0.0]))

    with overlay:
        patched = model.lm_head(hidden)

    restored = model.lm_head(hidden)

    assert patched[..., 2] == baseline[..., 2] + 0.5
    assert torch.equal(model.lm_head.weight, before_weight)
    assert torch.equal(restored, baseline)


def test_runtime_sparse_overlay_adjusts_embedding_rows_without_mutating_weights():
    model = _TinyModel()
    token_ids = torch.tensor([[1, 2, 1]])
    before_weight = model.model.embed_tokens.weight.detach().clone()
    baseline = model.model.embed_tokens(token_ids)
    overlay = RuntimeSparseOverlay(model)
    overlay.add_embed_delta(1, torch.tensor([1.0, 0.0, -1.0]))

    with overlay:
        patched = model.model.embed_tokens(token_ids)

    restored = model.model.embed_tokens(token_ids)

    expected_delta = torch.tensor([1.0, 0.0, -1.0])
    assert torch.equal(patched[0, 0], baseline[0, 0] + expected_delta)
    assert torch.equal(patched[0, 1], baseline[0, 1])
    assert torch.equal(patched[0, 2], baseline[0, 2] + expected_delta)
    assert torch.equal(model.model.embed_tokens.weight, before_weight)
    assert torch.equal(restored, baseline)

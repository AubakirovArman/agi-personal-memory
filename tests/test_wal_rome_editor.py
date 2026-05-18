from types import SimpleNamespace

import torch
from torch import nn

from agim.eval.easyedit_budget import patch_artifact_from_backup
from agim.eval.easyedit_metrics import edit_nt_metrics
from agim.model.wal_rome_editor import WALRomeEditor


class _Batch:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    eos_token_id = None

    def __init__(self):
        self.vocab = {
            "Alice": 1,
            "is": 2,
            "Paris": 3,
            "Rome": 4,
            "born": 5,
            "in": 6,
        }

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(tok, len(self.vocab) + 1)
                for tok in text.split()]

    def __call__(self, text, return_tensors="pt"):
        return _Batch(torch.tensor([self.encode(text)], dtype=torch.long))


class _MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class _Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = _MLP(hidden)

    def forward(self, hidden, attention_mask=None):
        return (hidden + self.mlp(hidden),)


class _Inner(nn.Module):
    def __init__(self, vocab_size, hidden, layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([_Layer(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=16, hidden=8, layers=3):
        super().__init__()
        self.model = _Inner(vocab_size, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=False):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_wal_rome_updates_sparse_ffn_rows_and_rolls_back():
    torch.manual_seed(0)
    model = _TinyCausalLM()
    tok = _Tokenizer()
    editor = WALRomeEditor(
        model, tok, K=8, lmax=2, device="cpu", target_layer=1,
        top_rows=3, clamp_rome=0.4,
    )
    editor.build_vocab(sample_size=128)
    before = model.model.layers[1].mlp.down_proj.weight.detach().clone()

    backup = editor.apply_edit(
        "Alice", "Paris", prompt="Alice is", wal_encode_updates=False)

    changed_rows = {
        row_id for (layer_idx, row_id) in backup["ffn_backup"]
        if layer_idx == 1
    }
    assert 1 <= len(changed_rows) <= 3
    assert torch.any(model.model.layers[1].mlp.down_proj.weight != before)

    nt = edit_nt_metrics(editor, backup, eos_id=None)
    assert nt["edited_ffn_rows_count"] == len(changed_rows)
    assert nt["ffn_down_proj_non_edited_max"] == 0.0

    editor.rollback(backup)
    assert torch.allclose(model.model.layers[1].mlp.down_proj.weight, before)


def test_wal_rome_backup_becomes_patch_artifact_rows():
    torch.manual_seed(1)
    model = _TinyCausalLM()
    tok = _Tokenizer()
    editor = WALRomeEditor(
        model, tok, K=8, lmax=2, device="cpu", target_layer=2,
        top_rows=2, clamp_rome=0.3,
    )
    editor.build_vocab(sample_size=128)
    backup = editor.apply_edit(
        "Alice", "Paris", prompt="Alice is", wal_encode_updates=False)
    args = SimpleNamespace(
        edit_backend="wal_rome",
        wal_encode_updates=True,
        sequential_edit=False,
        method_profile_id=None,
    )
    fact = {
        "case_id": 9,
        "requested_rewrite": {
            "subject": "Alice",
            "relation_id": "P19",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "Rome"},
        },
    }

    artifact = patch_artifact_from_backup(editor, args, fact, backup)

    assert artifact.method_profile_id == "single_wal_rome"
    assert artifact.row_counts() == {"model.layers.2.mlp.down_proj": 2}
    assert artifact.metadata["edit_locus"] == "mlp.down_proj"

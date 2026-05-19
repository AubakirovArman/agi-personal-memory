import torch
import pytest

from agim.eval.easyedit_official_runner import (
    attach_locality_acc,
    contextual_target_ids,
    easyedit_record,
    extract_portability,
    ngram_entropy,
    parse_retention_steps,
    summarize_official,
)
from agim.model.wal_dual_editor import WALDualLayerEditor
from agim.model.wal_dual_helpers import constrained_projection_key
from agim.model.wal_editor import WalLmHeadEditor


def test_extract_portability_normalizes_knowedit_shapes():
    fact = {
        "portability": {
            "one_hop": [
                {"New Question": "Where was Alice born?", "New Answer": [["Paris"]]},
                {"question": "Alice's birthplace?", "answer": "Paris"},
            ]
        }
    }

    assert extract_portability(fact) == {
        "one_hop": {
            "prompt": ["Where was Alice born?", "Alice's birthplace?"],
            "ground_truth": ["Paris", "Paris"],
        }
    }


def test_easyedit_record_keeps_all_rephrase_prompts():
    fact = {
        "requested_rewrite": {
            "subject": "Alice",
            "prompt": "{} was born in",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "Rome"},
        },
        "paraphrase_prompts": ["Alice birthplace:", "Birth city of Alice:"],
    }

    record = easyedit_record(fact, locality_limit=None)

    assert record["rephrase_prompt"] == "Alice birthplace:"
    assert record["rephrase_prompts"] == [
        "Alice birthplace:",
        "Birth city of Alice:",
    ]


def test_attach_locality_acc_uses_pre_post_consistency():
    pre = {"locality": {"neighborhood_output": [[1, 2], [3, 4]]}}
    post = {"locality": {"neighborhood_output": [[1, 2], [3, 0]]}}
    record = {"locality": {"neighborhood": {}}}

    attach_locality_acc(pre, post, record)

    assert post["locality"]["neighborhood_acc"] == [1.0, 0.5]
    assert "neighborhood_output" not in post["locality"]
    assert "locality" not in pre


def test_summarize_official_includes_new_metric_groups():
    rows = [
        {
            "relation_id": "P103",
            "pre": {"rewrite_acc": [0.0], "rephrase_acc": [0.0]},
            "post": {
                "rewrite_acc": [1.0],
                "rephrase_acc": [1.0],
                "rephrase_all_acc": [1.0, 0.0],
                "locality": {"neighborhood_acc": [1.0, 0.0]},
                "portability": {"one_hop_acc": [1.0]},
            },
            "generation": {
                "rewrite_acc": [1.0],
                "rephrase_acc": [0.0],
                "rephrase_all_acc": [0.0, 1.0],
            },
            "contextual_generation": {
                "rewrite_acc": [1.0],
                "rephrase_acc": [0.5],
                "rephrase_all_acc": [0.5, 1.0],
            },
            "NT": {
                "lm_head_non_edited_max": 0.0,
                "embed_non_edited_max": 0.0,
                "edited_lm_rows_count": 3,
                "edited_embed_rows_count": 2,
                "edited_lm_delta_l2_mean": 1.25,
                "edited_lm_delta_l2_max": 2.5,
                "edited_embed_delta_l2_mean": 0.5,
                "edited_embed_delta_l2_max": 1.0,
                "eos_row_changed": True,
            },
            "probability": {
                "rewrite_acc": 1.0,
                "rephrase_acc": 0.0,
                "rephrase_all_acc": [0.0, 1.0],
                "locality": {"neighborhood_acc": [1.0, 0.0]},
            },
            "fluency": {"ngram_entropy": 1.25},
        }
    ]

    summary = summarize_official(rows)

    assert summary["post"]["portability"]["mean_acc"] == 1.0
    assert summary["post_generation_contextual"] == {
        "rewrite_acc": 1.0,
        "rephrase_acc": 0.5,
        "rephrase_all_acc": 0.75,
    }
    assert summary["NT"] == {
        "lm_head_non_edited_max": 0.0,
        "embed_non_edited_max": 0.0,
        "edited_lm_rows_avg": 3.0,
        "edited_embed_rows_avg": 2.0,
        "eos_row_changed_rate": 1.0,
        "edited_lm_delta_l2_mean": 1.25,
        "edited_lm_delta_l2_max": 2.5,
        "edited_embed_delta_l2_mean": 0.5,
        "edited_embed_delta_l2_max": 1.0,
    }
    assert summary["post_probability"] == {
        "rewrite_acc": 1.0,
        "rephrase_acc": 0.0,
        "rephrase_all_acc": 0.5,
        "locality_acc": 0.5,
    }
    assert summary["post_fluency"]["ngram_entropy"] == 1.25
    assert summary["post"]["rephrase_all_acc"] == 0.5
    assert summary["metrics_by_relation_id"]["P103"]["n"] == 1
    assert summary["metrics_by_relation_id"]["P103"]["rewrite_acc"] == 1.0
    assert summary["metrics_by_relation_id"]["P103"]["rephrase_all_acc"] == 0.5
    assert summary["metrics_by_relation_id"]["P103"]["locality_acc"] == 0.5


def test_ngram_entropy_is_positive_for_varied_text():
    assert ngram_entropy(["alpha beta gamma delta"]) > 0.0
    assert ngram_entropy(["alpha"]) == 0.0


class _TokenBatch:
    def __init__(self, input_ids):
        self.input_ids = torch.tensor([input_ids])


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        if text == "The language is":
            return _TokenBatch([10, 11, 12])
        if text == "The language is English":
            return _TokenBatch([10, 11, 12, 99])
        return _TokenBatch(self.encode(text, add_special_tokens=False))

    def encode(self, text, add_special_tokens=False):
        if text == "English":
            return [42]
        return []


def test_contextual_target_ids_follow_prompt_space_target_suffix():
    assert contextual_target_ids(_FakeTokenizer(), "The language is", "English") == [99]


def test_parse_retention_steps_ignores_steps_beyond_total():
    assert parse_retention_steps("1,10,50", total=12) == [1, 10]
    assert parse_retention_steps("", total=12) == []

    with pytest.raises(ValueError):
        parse_retention_steps("0", total=12)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class _TinyTokenizer:
    eos_token_id = 7


def test_wal_non_target_diff_uses_snapshotted_rows():
    model = _TinyModel()
    editor = WalLmHeadEditor(model, _TinyTokenizer(), device="cpu")
    editor.snapshot_non_target({1, 7}, sample_size=20)

    assert editor._nt_snapshot
    assert 1 not in editor._nt_snapshot
    assert 7 not in editor._nt_snapshot
    assert editor.measure_non_target_diff() == 0.0

    rid = next(iter(editor._nt_snapshot))
    model.lm_head.weight.data[rid, 0] += 1.0

    assert editor.measure_non_target_diff() == pytest.approx(1.0)


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)


class _TinyDualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.model = _TinyBackbone()


def test_wal_dual_non_target_diff_covers_lm_head_and_embed():
    model = _TinyDualModel()
    editor = WALDualLayerEditor(model, _TinyTokenizer(), device="cpu")
    editor.snapshot_non_target({1, 7}, embed_exclude={2}, sample_size=20)

    assert editor.measure_non_target_diffs() == {
        "lm_head_non_edited_max": 0.0,
        "embed_non_edited_max": 0.0,
    }

    lm_rid = next(iter(editor._lm_nt_snapshot))
    emb_rid = next(iter(editor._emb_nt_snapshot))
    assert lm_rid not in {1, 7}
    assert emb_rid != 2

    model.lm_head.weight.data[lm_rid, 0] += 0.5
    model.model.embed_tokens.weight.data[emb_rid, 0] += 0.25

    diffs = editor.measure_non_target_diffs()
    assert diffs["lm_head_non_edited_max"] == pytest.approx(0.5)
    assert diffs["embed_non_edited_max"] == pytest.approx(0.25)
    assert editor.measure_non_target_diff() == pytest.approx(0.5)


def test_wal_dual_project_away_reduces_positive_basis_component():
    key = torch.tensor([1.0, 1.0])
    basis = [torch.tensor([1.0, 0.0])]

    projected = WALDualLayerEditor._project_away(key, basis, strength=1.0)

    assert torch.dot(projected, basis[0]) == pytest.approx(0.0)
    assert projected.norm().item() == pytest.approx(1.0)


def test_wal_dual_orthogonal_projection_handles_duplicate_basis():
    key = torch.tensor([1.0, 1.0])
    basis = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]

    projected = WALDualLayerEditor._project_away(
        key, basis, strength=1.0, mode="orthogonal")

    assert torch.dot(projected, torch.tensor([1.0, 0.0])) == pytest.approx(0.0)
    assert torch.dot(projected, torch.tensor([0.0, 1.0])) == pytest.approx(1.0)
    assert projected.norm().item() == pytest.approx(1.0)


def test_wal_dual_orthogonal_projection_full_span_returns_zero_vector():
    key = torch.tensor([1.0, 1.0])
    basis = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

    projected = WALDualLayerEditor._project_away(
        key, basis, strength=1.0, mode="orthogonal")

    assert projected.tolist() == pytest.approx([0.0, 0.0])


def test_wal_dual_combine_positive_keys_moves_toward_paraphrase_basis():
    primary = torch.tensor([1.0, 0.0])
    positive = torch.tensor([0.0, 1.0])

    combined = WALDualLayerEditor._combine_positive_keys(
        primary, [positive], weight=1.0)

    assert torch.dot(combined, primary) == pytest.approx(2 ** -0.5)
    assert torch.dot(combined, positive) == pytest.approx(2 ** -0.5)
    assert combined.norm().item() == pytest.approx(1.0)


def test_constrained_projection_key_strength_zero_returns_normalized_key():
    key = torch.tensor([3.0, 4.0])
    basis = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

    projected = constrained_projection_key(key, basis, pos_k=1, neg_k=1, strength=0.0)

    assert torch.allclose(projected, torch.tensor([0.6, 0.8]), atol=1e-6)


def test_constrained_projection_key_selects_pos_and_neg_basis():
    key = torch.tensor([0.5, -1.0])
    basis = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-0.8, 0.6]),
    ]

    projected = constrained_projection_key(key, basis, pos_k=1, neg_k=1, strength=1.0)
    projected = projected / (projected.norm() + 1e-8)
    normalized_key = key / (key.norm() + 1e-8)

    assert projected.norm().item() == pytest.approx(1.0)
    assert torch.dot(projected, normalized_key).item() < 0.99


def test_wal_dual_rollback_restores_history_basis_length():
    editor = WALDualLayerEditor(_TinyDualModel(), _TinyTokenizer(), device="cpu")
    editor._edit_key_basis = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]

    editor.rollback({"history_len": 1})

    assert len(editor._edit_key_basis) == 1
    assert torch.equal(editor._edit_key_basis[0], torch.tensor([1.0, 0.0]))

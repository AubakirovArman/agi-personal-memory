import torch
import pytest

from agim.eval.easyedit_official_runner import (
    attach_locality_acc,
    contextual_target_ids,
    extract_portability,
    ngram_entropy,
    summarize_official,
)
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
            "pre": {"rewrite_acc": [0.0], "rephrase_acc": [0.0]},
            "post": {
                "rewrite_acc": [1.0],
                "rephrase_acc": [1.0],
                "locality": {"neighborhood_acc": [1.0, 0.0]},
                "portability": {"one_hop_acc": [1.0]},
            },
            "generation": {"rewrite_acc": [1.0], "rephrase_acc": [0.0]},
            "contextual_generation": {"rewrite_acc": [1.0], "rephrase_acc": [0.5]},
            "probability": {
                "rewrite_acc": 1.0,
                "rephrase_acc": 0.0,
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
    }
    assert summary["post_probability"] == {
        "rewrite_acc": 1.0,
        "rephrase_acc": 0.0,
        "locality_acc": 0.5,
    }
    assert summary["post_fluency"]["ngram_entropy"] == 1.25


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

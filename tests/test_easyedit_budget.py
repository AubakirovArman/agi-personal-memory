import torch

from agim.eval.easyedit_budget import evaluate_edit_budget, norm_budget_policy
from agim.eval.easyedit_cli import build_parser


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.model = _Backbone()


class _Editor:
    def __init__(self):
        self.model = _Model()


def test_norm_budget_policy_is_disabled_by_default():
    args = build_parser().parse_args([])

    assert norm_budget_policy(args) is None


def test_evaluate_edit_budget_returns_no_commit_decision():
    editor = _Editor()
    args = build_parser().parse_args(["--max-row-delta-norm", "0.5"])
    fact = {
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Alice",
            "relation_id": "P17",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "Rome"},
        },
    }
    before = torch.zeros(2)
    editor.model.lm_head.weight.data[1, :] = torch.ones(2)
    backup = {"lm_backup": {1: before}, "emb_backup": {}}

    decision = evaluate_edit_budget(editor, args, fact, backup)

    assert decision is not None
    assert decision["no_commit"] is True
    assert decision["reasons"][0]["metric"] == "max_delta_norm"

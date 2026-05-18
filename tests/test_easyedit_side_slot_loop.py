from types import SimpleNamespace

import torch

from agim.eval.easyedit_side_slot_loop import run_sequential_side_slot


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight.data.zero_()


class _Editor:
    def __init__(self):
        self.model = _Model()

    def rollback(self, backup):
        for row_id, before in backup.get("lm_backup", {}).items():
            self.model.lm_head.weight.data[row_id, :] = before


def test_side_slot_loop_keeps_base_frozen_and_evaluates_overlay():
    editor = _Editor()
    args = SimpleNamespace(
        edit_backend="side_slot",
        sequential_edit=True,
        retention_steps="1",
        test_fluency=False,
        method_profile_id=None,
        wal_encode_updates=True,
    )
    fact = {
        "case_id": 1,
        "requested_rewrite": {
            "subject": "Alice",
            "relation_id": "P17",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "Rome"},
        },
    }
    record = {"prompt": "Alice was born in"}

    metrics, _times, retention = run_sequential_side_slot(
        args=args,
        model=editor.model,
        tok=None,
        hparams=None,
        editor=editor,
        facts=[fact],
        records=[record],
        compute_edit_quality=None,
        test_prediction_acc=None,
        device_id=0,
        apply_one=_apply_one,
        compute_pre=_compute_pre,
        post_bundle=_post_bundle,
        base_row=_base_row,
        budget_status=lambda _budget: {},
    )

    assert torch.equal(editor.model.lm_head.weight.data, torch.zeros_like(
        editor.model.lm_head.weight.data))
    assert metrics[0]["edit_backend"] == "side_slot"
    assert metrics[0]["side_slot_id"] == "case-1"
    assert metrics[0]["post"]["rewrite_acc"] == [1.0]
    assert retention["side_slot_summary"] == {"slots": 1, "enabled": 1}
    assert retention["relation_slot_summary"] == {"P17": {"slots": 1, "enabled": 1}}


def _apply_one(editor, _args, _fact, _record):
    before = editor.model.lm_head.weight.data[1, :].clone()
    editor.model.lm_head.weight.data[1, :] += torch.tensor([1.0, 0.0])
    return {"lm_backup": {1: before}, "emb_backup": {}}, 0.1


def _compute_pre(_args, _model, _tok, _hparams, _record, _compute, _device_id):
    return {"rewrite_acc": [0.0], "rephrase_acc": [0.0]}


def _post_bundle(_args, model, *_unused, **_kwargs):
    hidden = torch.tensor([[[1.0, 0.0]]])
    passed = float(model.lm_head(hidden)[0, 0, 1].item() == 1.0)
    return {
        "pre": {"rewrite_acc": [0.0], "rephrase_acc": [0.0]},
        "post": {
            "rewrite_acc": [passed],
            "rephrase_acc": [passed],
            "rephrase_all_acc": [passed],
            "locality": {"neighborhood_acc": [1.0]},
        },
        "generation": {"rewrite_acc": [0.0]},
        "contextual_generation": {"rewrite_acc": [passed]},
    }


def _base_row(fact, record, idx, edit_time, edit_status=None):
    row = {
        "case_id": fact.get("case_id", idx),
        "relation_id": fact["requested_rewrite"]["relation_id"],
        "requested_rewrite": record,
        "edit_time_s": edit_time,
    }
    if edit_status:
        row.update(edit_status)
    return row

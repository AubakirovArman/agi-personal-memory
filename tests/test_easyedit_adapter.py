from types import SimpleNamespace

import torch

from agim.integrations.easyedit_agimwal import (
    AGIMWALHyperParams,
    apply_agimwal_request,
    apply_agimwal_to_model,
    easyedit_registration_snippet,
    easyedit_weight_names,
)


class FakeEditor:
    def __init__(self):
        self.calls = []
        self.nt_sample_size = None
        self.built = False

    def build_vocab(self):
        self.built = True

    def apply_edit(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


def test_apply_request_normalizes_easyedit_fields():
    editor = FakeEditor()
    hparams = AGIMWALHyperParams(
        model_name="fake",
        device="cpu",
        use_positive_prompts=True,
        positive_constraint_mode="ridge",
    )

    result = apply_agimwal_request(
        editor,
        {
            "prompt": "The capital of {} is",
            "subject": "Norpacia",
            "target_new": {"str": "Virelia"},
            "ground_truth": "OldCity",
            "relation_id": "capital",
            "rephrase_prompts": ["{} has its capital in"],
            "locality": {"neighborhood": {"prompt": ["The capital of France is"]}},
        },
        hparams,
    )

    assert result == {"ok": True}
    call = editor.calls[0]
    assert call["prompt"] == "The capital of Norpacia is"
    assert call["target"] == "Virelia"
    assert call["old_target"] == "OldCity"
    assert call["relation"] == "capital"
    assert call["positive_prompts"] == ["Norpacia has its capital in"]
    assert call["neg_prompts"] == ["The capital of France is"]
    assert call["positive_constraint_mode"] == "ridge"


def test_apply_to_model_uses_easyedit_backup_format():
    hparams = AGIMWALHyperParams(model_name="fake", device="cpu")
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.ones(2, 3)),
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(4, 3))),
    )
    editor = FakeEditor()

    edited, backup = apply_agimwal_to_model(
        model,
        tok=object(),
        request={"prompt": "A is", "subject": "A", "target_new": "B"},
        hparams=hparams,
        return_orig_weights=True,
        editor_factory=lambda *_args: editor,
    )

    assert edited is model
    assert editor.built
    assert editor.nt_sample_size == 500
    assert sorted(backup) == ["lm_head.weight", "model.embed_tokens.weight"]
    assert torch.equal(backup["lm_head.weight"], torch.ones(2, 3))


def test_wal_rome_backup_names_cover_auto_locate_candidates():
    hparams = AGIMWALHyperParams(
        model_name="fake",
        backend="wal_rome",
        rome_auto_locate=True,
        rome_candidate_layers="2,4",
    )
    assert easyedit_weight_names(object(), hparams) == [
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.4.mlp.down_proj.weight",
    ]


def test_registration_snippet_names_easyedit_contract():
    snippet = easyedit_registration_snippet()
    assert "AGIMWALHyperParams" in snippet
    assert "ALG_DICT['AGIMWAL']" in snippet

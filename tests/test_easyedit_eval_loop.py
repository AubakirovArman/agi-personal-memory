from __future__ import annotations

import json
import types

import pytest

from agim.eval import easyedit_eval_loop as loop
from agim.eval.easyedit_cli import build_parser


def _build_fact():
    return {
        "case_id": 1,
        "requested_rewrite": {
            "subject": "Alice",
            "relation_id": "P17",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "London"},
        },
    }


def _build_record():
    return {
        "prompt": "Alice was born in",
        "rephrase_prompts": ["Where was Alice born?"],
        "locality": {"neighborhood": {"prompt": ["q1", "q2"]}},
    }


class _DummyEditor:
    def __init__(self):
        self.applied = 0
        self.rolled_back = 0

    def apply_edit(self, *_args, **_kwargs):
        self.applied += 1
        return {
            "lm_backup": {0: 0.0},
            "emb_backup": {},
        }

    def rollback(self, _backup):
        self.rolled_back += 1


def _fake_post_bundle_factory(metrics_by_profile):
    def _fake_post_bundle(args, _model, _tok, _hparams,
                         _compute_edit_quality, _test_prediction_acc,
                         _record, pre, _device_id, fluency=False):
        profile = getattr(args, "positive_profile", "off")
        metrics = metrics_by_profile[profile]
        return {
            "pre": pre,
            "post": {
                "rewrite_acc": [metrics["rewrite"]],
                "rephrase_all_acc": [metrics["ps_all"]],
                "locality": {"neighborhood_acc": [metrics["locality"]]},
            },
            "generation": {"rewrite_acc": [metrics["generation"]]},
        }

    return _fake_post_bundle


def test_parse_candidate_grid_utility():
    assert loop._parse_candidate_grid("safe, positive_w025,,conservative") == [
        "safe", "positive_w025", "conservative"
    ]


def test_run_single_selects_highest_candidate_score(monkeypatch):
    editor = _DummyEditor()
    args = build_parser().parse_args([
        "--candidate-grid", "safe,positive_w025",
        "--candidate-locality-min", "0.95",
    ])

    fake_post_bundle = _fake_post_bundle_factory({
        "off": {"rewrite": 0.99, "ps_all": 0.20, "locality": 0.98, "generation": 0.0},
        "w025": {"rewrite": 0.98, "ps_all": 0.55, "locality": 0.97, "generation": 0.0},
    })

    monkeypatch.setattr(loop, "_post_bundle", fake_post_bundle)
    monkeypatch.setattr(loop, "edit_nt_metrics", lambda *_, **__: {"nt": "ok"})
    monkeypatch.setattr(loop, "print_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loop, "summarize_official", lambda *_args, **_kwargs: {})

    metrics, _, _ = loop.run_single(
        args=args,
        model=types.SimpleNamespace(),
        tok=types.SimpleNamespace(eos_token_id=1),
        hparams=types.SimpleNamespace(),
        editor=editor,
        facts=[_build_fact()],
        records=[_build_record()],
        compute_edit_quality=lambda *_, **__: {"pre": [0.1]},
        test_prediction_acc=lambda *_: 0.0,
        device_id=0,
    )

    row = metrics[0]
    assert row["candidate_grid_selected"] == "positive_w025"
    assert row["candidate_grid_evaluated"] == 2
    assert row["candidate_grid"] == ["safe", "positive_w025"]
    assert row["NT"]["nt"] == "ok"


def test_run_single_marks_no_candidates_passing_as_fallback(monkeypatch):
    editor = _DummyEditor()
    args = build_parser().parse_args([
        "--candidate-grid", "safe,positive_w025",
        "--candidate-locality-min", "1.0",
    ])

    fake_post_bundle = _fake_post_bundle_factory({
        "off": {"rewrite": 0.99, "ps_all": 0.20, "locality": 0.98, "generation": 0.0},
        "w025": {"rewrite": 0.98, "ps_all": 0.55, "locality": 0.97, "generation": 0.0},
    })
    monkeypatch.setattr(loop, "_post_bundle", fake_post_bundle)
    monkeypatch.setattr(loop, "edit_nt_metrics", lambda *_, **__: {"nt": "ok"})
    monkeypatch.setattr(loop, "print_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loop, "summarize_official", lambda *_args, **_kwargs: {})

    metrics, _, _ = loop.run_single(
        args=args,
        model=types.SimpleNamespace(),
        tok=types.SimpleNamespace(eos_token_id=1),
        hparams=types.SimpleNamespace(),
        editor=editor,
        facts=[_build_fact()],
        records=[_build_record()],
        compute_edit_quality=lambda *_, **__: {"pre": [0.1]},
        test_prediction_acc=lambda *_: 0.0,
        device_id=0,
    )

    row = metrics[0]
    assert row["edit_status"] == "no_commit"
    assert row["candidate_rejected_reason"] == "all_candidates_rejected_by_thresholds"
    assert row["candidate_grid_selected"] == "safe"


def test_run_single_uses_relation_profile_map(monkeypatch, tmp_path):
    map_path = tmp_path / "relation_profiles.json"
    map_path.write_text(json.dumps({"P17": {"positive_profile": "w025"}}), encoding="utf-8")
    used_profiles = []

    def fake_post_bundle(args, _model, _tok, _hparams,
                         _compute_edit_quality, _test_prediction_acc,
                         _record, pre, _device_id, fluency=False):
        used_profiles.append(getattr(args, "positive_profile", "off"))
        return {
            "pre": pre,
            "post": {
                "rewrite_acc": [0.97],
                "rephrase_all_acc": [0.32],
                "locality": {"neighborhood_acc": [0.96]},
            },
            "generation": {"rewrite_acc": [0.0]},
        }

    editor = _DummyEditor()
    args = build_parser().parse_args([
        "--relation-profile-map",
        str(map_path),
    ])
    monkeypatch.setattr(loop, "_post_bundle", fake_post_bundle)
    monkeypatch.setattr(loop, "edit_nt_metrics", lambda *_, **__: {"nt": "ok"})
    monkeypatch.setattr(loop, "print_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loop, "summarize_official", lambda *_args, **kwargs: {})

    metrics, _, _ = loop.run_single(
        args=args,
        model=types.SimpleNamespace(),
        tok=types.SimpleNamespace(eos_token_id=1),
        hparams=types.SimpleNamespace(),
        editor=editor,
        facts=[_build_fact()],
        records=[_build_record()],
        compute_edit_quality=lambda *_, **__: {"pre": [0.1]},
        test_prediction_acc=lambda *_: 0.0,
        device_id=0,
    )

    row = metrics[0]
    assert row["relation_profile"] == {"positive_profile": "w025"}
    assert row["candidate_selected"] is None
    assert used_profiles == ["w025"]


def test_parse_relation_profile_map_missing_path_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="relation-profile-map path not found"):
        loop._parse_relation_profile_map(str(tmp_path / "unknown.json"))


def test_apply_relation_profile_rejects_unknown_named_profile(monkeypatch, tmp_path):
    relation_args = build_parser().parse_args([])
    with pytest.raises(ValueError, match="Unknown positive profile in relation map"):
        loop._apply_relation_profile(
            relation_args,
            {"P17": {"positive_profile": "nope"}},
            "P17",
        )


def test_run_evaluation_loop_side_slot_uses_relation_profile_map(monkeypatch, tmp_path):
    map_path = tmp_path / "relation_profiles.json"
    map_path.write_text(json.dumps({"P17": {"positive_profile": "w025"}}, ensure_ascii=False), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_sequential_side_slot(
        *,
        args_by_idx=None,
        relation_profiles=None,
        **_kwargs,
    ):
        captured["relation_profiles"] = relation_profiles
        captured["positive_profiles"] = [getattr(a, "positive_profile", "off") for a in args_by_idx]
        return [
            {
                "case_id": 1,
                "relation_id": "P17",
                "edit_backend": "side_slot",
            }
        ], [0.1], {}

    import agim.eval.easyedit_side_slot_loop as side_slot_loop

    monkeypatch.setattr(
        side_slot_loop,
        "run_sequential_side_slot",
        _fake_run_sequential_side_slot,
    )

    facts = [_build_fact()]
    records = [_build_record()]
    args = build_parser().parse_args([
        "--sequential-edit",
        "--edit-backend",
        "side_slot",
        "--relation-profile-map",
        str(map_path),
    ])
    metrics, _, _ = loop.run_evaluation_loop(
        args=args,
        model=types.SimpleNamespace(),
        tok=types.SimpleNamespace(),
        hparams=types.SimpleNamespace(),
        editor=_DummyEditor(),
        facts=facts,
        records=records,
        compute_edit_quality=lambda *_: {"rewrite_acc": [0.0]},
        test_prediction_acc=lambda *_: 0.0,
        device_id=0,
    )

    assert captured["positive_profiles"] == ["w025"]
    assert captured["relation_profiles"] == [{"positive_profile": "w025"}]
    assert metrics[0]["edit_backend"] == "side_slot"

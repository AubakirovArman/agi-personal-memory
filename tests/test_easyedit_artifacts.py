from types import SimpleNamespace

import torch
import pytest

from agim.eval.easyedit_cli import build_parser
from agim.eval.easyedit_dry_run import dry_run_payload
from agim.eval.easyedit_failures import collect_failures, failure_summary
from agim.eval.easyedit_payload import build_payload
from agim.eval.easyedit_presets import apply_preset
from agim.eval.easyedit_run_metadata import method_profile_id, parse_failure_families
from agim.model.wal_dual_editor import WALDualLayerEditor


def test_easyedit_random_50_preset_sets_reproducible_sample():
    args = apply_preset(build_parser().parse_args(["--preset", "random_50_seed_43"]))

    assert args.n == 50
    assert args.sample_policy == "random"
    assert args.seed == 43
    assert args.output.endswith("random_50_seed_43.json")


def test_dry_run_payload_summarizes_selected_records():
    args = SimpleNamespace(
        model="llama",
        device="cuda:2",
        dataset="counterfact.json",
        sample_policy="random",
        seed=42,
        n=1,
    )
    fact = {
        "case_id": 7,
        "requested_rewrite": {
            "relation_id": "P17",
            "subject": "Alice",
            "target_new": {"str": "Paris"},
            "target_true": {"str": "Rome"},
        },
    }
    record = {
        "prompt": "Alice was born in",
        "rephrase_prompts": ["Birthplace?", "Where born?"],
        "locality": {"neighborhood": {"prompt": ["n1", "n2", "n3"]}},
    }

    payload = dry_run_payload(
        args=args,
        dataset_sha256="abc",
        all_facts=[fact],
        facts=[fact],
        records=[record],
        locality_limit=None,
    )

    assert payload["would_load_model"] is False
    assert payload["dataset"]["relation_counts"] == {"P17": 1}
    assert payload["record_stats"]["rephrase_prompt_counts"]["max"] == 2
    assert payload["record_stats"]["locality_prompt_counts"]["mean"] == 3.0


def test_failure_summary_extracts_rephrase_and_locality_failures():
    rows = [
        {
            "case_id": 1,
            "relation_id": "P17",
            "requested_rewrite": {
                "subject": "Alice",
                "prompt": "Alice was born in",
                "target_new": {"str": "Paris"},
                "target_true": {"str": "Rome"},
            },
            "post": {
                "rewrite_acc": [1.0],
                "rephrase_acc": [1.0],
                "rephrase_all_acc": [1.0, 0.0],
                "locality": {"neighborhood_acc": [1.0, 0.0]},
            },
            "generation": {
                "rewrite_acc": [0.0],
                "rephrase_acc": [1.0],
                "rephrase_all_acc": [1.0],
            },
        }
    ]

    families = parse_failure_families("tf,vanilla_gen")
    failures = collect_failures(rows, families)
    summary = failure_summary(rows, families)

    assert failures[0]["failure_modes"] == [
        "tf_ps_all",
        "tf_locality",
        "gen_rewrite",
    ]
    assert summary["n_failed_cases"] == 1
    assert summary["failure_families"] == ["tf", "vanilla_gen"]
    assert summary["failed_by_relation_id"] == {"P17": 1}


def test_failure_summary_defaults_exclude_vanilla_generation():
    rows = [
        {
            "case_id": 1,
            "post": {"rewrite_acc": [1.0]},
            "generation": {"rewrite_acc": [0.0]},
        }
    ]

    assert collect_failures(rows) == []
    assert failure_summary(rows)["failure_families"] == ["tf", "ctx_gen", "prob"]


def test_method_profile_id_names_common_operating_points():
    assert method_profile_id(build_parser().parse_args([])) == "single_loc"
    assert method_profile_id(build_parser().parse_args(["--neg-prompt-limit", "4"])) == "single_ps"
    assert method_profile_id(build_parser().parse_args(["--sequential-edit"])) == "seq_tuned"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--history-slot-mode", "relation",
    ])) == "seq_relation_slots"


def test_build_payload_emits_schema_profile_and_digest_metadata():
    args = build_parser().parse_args(["--n", "1", "--failure-families", "tf"])
    metrics = [{"post": {"rewrite_acc": [0.0]}, "generation": {"rewrite_acc": [1.0]}}]

    payload = build_payload(
        args=args,
        metrics=metrics,
        retention={},
        summary={"post": {"rewrite_acc": 0.0}},
        elapsed=1.0,
        dataset_sha256="abc",
        all_facts=[{"case_id": 1}],
        facts=[{"case_id": 1, "requested_rewrite": {"relation_id": "P17"}}],
        locality_limit=None,
    )

    assert payload["artifact_schema_version"] == "easyedit_official.v2"
    assert payload["method_profile_id"] == "single_loc"
    assert payload["base_model_digest"].startswith("sha256:")
    assert payload["atoms_digest"] is None
    assert payload["failure_analysis"]["failure_families"] == ["tf"]


def test_failure_summary_accepts_string_targets_from_easyedit_rows():
    rows = [
        {
            "case_id": 2,
            "relation_id": "P19",
            "requested_rewrite": {
                "subject": "Bob",
                "prompt": "Bob was born in",
                "target_new": "Berlin",
                "target_true": "Paris",
            },
            "post": {"rewrite_acc": [0.0]},
        }
    ]

    failure = collect_failures(rows)[0]

    assert failure["target_new"] == "Berlin"
    assert failure["target_true"] == "Paris"


def test_wal_dual_relation_history_basis_and_rollback():
    editor = WALDualLayerEditor(object(), object(), device="cpu")
    global_key = torch.tensor([1.0, 0.0])
    p17_key = torch.tensor([0.0, 1.0])
    editor._edit_key_basis = [global_key]
    editor._relation_key_basis = {"P17": [p17_key]}

    assert torch.equal(editor._history_basis(10)[0], global_key)
    assert torch.equal(editor._history_basis(10, "P17", "relation")[0], p17_key)

    editor._relation_key_basis["P17"].append(torch.tensor([1.0, 1.0]))
    editor.rollback({"history_len": 0, "relation_key": "P17", "relation_history_len": 1})

    assert editor._edit_key_basis == []
    assert len(editor._relation_key_basis["P17"]) == 1
    assert torch.equal(editor._relation_key_basis["P17"][0], p17_key)


def test_wal_dual_projected_positive_keys_respect_protected_basis():
    primary = torch.tensor([1.0, 0.0])
    positive = torch.tensor([1.0, 1.0])
    protected = [torch.tensor([1.0, 0.0])]

    combined = WALDualLayerEditor._combine_positive_keys(
        primary,
        [positive],
        weight=1.0,
        protected_basis=protected,
        projection_strength=1.0,
        projection_mode="orthogonal",
    )

    assert torch.dot(combined, primary) == pytest.approx(2 ** -0.5)
    assert torch.dot(combined, torch.tensor([0.0, 1.0])) == pytest.approx(2 ** -0.5)

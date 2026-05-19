from types import SimpleNamespace

import torch
import pytest

from agim.eval.easyedit_cli import build_parser
from agim.eval.easyedit_dry_run import dry_run_payload
from agim.eval.easyedit_failures import collect_failures, failure_summary
from agim.eval.easyedit_payload import build_payload
from agim.eval.easyedit_presets import apply_preset
from agim.eval.easyedit_relation_banks import (
    preload_relation_protected_banks,
    relation_locality_prompts,
)
from agim.eval.easyedit_run_metadata import method_profile_id, parse_failure_families
from agim.model.wal_dual_editor import WALDualLayerEditor


def test_easyedit_random_50_preset_sets_reproducible_sample():
    args = apply_preset(build_parser().parse_args(["--preset", "random_50_seed_43"]))

    assert args.n == 50
    assert args.sample_policy == "random"
    assert args.seed == 43
    assert args.output.endswith("random_50_seed_43.json")


def test_psall_improvement_presets_are_wiredup():
    args_objective = apply_preset(build_parser().parse_args([
        "--preset",
        "ablation_objective_balance_seed42",
    ]))
    assert args_objective.positive_profile == "w015"
    assert args_objective.anti_profile == "target_low"
    assert args_objective.n == 50
    assert args_objective.seed == 42

    args_rerank = apply_preset(build_parser().parse_args([
        "--preset",
        "ablation_decode_rerank_seed42",
    ]))
    assert args_rerank.candidate_grid == "safe,positive_w025"
    assert args_rerank.candidate_locality_min == 0.95
    assert args_rerank.candidate_rewrite_min == 0.95
    assert args_rerank.anti_profile == "target_low"


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
    assert build_parser().parse_args([]).clamp_eos == 0.0


def test_cli_supports_constrained_projection_and_anti_scope():
    args = build_parser().parse_args([
        "--clamp-anti-scope",
        "subject",
        "--positive-constraint-mode",
        "constrained",
        "--positive-constraint-k-pos",
        "5",
        "--positive-constraint-k-neg",
        "2",
    ])

    assert args.clamp_anti_scope == "subject"
    assert args.positive_constraint_mode == "constrained"
    assert args.positive_constraint_k_pos == 5
    assert args.positive_constraint_k_neg == 2
    assert args.anti_profile == "off"
    assert args.positive_profile == "off"
    assert build_parser().parse_args(["--anti-profile", "target_low"]).anti_profile == "target_low"
    assert (
        build_parser().parse_args(["--positive-profile", "w025"]).positive_profile
        == "w025"
    )
    assert build_parser().parse_args([
        "--relation-profile-map",
        "/tmp/relation_profile_map.json",
    ]).relation_profile_map == "/tmp/relation_profile_map.json"
    assert build_parser().parse_args([
        "--candidate-grid",
        "safe,positive_w025",
    ]).candidate_grid == "safe,positive_w025"
    assert build_parser().parse_args([
        "--candidate-locality-min",
        "0.95",
        "--candidate-rewrite-min",
        "0.9",
        "--candidate-rerank-metric",
        "rewrite_then_psall",
    ]).candidate_locality_min == 0.95
    assert method_profile_id(build_parser().parse_args([])) == "single_loc"
    assert method_profile_id(build_parser().parse_args(["--neg-prompt-limit", "4"])) == "single_ps"
    assert method_profile_id(build_parser().parse_args(["--sequential-edit"])) == "seq_tuned"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--edit-backend", "side_slot",
    ])) == "seq_side_slot"
    assert method_profile_id(build_parser().parse_args([
        "--edit-backend", "wal_rome",
    ])) == "single_wal_rome"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--edit-backend", "wal_rome",
    ])) == "seq_wal_rome"
    assert method_profile_id(build_parser().parse_args([
        "--edit-backend", "wal_memit",
    ])) == "single_wal_memit"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--edit-backend", "wal_memit",
    ])) == "seq_wal_memit"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--history-slot-mode", "relation",
    ])) == "seq_relation_slots"
    assert method_profile_id(build_parser().parse_args([
        "--sequential-edit", "--relation-protected-mode", "preload",
    ])) == "seq_relation_protected_preload"
    assert method_profile_id(build_parser().parse_args([
        "--no-wal-encode-updates",
    ])) == "single_exact_additive"
    assert method_profile_id(build_parser().parse_args([
        "--anti-profile",
        "target_low",
    ])) == "single_loc_anti_target_low"
    assert method_profile_id(build_parser().parse_args([
        "--positive-profile",
        "w025",
    ])) == "single_loc_pos_w025"
    assert method_profile_id(build_parser().parse_args([
        "--anti-profile",
        "subject_low",
        "--positive-profile",
        "w025_ridge",
    ])) == "single_loc_anti_subject_low_pos_w025_ridge"


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
    assert payload["hyperparams"]["wal_encode_updates"] is True
    assert payload["hyperparams"]["relation_protected_mode"] == "none"
    assert payload["hyperparams"]["anti_profile"] == "off"
    assert payload["hyperparams"]["positive_profile"] == "off"
    assert payload["hyperparams"]["candidate_grid"] == ""
    assert payload["hyperparams"]["candidate_rerank_metric"] == "psall_guarded"
    assert payload["hyperparams"]["relation_profile_map"] == ""
    assert payload["relation_protected_banks"] == {
        "mode": "none", "state_namespace": "default", "relations": {}}
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


def test_wal_dual_relation_protected_bank_and_rollback():
    editor = WALDualLayerEditor(object(), object(), device="cpu")
    editor._add_relation_protected_keys(
        "P17",
        [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 3.0])],
        limit=1,
    )

    assert len(editor._relation_protected_basis["P17"]) == 1
    bank = editor._relation_protected_bank("P17", 10)
    assert torch.equal(bank[0], torch.tensor([0.0, 1.0]))

    editor._add_relation_protected_keys("P17", [torch.tensor([1.0, 1.0])], limit=2)
    editor.rollback({"relation_key": "P17", "relation_protected_len": 1})

    assert len(editor._relation_protected_basis["P17"]) == 1
    assert torch.equal(editor._relation_protected_basis["P17"][0], torch.tensor([0.0, 1.0]))


def test_relation_protected_preload_groups_locality_prompts():
    facts = [
        {"requested_rewrite": {"relation_id": "P17"}},
        {"requested_rewrite": {"relation_id": "P17"}},
        {"requested_rewrite": {"relation_id": "P19"}},
    ]
    records = [
        {"locality": {"neighborhood": {"prompt": ["a", "b"]}}},
        {"locality": {"neighborhood": {"prompt": ["a", "c"]}}},
        {"locality": {"neighborhood": {"prompt": ["d"]}}},
    ]

    assert relation_locality_prompts(facts, records, prompt_limit=2) == {
        "P17": ["a", "b", "c"],
        "P19": ["d"],
    }


class _FakeRelationBankEditor:
    def __init__(self):
        self._relation_protected_basis = {}

    def _prompt_keys(self, prompts, limit):
        return [torch.tensor([float(i + 1), 0.0]) for i, _ in enumerate(prompts[:limit])]

    def _activate_state_namespace(self, namespace):
        self.state_namespace = namespace

    def _sync_active_state(self):
        self.synced = True

    _add_relation_protected_keys = WALDualLayerEditor._add_relation_protected_keys


def test_preload_relation_protected_banks_adds_keys():
    editor = _FakeRelationBankEditor()
    args = SimpleNamespace(
        relation_protected_mode="preload",
        relation_protected_prompt_limit=1,
        max_relation_protected_keys=8,
    )
    facts = [{"requested_rewrite": {"relation_id": "P17"}}]
    records = [{"locality": {"neighborhood": {"prompt": ["a", "b"]}}}]

    summary = preload_relation_protected_banks(editor, args, facts, records)

    assert summary == {
        "mode": "preload",
        "state_namespace": "default",
        "relations": {"P17": {"prompts": 1, "keys": 1}},
    }
    assert len(editor._relation_protected_basis["P17"]) == 1


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

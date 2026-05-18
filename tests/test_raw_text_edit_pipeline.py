import json

import pytest

from agim.eval.raw_text_edit_pipeline import (
    parse_raw_update,
    patch_drafts_payload,
    proposals_payload,
    score_raw_text_case,
    scored_raw_text_payload,
    summarize_scored_raw_text_rows,
)
from agim.model.patch_service import PatchService


def test_parse_relation_sentence_to_requested_rewrite():
    proposal = parse_raw_update("The capital of France is Berlin.")

    assert proposal.subject == "France"
    assert proposal.relation_id == "capital"
    assert proposal.target_new == "Berlin"
    assert proposal.prompt == "The capital of France is"
    assert proposal.to_requested_rewrite("Paris")["requested_rewrite"] == {
        "subject": "France",
        "prompt": "The capital of {} is",
        "relation_id": "capital",
        "target_new": {"str": "Berlin"},
        "target_true": {"str": "Paris"},
    }


def test_parse_simple_is_sentence():
    proposal = parse_raw_update("Alice is a physicist.")

    assert proposal.subject == "Alice"
    assert proposal.relation_id == "is"
    assert proposal.target_new == "a physicist"
    assert proposal.confidence == 0.6


def test_parse_json_update():
    raw = json.dumps({
        "subject": "France",
        "relation_id": "capital city",
        "target_new": "Berlin",
    })

    proposal = parse_raw_update(raw)

    assert proposal.relation_id == "capital_city"
    assert proposal.prompt == "The capital city of France is"
    assert proposal.parser == "json"


def test_proposals_payload_contains_requested_rewrites():
    payload = proposals_payload([
        "The capital of France is Berlin.",
        "Alice is a physicist.",
    ])

    assert payload["artifact_schema_version"] == "raw_text_edit_proposals.v1"
    assert payload["n"] == 2
    assert payload["requested_rewrites"][0]["requested_rewrite"]["subject"] == "France"


def test_parse_raw_update_rejects_unparseable_text():
    with pytest.raises(ValueError):
        parse_raw_update("No structured update here")


def test_raw_text_proposal_builds_patch_service_draft():
    proposal = parse_raw_update("The capital of France is Berlin.")
    artifact = proposal.to_patch_artifact(
        base_model_digest="model-sha",
        target_true="Paris",
    )
    service = PatchService()

    record = service.propose_patch(artifact)

    assert record["status"] == "proposed"
    assert record["row_counts"] == {}
    assert artifact.patch_id.startswith("raw-")
    assert artifact.target_true == "Paris"
    assert artifact.metadata["requires_backend_materialization"] is True
    assert artifact.metadata["requested_rewrite"]["prompt"] == "The capital of {} is"


def test_patch_drafts_payload_is_reproducible():
    payload = patch_drafts_payload(
        ["The capital of France is Berlin."],
        base_model_digest="model-sha",
        target_true_by_subject={"France": "Paris"},
    )

    artifact = payload["artifacts"][0]
    assert payload["artifact_schema_version"] == "raw_text_patch_drafts.v1"
    assert artifact["base_model_digest"] == "model-sha"
    assert artifact["target_true"] == "Paris"
    assert artifact["rows"] == []


def test_score_raw_text_case_checks_outputs_and_canaries():
    proposal = parse_raw_update("The capital of France is Berlin.")

    row = score_raw_text_case(proposal.__dict__, {
        "case_id": 0,
        "direct_output": "The capital of France is Berlin.",
        "patch_status": "applied",
        "canary_results": {"rewrite": True, "locality": False},
        "rollback_verified": True,
    })

    assert row["direct_success"] is True
    assert row["service_ready"] == 1.0
    assert row["canary_pass_rate"] == 0.5
    assert row["raw_text_composite_acc"] == 0.875


def test_scored_raw_text_payload_summarizes_model_outputs():
    adapter = proposals_payload([
        "The capital of France is Berlin.",
        "Alice is a physicist.",
    ])
    outputs = {
        "artifact_schema_version": "raw_text_model_outputs.v1",
        "cases": [
            {"case_id": 0, "direct_output": "Berlin", "patch_status": "applied"},
            {"case_id": 1, "direct_output": "chemist", "patch_status": "proposed"},
        ],
    }

    payload = scored_raw_text_payload(adapter, outputs, "raw.json")

    assert payload["artifact_schema_version"] == "raw_text_scored_outputs.v1"
    assert payload["summary"]["rewrite_acc"] == 0.5
    assert payload["summary"]["service_ready_rate"] == 0.5


def test_summarize_scored_raw_text_rows_reports_composite():
    summary = summarize_scored_raw_text_rows([
        {"direct_success": True, "prompt_success": None, "service_ready": 1.0,
         "canary_pass_rate": 1.0, "rollback_verified": None,
         "raw_text_composite_acc": 1.0},
        {"direct_success": False, "prompt_success": None, "service_ready": 0.0,
         "canary_pass_rate": 0.5, "rollback_verified": None,
         "raw_text_composite_acc": 0.166667},
    ])

    assert summary["raw_text_composite_acc"] == 0.583333

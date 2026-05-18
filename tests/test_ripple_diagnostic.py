from agim.eval.ripple_diagnostic import (
    diagnostic_payload,
    normalize_ripple_record,
    ripple_dataset_payload,
    score_ripple_case,
    scored_ripple_payload,
    summarize_scored_ripple_rows,
    summarize_ripple_by_relation,
    summarize_ripple_rows,
)


def _row(case_id, relation_id, rewrite, locality, prob_locality=None):
    row = {
        "case_id": case_id,
        "relation_id": relation_id,
        "post": {
            "rewrite_acc": [rewrite],
            "locality": {"neighborhood_acc": locality},
        },
    }
    if prob_locality is not None:
        row["probability"] = {"locality": {"neighborhood_acc": prob_locality}}
    return row


def test_summarize_ripple_rows_reports_break_rate():
    rows = [
        _row(1, "P17", 1.0, [1.0, 1.0], [1.0]),
        _row(2, "P17", 1.0, [0.0, 1.0], [0.0]),
        _row(3, "P19", 0.0, [0.0, 0.0], [1.0]),
    ]

    summary = summarize_ripple_rows(rows)

    assert summary["n"] == 3
    assert summary["direct_rewrite_acc"] == 0.666667
    assert summary["related_preservation_acc"] == 0.5
    assert summary["prob_related_preservation_acc"] == 0.666667
    assert summary["ripple_break_rate"] == 0.333333


def test_summarize_ripple_by_relation_groups_rows():
    rows = [
        _row(1, "P17", 1.0, [1.0]),
        _row(2, "P19", 0.0, [0.0]),
    ]

    by_relation = summarize_ripple_by_relation(rows)

    assert by_relation["P17"]["direct_rewrite_acc"] == 1.0
    assert by_relation["P19"]["related_preservation_acc"] == 0.0


def test_diagnostic_payload_preserves_source_metadata():
    artifact = {
        "artifact_schema_version": "easyedit_official.v2",
        "method_profile_id": "single_loc",
        "metrics": [_row(1, "P17", 1.0, [1.0])],
    }

    payload = diagnostic_payload(artifact, "result.json")

    assert payload["artifact_schema_version"] == "ripple_style_diagnostic.v1"
    assert payload["source"] == "result.json"
    assert payload["source_method_profile_id"] == "single_loc"
    assert "not an official RippleEdits" in payload["caveat"]


def test_normalize_ripple_record_keeps_related_facts_and_locality():
    record = {
        "id": "rp-1",
        "requested_rewrite": {
            "prompt": "The capital of {} is",
            "subject": "France",
            "relation_id": "P36",
            "target_new": {"str": "Berlin"},
        },
        "related_facts": [
            {"prompt": "France's government sits in", "ground_truth": "Berlin"}
        ],
        "neighborhood_prompts": ["The capital of Germany is"],
    }

    case = normalize_ripple_record(record, 0)

    assert case["source_record_id"] == "rp-1"
    assert case["request"]["prompt"] == "The capital of France is"
    assert case["related_facts"][0]["ground_truth"] == "Berlin"
    assert case["locality"]["neighborhood"]["prompt"] == ["The capital of Germany is"]


def test_ripple_dataset_payload_has_adapter_caveat():
    payload = ripple_dataset_payload([
        {
            "prompt": "{} is",
            "subject": "A",
            "target_new": "B",
            "ripple_prompts": ["What follows from A?"],
        }
    ], "ripple.json")

    assert payload["artifact_schema_version"] == "ripple_dataset_adapter.v1"
    assert payload["n"] == 1
    assert "not a scored RippleEdits" in payload["caveat"]


def test_score_ripple_case_checks_direct_related_and_locality_outputs():
    case = normalize_ripple_record({
        "requested_rewrite": {
            "prompt": "{} is",
            "subject": "A",
            "target_new": "B",
        },
        "related_facts": [
            {"prompt": "A implies", "ground_truth": "C"},
            {"prompt": "A also implies", "ground_truth": "D"},
        ],
        "neighborhood_prompts": ["Neighbor prompt"],
    }, 3)
    row = score_ripple_case(case, {
        "direct_output": "A is B",
        "related_outputs": ["therefore C", "miss"],
        "locality_outputs": ["stable answer"],
    })

    assert row["case_id"] == 3
    assert row["direct_success"] is True
    assert row["related_scores"] == [True, False]
    assert row["related_acc"] == 0.5
    assert row["ripple_break"] is True


def test_scored_ripple_payload_summarizes_outputs():
    adapter = ripple_dataset_payload([
        {
            "prompt": "{} is",
            "subject": "A",
            "target_new": "B",
            "related_facts": [{"prompt": "A implies", "ground_truth": "C"}],
        }
    ], "ripple.json")
    outputs = {
        "artifact_schema_version": "ripple_model_outputs.v1",
        "cases": [{
            "case_id": 0,
            "direct_output": "A is B",
            "related_outputs": ["C follows"],
        }],
    }

    payload = scored_ripple_payload(adapter, outputs, "adapter.json")

    assert payload["artifact_schema_version"] == "ripple_scored_outputs.v1"
    assert payload["summary"]["direct_rewrite_acc"] == 1.0
    assert payload["summary"]["related_acc"] == 1.0
    assert "documented model-editing run" in payload["caveat"]


def test_summarize_scored_ripple_rows_reports_break_rate():
    summary = summarize_scored_ripple_rows([
        {"direct_success": True, "related_acc": 0.0,
         "locality_response_rate": 1.0, "ripple_break": True},
        {"direct_success": False, "related_acc": 1.0,
         "locality_response_rate": 0.0, "ripple_break": False},
    ])

    assert summary["direct_rewrite_acc"] == 0.5
    assert summary["related_acc"] == 0.5
    assert summary["ripple_break_rate"] == 0.5

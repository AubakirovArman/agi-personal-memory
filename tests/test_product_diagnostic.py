from agim.eval.product_diagnostic import (
    diagnostic_payload,
    normalize_product_record,
    product_dataset_payload,
    score_product_case,
    scored_product_payload,
    summarize_scored_product_rows,
    summarize_product_by_relation,
    summarize_product_rows,
)


def _row(case_id, relation_id, rewrite, ps_all, locality, portability=None):
    post = {
        "rewrite_acc": [rewrite],
        "rephrase_all_acc": ps_all,
        "locality": {"neighborhood_acc": locality},
    }
    if portability is not None:
        post["portability"] = {"one_hop_acc": portability}
    return {"case_id": case_id, "relation_id": relation_id, "post": post}


def test_product_summary_combines_available_signals():
    rows = [
        _row(1, "P17", 1.0, [1.0, 0.0], [1.0], [1.0]),
        _row(2, "P17", 0.0, [0.0, 0.0], [0.0], [0.0]),
    ]

    summary = summarize_product_rows(rows)

    assert summary["rewrite_acc"] == 0.5
    assert summary["paraphrase_all_acc"] == 0.25
    assert summary["locality_acc"] == 0.5
    assert summary["portability_acc"] == 0.5
    assert summary["product_composite_acc"] == 0.4375


def test_product_summary_groups_by_relation():
    rows = [
        _row(1, "P17", 1.0, [1.0], [1.0]),
        _row(2, "P19", 0.0, [0.0], [0.0]),
    ]

    by_relation = summarize_product_by_relation(rows)

    assert by_relation["P17"]["product_composite_acc"] == 1.0
    assert by_relation["P19"]["rewrite_acc"] == 0.0


def test_product_payload_keeps_external_benchmark_caveat():
    artifact = {
        "artifact_schema_version": "easyedit_official.v2",
        "method_profile_id": "single_loc",
        "metrics": [_row(1, "P17", 1.0, [1.0], [1.0])],
    }

    payload = diagnostic_payload(artifact, "result.json")

    assert payload["artifact_schema_version"] == "product_diagnostic.v1"
    assert payload["source_method_profile_id"] == "single_loc"
    assert "not an external KnowEdit" in payload["caveat"]


def test_normalize_product_record_keeps_knowedit_groups():
    record = {
        "id": "ke-1",
        "subject": "France",
        "prompt": "The capital of France is",
        "target_new": "Berlin",
        "ground_truth": "Paris",
        "portability": {
            "Reasoning": [
                {"prompt": "France's new capital is in", "ground_truth": "Germany"}
            ]
        },
        "locality": {
            "Relation_Specificity": [
                {"prompt": ["The capital of Germany is"], "ground_truth": ["Berlin"]}
            ]
        },
    }

    case = normalize_product_record(record, 0)

    assert case["source_record_id"] == "ke-1"
    assert case["request"]["target_true"] == "Paris"
    assert case["portability"]["Reasoning"]["ground_truth"] == ["Germany"]
    assert case["locality"]["Relation_Specificity"]["prompt"] == [
        "The capital of Germany is"
    ]


def test_product_dataset_payload_has_adapter_caveat():
    payload = product_dataset_payload([
        {"concept": "A", "text": "{} is", "labels": ["B"]},
    ], source="knowedit.json", benchmark_name="knowedit")

    assert payload["artifact_schema_version"] == "product_dataset_adapter.v1"
    assert payload["benchmark_name"] == "knowedit"
    assert payload["cases"][0]["request"]["subject"] == "A"
    assert "not a scored external leaderboard" in payload["caveat"]


def test_score_product_case_checks_rewrite_locality_and_portability():
    case = normalize_product_record({
        "subject": "France",
        "prompt": "The capital of France is",
        "target_new": "Berlin",
        "locality": {"Relation": [
            {"prompt": "The capital of Germany is", "ground_truth": "Berlin"}
        ]},
        "portability": {"Reasoning": [
            {"prompt": "France's new capital is in", "ground_truth": "Germany"}
        ]},
    }, 1)
    row = score_product_case(case, {
        "direct_output": "Berlin",
        "locality_outputs": {"Relation": ["Berlin"]},
        "portability_outputs": {"Reasoning": ["Germany"]},
    })

    assert row["direct_success"] is True
    assert row["locality_acc"] == 1.0
    assert row["portability_acc"] == 1.0
    assert row["product_composite_acc"] == 1.0


def test_scored_product_payload_summarizes_outputs():
    adapter = product_dataset_payload([
        {"concept": "A", "text": "{} is", "labels": ["B"]},
    ])
    outputs = {
        "artifact_schema_version": "product_model_outputs.v1",
        "cases": [{"case_id": 0, "direct_output": "A is B"}],
    }

    payload = scored_product_payload(adapter, outputs, "adapter.json")

    assert payload["artifact_schema_version"] == "product_scored_outputs.v1"
    assert payload["summary"]["rewrite_acc"] == 1.0
    assert "documented model-editing run" in payload["caveat"]


def test_summarize_scored_product_rows_reports_composite():
    summary = summarize_scored_product_rows([
        {"direct_success": True, "locality_acc": 1.0,
         "portability_acc": 0.0, "product_composite_acc": 0.666667},
        {"direct_success": False, "locality_acc": 0.0,
         "portability_acc": 1.0, "product_composite_acc": 0.333333},
    ])

    assert summary["rewrite_acc"] == 0.5
    assert summary["product_composite_acc"] == 0.5

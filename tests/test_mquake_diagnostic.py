from agim.eval.mquake_diagnostic import (
    diagnostic_payload,
    mquake_dataset_payload,
    normalize_mquake_record,
    score_mquake_case,
    scored_mquake_payload,
    summarize_scored_mquake_rows,
    summarize_mquake_by_relation,
    summarize_mquake_rows,
)


def _row(case_id, relation_id, rewrite, one_hop):
    return {
        "case_id": case_id,
        "relation_id": relation_id,
        "post": {
            "rewrite_acc": [rewrite],
            "portability": {"one_hop_acc": one_hop},
        },
    }


def test_summarize_mquake_rows_reports_composite_and_hop_failures():
    rows = [
        _row(1, "P17", 1.0, [1.0]),
        _row(2, "P17", 1.0, [0.0]),
        _row(3, "P19", 0.0, [0.0]),
    ]

    summary = summarize_mquake_rows(rows)

    assert summary["n"] == 3
    assert summary["direct_rewrite_acc"] == 0.666667
    assert summary["multi_hop_acc"] == 0.333333
    assert summary["composite_acc"] == 0.5
    assert summary["direct_success_hop_fail_rate"] == 0.333333


def test_summarize_mquake_by_relation_groups_rows():
    rows = [
        _row(1, "P17", 1.0, [1.0]),
        _row(2, "P19", 0.0, [0.0]),
    ]

    by_relation = summarize_mquake_by_relation(rows)

    assert by_relation["P17"]["multi_hop_acc"] == 1.0
    assert by_relation["P19"]["direct_rewrite_acc"] == 0.0


def test_mquake_diagnostic_payload_preserves_source_metadata():
    artifact = {
        "artifact_schema_version": "easyedit_official.v2",
        "method_profile_id": "single_loc",
        "metrics": [_row(1, "P17", 1.0, [1.0])],
    }

    payload = diagnostic_payload(artifact, "result.json")

    assert payload["artifact_schema_version"] == "mquake_style_diagnostic.v1"
    assert payload["source"] == "result.json"
    assert payload["source_method_profile_id"] == "single_loc"
    assert "not an official MQuAKE" in payload["caveat"]


def test_normalize_mquake_record_keeps_multi_edit_and_hop_questions():
    record = {
        "id": "mq-1",
        "requested_rewrite": [
            {
                "prompt": "The capital of {} is",
                "subject": "France",
                "relation_id": "P36",
                "target_new": {"str": "Berlin"},
                "target_true": {"str": "Paris"},
            },
            {
                "prompt": "{} is located in",
                "subject": "Berlin",
                "relation_id": "P17",
                "target_new": {"str": "Germany"},
            },
        ],
        "questions": ["Which country contains France's new capital?"],
        "new_answer": "Germany",
    }

    case = normalize_mquake_record(record, 0)

    assert case["source_record_id"] == "mq-1"
    assert len(case["requests"]) == 2
    assert case["requests"][0]["prompt"] == "The capital of France is"
    assert case["portability"]["multi_hop"]["ground_truth"] == ["Germany"]


def test_mquake_dataset_payload_has_adapter_caveat():
    payload = mquake_dataset_payload([
        {
            "requested_rewrite": {
                "prompt": "{} is",
                "subject": "A",
                "target_new": "B",
            },
            "questions": ["Where is B?"],
            "new_answer": "C",
        }
    ], "mquake.json")

    assert payload["artifact_schema_version"] == "mquake_dataset_adapter.v1"
    assert payload["n"] == 1
    assert "requires a model editor evaluation pass" in payload["caveat"]


def test_score_mquake_case_checks_direct_and_multi_hop_outputs():
    case = normalize_mquake_record({
        "requested_rewrite": [
            {"prompt": "{} is", "subject": "A", "target_new": "B"},
            {"prompt": "{} is in", "subject": "B", "target_new": "C"},
        ],
        "questions": ["Where does A lead?"],
        "new_answer": "C",
    }, 7)
    row = score_mquake_case(case, {
        "direct_outputs": ["A is B", {"text": "B is not D"}],
        "hop_outputs": ["The answer is C."],
    })

    assert row["case_id"] == 7
    assert row["direct_scores"] == [True, False]
    assert row["hop_scores"] == [True]
    assert row["direct_acc"] == 0.5
    assert row["multi_hop_acc"] == 1.0


def test_scored_mquake_payload_summarizes_outputs():
    adapter = mquake_dataset_payload([
        {
            "requested_rewrite": {
                "prompt": "{} is",
                "subject": "A",
                "target_new": "B",
            },
            "questions": ["Where is B?"],
            "new_answer": "C",
        }
    ], "mquake.json")
    outputs = {
        "artifact_schema_version": "mquake_model_outputs.v1",
        "cases": [{
            "case_id": 0,
            "direct_outputs": ["A is B"],
            "hop_outputs": ["C is here"],
        }],
    }

    payload = scored_mquake_payload(adapter, outputs, "adapter.json")

    assert payload["artifact_schema_version"] == "mquake_scored_outputs.v1"
    assert payload["summary"]["direct_rewrite_acc"] == 1.0
    assert payload["summary"]["multi_hop_acc"] == 1.0
    assert "documented model-editing run" in payload["caveat"]


def test_summarize_scored_mquake_rows_reports_all_success_rates():
    summary = summarize_scored_mquake_rows([
        {"direct_acc": 1.0, "multi_hop_acc": 0.0,
         "all_direct_success": True, "all_hop_success": False},
        {"direct_acc": 0.0, "multi_hop_acc": 1.0,
         "all_direct_success": False, "all_hop_success": True},
    ])

    assert summary["composite_acc"] == 0.5
    assert summary["all_direct_success_rate"] == 0.5
    assert summary["all_hop_success_rate"] == 0.5

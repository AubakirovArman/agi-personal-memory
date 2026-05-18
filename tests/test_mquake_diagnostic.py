from agim.eval.mquake_diagnostic import (
    diagnostic_payload,
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

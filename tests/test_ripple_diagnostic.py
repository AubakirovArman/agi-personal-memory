from agim.eval.ripple_diagnostic import (
    diagnostic_payload,
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

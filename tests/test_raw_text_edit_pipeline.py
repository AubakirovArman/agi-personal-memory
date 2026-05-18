import json

import pytest

from agim.eval.raw_text_edit_pipeline import parse_raw_update, proposals_payload


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

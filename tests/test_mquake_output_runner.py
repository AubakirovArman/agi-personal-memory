from types import SimpleNamespace

from agim.eval.mquake_output_runner import output_payload, run_mquake_case


class _Editor:
    def __init__(self):
        self.applied = []
        self.rolled_back = []

    def apply_edit(self, subject, target, relation, **kwargs):
        backup = {"subject": subject, "target": target, "relation": relation}
        self.applied.append((backup, kwargs))
        return backup

    def rollback(self, backup):
        self.rolled_back.append(backup)


def _args():
    return SimpleNamespace(
        n=2,
        model="llama",
        device="cpu",
        clamp_lm=0.2,
        clamp_embed=0.06,
        clamp_eos=0.0,
        clamp_anti=0.06,
        target_token_mode="contextual",
        direct_max_new_tokens=12,
        hop_max_new_tokens=24,
    )


def test_run_mquake_case_applies_requests_and_rolls_back():
    editor = _Editor()
    case = {
        "case_id": 7,
        "requests": [
            {"subject": "Alice", "target_new": "Paris", "target_true": "Rome",
             "relation_id": "P17", "prompt": "Alice was born in"},
            {"subject": "Paris", "target_new": "France", "target_true": "Italy",
             "relation_id": "P131", "prompt": "Paris is located in"},
        ],
        "portability": {"multi_hop": {"prompt": ["Where was Alice born?"]}},
    }

    row = run_mquake_case(case, _args(), editor, lambda prompt, _limit: f"{prompt} ok")

    assert row["case_id"] == 7
    assert row["n_requests"] == 2
    assert row["n_hops"] == 1
    assert row["direct_outputs"][0]["text"] == "Alice was born in ok"
    assert len(editor.applied) == 2
    assert editor.rolled_back == [
        {"subject": "Paris", "target": "France", "relation": "P131"},
        {"subject": "Alice", "target": "Paris", "relation": "P17"},
    ]


def test_output_payload_records_schema_and_cases():
    editor = _Editor()
    adapter = {
        "artifact_schema_version": "mquake_dataset_adapter.v1",
        "source": "adapter.json",
        "cases": [
            {
                "case_id": 1,
                "requests": [
                    {"subject": "Alice", "target_new": "Paris",
                     "target_true": "Rome", "relation_id": "P17",
                     "prompt": "Alice was born in"},
                ],
                "portability": {"multi_hop": {"prompt": ["Where?"]}},
            },
        ],
    }

    payload = output_payload(
        adapter_payload=adapter,
        cases=adapter["cases"],
        args=_args(),
        editor=editor,
        generate=lambda _prompt, _limit: "Paris",
    )

    assert payload["artifact_schema_version"] == "mquake_model_outputs.v1"
    assert payload["adapter_schema_version"] == "mquake_dataset_adapter.v1"
    assert payload["n"] == 1
    assert payload["cases"][0]["direct_outputs"][0]["text"] == "Paris"

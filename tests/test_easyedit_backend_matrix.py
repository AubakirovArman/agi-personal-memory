from pathlib import Path

from agim.eval.easyedit_backend_matrix import (
    backend_output_path,
    backend_skip_reason,
    parse_backend_list,
    run_backend_comparison,
)
from agim.eval.easyedit_cli import build_parser


def test_backend_comparison_helpers_parse_and_skip_offline_backends():
    args = build_parser().parse_args([
        "--compare-backends", "dual_row,wal_rome,wal_memit,side_slot",
    ])

    assert parse_backend_list(args.compare_backends) == [
        "dual_row", "wal_rome", "wal_memit", "side_slot",
    ]
    assert backend_skip_reason(args, "side_slot") == (
        "side_slot comparison requires --sequential-edit"
    )
    assert str(backend_output_path(Path("results/matrix.json"), "wal_rome")) == (
        "results/matrix.wal_rome.json"
    )


def test_run_backend_comparison_writes_completed_and_skipped_rows():
    args = build_parser().parse_args([
        "--compare-backends", "dual_row,wal_memit",
        "--output", "results/matrix.json",
    ])

    def fake_run_backend_once(**kwargs):
        return {
            "method_profile_id": f"single_{kwargs['args'].edit_backend}",
            "summary": {"post": {"rewrite_acc": 1.0}},
            "time_s": 1.25,
        }

    payload = run_backend_comparison(
        args=args,
        model=object(),
        tok=object(),
        hparams=object(),
        facts=[{"case_id": 1}],
        records=[{"prompt": "x"}],
        dataset_sha256="abc",
        all_facts=[],
        locality_limit=None,
        compute_edit_quality=object(),
        test_prediction_acc=object(),
        summary_metrics=object(),
        device_id=0,
        run_backend_once=fake_run_backend_once,
    )

    assert payload["artifact_schema_version"] == "easyedit_backend_matrix.v1"
    assert payload["rows"][0]["status"] == "completed"
    assert payload["rows"][0]["output"] == "results/matrix.dual_row.json"
    assert payload["rows"][1]["status"] == "completed"

"""Run AGIM/WAL edits through official EasyEdit evaluation functions."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.eval.easyedit_counterfact import load_dataset, select_facts
from agim.model.wal_dual_editor import WALDualLayerEditor
from agim.model.wal_rome_editor import WALRomeEditor
from agim.model.wal_memit_editor import WALMemitEditor

from .easyedit_backend_matrix import run_backend_comparison
from .easyedit_bundle import post_edit_bundle
from .easyedit_cli import build_parser, print_final_summary
from .easyedit_dry_run import dry_run_payload, write_dry_run_summary
from .easyedit_eval_loop import run_evaluation_loop
from .easyedit_failures import write_failures_only
from .easyedit_loader import DEFAULT_EASYEDIT_ROOT, load_easyedit_official
from .easyedit_metrics import (
    attach_locality_acc,
    contextual_target_ids,
    ngram_entropy,
)
from .easyedit_payload import build_payload
from .easyedit_presets import apply_preset
from .easyedit_records import easyedit_record, extract_portability
from .easyedit_relation_banks import preload_relation_protected_banks
from .easyedit_run_metadata import parse_failure_families
from .easyedit_summary import summarize_official
from .easyedit_utils import jsonable, parse_device_id, parse_retention_steps


__all__ = [
    "DEFAULT_EASYEDIT_ROOT",
    "attach_locality_acc",
    "contextual_target_ids",
    "easyedit_record",
    "extract_portability",
    "load_easyedit_official",
    "ngram_entropy",
    "parse_device_id",
    "parse_retention_steps",
    "post_edit_bundle",
    "summarize_official",
]


def main() -> int:
    args = apply_preset(build_parser().parse_args())
    parse_failure_families(args.failure_families)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device_id = parse_device_id(args.device)
    locality_limit = None if args.locality_limit == 0 else args.locality_limit
    all_facts, dataset_sha256 = load_dataset(args.dataset)
    facts = select_facts(all_facts, args.n, args.sample_policy, args.seed)
    records = [easyedit_record(fact, locality_limit) for fact in facts]
    if args.dry_run_summary:
        payload = dry_run_payload(
            args=args,
            dataset_sha256=dataset_sha256,
            all_facts=all_facts,
            facts=facts,
            records=records,
            locality_limit=locality_limit,
        )
        output = write_dry_run_summary(args, payload)
        print(f"Dry-run summary saved {output}")
        return 0

    compute_edit_quality, test_prediction_acc, summary_metrics = load_easyedit_official(
        args.easyedit_root
    )
    hparams = SimpleNamespace(alg_name="AGIM_WAL", max_length=512)

    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    model.eval()

    if args.compare_backends:
        payload = run_backend_comparison(
            args=args,
            model=model,
            tok=tok,
            hparams=hparams,
            facts=facts,
            records=records,
            dataset_sha256=dataset_sha256,
            all_facts=all_facts,
            locality_limit=locality_limit,
            compute_edit_quality=compute_edit_quality,
            test_prediction_acc=test_prediction_acc,
            summary_metrics=summary_metrics,
            device_id=device_id,
            run_backend_once=_run_backend_once,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
        print(f"Backend comparison saved {output}")
        return 0

    _run_backend_once(
        args=args,
        model=model,
        tok=tok,
        hparams=hparams,
        facts=facts,
        records=records,
        dataset_sha256=dataset_sha256,
        all_facts=all_facts,
        locality_limit=locality_limit,
        compute_edit_quality=compute_edit_quality,
        test_prediction_acc=test_prediction_acc,
        summary_metrics=summary_metrics,
        device_id=device_id,
    )
    return 0


def _run_backend_once(
    *,
    args,
    model,
    tok,
    hparams,
    facts,
    records,
    dataset_sha256,
    all_facts,
    locality_limit,
    compute_edit_quality,
    test_prediction_acc,
    summary_metrics,
    device_id,
) -> dict[str, Any]:
    editor = _build_editor(args, model, tok)
    editor.nt_sample_size = args.nt_sample_size
    editor.build_vocab()
    relation_bank_summary = preload_relation_protected_banks(
        editor, args, facts, records)
    _print_run_header(args, records, locality_limit)

    t0 = time.time()
    metrics, _edit_times, retention = run_evaluation_loop(
        args=args,
        model=model,
        tok=tok,
        hparams=hparams,
        editor=editor,
        facts=facts,
        records=records,
        compute_edit_quality=compute_edit_quality,
        test_prediction_acc=test_prediction_acc,
        device_id=device_id,
    )
    elapsed = time.time() - t0
    summary = summarize_official(metrics)
    if args.write_easyedit_log:
        summary_metrics(metrics)

    payload = build_payload(
        args=args,
        metrics=metrics,
        retention=retention,
        summary=summary,
        elapsed=elapsed,
        dataset_sha256=dataset_sha256,
        all_facts=all_facts,
        facts=facts,
        locality_limit=locality_limit,
        model=model,
        editor=editor,
        relation_bank_summary=relation_bank_summary,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    if args.save_failures_only:
        failure_output = write_failures_only(args, metrics, summary)
        print(f"Failures-only artifact saved {failure_output}")
    print_final_summary(summary, retention, elapsed, len(metrics), output)
    return payload


def _print_run_header(args, records: list[dict], locality_limit: int | None) -> None:
    print(
        f"\nOfficial EasyEdit metric run ({len(records)} facts, "
        f"locality={'all' if locality_limit is None else locality_limit}, "
        f"target_token_mode={args.target_token_mode}, "
        f"wal_encode_updates={args.wal_encode_updates}):\n",
        flush=True,
    )


def _build_editor(args, model, tok):
    if args.edit_backend == "wal_rome":
        return WALRomeEditor(
            model,
            tok,
            device=args.device,
            target_layer=args.rome_target_layer,
            candidate_layers=_parse_rome_layers(args.rome_candidate_layers),
            top_rows=args.rome_top_rows,
            clamp_rome=args.rome_clamp,
            auto_locate=args.rome_auto_locate,
        )
    if args.edit_backend == "wal_memit":
        return WALMemitEditor(
            model,
            tok,
            target_layer=args.rome_target_layer,
            device=args.device,
        )
    return WALDualLayerEditor(model, tok, device=args.device)


def _parse_rome_layers(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

"""Run AGIM/WAL edits through official EasyEdit evaluation functions."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.eval.easyedit_counterfact import load_dataset, select_facts
from agim.model.wal_dual_editor import WALDualLayerEditor

from .easyedit_bundle import post_edit_bundle
from .easyedit_cli import build_parser, print_final_summary
from .easyedit_eval_loop import run_evaluation_loop
from .easyedit_loader import DEFAULT_EASYEDIT_ROOT, load_easyedit_official
from .easyedit_metrics import (
    attach_locality_acc,
    contextual_target_ids,
    ngram_entropy,
)
from .easyedit_payload import build_payload
from .easyedit_records import easyedit_record, extract_portability
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
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device_id = parse_device_id(args.device)
    locality_limit = None if args.locality_limit == 0 else args.locality_limit
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

    editor = WALDualLayerEditor(model, tok, device=args.device)
    editor.build_vocab()

    all_facts, dataset_sha256 = load_dataset(args.dataset)
    facts = select_facts(all_facts, args.n, args.sample_policy, args.seed)
    records = [easyedit_record(fact, locality_limit) for fact in facts]
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
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print_final_summary(summary, retention, elapsed, len(metrics), output)
    return 0


def _print_run_header(args, records: list[dict], locality_limit: int | None) -> None:
    print(
        f"\nOfficial EasyEdit metric run ({len(records)} facts, "
        f"locality={'all' if locality_limit is None else locality_limit}, "
        f"target_token_mode={args.target_token_mode}):\n",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

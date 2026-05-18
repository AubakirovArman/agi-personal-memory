"""Run MQuAKE adapter cases through the current AGIM edit backend."""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agim.eval.easyedit_counterfact import LLAMA, git_sha
from agim.model.wal_dual_editor import WALDualLayerEditor

from .easyedit_utils import jsonable


GenerateFn = Callable[[str, int], str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="MQuAKE adapter JSON")
    parser.add_argument("--output", required=True, help="Model-output JSON")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model", default=LLAMA)
    parser.add_argument(
        "--device",
        default=os.getenv("AGIM_DEVICE", "cuda"),
        help="CUDA device id for generation (set AGIM_DEVICE)",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--clamp_lm", type=float, default=0.20)
    parser.add_argument("--clamp_embed", type=float, default=0.06)
    parser.add_argument("--clamp_eos", type=float, default=0.0)
    parser.add_argument("--clamp_anti", type=float, default=0.06)
    parser.add_argument("--target-token-mode",
                        choices=["standalone", "contextual", "both"],
                        default="contextual")
    parser.add_argument("--direct-max-new-tokens", type=int, default=12)
    parser.add_argument("--hop-max-new-tokens", type=int, default=24)
    return parser


def output_payload(
    *,
    adapter_payload: dict[str, Any],
    cases: list[dict[str, Any]],
    args,
    editor,
    generate: GenerateFn,
) -> dict[str, Any]:
    started = time.time()
    rows = []
    selected = cases[:max(args.n, 0)]
    for idx, case in enumerate(selected, start=1):
        rows.append(run_mquake_case(case, args, editor, generate))
        if idx % 10 == 0 or idx == len(selected):
            print(f"  [{idx}/{len(selected)}] MQuAKE output cases", flush=True)
    return {
        "artifact_schema_version": "mquake_model_outputs.v1",
        "adapter_schema_version": adapter_payload.get("artifact_schema_version"),
        "adapter_source": adapter_payload.get("source"),
        "n": len(rows),
        "model": args.model,
        "device": args.device,
        "git_sha": git_sha(),
        "command": " ".join(sys.argv),
        "method_profile_id": "mquake_dual_row_outputs",
        "hyperparams": {
            "clamp_lm": args.clamp_lm,
            "clamp_embed": args.clamp_embed,
            "clamp_eos": args.clamp_eos,
            "clamp_anti": args.clamp_anti,
            "target_token_mode": args.target_token_mode,
            "direct_max_new_tokens": args.direct_max_new_tokens,
            "hop_max_new_tokens": args.hop_max_new_tokens,
        },
        "time_s": round(time.time() - started, 2),
        "cases": rows,
        "caveat": (
            "MQuAKE adapter model outputs from the current AGIM dual-row "
            "backend. Score with agim.eval.mquake_diagnostic --score-adapter."
        ),
    }


def run_mquake_case(
    case: dict[str, Any],
    args,
    editor,
    generate: GenerateFn,
) -> dict[str, Any]:
    backups = []
    for request in case.get("requests", []):
        backups.append(editor.apply_edit(
            request.get("subject", ""),
            request.get("target_new", ""),
            request.get("relation_id", ""),
            prompt=request.get("prompt", ""),
            clamp_lm=args.clamp_lm,
            clamp_embed=args.clamp_embed,
            clamp_eos=args.clamp_eos,
            clamp_anti=args.clamp_anti,
            old_target=request.get("target_true", ""),
            target_token_mode=args.target_token_mode,
        ))
    try:
        direct_outputs = [
            _output_row(request.get("prompt", ""),
                        generate(request.get("prompt", ""),
                                 args.direct_max_new_tokens))
            for request in case.get("requests", [])
        ]
        hop_prompts = case.get("portability", {}).get("multi_hop", {}).get(
            "prompt", [])
        hop_outputs = [
            _output_row(prompt, generate(prompt, args.hop_max_new_tokens))
            for prompt in hop_prompts
        ]
    finally:
        for backup in reversed(backups):
            editor.rollback(backup)
    return {
        "case_id": case.get("case_id"),
        "n_requests": len(case.get("requests", [])),
        "n_hops": len(case.get("portability", {}).get("multi_hop", {}).get(
            "prompt", [])),
        "direct_outputs": direct_outputs,
        "hop_outputs": hop_outputs,
    }


def generate_text(model, tok, device: str, prompt: str,
                  max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=False,
        )
    return tok.decode(generated[0, input_len:], skip_special_tokens=True).strip()


def _output_row(prompt: str, text: str) -> dict[str, str]:
    return {"prompt": prompt, "text": text}


def main() -> int:
    args = build_parser().parse_args()
    adapter_path = Path(args.adapter)
    adapter_payload = json.loads(adapter_path.read_text())
    cases = adapter_payload.get("cases", [])
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
    payload = output_payload(
        adapter_payload=adapter_payload,
        cases=cases,
        args=args,
        editor=editor,
        generate=lambda prompt, limit: generate_text(
            model, tok, args.device, prompt, limit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    print(f"MQuAKE model outputs saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

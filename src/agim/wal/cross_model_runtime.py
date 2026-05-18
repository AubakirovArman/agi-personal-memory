from __future__ import annotations

from .cross_model_common import ModelCandidate, _sha256_json


def run_local_causal_lm_smoke(candidate: ModelCandidate) -> dict[str, object]:
    """Load a local small CausalLM and run a minimal deterministic inference."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "BLOCKED",
            "pass": False,
            "reason": "RUNTIME_DEPENDENCY_MISSING",
            "error": str(exc),
        }

    prompt = "WAL small-model smoke test. Answer with one token: OK"
    try:
        tokenizer = AutoTokenizer.from_pretrained(candidate.path, local_files_only=True)
        load_kwargs: dict[str, object] = {"local_files_only": True}
        if torch.cuda.is_available():
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(candidate.path, **load_kwargs)
        model.eval()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_finite = bool(torch.isfinite(outputs.logits).all().item())
            generated = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated[0][inputs.input_ids.shape[1] :].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        checksum = _sha256_json(
            {
                "model_type": candidate.model_type,
                "architectures": candidate.architectures,
                "prompt": prompt,
                "generated_ids": generated_ids,
            }
        )[:16]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "PASS" if logits_finite and generated_ids else "FAIL",
            "pass": logits_finite and bool(generated_ids),
            "reason": None if logits_finite and generated_ids else "INFERENCE_OUTPUT_INVALID",
            "device": str(device),
            "dtype": str(load_kwargs["dtype"]),
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_token_count": len(generated_ids),
            "generated_text": generated_text,
            "logits_finite": logits_finite,
            "behavioral_checksum": checksum,
        }
    except Exception as exc:  # pragma: no cover - hardware/model dependent
        message = str(exc)
        lowered = message.lower()
        if "out of memory" in lowered or "oom" in lowered:
            status = "BLOCKED"
            reason = "RESOURCE_LIMIT_OOM"
        elif "unrecognized configuration" in lowered or "unsupported" in lowered:
            status = "UNSUPPORTED"
            reason = "UNSUPPORTED_CONFIG"
        else:
            status = "FAIL"
            reason = "LOCAL_MODEL_RUNTIME_ERROR"
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "status": status,
            "pass": False,
            "reason": reason,
            "error": message,
        }

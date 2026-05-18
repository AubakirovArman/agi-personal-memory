from __future__ import annotations

import torch


def generate(model, tok, prompt, max_tokens=10, temperature=None, rep_penalty=1.2):
    """Generation with repetition penalty (1.2 = best in sweep)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ilen = inputs.input_ids.shape[1]
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tok.eos_token_id,
        "repetition_penalty": rep_penalty,
    }
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return out[0, ilen:], tok.decode(out[0][ilen:], skip_special_tokens=True)


def token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like overlap between two tokenized strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))


def has_repetition(gen_ids: torch.Tensor, target_ids: list[int], threshold=2) -> bool:
    """Detect target token sequence repeated consecutively > threshold times."""
    t = torch.tensor(target_ids, device=gen_ids.device)
    n = len(target_ids)
    count = 0
    max_count = 0
    i = 0
    while i <= len(gen_ids) - n:
        if torch.equal(gen_ids[i:i+n], t):
            count += 1
            max_count = max(max_count, count)
            i += n
        else:
            count = 0
            i += 1
    return max_count > threshold, max_count


def evaluate_edit_hardened(model, tok, editor, fact, clamp=0.3):
    """One CounterFact edit — hardened evaluation."""
    rewrite = fact["requested_rewrite"]
    subject = rewrite["subject"]
    relation = rewrite["relation_id"]
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]
    prompt = rewrite["prompt"].format(subject)
    target_ids = tok.encode(target_new, add_special_tokens=False)
    num_tokens = len(target_ids)

    # ── Fix 2: Snapshot lm_head for NT measurement ──
    weight = model.lm_head.weight.data
    edited_tids = set(target_ids)
    nt_sample_size = min(500, weight.shape[0])
    nt_sample = []
    while len(nt_sample) < nt_sample_size:
        rid = torch.randint(0, weight.shape[0], (1,)).item()
        if rid not in nt_sample:
            nt_sample.append(rid)
    nt_before = {rid: weight[rid, :].clone() for rid in nt_sample if rid not in edited_tids}

    # ── BEFORE: snapshot neighborhood answers ──
    n_prompts = fact.get("neighborhood_prompts", [])[:4]
    n_before_ids = []
    n_before_texts = []
    for n_prompt in n_prompts:
        n_ids, n_text = generate(model, tok, n_prompt[:100], 8)
        n_before_ids.append(n_ids)
        n_before_texts.append(n_text.strip())

    # ── EDIT ──
    ok = editor.apply_edit(subject, target_new, relation, clamp_norm=clamp)
    if not ok:
        editor.rollback()
        return None

    # ═══ ES: Efficacy ═══
    # Protocol 1: AGIM — generate 10 tokens, substring match
    es_gen_ids, es_gen_text = generate(model, tok, prompt, 10)
    es_agim = 1.0 if target_new.lower() in es_gen_text.lower() else 0.0

    # Protocol 2: EasyEdit — generate len(target) tokens, token exact match
    ee_ids, ee_text = generate(model, tok, prompt, max(1, len(target_ids)))
    es_easyedit = 1.0 if ee_ids[:len(target_ids)].cpu().tolist() == target_ids else 0.0

    # Fix 3: Repetition check (on AGIM generation)
    is_rep, rep_count = has_repetition(es_gen_ids, target_ids, threshold=2)
    es_clean = es_agim and not is_rep

    # ═══ PS: Paraphrase ═══
    ps_agim = ps_easyedit = ps_rep_count = 0.0
    ps_n = 0
    for para in fact.get("paraphrase_prompts", [])[:2]:
        p_ids, p_text = generate(model, tok, para[:100], 10)
        if target_new.lower() in p_text.lower():
            ps_agim += 1
        if p_ids[:len(target_ids)].cpu().tolist() == target_ids:
            ps_easyedit += 1
        is_pr, _ = has_repetition(p_ids, target_ids, threshold=2)
        if is_pr:
            ps_rep_count += 1
        ps_n += 1
    ps_agim /= max(ps_n, 1)
    ps_easyedit /= max(ps_n, 1)
    ps_rep_rate = ps_rep_count / max(ps_n, 1)

    # ═══ NS: Neighborhood — правильная метрика (target NOT in neighbor answer) ═══
    ns_hits = 0  # NS_B: target_new NOT in neighbor answer
    ns_overlap_hits = 0  # NS_A: overlap for diagnostic
    ns_overlaps = []
    for i, n_prompt in enumerate(n_prompts):
        _, n_after_text = generate(model, tok, n_prompt[:100], 8)
        na = n_after_text.strip()
        # Correct NS: target_new should NOT appear in neighbor answer
        if target_new.lower() not in na.lower():
            ns_hits += 1
        # Diagnostic: token overlap (legacy)
        overlap = token_overlap(n_before_texts[i], na)
        ns_overlaps.append(overlap)
        if overlap > 0.3:
            ns_overlap_hits += 1
    ns = ns_hits / max(len(n_prompts), 1)
    ns_overlap_legacy = ns_overlap_hits / max(len(n_prompts), 1)

    # ═══ NT: Measured non-target diff (Fix 2) ═══
    nt_max = 0.0
    nt_count_nonzero = 0
    for rid, orig in nt_before.items():
        diff = (weight[rid, :] - orig.to(weight.device)).abs().max().item()
        nt_max = max(nt_max, diff)
        if diff > 1e-6:
            nt_count_nonzero += 1

    # ═══ Rollback ═══
    editor.rollback()
    _, rb_text = generate(model, tok, prompt, 8)
    rb_ok = target_true.lower() in rb_text.lower()

    return {
        "subject": subject, "new": target_new, "old": target_true,
        "num_tokens": num_tokens,
        # ES
        "ES_agim": es_agim,
        "ES_easyedit": es_easyedit,
        "ES_clean": es_clean,
        "ES_has_rep": is_rep,
        "ES_rep_count": rep_count,
        # PS
        "PS_agim": ps_agim,
        "PS_easyedit": ps_easyedit,
        "PS_rep_rate": ps_rep_rate,
        # NS (correct: target NOT in neighbor)
        "NS": ns,
        "NS_mean_overlap": ns_overlap_legacy,
        # NT (Fix 2)
        "NT_max": nt_max,
        "NT_nonzero_rows": nt_count_nonzero,
        # RB
        "RB": 1.0 if rb_ok else 0.0,
        # Samples
        "gen_agim": es_gen_text[:60],
        "gen_easyedit": ee_text[:60],
    }

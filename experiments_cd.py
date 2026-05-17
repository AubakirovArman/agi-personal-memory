"""Experiments C + D: negative projection + constrained ridge lm_head edit."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor
from agim.wal.encoder import wal_encode_scalar_gpu

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 50

print('Loading model...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval()
print('OK'); sys.stdout.flush()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N]

def gen(prompt, max_t=10, rep_pen=1.2):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_t, do_sample=False,
                             repetition_penalty=rep_pen, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()

def has_rep(text, target, thr=2):
    t = re.escape(target.lower())
    return bool(re.search(rf'({t})\1{{{thr},}}', text.lower()))

def tok_overlap(a, b):
    ta = set(a.lower().split()); tb = set(b.lower().split())
    return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0

def get_key(ids):
    """Get normalized last hidden state from token ids."""
    last_hidden = None
    def hook_fn(module, inp, out):
        nonlocal last_hidden
        hs = out[0] if isinstance(out, tuple) else out
        last_hidden = hs.detach().clone()
    handle = model.model.norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=ids.unsqueeze(0).to(model.device))
    handle.remove()
    if last_hidden is None: return None
    k = last_hidden[0, -1, :].float() if last_hidden.dim() == 3 else last_hidden[-1, :].float()
    return k / (k.norm() + 1e-8)


editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()
atoms_gpu = editor.atoms.to(DEV)
weight = model.lm_head.weight.data


def evaluate_config(label, clamp, use_neg_proj=False, use_ridge=False):
    """Run 50-fact eval with given config. Returns (ES, ES_cl, PS, NS, Comp)."""
    es = es_cl = ps = ns_h = ns_n = rep = 0
    t0 = time.time()

    for fact in data:
        rw = fact['requested_rewrite']
        s, rel = rw['subject'], rw['relation_id']
        tnew, told = rw['target_new']['str'], rw['target_true']['str']
        p = rw['prompt'].format(s)
        tids = tok.encode(tnew, add_special_tokens=False)
        prompt_ids = tok(p, return_tensors='pt').input_ids[0]

        # BEFORE: neighborhood answers
        nb = [gen(np[:100]) for np in fact.get('neighborhood_prompts', [])[:4]]

        # ── Build edit delta ──
        # Positive keys: edit prompt + paraphrase prompts
        pos_keys = []
        pos_keys.append(get_key(prompt_ids))

        # Negative keys: neighborhood prompts + prompt+target
        neg_keys = []
        for np in fact.get('neighborhood_prompts', [])[:4]:
            np_ids = tok(np[:100], return_tensors='pt').input_ids[0]
            nk = get_key(np_ids)
            if nk is not None:
                neg_keys.append(nk)

        # After-target key (already used in anti-boost)
        full_ids = torch.cat([prompt_ids, torch.tensor(tids)])
        after_key = get_key(full_ids)
        if after_key is not None:
            neg_keys.append(after_key)

        # ── Compute delta direction ──
        if use_neg_proj and len(neg_keys) > 0:
            # Experiment C: project away from negative subspace
            K_neg = torch.stack(neg_keys)  # [M, D]
            # SVD-based projection
            U, S, V = torch.svd(K_neg.T @ K_neg)
            # Take top directions that explain 95% of variance
            var = S.cumsum(0) / S.sum()
            k = max(1, (var < 0.95).sum().item())
            neg_basis = V[:, :k]  # [D, k]

            # Project each positive key away from negative subspace
            clean_deltas = []
            for pk in pos_keys:
                if pk is None: continue
                # Remove projection onto negative basis
                proj = neg_basis @ (neg_basis.T @ pk)
                clean = pk - proj
                if clean.norm() > 1e-8:
                    clean = clean / (clean.norm() + 1e-8)
                clean_deltas.append(clean)

            if clean_deltas:
                delta = torch.stack(clean_deltas).mean(0)
            else:
                delta = pos_keys[0]
        elif use_ridge and len(neg_keys) > 0:
            # Experiment D: constrained ridge regression
            X = torch.stack([pos_keys[0]] + neg_keys)  # [1+M, D]
            # Target: +margin for positive, 0 for negatives, -margin for after-target
            y = torch.tensor([1.0] + [0.0] * (len(neg_keys) - 1) + [-0.5])
            # Ridge: delta = X^T (X X^T + λI)^(-1) y
            lam = 0.1
            G = X @ X.T + lam * torch.eye(len(X))
            alpha = torch.linalg.solve(G, y)
            delta = X.T @ alpha
            if delta.norm() > 1e-8:
                delta = delta / (delta.norm() + 1e-8)
        else:
            delta = pos_keys[0] if pos_keys[0] is not None else None

        if delta is None:
            continue

        # ── Apply edit manually ──
        edited_tids = set()
        for i, tid in enumerate(tids):
            if tid not in editor._original_rows:
                editor._original_rows[tid] = weight[tid, :].clone()
            edited_tids.add(tid)
            row = weight[tid, :].float().to(DEV)
            boost = clamp * delta.to(DEV) / (2 ** i)
            _, _, recon = wal_encode_scalar_gpu(row + boost, atoms_gpu, editor.lmax)
            weight[tid, :] = recon.to(device=weight.device, dtype=weight.dtype)

        # EOS boost + anti-boost on after-target context
        eos_id = tok.eos_token_id
        if after_key is not None and eos_id is not None:
            if eos_id not in editor._original_rows:
                editor._original_rows[eos_id] = weight[eos_id, :].clone()
            eos_row = weight[eos_id, :].float().to(DEV)
            _, _, eos_r = wal_encode_scalar_gpu(
                eos_row + clamp * 0.8 * after_key.to(DEV), atoms_gpu, editor.lmax)
            weight[eos_id, :] = eos_r.to(device=weight.device, dtype=weight.dtype)
            # Anti-boost
            for tid in tids:
                if tid == eos_id: continue
                row2 = weight[tid, :].float().to(DEV)
                _, _, a_r = wal_encode_scalar_gpu(
                    row2 - clamp * 0.3 * after_key.to(DEV), atoms_gpu, editor.lmax)
                weight[tid, :] = a_r.to(device=weight.device, dtype=weight.dtype)

        # ── Evaluate ──
        a = gen(p)
        if tnew.lower() in a.lower():
            es += 1
            if not has_rep(a, tnew): es_cl += 1
            else: rep += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100]).lower(): ps += 1
        for j, np in enumerate(fact.get('neighborhood_prompts', [])[:4]):
            if tok_overlap(nb[j], gen(np[:100])) > 0.3: ns_h += 1
            ns_n += 1
        editor.rollback()

    n = len(data)
    es_v = es / n; ecl_v = es_cl / n; ps_v = ps / (n * 2)
    ns_v = ns_h / max(ns_n, 1); comp = (es_v + ps_v + ns_v) / 3
    e = time.time() - t0
    print(f'  {label:35s} ES={es_v:.0%} ES_cl={ecl_v:.0%} PS={ps_v:.0%} NS={ns_v:.0%} Comp={comp:.1%} Rep={rep/n:.0%} ({e:.0f}s)')
    sys.stdout.flush()
    return es_v, ecl_v, ps_v, ns_v, comp


# ═══ BASELINE (current anti-boost) ═══
print('\nExperiment C+D: negative projection + ridge\n')
print('BASELINE (anti-boost only):')
evaluate_config('baseline clamp=0.30', 0.30)

# ═══ Experiment C: negative projection ═══
print('\nExperiment C: negative projection with diff clamps:')
best_c = None
for clamp in [0.2, 0.3]:
    r = evaluate_config(f'C: neg_proj clamp={clamp}', clamp, use_neg_proj=True)
    if best_c is None or r[4] > best_c[4]: best_c = r + (clamp,)

# ═══ Experiment D: constrained ridge ═══
print('\nExperiment D: constrained ridge with diff clamps:')
best_d = None
for clamp in [0.2, 0.3]:
    r = evaluate_config(f'D: ridge clamp={clamp}', clamp, use_ridge=True)
    if best_d is None or r[4] > best_d[4]: best_d = r + (clamp,)

print(f'\n{"="*60}')
print(f'BEST: C (neg_proj)  clamp={best_c[5]:.2f} Comp={best_c[4]:.1%} ES={best_c[0]:.0%} NS={best_c[3]:.0%}')
print(f'BEST: D (ridge)     clamp={best_d[5]:.2f} Comp={best_d[4]:.1%} ES={best_d[0]:.0%} NS={best_d[3]:.0%}')

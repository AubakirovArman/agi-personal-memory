"""Dual-layer WAL: embed_tokens + lm_head. Our protocol, 50 facts."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 50

print('Loading...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval(); print('OK'); sys.stdout.flush()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N]

def gen(p, t=10, r=1.2):
    i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
    with torch.no_grad():
        o = model.generate(**i, max_new_tokens=t, do_sample=False,
                           repetition_penalty=r, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][il:], skip_special_tokens=True).strip()

def get_key(ids):
    last_h = None
    def hook(m, i, o):
        nonlocal last_h; hs = o[0] if isinstance(o, tuple) else o; last_h = hs.detach().clone()
    h = model.model.norm.register_forward_hook(hook)
    with torch.no_grad(): model(input_ids=ids.unsqueeze(0).to(model.device))
    h.remove()
    if last_h is None: return None
    return last_h[0, -1, :].float() if last_h.dim() == 3 else last_h[-1, :].float()

# Build WAL atoms from both lm_head and embed_tokens
print('Building atoms...'); sys.stdout.flush()
lm_flat = model.lm_head.weight.data.float().flatten()
emb_flat = model.model.embed_tokens.weight.data.float().flatten()
combined = torch.cat([lm_flat[:2_000_000], emb_flat[:2_000_000]])
atoms = build_atoms_kmeans(combined, 256, iters=5, device=torch.device(DEV))
atoms_gpu = atoms.to(DEV)
print(f'Atoms: {atoms.shape} [{atoms.min():.3f}, {atoms.max():.3f}]')
sys.stdout.flush()

# ── Sweep ──
print('\nDual-layer sweep (embed + lm_head):')
print('-' * 55)
sys.stdout.flush()

for clamp in [0.1, 0.15, 0.2, 0.25, 0.3]:
    es = es_cl = ps = ns_h = ns_n = rep = 0
    t0 = time.time()
    for fact in data:
        rw = fact['requested_rewrite']
        s, rel = rw['subject'], rw['relation_id']
        tnew = rw['target_new']['str']
        p = rw['prompt'].format(s)
        tids = tok.encode(tnew, add_special_tokens=False)
        pids = tok(p, return_tensors='pt').input_ids[0]
        subj_ids = tok.encode(s, add_special_tokens=False)

        # Save originals
        lm_backup = {}
        emb_backup = {}

        # ── lm_head: sequence-level boost ──
        for i, tid in enumerate(tids):
            if i == 0:
                k = get_key(pids)
            else:
                prev = torch.cat([pids, torch.tensor(tids[:i], device=pids.device)])
                k = get_key(prev)
            if k is None: continue
            k = k / (k.norm() + 1e-8)
            if tid not in lm_backup:
                lm_backup[tid] = model.lm_head.weight.data[tid, :].clone()
            row = model.lm_head.weight.data[tid, :].float().to(DEV)
            _, _, recon = wal_encode_scalar_gpu(row + clamp * k.to(DEV), atoms_gpu, 16)
            model.lm_head.weight.data[tid, :] = recon.to(
                device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)

        # ── embed_tokens: push subject toward target direction ──
        tdir = model.lm_head.weight.data[tids[0], :].float().to(DEV)
        tdir = tdir / (tdir.norm() + 1e-8)
        for sid in subj_ids:
            if sid not in emb_backup:
                emb_backup[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
            row = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
            _, _, recon = wal_encode_scalar_gpu(
                row + clamp * 0.3 * tdir, atoms_gpu, 16)
            model.model.embed_tokens.weight.data[sid, :] = recon.to(
                device=model.model.embed_tokens.weight.device,
                dtype=model.model.embed_tokens.weight.dtype)

        # ── EOS boost + anti-boost (lm_head) ──
        eos_id = tok.eos_token_id
        full_ids = torch.cat([pids, torch.tensor(tids, device=pids.device)])
        stop_k = get_key(full_ids)
        if stop_k is not None and eos_id is not None:
            stop_k = stop_k / (stop_k.norm() + 1e-8)
            if eos_id not in lm_backup:
                lm_backup[eos_id] = model.lm_head.weight.data[eos_id, :].clone()
            eos_row = model.lm_head.weight.data[eos_id, :].float().to(DEV)
            _, _, eos_r = wal_encode_scalar_gpu(
                eos_row + clamp * 0.8 * stop_k.to(DEV), atoms_gpu, 16)
            model.lm_head.weight.data[eos_id, :] = eos_r.to(
                device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
            # Anti-boost target tokens on lm_head
            for tid in tids:
                if tid == eos_id: continue
                row = model.lm_head.weight.data[tid, :].float().to(DEV)
                _, _, ar = wal_encode_scalar_gpu(
                    row - clamp * 0.3 * stop_k.to(DEV), atoms_gpu, 16)
                model.lm_head.weight.data[tid, :] = ar.to(
                    device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)

        # ── Anti-boost on embed_tokens too! ──
        if stop_k is not None:
            for sid in subj_ids:
                if sid not in emb_backup:
                    emb_backup[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
                row = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
                _, _, ar = wal_encode_scalar_gpu(
                    row - clamp * 0.15 * stop_k.to(DEV), atoms_gpu, 16)
                model.model.embed_tokens.weight.data[sid, :] = ar.to(
                    device=model.model.embed_tokens.weight.device,
                    dtype=model.model.embed_tokens.weight.dtype)

        # ── Evaluate ──
        a = gen(p)
        if tnew.lower() in a.lower():
            es += 1
            if not re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()):
                es_cl += 1
            else:
                rep += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100]).lower(): ps += 1
        for np in fact.get('neighborhood_prompts', [])[:4]:
            if tnew.lower() not in gen(np[:100]).lower(): ns_h += 1
            ns_n += 1

        # Rollback
        for tid, orig in lm_backup.items():
            model.lm_head.weight.data[tid, :] = orig
        for sid, orig in emb_backup.items():
            model.model.embed_tokens.weight.data[sid, :] = orig

    n = len(data); e = time.time() - t0
    ns_v = ns_h / max(ns_n, 1)
    comp = (es / n + ps / (n * 2) + ns_v) / 3
    print(f'  clamp={clamp:.2f}: ES={es/n:.0%} ES_cl={es_cl/n:.0%} PS={ps/(n*2):.0%} '
          f'NS={ns_v:.0%} Comp={comp:.1%} Rep={rep/n:.0%} ({e:.0f}s)')
    sys.stdout.flush()

"""Fine-tune dual-layer with clamp=0.15: vary embed boost, anti-boost, EOS, temp."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 50
CLAMP = 0.15  # fixed at best balance

print('Loading...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval(); print('OK'); sys.stdout.flush()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N]

def gen(p, t=10, r=1.2, temp=None):
    i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
    kw = {'max_new_tokens': t, 'pad_token_id': tok.eos_token_id, 'repetition_penalty': r}
    if temp and temp > 0: kw.update({'do_sample': True, 'temperature': temp, 'top_p': 0.9})
    else: kw['do_sample'] = False
    with torch.no_grad(): o = model.generate(**i, **kw)
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

print('Building atoms...'); sys.stdout.flush()
combined = torch.cat([
    model.lm_head.weight.data.float().flatten()[:2_000_000],
    model.model.embed_tokens.weight.data.float().flatten()[:2_000_000]])
atoms = build_atoms_kmeans(combined, 256, iters=5, device=torch.device(DEV))
atoms_gpu = atoms.to(DEV)
print(f'Atoms: {atoms.shape}'); sys.stdout.flush()

def run_config(label, temp=None, emb_boost=0.3, emb_anti=0.15, lm_anti=0.3, eos_boost=0.8):
    es = es_cl = ps = ns_h = ns_n = rep = 0; t0 = time.time()
    for fact in data:
        rw = fact['requested_rewrite']; s = rw['subject']; rel = rw['relation_id']
        tnew = rw['target_new']['str']; p = rw['prompt'].format(s)
        tids = tok.encode(tnew, add_special_tokens=False)
        pids = tok(p, return_tensors='pt').input_ids[0]
        subj_ids = tok.encode(s, add_special_tokens=False)
        lm_bu = {}; emb_bu = {}

        # lm_head boost (sequence-level)
        for i, tid in enumerate(tids):
            k = get_key(pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)]))
            if k is None: continue; k = k / (k.norm() + 1e-8)
            if tid not in lm_bu: lm_bu[tid] = model.lm_head.weight.data[tid, :].clone()
            r = model.lm_head.weight.data[tid, :].float().to(DEV)
            _, _, rec = wal_encode_scalar_gpu(r + CLAMP * k.to(DEV), atoms_gpu, 16)
            model.lm_head.weight.data[tid, :] = rec.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)

        # embed boost
        tdir = model.lm_head.weight.data[tids[0], :].float().to(DEV)
        tdir = tdir / (tdir.norm() + 1e-8)
        for sid in subj_ids:
            if sid not in emb_bu: emb_bu[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
            r = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
            _, _, rec = wal_encode_scalar_gpu(r + CLAMP * emb_boost * tdir, atoms_gpu, 16)
            model.model.embed_tokens.weight.data[sid, :] = rec.to(
                device=model.model.embed_tokens.weight.device, dtype=model.model.embed_tokens.weight.dtype)

        # Stop-token edit
        eos_id = tok.eos_token_id
        full_ids = torch.cat([pids, torch.tensor(tids, device=pids.device)])
        stop_k = get_key(full_ids)
        if stop_k is not None and eos_id is not None:
            stop_k = stop_k / (stop_k.norm() + 1e-8)
            # EOS boost
            if eos_id not in lm_bu: lm_bu[eos_id] = model.lm_head.weight.data[eos_id, :].clone()
            r = model.lm_head.weight.data[eos_id, :].float().to(DEV)
            _, _, rec = wal_encode_scalar_gpu(r + CLAMP * eos_boost * stop_k.to(DEV), atoms_gpu, 16)
            model.lm_head.weight.data[eos_id, :] = rec.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
            # lm_head anti-boost
            for tid in tids:
                if tid == eos_id: continue
                r = model.lm_head.weight.data[tid, :].float().to(DEV)
                _, _, rec = wal_encode_scalar_gpu(r - CLAMP * lm_anti * stop_k.to(DEV), atoms_gpu, 16)
                model.lm_head.weight.data[tid, :] = rec.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
            # embed anti-boost
            for sid in subj_ids:
                if sid not in emb_bu: emb_bu[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
                r = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
                _, _, rec = wal_encode_scalar_gpu(r - CLAMP * emb_anti * stop_k.to(DEV), atoms_gpu, 16)
                model.model.embed_tokens.weight.data[sid, :] = rec.to(
                    device=model.model.embed_tokens.weight.device, dtype=model.model.embed_tokens.weight.dtype)

        # Evaluate
        a = gen(p, temp=temp)
        if tnew.lower() in a.lower():
            es += 1
            if not re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()): es_cl += 1
            else: rep += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100], temp=temp).lower(): ps += 1
        for np in fact.get('neighborhood_prompts', [])[:4]:
            if tnew.lower() not in gen(np[:100], temp=temp).lower(): ns_h += 1; ns_n += 1

        # Rollback
        for tid, o in lm_bu.items(): model.lm_head.weight.data[tid, :] = o
        for sid, o in emb_bu.items(): model.model.embed_tokens.weight.data[sid, :] = o

    n = len(data); e = time.time() - t0
    ns_v = ns_h / max(ns_n, 1); comp = (es / n + ps / (n * 2) + ns_v) / 3
    print(f'  {label:40s} ES={es/n:.0%} PS={ps/(n*2):.0%} NS={ns_v:.0%} Comp={comp:.1%} Rep={rep/n:.0%} ({e:.0f}s)')
    sys.stdout.flush()

print(f'\nFine-tuning dual-layer (clamp={CLAMP}):')
print('-' * 65)
sys.stdout.flush()

# Baseline
run_config('BASELINE')

# Vary embed boost
for eb in [0.1, 0.5, 0.7]:
    run_config(f'emb_boost={eb}', emb_boost=eb)

# Vary embed anti-boost
for ea in [0.3, 0.5]:
    run_config(f'emb_anti={ea}', emb_anti=ea)

# Vary lm anti-boost
for la in [0.1, 0.5]:
    run_config(f'lm_anti={la}', lm_anti=la)

# Vary EOS boost
for eos in [0.5, 1.0]:
    run_config(f'eos={eos}', eos_boost=eos)

# Temperature
for t in [0.2, 0.3]:
    run_config(f'temp={t}', temp=t)

# Best combo
run_config('COMBO: eb=0.5 ea=0.3 eos=1.0', emb_boost=0.5, emb_anti=0.3, eos_boost=1.0)

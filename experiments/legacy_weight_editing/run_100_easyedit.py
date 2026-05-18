"""100 facts — EasyEdit protocol + AGIM protocol, dual-layer WAL, clamp=0.20."""
import os
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 100; CLAMP = 0.20

print('Loading...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f: facts = json.load(f)[:N]
combined = torch.cat([model.lm_head.weight.data.float().flatten()[:2_000_000],
                       model.model.embed_tokens.weight.data.float().flatten()[:2_000_000]])
atoms = build_atoms_kmeans(combined, 256, iters=5, device=torch.device(DEV)); atoms_gpu = atoms.to(DEV)
print(f'Ready. Atoms={atoms.shape}\n'); sys.stdout.flush()

def gen(p, t=10, r=1.2):
    i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
    with torch.no_grad(): o = model.generate(**i, max_new_tokens=t, do_sample=False, repetition_penalty=r, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][il:], skip_special_tokens=True).strip()

def get_key(ids):
    last_h = None
    def hook(m, i, o): nonlocal last_h; hs = o[0] if isinstance(o, tuple) else o; last_h = hs.detach().clone()
    h = model.model.norm.register_forward_hook(hook)
    with torch.no_grad(): model(input_ids=ids.unsqueeze(0).to(model.device))
    h.remove()
    return (last_h[0,-1,:].float() if last_h.dim()==3 else last_h[-1,:].float()) if last_h is not None else None

# Both protocols
es_a = es_ee = es_cl = ps_a = ps_ee = ns = ns_n = ns_a = ns_a_n = rep = 0
t0 = time.time()

for idx, fact in enumerate(facts):
    rw = fact['requested_rewrite']; s = rw['subject']; tnew = rw['target_new']['str']; told = rw['target_true']['str']
    p = rw['prompt'].format(s); tids = tok.encode(tnew, add_special_tokens=False)
    pids = tok(p, return_tensors='pt').input_ids[0]; sids = tok.encode(s, add_special_tokens=False)
    lm_bu = {}; emb_bu = {}

    # ── EDIT: EXACT copy from test_dual_layer.py ──
    for i, tid in enumerate(tids):
        k = get_key(pids if i == 0 else torch.cat([pids, torch.tensor(tids[:i], device=pids.device)]))
        if k is None: continue; k = k / (k.norm() + 1e-8)
        if tid not in lm_bu: lm_bu[tid] = model.lm_head.weight.data[tid, :].clone()
        row = model.lm_head.weight.data[tid, :].float().to(DEV)
        _, _, recon = wal_encode_scalar_gpu(row + CLAMP * k.to(DEV), atoms_gpu, 16)
        model.lm_head.weight.data[tid, :] = recon.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)

    tdir = model.lm_head.weight.data[tids[0], :].float().to(DEV); tdir = tdir / (tdir.norm() + 1e-8)
    for sid in sids:
        if sid not in emb_bu: emb_bu[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
        row = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
        _, _, recon = wal_encode_scalar_gpu(row + CLAMP * 0.3 * tdir, atoms_gpu, 16)
        model.model.embed_tokens.weight.data[sid, :] = recon.to(device=model.model.embed_tokens.weight.device, dtype=model.model.embed_tokens.weight.dtype)

    eid = tok.eos_token_id; full = torch.cat([pids, torch.tensor(tids, device=pids.device)]); sk = get_key(full)
    if sk is not None and eid is not None:
        sk = sk / (sk.norm() + 1e-8)
        if eid not in lm_bu: lm_bu[eid] = model.lm_head.weight.data[eid, :].clone()
        er = model.lm_head.weight.data[eid, :].float().to(DEV)
        _, _, rec = wal_encode_scalar_gpu(er + CLAMP * 0.8 * sk.to(DEV), atoms_gpu, 16)
        model.lm_head.weight.data[eid, :] = rec.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
        for tid in tids:
            if tid == eid: continue
            r2 = model.lm_head.weight.data[tid, :].float().to(DEV)
            _, _, ar = wal_encode_scalar_gpu(r2 - CLAMP * 0.3 * sk.to(DEV), atoms_gpu, 16)
            model.lm_head.weight.data[tid, :] = ar.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
        for sid in sids:
            if sid not in emb_bu: emb_bu[sid] = model.model.embed_tokens.weight.data[sid, :].clone()
            r3 = model.model.embed_tokens.weight.data[sid, :].float().to(DEV)
            _, _, ar = wal_encode_scalar_gpu(r3 - CLAMP * 0.15 * sk.to(DEV), atoms_gpu, 16)
            model.model.embed_tokens.weight.data[sid, :] = ar.to(device=model.model.embed_tokens.weight.device, dtype=model.model.embed_tokens.weight.dtype)

    # ── ES (AGIM substring) ──
    a = gen(p)
    if tnew.lower() in a.lower():
        es_a += 1
        if not re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()): es_cl += 1
        else: rep += 1

    # ── ES (EasyEdit token-exact) ──
    inp = tok(p, return_tensors='pt').to(model.device); ilen = inp.input_ids.shape[1]
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    if out[0, ilen:].cpu().tolist() == tids: es_ee += 1

    # ── PS (both protocols) ──
    for pa in fact.get('paraphrase_prompts', [])[:2]:
        if tnew.lower() in gen(pa[:100]).lower(): ps_a += 1
        inp2 = tok(pa[:100], return_tensors='pt').to(model.device); ilen2 = inp2.input_ids.shape[1]
        with torch.no_grad(): out2 = model.generate(**inp2, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        if out2[0, ilen2:].cpu().tolist() == tids: ps_ee += 1

    # ── NS: target NOT in neighbor (correct metric) ──
    for np in fact.get('neighborhood_prompts', [])[:4]:
        if tnew.lower() not in gen(np[:100]).lower(): ns += 1; ns_n += 1

    # Rollback
    for tid, o in lm_bu.items(): model.lm_head.weight.data[tid, :] = o
    for sid, o in emb_bu.items(): model.model.embed_tokens.weight.data[sid, :] = o

    if (idx + 1) % 25 == 0:
        nd = idx + 1; e = time.time() - t0
        print(f'  [{nd}/{N}] ES_a={es_a/nd:.0%} ES_ee={es_ee/nd:.0%} PS_a={ps_a/(nd*2):.0%} NS={ns/max(ns_n,1):.0%} ({e:.0f}s)')
        sys.stdout.flush()

n = len(facts); e = time.time() - t0
es_av = es_a/n; es_clv = es_cl/n; es_eev = es_ee/n; ps_av = ps_a/(n*2); ps_eev = ps_ee/(n*2); ns_v = ns/max(ns_n,1)
comp_a = (es_av + ps_av + ns_v)/3; comp_e = (es_eev + ps_eev + ns_v)/3

print(f'\n===== 100 FACTS (clamp={CLAMP}) =====')
print(f'  AGIM:     ES={es_av:.1%} PS={ps_av:.1%} NS={ns_v:.1%} Comp={comp_a:.1%}')
print(f'  EasyEdit: ES={es_eev:.1%} PS={ps_eev:.1%} NS={ns_v:.1%} Comp={comp_e:.1%}')
print(f'  ES_clean={es_clv:.1%} Rep={rep/n:.0%} Time={e:.0f}s')
out_path = 'results/local_protocol/easyedit_100.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path,'w') as f: json.dump({'n':n,'AGIM':{'ES':round(es_av,4),'PS':round(ps_av,4),'NS':round(ns_v,4),'Comp':round(comp_a,4)},'EasyEdit':{'ES':round(es_eev,4),'PS':round(ps_eev,4),'NS':round(ns_v,4),'Comp':round(comp_e,4)},'ES_clean':round(es_clv,4),'rep':round(rep/n,4),'time_s':round(e,1)},f,indent=2)
print(f'Saved {out_path}')

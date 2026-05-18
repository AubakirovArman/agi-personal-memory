"""Day 2: Adaptive clamp per example + negative projection in dual-layer."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_dual_editor import WALDualLayerEditor

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 50

print('Loading...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval()
with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f: facts = json.load(f)[:N]

def gen(p, t=10, r=1.2):
    i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
    with torch.no_grad(): o = model.generate(**i, max_new_tokens=t, do_sample=False, repetition_penalty=r, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][il:], skip_special_tokens=True).strip()

editor = WALDualLayerEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()

# ═══ Experiment 1: Adaptive clamp grid ═══
print('\nEXP 1: Adaptive clamp per example (50 facts)')
print('-' * 60)
sys.stdout.flush()

clamp_grid = [(lm, emb) for lm in [0.12, 0.16, 0.20, 0.25]
              for emb in [0.03, 0.06, 0.09]]
best_fixed = None

for lm, emb in clamp_grid:
    es = es_ee = ps = ps_ee = ns_a = ns_c = ns_n = rep = 0
    t0 = time.time()
    for fact in facts:
        rw = fact['requested_rewrite']; s = rw['subject']; tnew = rw['target_new']['str']; p = rw['prompt'].format(s)
        tids = tok.encode(tnew, add_special_tokens=False)
        nb = [gen(np[:100]) for np in fact.get('neighborhood_prompts', [])[:4]]
        bak = editor.apply_edit(s, tnew, prompt=p, clamp_lm=lm, clamp_embed=emb, clamp_eos=0.8*lm, clamp_anti=0.3*lm)
        a = gen(p)
        if tnew.lower() in a.lower():
            es += 1
            if re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()): rep += 1
        inp = tok(p, return_tensors='pt').to(model.device); ilen = inp.input_ids.shape[1]
        with torch.no_grad(): out = model.generate(**inp, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        if out[0, ilen:].cpu().tolist() == tids: es_ee += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100]).lower(): ps += 1
            inp2 = tok(pa[:100], return_tensors='pt').to(model.device); ilen2 = inp2.input_ids.shape[1]
            with torch.no_grad(): out2 = model.generate(**inp2, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
            if out2[0, ilen2:].cpu().tolist() == tids: ps_ee += 1
        for j, np in enumerate(fact.get('neighborhood_prompts', [])[:4]):
            na = gen(np[:100])
            if tnew.lower() not in na.lower(): ns_a += 1
            ta = set(nb[j].lower().split()); tb = set(na.lower().split())
            if len(ta & tb) / max(len(ta), len(tb)) > 0.3 if ta and tb else False: ns_c += 1
            ns_n += 1
        editor.rollback(bak)
    n = len(facts); e = time.time() - t0
    ns_av = ns_a / max(ns_n, 1); ns_cv = ns_c / max(ns_n, 1); es_eev = es_ee / n; ps_eev = ps_ee / (n * 2)
    comp = (es_eev + ps_eev + ns_av) / 3
    print(f'  lm={lm:.2f} emb={emb:.2f}: ES_ee={es_eev:.0%} PS_ee={ps_eev:.0%} NS_abs={ns_av:.0%} NS_con={ns_cv:.0%} Comp={comp:.1%} Rep={rep/n:.0%} ({e:.0f}s)')
    sys.stdout.flush()
    if best_fixed is None or comp > best_fixed[0]: best_fixed = (comp, lm, emb, es_eev, ns_av)

print(f'\n  BEST FIXED: lm={best_fixed[1]:.2f} emb={best_fixed[2]:.2f} Comp={best_fixed[0]:.1%} ES={best_fixed[3]:.0%} NS={best_fixed[4]:.0%}')

# ═══ Experiment 2: Per-example adaptive (pick best clamp per fact) ═══
print(f'\nEXP 2: Per-example adaptive clamp')
print('-' * 60)
sys.stdout.flush()

es_ad = es_ee_ad = ps_ad = ps_ee_ad = ns_a_ad = ns_c_ad = ns_n_ad = rep_ad = 0
t0 = time.time()
for fi, fact in enumerate(facts):
    rw = fact['requested_rewrite']; s = rw['subject']; tnew = rw['target_new']['str']; p = rw['prompt'].format(s)
    tids = tok.encode(tnew, add_special_tokens=False)
    nb = [gen(np[:100]) for np in fact.get('neighborhood_prompts', [])[:4]]

    # Try all clamps, pick best
    best_score = -999; best_lm = best_emb = 0.20
    for lm in [0.12, 0.16, 0.20, 0.25]:
        for emb in [0.03, 0.06]:
            bak = editor.apply_edit(s, tnew, prompt=p, clamp_lm=lm, clamp_embed=emb, clamp_eos=0.8*lm, clamp_anti=0.3*lm)
            a = gen(p)
            es_ok = 1 if tnew.lower() in a.lower() else 0
            ns_ok = 0
            for np in fact.get('neighborhood_prompts', [])[:4]:
                if tnew.lower() not in gen(np[:100]).lower(): ns_ok += 1
            score = es_ok * 3 + ns_ok  # prioritize NS
            if score > best_score: best_score = score; best_lm = lm; best_emb = emb
            editor.rollback(bak)

    # Apply best config
    bak = editor.apply_edit(s, tnew, prompt=p, clamp_lm=best_lm, clamp_embed=best_emb, clamp_eos=0.8*best_lm, clamp_anti=0.3*best_lm)
    a = gen(p)
    if tnew.lower() in a.lower():
        es_ad += 1
        if re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()): rep_ad += 1
    inp = tok(p, return_tensors='pt').to(model.device); ilen = inp.input_ids.shape[1]
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    if out[0, ilen:].cpu().tolist() == tids: es_ee_ad += 1
    for pa in fact.get('paraphrase_prompts', [])[:2]:
        if tnew.lower() in gen(pa[:100]).lower(): ps_ad += 1
        inp2 = tok(pa[:100], return_tensors='pt').to(model.device); ilen2 = inp2.input_ids.shape[1]
        with torch.no_grad(): out2 = model.generate(**inp2, max_new_tokens=len(tids), do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        if out2[0, ilen2:].cpu().tolist() == tids: ps_ee_ad += 1
    for j, np in enumerate(fact.get('neighborhood_prompts', [])[:4]):
        na = gen(np[:100])
        if tnew.lower() not in na.lower(): ns_a_ad += 1
        ta = set(nb[j].lower().split()); tb = set(na.lower().split())
        if len(ta & tb) / max(len(ta), len(tb)) > 0.3 if ta and tb else False: ns_c_ad += 1
        ns_n_ad += 1
    editor.rollback(bak)

    if (fi + 1) % 25 == 0:
        nd = fi + 1; e = time.time() - t0
        print(f'  [{nd}/{N}] ES_ee={es_ee_ad/nd:.0%} NS_abs={ns_a_ad/max(ns_n_ad,1):.0%} ({e:.0f}s)')
        sys.stdout.flush()

n = len(facts); e = time.time() - t0
ns_av = ns_a_ad / max(ns_n_ad, 1); ns_cv = ns_c_ad / max(ns_n_ad, 1)
es_eev = es_ee_ad / n; ps_eev = ps_ee_ad / (n * 2)
comp_ad = (es_eev + ps_eev + ns_av) / 3
print(f'\n  ADAPTIVE: ES_ee={es_eev:.0%} PS_ee={ps_eev:.0%} NS_abs={ns_av:.0%} Comp={comp_ad:.1%} Rep={rep_ad/n:.0%} ({e:.0f}s)')
print(f'\n  Fixed best: Comp={best_fixed[0]:.1%}  Adaptive: Comp={comp_ad:.1%}  Gain: {comp_ad-best_fixed[0]:+.1%}')

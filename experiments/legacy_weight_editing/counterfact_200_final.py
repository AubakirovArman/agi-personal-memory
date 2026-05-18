"""Final CounterFact 200 with anti-boost + rep_penalty=1.2."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 200

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

editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()

print(f'\nCounterFact {N} — clamp=0.30 + anti-boost + rep_pen=1.2')
print('-' * 60)
sys.stdout.flush()

es = es_cl = es_ee = ps = ps_ee = ns_h = ns_n = rb_ok = 0
rep_facts = 0
t0 = time.time()

for i, fact in enumerate(data):
    rw = fact['requested_rewrite']
    s, rel = rw['subject'], rw['relation_id']
    tnew, told = rw['target_new']['str'], rw['target_true']['str']
    p = rw['prompt'].format(s)
    tids = tok.encode(tnew, add_special_tokens=False)

    # BEFORE: neighborhood answers
    nb = [gen(np[:100]) for np in fact.get('neighborhood_prompts', [])[:4]]

    # NT: sample rows before
    weight = model.lm_head.weight.data
    edited_set = set(tids)
    nt_sample = [rid for rid in
                 [torch.randint(0, weight.shape[0], (1,)).item() for _ in range(200)]
                 if rid not in edited_set][:100]
    nt_before = {rid: weight[rid, :].clone() for rid in nt_sample}

    if not editor.apply_edit(s, tnew, rel, clamp_norm=0.3):
        continue

    # ES (substring)
    a = gen(p)
    if tnew.lower() in a.lower():
        es += 1
        if not has_rep(a, tnew):
            es_cl += 1
        else:
            rep_facts += 1

    # ES (EasyEdit — token exact)
    inp_ee = tok(p, return_tensors='pt').to(model.device)
    ilen_ee = inp_ee.input_ids.shape[1]
    with torch.no_grad():
        out_ee = model.generate(**inp_ee, max_new_tokens=len(tids), do_sample=False,
                                repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    if out_ee[0, ilen_ee:].cpu().tolist() == tids:
        es_ee += 1

    # PS (substring + EasyEdit)
    for pa in fact.get('paraphrase_prompts', [])[:2]:
        if tnew.lower() in gen(pa[:100]).lower():
            ps += 1
        inp_pee = tok(pa[:100], return_tensors='pt').to(model.device)
        ilen_pee = inp_pee.input_ids.shape[1]
        with torch.no_grad():
            out_pee = model.generate(**inp_pee, max_new_tokens=len(tids),
                                     do_sample=False, repetition_penalty=1.2,
                                     pad_token_id=tok.eos_token_id)
        if out_pee[0, ilen_pee:].cpu().tolist() == tids:
            ps_ee += 1

    # NS (overlap with before)
    for j, np in enumerate(fact.get('neighborhood_prompts', [])[:4]):
        if tok_overlap(nb[j], gen(np[:100])) > 0.3:
            ns_h += 1
        ns_n += 1

    # NT measurement
    nt_max = 0.0
    for rid, orig in nt_before.items():
        diff = (weight[rid, :] - orig.to(weight.device)).abs().max().item()
        nt_max = max(nt_max, diff)

    # Rollback
    editor.rollback()
    if told.lower() in gen(p).lower():
        rb_ok += 1

    if (i + 1) % 40 == 0:
        nd = i + 1
        e = time.time() - t0
        print(f'  [{nd}/{N}] ES={es/nd:.0%} ES_cl={es_cl/nd:.0%} '
              f'ES_ee={es_ee/nd:.0%} PS={ps/(nd*2):.0%} '
              f'NS={ns_h/max(ns_n,1):.0%} Rep={rep_facts}/{nd} ({e:.0f}s)')
        sys.stdout.flush()

nd = len(data)
e = time.time() - t0
es_v = es / nd; ecl_v = es_cl / nd; eee_v = es_ee / nd
ps_v = ps / (nd * 2); psee_v = ps_ee / (nd * 2); ns_v = ns_h / max(ns_n, 1)
comp_agim = (es_v + ps_v + ns_v) / 3
comp_ee = (eee_v + psee_v + ns_v) / 3
rep_r = rep_facts / nd

print(f'\n{"=" * 60}')
print(f'CounterFact {N} — ANTI-BOOST RESULTS')
print(f'{"=" * 60}')
print(f'  AGIM proto:  ES={es_v:.1%} ES_cl={ecl_v:.1%} PS={ps_v:.1%} NS={ns_v:.1%} Comp={comp_agim:.1%}')
print(f'  EasyEdit:    ES={eee_v:.1%} PS={psee_v:.1%} NS={ns_v:.1%} Comp={comp_ee:.1%}')
print(f'  Repetition:  {rep_facts}/{nd} ({rep_r:.0%})')
print(f'  RB:          {rb_ok/nd:.1%}')
print(f'  NT max:      {nt_max:.8f}')
print(f'  Time:        {e:.0f}s ({e/60:.1f}min)')

with open('results/counterfact_200_antibost.json', 'w') as f:
    json.dump({
        'n': nd,
        'AGIM': {'ES': round(es_v, 4), 'ES_clean': round(ecl_v, 4),
                 'PS': round(ps_v, 4), 'NS': round(ns_v, 4),
                 'Composite': round(comp_agim, 4)},
        'EasyEdit': {'ES': round(eee_v, 4), 'PS': round(psee_v, 4),
                     'NS': round(ns_v, 4), 'Composite': round(comp_ee, 4)},
        'repetition_rate': round(rep_r, 4),
        'NT_max': nt_max,
        'RB': round(rb_ok / nd, 4),
        'time_s': round(e, 1)
    }, f, indent=2)
print('Saved to results/counterfact_200_antibost.json')

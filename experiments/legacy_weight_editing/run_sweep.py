"""One-shot sweep: rep_penalty + clamp optimization. Load model ONCE."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'
DEV = 'cuda:3'
N_FACTS = 30

print('Loading model ONCE...')
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval()
print('OK')

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N_FACTS]

editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()


def gen(prompt, max_t=10, rep_pen=1.2, temp=None):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    ilen = inputs.input_ids.shape[1]
    kwargs = {'max_new_tokens': max_t, 'pad_token_id': tok.eos_token_id,
              'repetition_penalty': rep_pen}
    if temp and temp > 0:
        kwargs.update({'do_sample': True, 'temperature': temp, 'top_p': 0.9})
    else:
        kwargs['do_sample'] = False
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def has_rep(text, target, threshold=2):
    t = re.escape(target.lower())
    return bool(re.search(rf'({t})\1{{{threshold},}}', text.lower()))


def tok_overlap(a, b):
    ta = set(a.lower().split()); tb = set(b.lower().split())
    return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0


# ═══ EXP 1: rep_penalty sweep ═══
print('\nEXP 1: rep_penalty sweep (clamp=0.3)')
print('-' * 55)
configs = [
    (None, 1.0, 'baseline greedy'),
    (None, 1.15, 'greedy rep=1.15'),
    (None, 1.2, 'greedy rep=1.2'),
    (0.3, 1.0, 'T=0.3'),
    (0.3, 1.15, 'T=0.3 rep=1.15'),
]
best_r = None
for temp, rp, label in configs:
    es = es_cl = ps = 0
    t0 = time.time()
    for fact in data:
        rw = fact['requested_rewrite']
        s, rel = rw['subject'], rw['relation_id']
        tnew = rw['target_new']['str']
        p = rw['prompt'].format(s)
        editor.apply_edit(s, tnew, rel, clamp_norm=0.3)
        a = gen(p, temp=temp, rep_pen=rp)
        if tnew.lower() in a.lower():
            es += 1
            if not has_rep(a, tnew):
                es_cl += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100], temp=temp, rep_pen=rp).lower():
                ps += 1
        editor.rollback()
    n = len(data)
    e_v, ec_v, ps_v = es / n, es_cl / n, ps / (n * 2)
    print(f'  {label:22s} ES={e_v:.0%} ES_cl={ec_v:.0%} PS={ps_v:.0%} ({time.time()-t0:.0f}s)')
    if best_r is None or ec_v > best_r[0]:
        best_r = (ec_v, e_v, ps_v, temp, rp, label)

# ═══ EXP 2: clamp sweep ═══
t_best, rp_best = best_r[3], best_r[4]
print(f'\nEXP 2: clamp sweep ({best_r[5]}, rep_pen={rp_best})')
print('-' * 55)
best_c = None
for clamp in [0.1, 0.15, 0.2, 0.25, 0.3]:
    es = es_cl = ps = ns_h = ns_n = 0
    t0 = time.time()
    for fact in data:
        rw = fact['requested_rewrite']
        s, rel = rw['subject'], rw['relation_id']
        tnew = rw['target_new']['str']
        p = rw['prompt'].format(s)
        nb = [gen(np[:100], rep_pen=rp_best, temp=t_best)
              for np in fact.get('neighborhood_prompts', [])[:4]]
        editor.apply_edit(s, tnew, rel, clamp_norm=clamp)
        a = gen(p, rep_pen=rp_best, temp=t_best)
        if tnew.lower() in a.lower():
            es += 1
            if not has_rep(a, tnew):
                es_cl += 1
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            if tnew.lower() in gen(pa[:100], rep_pen=rp_best, temp=t_best).lower():
                ps += 1
        for i, np in enumerate(fact.get('neighborhood_prompts', [])[:4]):
            na = gen(np[:100], rep_pen=rp_best, temp=t_best)
            if tok_overlap(nb[i], na) > 0.3:
                ns_h += 1
            ns_n += 1
        editor.rollback()
    n = len(data)
    e_v, ec_v, ps_v = es / n, es_cl / n, ps / (n * 2)
    ns_v = ns_h / max(ns_n, 1)
    comp = (e_v + ps_v + ns_v) / 3
    print(f'  clamp={clamp:.2f}: ES={e_v:.0%} ES_cl={ec_v:.0%} PS={ps_v:.0%} '
          f'NS={ns_v:.0%} Comp={comp:.1%} ({time.time()-t0:.0f}s)')
    if best_c is None or comp > best_c[0]:
        best_c = (comp, clamp, e_v, ec_v, ps_v, ns_v)

# ═══ FINAL ═══
print(f'\n{"="*55}')
print(f'BEST CONFIGURATION')
print(f'{"="*55}')
print(f'  Generation:  {best_r[5]} (rep_pen={rp_best})')
print(f'  ES:          {best_r[1]:.0%}')
print(f'  ES_clean:    {best_r[0]:.0%}')
print(f'  PS:          {best_r[2]:.0%}')
print(f'  Clamp:       {best_c[1]:.2f}')
print(f'  NS:          {best_c[5]:.0%}')
print(f'  Composite:   {best_c[0]:.1%}')

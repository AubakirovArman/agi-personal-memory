"""Experiment B: anti-boost sweep."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'
N = 30

print('Loading model...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval()
print('OK'); sys.stdout.flush()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N]

def gen(prompt, max_t=10, rep_pen=1.2):
    inputs = tok(prompt, return_tensors='pt').to(model.device); ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_t, do_sample=False, repetition_penalty=rep_pen, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()

def has_rep(text, target, thr=2):
    t = re.escape(target.lower())
    return bool(re.search(rf'({t})\1{{{thr},}}', text.lower()))

def tok_overlap(a, b):
    ta = set(a.lower().split()); tb = set(b.lower().split())
    return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0

editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()

print('Sweep (EOS + anti-boost):')
best = None
for clamp in [0.15, 0.2, 0.25, 0.3]:
    es=es_cl=ps=ns_h=ns_n=0; t0=time.time()
    for fact in data:
        rw=fact['requested_rewrite']; s,rel=rw['subject'],rw['relation_id']; tnew=rw['target_new']['str']
        p=rw['prompt'].format(s)
        nb=[gen(np[:100]) for np in fact.get('neighborhood_prompts',[])[:4]]
        if not editor.apply_edit(s,tnew,rel,clamp_norm=clamp): continue
        a=gen(p)
        if tnew.lower() in a.lower():
            es+=1
            if not has_rep(a,tnew): es_cl+=1
        for pa in fact.get('paraphrase_prompts',[])[:2]:
            if tnew.lower() in gen(pa[:100]).lower(): ps+=1
        for i,np in enumerate(fact.get('neighborhood_prompts',[])[:4]):
            if tok_overlap(nb[i], gen(np[:100]))>0.3: ns_h+=1; ns_n+=1
        editor.rollback()
    n=len(data); e_v=es/n; ec_v=es_cl/n; ps_v=ps/(n*2); ns_v=ns_h/max(ns_n,1)
    comp=(e_v+ps_v+ns_v)/3
    print(f'  clamp={clamp:.2f}: ES={e_v:.0%} ES_cl={ec_v:.0%} PS={ps_v:.0%} NS={ns_v:.0%} Comp={comp:.1%} ({time.time()-t0:.0f}s)')
    if best is None or comp>best[0]: best=(comp,clamp,e_v,ec_v,ps_v,ns_v)
    sys.stdout.flush()

print(f'\nBEST: clamp={best[1]:.2f} Comp={best[0]:.1%} ES={best[2]:.0%} ES_cl={best[3]:.0%} NS={best[5]:.0%}')

"""Test remaining EasyEdit ideas with CORRECT NS metric (target NOT in neighbor)."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from agim.model.wal_editor import WalLmHeadEditor

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 50

print('Loading model...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval(); print('OK'); sys.stdout.flush()

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f:
    data = json.load(f)[:N]


class TargetStopCriteria(StoppingCriteria):
    """Fix 7: Stop when target tokens generated."""
    def __init__(self, target_ids, tokenizer):
        self.target_ids = target_ids
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < len(self.target_ids): return False
        return input_ids[0, -len(self.target_ids):].cpu().tolist() == self.target_ids


def gen(prompt, max_t=10, rep_pen=1.2, temp=None, early_stop_tids=None):
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    ilen = inputs.input_ids.shape[1]
    kwargs = {'max_new_tokens': max_t, 'pad_token_id': tok.eos_token_id,
              'repetition_penalty': rep_pen}
    if temp and temp > 0:
        kwargs.update({'do_sample': True, 'temperature': temp, 'top_p': 0.9})
    else:
        kwargs['do_sample'] = False
    if early_stop_tids:
        kwargs['stopping_criteria'] = StoppingCriteriaList([TargetStopCriteria(early_stop_tids, tok)])
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tok.decode(out[0][ilen:], skip_special_tokens=True).strip()


def has_rep(text, target, thr=2):
    t = re.escape(target.lower())
    return bool(re.search(rf'({t})\1{{{thr},}}', text.lower()))


def eval_config(label, clamp=0.3, temp=None, rep_pen=1.2, early_stop=False):
    """Evaluate one config with correct NS metric (target NOT in neighbor)."""
    es = es_cl = es_ee = ps = ps_ee = ns = ns_n = rep = 0
    t0 = time.time()

    for fact in data:
        rw = fact['requested_rewrite']
        s, rel = rw['subject'], rw['relation_id']
        tnew = rw['target_new']['str']
        p = rw['prompt'].format(s)
        tids = tok.encode(tnew, add_special_tokens=False)

        editor.apply_edit(s, tnew, rel, clamp_norm=clamp)

        # ES (substring)
        est = tids if early_stop else None
        a = gen(p, temp=temp, rep_pen=rep_pen, early_stop_tids=est)
        if tnew.lower() in a.lower():
            es += 1
            if not has_rep(a, tnew): es_cl += 1
            else: rep += 1

        # ES EasyEdit
        inp = tok(p, return_tensors='pt').to(model.device); il = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=len(tids), do_sample=False,
                                 repetition_penalty=rep_pen, pad_token_id=tok.eos_token_id)
        if out[0, il:].cpu().tolist() == tids: es_ee += 1

        # PS
        for pa in fact.get('paraphrase_prompts', [])[:2]:
            pa_ans = gen(pa[:100], temp=temp, rep_pen=rep_pen, early_stop_tids=est)
            if tnew.lower() in pa_ans.lower(): ps += 1
            inp2 = tok(pa[:100], return_tensors='pt').to(model.device); il2 = inp2.input_ids.shape[1]
            with torch.no_grad():
                out2 = model.generate(**inp2, max_new_tokens=len(tids), do_sample=False,
                                      repetition_penalty=rep_pen, pad_token_id=tok.eos_token_id)
            if out2[0, il2:].cpu().tolist() == tids: ps_ee += 1

        # NS: target NOT in neighbor (correct metric)
        for np in fact.get('neighborhood_prompts', [])[:4]:
            na = gen(np[:100], temp=temp, rep_pen=rep_pen)
            if tnew.lower() not in na.lower(): ns += 1
            ns_n += 1

        editor.rollback()

    n = len(data)
    es_v = es / n; ecl_v = es_cl / n; eee_v = es_ee / n
    ps_v = ps / (n * 2); psee_v = ps_ee / (n * 2)
    ns_v = ns / max(ns_n, 1)
    comp_agim = (es_v + ps_v + ns_v) / 3
    comp_ee = (eee_v + psee_v + ns_v) / 3
    e = time.time() - t0
    print(f'  {label:35s} ES={es_v:.0%} ES_cl={ecl_v:.0%} PS={ps_v:.0%} NS={ns_v:.0%} '
          f'Comp_AGIM={comp_agim:.1%} Comp_EE={comp_ee:.1%} Rep={rep/n:.0%} ({e:.0f}s)')
    sys.stdout.flush()
    return comp_ee, es_v, ns_v, ecl_v


editor = WalLmHeadEditor(model, tok, K=256, lmax=16, device=DEV)
editor.build_vocab()

print('\n===== Testing Remaining Ideas (correct NS metric) =====')
print('-' * 65)
sys.stdout.flush()

results = []

# Baseline: current best config
r = eval_config('BASELINE clamp=0.3 rep=1.2', clamp=0.3, rep_pen=1.2)
results.append(('BASELINE', r))

# Fix 3: Temperature sweep
for temp in [0.2, 0.3, 0.5]:
    r = eval_config(f'Fix3 T={temp} clamp=0.3', clamp=0.3, temp=temp)
    results.append((f'Fix3 T={temp}', r))

# Fix 7: Early stopping
r = eval_config('Fix7 early_stop clamp=0.3', clamp=0.3, early_stop=True)
results.append(('Fix7 early_stop', r))

# Fix 5: Truncate (use len(tids) tokens in generation check) — already in ES_ee

# Lower clamp + early stop combo
for clamp in [0.2, 0.25]:
    r = eval_config(f'clamp={clamp} + early_stop', clamp=clamp, early_stop=True)
    results.append((f'clamp={clamp}+stop', r))

# Best combo: T + early_stop
r = eval_config('T=0.2 + early_stop clamp=0.3', clamp=0.3, temp=0.2, early_stop=True)
results.append(('T=0.2+stop', r))

print(f'\n===== RANKING (by EasyEdit Composite) =====')
results.sort(key=lambda x: x[1][0], reverse=True)
for name, (comp_ee, es, ns, ecl) in results:
    print(f'  {name:30s} Comp_EE={comp_ee:.1%} ES={es:.0%} NS={ns:.0%} ES_cl={ecl:.0%}')

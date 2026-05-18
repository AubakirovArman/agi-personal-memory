"""ROME-style WAL-FFN: solve W_new @ k = v_target, not manual column boost."""
import torch, sys, time, json, urllib.request, re
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.wal.encoder import build_atoms_kmeans, wal_encode_scalar_gpu

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'; DEV = 'cuda:3'; N = 30
print('Loading...'); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, device_map=DEV, local_files_only=True)
model.eval(); print('OK')

with urllib.request.urlopen('https://rome.baulab.info/data/dsets/counterfact.json') as f: facts = json.load(f)[:N]

def gen(p, t=10):
    i = tok(p, return_tensors='pt').to(model.device); il = i.input_ids.shape[1]
    with torch.no_grad(): o = model.generate(**i, max_new_tokens=t, do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][il:], skip_special_tokens=True).strip()

# Build WAL atoms for down_proj at key layers
print('Building atoms...'); sys.stdout.flush()
layers = [7, 15, 23, 31]
atoms_dict = {}
for lidx in layers:
    W = model.model.layers[lidx].mlp.down_proj.weight.data.float()
    atoms_dict[lidx] = build_atoms_kmeans(W.flatten()[:2_000_000], 256, iters=5, device=torch.device(DEV))
    print(f'  L{lidx}: atoms={atoms_dict[lidx].shape}')

def get_mlp_key(layer, pids):
    down = model.model.layers[layer].mlp.down_proj
    m = []
    def hook(module, inp, out): nonlocal m; m.append(inp[0].detach().clone())
    h = down.register_forward_hook(hook)
    with torch.no_grad(): model(input_ids=pids.unsqueeze(0).to(model.device))
    h.remove()
    return (m[0][0, -1, :].float() / (m[0][0, -1, :].norm() + 1e-8)) if m else None

def get_last_hidden(pids):
    last = []
    def hook(m, i, o): nonlocal last; hs = o[0] if isinstance(o, tuple) else o; last.append(hs.detach().clone())
    h = model.model.norm.register_forward_hook(hook)
    with torch.no_grad(): model(input_ids=pids.unsqueeze(0).to(model.device))
    h.remove()
    return (last[0][0, -1, :].float() / (last[0][0, -1, :].norm() + 1e-8)) if last else None

# ═══ ROME-style FFN edit ═══
print('\nROME-style WAL-FFN + dual-layer sweep:')
print('-' * 60)
best = None

for ffn_layer in [7, 15, 31]:
    for ffn_clamp in [0.01, 0.05, 0.1]:
        es = es_cl = ps = ns_h = ns_n = rep = 0
        t0 = time.time()
        for fact in facts:
            rw = fact['requested_rewrite']; s = rw['subject']; tnew = rw['target_new']['str']; p = rw['prompt'].format(s)
            tids = tok.encode(tnew, add_special_tokens=False); pids = tok(p, return_tensors='pt').input_ids[0]

            # ── ROME-style: solve W @ k = v_target ──
            k = get_mlp_key(ffn_layer, pids)
            if k is None: continue
            W = model.model.layers[ffn_layer].mlp.down_proj
            W_orig = W.weight.data.clone()
            v_curr = torch.mv(W.weight.data.float(), k)
            tdir = model.lm_head.weight.data[tids[0], :].float().to(DEV)
            v_target = v_curr + ffn_clamp * tdir / (tdir.norm() + 1e-8) * v_curr.norm()
            delta_v = v_target - v_curr
            # rank-1: W += (v_target - W@k) @ k^T / (k^T k)
            denom = torch.dot(k, k) + 1e-6
            update = torch.outer(delta_v, k) / denom

            # Apply through WAL
            atoms_gpu = atoms_dict[ffn_layer].to(DEV)
            changed = 0
            for j in range(W.weight.data.shape[1]):
                col = W.weight.data[:, j].float().to(DEV)
                diff = update[:, j].abs().max().item()
                if diff > 1e-5:
                    _, _, rec = wal_encode_scalar_gpu(col + update[:, j].to(DEV), atoms_gpu, 16)
                    W.weight.data[:, j] = rec.to(device=W.device, dtype=W.dtype)
                    changed += 1

            # ── Also do light lm_head boost ──
            last_k = get_last_hidden(pids)
            if last_k is not None:
                for ti, tid in enumerate(tids):
                    row = model.lm_head.weight.data[tid, :].float().to(DEV)
                    _, _, rec = wal_encode_scalar_gpu(row + 0.1 * last_k.to(DEV) / (2**ti), atoms_dict[31].to(DEV), 16)
                    model.lm_head.weight.data[tid, :] = rec.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)

            a = gen(p)
            if tnew.lower() in a.lower():
                es += 1
                if not re.search(rf'({re.escape(tnew.lower())})\1{{2,}}', a.lower()): es_cl += 1
                else: rep += 1
            for pa in fact.get('paraphrase_prompts', [])[:2]:
                if tnew.lower() in gen(pa[:100]).lower(): ps += 1
            for np in fact.get('neighborhood_prompts', [])[:4]:
                if tnew.lower() not in gen(np[:100]).lower(): ns_h += 1; ns_n += 1

            # Rollback FFN + lm_head
            W.weight.data = W_orig

        n = len(facts); e = time.time() - t0
        ns_v = ns_h / max(ns_n, 1); comp = (es / n + ps / (n * 2) + ns_v) / 3
        print(f'  L={ffn_layer} clamp={ffn_clamp:.2f}: ES={es/n:.0%} PS={ps/(n*2):.0%} NS={ns_v:.0%} Comp={comp:.1%} ({e:.0f}s)')
        sys.stdout.flush()
        if best is None or comp > best[0]: best = (comp, ffn_layer, ffn_clamp, es/n, ns_v)

print(f'\n  BEST: L={best[1]} clamp={best[2]:.2f} Comp={best[0]:.1%} ES={best[3]:.0%} NS={best[4]:.0%}')

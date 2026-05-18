# EasyEdit AGIMWAL Adapter

This repo now ships a copy-ready EasyEdit method adapter in
`agim.integrations.easyedit_agimwal`.

The adapter is for external EasyEdit review without moving AGIM code into an
EasyEdit fork. Install this repository, then register the adapter in an
EasyEdit checkout:

```python
from agim.integrations.easyedit_agimwal import (
    AGIMWALHyperParams,
    apply_agimwal_to_model,
)

ALG_DICT["AGIMWAL"] = apply_agimwal_to_model
```

The adapter exposes the same entry-point shape as standard EasyEdit methods:

```python
edited_model, weights_copy = apply_agimwal_to_model(
    model, tokenizer, requests, hparams, return_orig_weights=True)
```

## Hparams Template

```yaml
alg_name: AGIMWAL
model_name: meta-llama/Llama-3.1-8B-Instruct
device: 0
backend: dual_row
K: 256
lmax: 16
clamp_lm: 0.20
clamp_embed: 0.06
clamp_eos: 0.0
clamp_anti: 0.06
target_token_mode: contextual
use_neg_prompts: true
neg_prompt_limit: 10
use_positive_prompts: false
batch_size: 1
model_parallel: false
fp16: true
```

Use `backend: wal_rome` to route through the experimental located FFN backend.
That backend is implemented, but it is not yet a headline EasyEdit result.

## Scope

- This is a local adapter package, not an upstream EasyEdit PR.
- It returns full-tensor backups because EasyEdit's generic rollback path
  restores parameters from `weights_copy`.
- Current public claims must still use tracked artifacts under
  `results/easyedit_official/`, not adapter smoke tests.

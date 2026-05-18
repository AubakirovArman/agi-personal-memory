# Verify Path B Legacy Substrate Tests

This document covers older WAL/ROME substrate checks. They are useful for
debugging editing components, but they are not the current flagship Path B
evidence.

## Legacy Scope

Legacy substrate tests include direct `WALWeightEditor`, `ROMEEditor`, Gemma E2E
checks, and older local CounterFact scripts. They can show that a specific
editing path mutates a model and rolls back locally. They do not replace the
current EasyEdit-compatible artifacts.

The heavy local-model E2E files are marked with `pytest.mark.substrate` in
addition to `slow`/`gpu`/model-specific markers. Use that marker when explicitly
debugging legacy model-editing substrate behavior.

## Optional Local E2E Pattern

Use explicit legacy-only placeholders and keep this flow on local machines only:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_editor import ROMEEditor

model_path = "/path/to/local/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype="auto", device_map=device, local_files_only=True)
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

editor = ROMEEditor(model, tok, device=device)
editor.apply_edit(
    subject="Zanikland",
    relation="capital",
    target="Blorptown",
    target_layer=5,
)
editor.rollback()
```

For legacy smoke scripts, environment-variable defaults are intentionally not
enforced here because this area is not used for current Path B claims.

## Legacy Artifact Boundary

`results/local_protocol/official_eval_1000.json` is a historical AGIM-local
CounterFact diagnostic. It is not the current official-compatible EasyEdit
result and should not be compared directly with EasyEdit leaderboards.

## Use Current Path B Docs For Claims

- `docs/VERIFY_PATH_B_CURRENT.md`
- `BENCHMARK.md`
- `CURRENT_STATUS.md`
- `docs/CLAIMS_AND_EVIDENCE.md`

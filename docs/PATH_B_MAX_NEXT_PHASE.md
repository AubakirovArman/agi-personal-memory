# Path B: Next Phase (Immediate, High-Impact)

Scope: `agi_personal_memory` (as of 2026-05-18).

## Что уже подтверждено уже сейчас

- Single-edit Path B работает по EasyEdit-совместимому раннеру на n=50/n=1000:  
  `results/easyedit_official/current/random_50_report_2026-05-18.md`,  
  `results/easyedit_official/current/random_200_report_2026-05-18.md`,  
  `results/easyedit_official/current/random_1000_report_2026-05-18.md`.
- Side-slot уже есть как runnable путь в n=50 sequential comparison (через matrix):  
  `results/easyedit_official/ablations/backend_matrix_sequential_random_50_report_2026-05-18.md`.
- Последовательность и governance-подготовка есть, но не закрыта до production-ready proof packet.

## Что всё ещё реально отсутствует на hard-gate уровне

1. Gate 5: public PatchService/governance release packet с сохранённым lifecycle-следом.
2. Req. 23: constrained `K_pos/K_neg`.
3. Req. 24: relation-protected banks.
4. Req. 25: ENCORE-like budget/no-commit guard.
5. Req. 26: patch conflict detector.
6. Req. 30: relation-aware slot sharding.
7. Req. 39, 40: PatchService public API lifecycle + governance/adapter boundary.

## Очередность на ближайший блок (без локальной метрики №1 vs EasyEdit-claim сравнения)

1. Собрать публичный PatchService/governance proof packet (`claims lock`, tenant context, relation filters, immutable audit bundle).
2. Закрыть req. 23, req. 25 с новыми артефактами.
3. Закрыть req. 26, req. 30.
4. Обновить `PATH_B_MAX_*` status-документы и `CLAIMS_AND_EVIDENCE.md` под новую публичную boundary.

Финальный Gate 5 production run:

```bash
cd /mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory
export AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1
export AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER="<PRODUCTION_IMMUTABLE_PROVIDER>"
export AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE="object_lock"
export AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL="https://<PUBLIC_HOST>/api"
export AGIM_GATE5_PUBLIC_API_SMOKE=1
bash scripts/run_path_b_max_bootstrap.sh 5
```

Требуемый evidence после прогона:

- `results/easyedit_official/governance/path_b_max_gate5_public_release.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_receipt.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_bundle.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_transport_manifest.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_index.json`
- `results/easyedit_official/governance/path_b_max_gate5_public_api*` smoke trace artifact (actual filename depends on script output)
- обновлённые `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`, `PATH_B_MAX_COMPLETION_MATRIX.md`, `PATH_B_MAX_STATUS_BOARD.md`, `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`

## Что использовать как source-of-truth статуса

- `docs/PATH_B_MAX_ACTION_PLAN.md`
- `docs/PATH_B_MAX_HARDGATE_QUEUE.md`
- `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`
- `docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`
- `docs/PATH_B_MAX_COMPLETION_MATRIX.md`
- `docs/PATH_B_COMPLETION_AUDIT.md`
- `docs/CLAIMS_AND_EVIDENCE.md`

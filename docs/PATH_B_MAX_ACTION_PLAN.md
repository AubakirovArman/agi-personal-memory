# Path B Max Action Plan (Objective Executor)

Цель: довести репозиторий до состояния, где каждый пункт из
`sites/deep-research-report (5).md` имеет:
- источник правды (команда),
- воспроизводимый артефакт,
- явный статус покрытия,
- и критерий закрытия.

Используется вместе с
- [PATH_B_MAX_EXECUTION_CHECKLIST.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_CHECKLIST.md)
- [PATH_B_MAX_EXECUTION_RUNBOOK.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_RUNBOOK.md)
- [PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md)
- [PATH_B_MAX_GATE_COMMANDS.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_GATE_COMMANDS.md)

## Hard gates (blocking beta-ready)

1. `wal_memit` official n=50 baseline
   - Evidence target: `results/easyedit_official/current/random_50_seed_42_wal_memit*.json` + markdown summary
   - Command target: `python -m agim.eval.easyedit_official_runner --edit-backend wal_memit ...`
   - Status: not done
  - Done criteria:
    - JSON + MD pair contains `artifact_schema_version`, `method_profile_id`, `base_model_digest`, `atoms_digest`
    - CLI args pinned in payload: `model`, `n`, `seed`, `sample-policy`, `target-token-mode`, `method-profile-id`
    - `failure_analysis` + patch-delta summary included
  - After close: update status in `PATH_B_MAX_EXECUTION_CHECKLIST.md` and `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.

2. Full backend comparison (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) on same dataset/profile
   - Evidence target: `results/easyedit_official/ablations/*backend_matrix*.md`
   - Command target: `python -m agim.eval.easyedit_official_runner ... --compare-backends ...`
   - Status: partial
  - Done criteria:
    - one matrix command with `--sample-policy`, `--n`, `--seed` fixed for all backends
    - per-backend JSON artifacts for the same sample set and `method_profile_id`
    - report includes side-by-side metric deltas and failure family split
  - After close: update status in `PATH_B_MAX_EXECUTION_CHECKLIST.md` and `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.

3. External model-output chain coverage (minimum 4 channels)
   - RippleEdits
   - MQuAKE
   - raw-text
   - one product-like benchmark
   - Evidence target: `results/external_benchmark_runs/*` with raw output + scored report
   - Status: partial
  - Done criteria:
     - model-output JSON/JSONL for each channel with reproducible `command` + artifact hash
     - per-channel scored markdown with metric definition
     - at least one baseline baseline-vs-new comparison or equivalent benchmark delta
  - After close: update status in `PATH_B_MAX_EXECUTION_CHECKLIST.md`, `PATH_B_COMPLETION_AUDIT.md` and `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.

4. Side-slot retention hardening (`10/50/100`, seeds `42/43/44`) with locality evidence
   - Evidence target: `results/easyedit_official/sequential/` family
   - Command target: `python -m agim.eval.easyedit_official_runner --sequential-edit --edit-backend side_slot ...`
   - Status: partial
  - Done criteria:
     - side-slot runs for `n=10`, `n=50`, `n=100` each for seeds `42/43/44`
     - `save-neighbor-text` + `save-failures-only` captured for all runs
     - retention table per n in one markdown summary
  - After close: update status in `PATH_B_MAX_EXECUTION_CHECKLIST.md`, `PATH_B_COMPLETION_AUDIT.md` and `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.

5. PatchService/governance proof packet
   - Evidence target: release-gate packet with public API contract and immutable audit trail
   - Command/docs target: patch_service/governance docs + runner outputs
   - Status: partial
  - Done criteria:
     - documented public lifecycle operations: `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`
     - signed governance chain artifact for one concrete patch path
     - claims docs updated: unsafe claims removed from `README`, `CURRENT_STATUS`, `BENCHMARK`, `CLAIMS_AND_EVIDENCE`
  - After close: update status in `PATH_B_MAX_EXECUTION_CHECKLIST.md`, `PATH_B_COMPLETION_AUDIT.md`, `PATH_B_MAX_PROMPT_ARTIFACT_AUDIT.md`, and `PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.

## Операционный порядок на ближайшие 2 этапа

1. Закрыть Gate 1
2. Закрыть Gate 2
3. Закрыть Gate 3
4. Закрыть Gate 4
5. Закрыть Gate 5

## Текущие обязательные обновления в документах

- Учитывать, что локальный `results/local_protocol/*` не является `official` путем.
- Не добавлять новые `Done` без конкретного `JSON+MD` из `results/easyedit_official/*` или `results/external_benchmark_runs/*`.
  - Все новые команды публиковать с окружением (`--model`, `--sample-policy`, `--seed`, `--n`, `--method-profile-id`, `--edit-backend`), чтобы блокировать неоднозначность сравнения.
- Для hard-gate доказательств указывать путь к сохранённому payload и его checksum.

## Рекомендуемый статус после каждой итерации

- Done: только если есть официальные артефакты и явное согласование со статусом чеклиста.
- Partial: только если есть code path/adapter без tracked full output chain.
- Not Started: нет ни кода, ни артефакта для требования.
- In Progress: есть артефакт, но требуются seed/раунды/метрические разрезы.

## Быстрый triage после каждого run

- Провести один run → записать output в `results/...` по соглашению naming.
- Добавить краткую запись в `docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`.
- Обновить соответствующую строку в `docs/PATH_B_COMPLETION_AUDIT.md`.
- Обновить boundary claims в `docs/CLAIMS_AND_EVIDENCE.md`.

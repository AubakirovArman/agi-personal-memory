# Path B Maximal Execution Checklist

Источник: `/mnt/hf_model_weights/arman/3bit/sites/deep-research-report (5).md`
Дата: 2026-05-18

Цель:
- Для всех 40 требований deep-research создать traceability-by-evidence: required command/artifact/test/gate → concrete file → статус.
- Зафиксировать блокеры как закрытые только при наличии официальных артефактов (JSON/MD/outputs), а не только «планов в коде».

Операционный порядок блокеров:
- [PATH_B_MAX_EXECUTION_RUNBOOK.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_EXECUTION_RUNBOOK.md)
- Исполнительный маршрутизатор: [PATH_B_MAX_ACTION_PLAN.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_ACTION_PLAN.md)
- Completion matrix: [PATH_B_MAX_COMPLETION_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_COMPLETION_MATRIX.md)
- Audit matrix: [PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md)
- Gate commands: [PATH_B_MAX_GATE_COMMANDS.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_GATE_COMMANDS.md)
- Операционный queue: [PATH_B_MAX_HARDGATE_QUEUE.md](/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/docs/PATH_B_MAX_HARDGATE_QUEUE.md)

## Чеклист (1..40)

1. Source-of-truth разносится в один активный источник  
   - Артефакт: `CURRENT_STATUS.md`, `BENCHMARK.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`  
   - Статус: Done  
   - Следующий шаг: поддерживать синхронно при любом новом артефакте

2. README без смешивания Path A и Path B  
   - Артефакт: `README.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md`  
   - Статус: Done  
   - Следующий шаг: сохранить разделение Quick-start для каждого пути при обновлении команд

3. Документация verify заменена на 3-track вариант  
   - Артефакт: `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md`, `docs/VERIFY_PATH_A.md`, `docs/HOW_TO_VERIFY.md`  
   - Статус: Done  
   - Следующий шаг: поддерживать 3-track ссылочную модель при любом появлении новых инструкций

4. Legacy-контур WALWeight/ROME помечен как исторический  
   - Артефакт: `docs/PATH_B_COMPLETION_AUDIT.md`, `results/local_protocol/README.md`  
   - Статус: Partial  
   - Следующий шаг: явно отделить legacy E2E тесты в отчетах/CI runbook

5. Operating profiles и `method_profile_id` в метаданных  
   - Артефакт: `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py`  
   - Статус: Done  
   - Следующий шаг: требовать profile_id в каждом новом официальном run и отчетах

6. `artifact_schema_version`  
   - Артефакт: `src/agim/eval/easyedit_run_metadata.py`  
   - Статус: Done  
   - Следующий шаг: обновить changelog при schema changes

7. Убрать machine-specific defaults из публичных путей  
   - Артефакт: `docs/VERIFY_PATH_B_CURRENT.md`, `src/agim/eval/easyedit_cli.py`  
   - Статус: Done  
   - Следующий шаг: пройтись по README-скриптам и убрать оставшиеся абсолютные пути в public guidance

8. Один reproducibility bundle на базовые режимы  
   - Артефакт: `docs/VERIFY_PATH_B_CURRENT.md`, `results/easyedit_official/current/`, `results/easyedit_official/sequential/`  
   - Статус: Done  
   - Следующий шаг: добавить готовый single command script для полного набора baseline + отчётов

9. `base_model_digest` + `atoms_digest` в payload  
   - Артефакт: `src/agim/eval/easyedit_payload.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/model/patch_artifact.py`  
   - Статус: Partial  
   - Следующий шаг: enforce validation на уровне patch apply/runner для hard mismatch

10. Durable PatchArtifact для Path B  
   - Артефакт: `src/agim/model/patch_artifact.py`, `tests/test_patch_artifact.py`  
   - Статус: Partial  
   - Следующий шаг: связать Artifact lifecycle с official runner outputs и canary suite

11. Тест save/reload/re-apply/rollback  
   - Артефакт: `tests/test_patch_service.py`, `tests/test_patch_artifact.py`  
   - Статус: Partial  
   - Следующий шаг: добавить тест полного loop: `save patch → reload process → apply → outputs parity → rollback`

12. Раздельный `failures_only` по метрикам  
   - Артефакт: `src/agim/eval/easyedit_records.py`, `src/agim/eval/easyedit_official_runner.py`  
   - Статус: Partial/Not Started  
   - Следующий шаг: добавить селектор metric family в CLI + тест на `tf/ctx_gen/prob/vanilla_gen`

13. `random_200` seed-coverage  
   - Артефакт: `results/easyedit_official/current/random_200_report_2026-05-18.md` и JSON/seed артефакты  
   - Статус: Done  
   - Следующий шаг: сохранить полный `failures`/`dry_run` bundle для 200 run

14. `random_1000` seed-coverage  
   - Артефакт: `results/easyedit_official/current/random_1000_report_2026-05-18.md`, `results/easyedit_official/current/random_1000_seed_42.json`  
   - Статус: Done  
   - Следующий шаг: добавить seed-сравнение по `seed_44/seed_43` для более плотной стабилизации

15. `target_token_mode` matrix  
   - Артефакт: `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`  
   - Статус: Done  
   - Следующий шаг: включить `both` fixed primary sequence как отдельный named mode

16. Component ablations  
   - Артефакт: `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md`  
   - Статус: Done  
   - Следующий шаг: продлить на разные profiles (`seq_tuned`, `seq_positive`) при необходимости

17. Exact-additive baseline  
   - Артефакт: `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md`  
   - Статус: Done  
   - Следующий шаг: закрепить как baseline в release notes

18. Deterministic NT mode  
   - Артефакт: `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py`  
   - Статус: Partially Done  
   - Следующий шаг: зафиксировать `random_seed` и `nt_snapshot_path` в обязательный bundle summary

19. Метрики norm growth и reuse  
   - Артефакт: `results/easyedit_official/current/easyedit_agim_status_2026-05-18.md`  
   - Статус: Partially Done  
   - Следующий шаг: добавить cumulative edit norm/row overlap в единый audit schema

20. Убрать глобальный EOS из default профиля  
   - Артефакт: `src/agim/eval/easyedit_presets.py`, `results/easyedit_official/ablations/eos_default_report_2026-05-18.md`  
   - Статус: Done  
   - Следующий шаг: сохранить EOS-control как optional non-persistent policy path

21. Убрать global anti-rep из глобальных строк  
   - Артефакт: `src/agim/eval/easyedit_presets.py`, `results/easyedit_official/ablations/component_random_200_seed_42_dual_no_anti*.json`  
   - Статус: Not Started  
   - Следующий шаг: заменить на decode-time или patch-local stop rules с контролем locality

22. Fix `target_token_mode="both"` primary sequence  
   - Артефакт: `results/easyedit_official/ablations/both_primary_fixed_report_2026-05-18.md`, `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_eval_loop.py`  
   - Статус: Done (partial design evidence)  
   - Следующий шаг: закрыть с dedicated run для `both` baseline в backend matrix

23. Constrained K_pos/K_neg key solve  
   - Артефакт: `src/agim/model/wal_dual_editor.py`, связанные ablations  
   - Статус: Not Started  
   - Следующий шаг: реализовать constrained update solver и добавить ablation сравнений с K_neg penalties

24. Relation-specific protected banks  
   - Артефакт: `results/easyedit_official/sequential/relation_protected_bank_report_2026-05-18.md`  
   - Статус: Partially Done  
   - Следующий шаг: расширить failure-priority sampling по worst-relation кейсам

25. ENCORE-style norm budgets и ранний stop  
   - Артефакт: `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_side_slot_loop.py` (проектные заготовки)  
   - Статус: Not Started  
   - Следующий шаг: внедрить per-backend budget guard и reject-commit path

26. Patch conflict detector  
   - Артефакт: `src/agim/model` (заготовки для conflict checks)  
   - Статус: Not Started  
   - Следующий шаг: внедрить rule checks (subject overlap, target overlap, EOS/control rows, relation overlap)

27. Runtime sparse overlay  
   - Артефакт: `src/agim/model` overlay experiments и current API hooks  
   - Статус: Partially Done  
   - Следующий шаг: перевести apply flow на overlay-by-default для lm_head/embed

28. Tenant/session mutable state isolation  
   - Артефакт: `src/agim/model` и runner glue  
   - Статус: Not Started  
   - Следующий шаг: ввести patch namespace + patch stack per tenant/session

29. Side-slot memory для sequential  
   - Артефакт: `results/easyedit_official/sequential/side_slot_random_50_report_2026-05-18.md`  
   - Статус: In Progress  
   - Следующий шаг: закрыть side-slot retention 10/50/100 edit benchmark

30. Relation sharding isolation для slots  
   - Артефакт: `src/agim/model` и `src/agim/eval/easyedit_side_slot_loop.py`  
   - Статус: Not Started  
   - Следующий шаг: добавить relation-aware slot allocator

31. Side-slot retention 10/50/100 сравнение  
   - Артефакт: `results/easyedit_official/sequential/`  
   - Статус: Not Started  
   - Следующий шаг: провести side-slot runs и сравнить с tuned in-place baseline

32. `WALRomeEditor` как located FFN backend  
   - Артефакт: `src/agim/model/wal_rome_editor.py`, `results/easyedit_official/ablations/wal_rome_smoke_5_report_2026-05-18.md`  
   - Статус: Partially Done  
   - Следующий шаг: поднять от smoke к n=50 quality baseline

33. `WALMemitBatchEditor` для consolidation  
   - Артефакт: `src/agim/model/wal_memit_batch_editor.py`, `tests/test_wal_memit_batch_editor.py`  
   - Статус: Done (infrastructure)  
   - Следующий шаг: закрыть официальную n=50 quality baseline для `wal_memit`

34. Backend сравнение `dual_row`, `wal_rome`, `wal_memit`, `side_slot`  
   - Артефакт: `src/agim/eval/easyedit_backend_matrix.py`, `results/easyedit_official/ablations/backend_matrix_random_50_report_2026-05-18.md`  
   - Статус: In Progress  
   - Следующий шаг: сделать независимый n=50 и n=1000 matrix для `wal_memit` и side_slot

35. RippleEdits diagnostic обязательный  
   - Артефакт: `src/agim/eval/ripple_diagnostic.py`, `results/external_benchmark_*`  
   - Статус: Partial (адптеры есть, model-output run нет)  
   - Следующий шаг: выполнить tracked model-output RippleEdits run

36. MQuAKE второй уровень  
   - Артефакт: `src/agim/eval/mquake_diagnostic.py`, `results/external_benchmark_adapters/mquake_cf_3k_v2_first50_adapter.json`  
   - Статус: Partial  
   - Следующий шаг: добавить официальный model-output MQuAKE full-sized/внешний comparison

37. AKEW-style raw-text pipeline  
   - Артефакт: `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py`  
   - Статус: Partial  
   - Следующий шаг: добавить первый tracked raw-text model-output run

38. Product-facing benchmark (ScEdit / UniEdit / KnowEdit / MLaKE)  
   - Артефакт: `results/other_benchmarks/`, `src/agim/eval/product_diagnostic.py`  
   - Статус: Not Started  
   - Следующий шаг: выбрать и закрыть один benchmark с trackable output chain

39. PatchService API lifecycle  
   - Артефакт: `src/agim/model/patch_service.py`, `src/agim/model/patch_governance.py`, `tests/test_patch_service.py`  
   - Статус: Partially Done  
   - Следующий шаг: сделать все endpoint-операции через публичный контракт и добавить E2E happy-path

40. Governance и public API package  
   - Артефакт: `src/agim/model/patch_governance.py`, adapter/docs readiness files  
   - Статус: Partially Done  
   - Следующий шаг: завершить approvals/signatures/audit trail/tenant isolation и выкатить адаптер как пакет

## Ключевые блокеры сейчас

1. `wal_memit` n=50 quality baseline по отдельному run отсутствует.  
2. `wal_rome` и `wal_memit` не закрывают полноценный n=50/n=1000 quality benchmark в сравнительном матричном формате с устойчивыми профилями.  
3. Нехватка внешних model-output ранов для RippleEdits / AKEW-style / ScEdit-like benchmark.  
4. Sequential/locality до текущего времени не доведены до product-grade, кроме диагностического coverage.

## Дополнительная техническая уборка (до релиза)

1. Убрать жёсткие `cuda:x` дефолты из user-facing CLI и документации, где это влияет на reproducibility (`src/agim/eval/*.py`, `docs/VERIFY_PATH_B_CURRENT.md`, `docs/VERIFY_PATH_B_LEGACY.md`, `docs/STEP_BY_STEP_GUIDE.md`).
2. Проверить, что `local_files_only` в публичных путях всегда явно управляется env/флагом, а не зашит как всегда-true.
3. Разделить legacy-продуктивные скрипты (CounterFact/local протокол) и Path B текущие потоки в отдельных папках с явными ярлыками `legacy`/`current`.
4. Ввести "переход на env-конфигы" для всех `device`/`easyedit-root` в тех местах, где сейчас есть `/path/to/...`, `--easyedit-root`, `AGIM_DEVICE` и т.п. с машинно-специфичным поведением.
5. Свести `TODO` в runtime WAL слоях (`src/agim/wal/v1/runtime{.py}` и `runtime_persistence.py`) в отдельный issue или backlog item, чтобы они не выглядели как незакрытые блокеры Path B.

## Блокер-блок исполнения (чёткий порядок)

1. Закрыть `wal_memit` baseline в официальном формате.
   - Выполнить команду из `VERIFY_PATH_B_CURRENT.md` для `--edit-backend wal_memit` на n=50.
   - Сохранить JSON и README-отчет по метрикам (baseline + failures).

2. Закрыть backend matrix для качества.
   - Выполнить сравнительный matrix run с `dual_row,wal_rome,wal_memit,side_slot`.
   - Убедиться, что есть per-backend артефакты и сравнительный md-отчёт.

3. Получить внешний consequence coverage с trackable outputs.
   - RippleEdits run с модельным output chain.
   - Полноценный MQuAKE run (не только диагностический scorer-pass).
   - Один raw-text benchmark run с external-style dataset.
   - Один продуктовый бенч (ScEdit/KnowEdit/UniEdit/MLaKE) с tracked output.

4. Sequential product-hardening.
   - Side-slot retention rerun для 10/50/100 edits.
   - Уровень локальности в отчёте с raw pre/post neighbor evidence.

5. Governance finalization.
   - Завершить публичный PatchService contract с `propose/simulate/run_canaries/approve/apply/rollback/inspect/diff`.
   - Подготовить release-gate proof package, где unsafe claims удалены из README/claims tables.

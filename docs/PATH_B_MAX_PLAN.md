# Path B Maximal Execution Plan

Источник: `sites/deep-research-report (5).md`
Актуализация: 2026-05-18

Execution evidence is tracked by
`docs/PATH_B_MAX_EXECUTION_CHECKLIST.md`,
`docs/PATH_B_MAX_EXECUTION_RUNBOOK.md`,
`docs/PATH_B_MAX_HARDGATE_QUEUE.md`,
`docs/PATH_B_MAX_PROMPT_TO_ARTIFACT_AUDIT_MATRIX.md`.
Use this plan for planning context; use the checklist and audit matrix for hard completion.
Current production status is now tracked in `PATH_B_MAX_MAX_PLAN.md`,
`PATH_B_MAX_COMPLETION_MATRIX.md`,
`PATH_B_MAX_EXECUTION_CHECKLIST.md`,
`PATH_B_MAX_EXECUTION_LEDGER.md`.
Use this file as historical planning context and milestone intent, not as the canonical closure ledger.

## Критерий завершения

Path B считается production-ready для beta только после выполнения блоков по очереди:

1. Источник истины и доки синхронизированы.
2. Official EasyEdit-метрики стабильны для `wal_rome`, `wal_memit`, `dual_row` в профилях single и sequential.
3. Появился durable PatchArtifact lifecycle с проверкой, apply, rollback и audit.
4. Есть внешние диагностики Beyond-CounterFact (минимум Ripple + MQuAKE или эквивалентный). 
5. Есть patch lifecycle API с governance-ограничениями.

Hard blocker gates в этой фазе:

- Gate 1: `wal_memit` official `n=50` baseline.
- Gate 2: full backend matrix (`dual_row`, `wal_rome`, `wal_memit`, `side_slot`) на одном датасете и профиле.
- Gate 3: внешние model-output chains: RippleEdits, MQuAKE, raw-text, один product-like benchmark.
- Gate 4: side-slot sequential retention на `n=10/50/100` с `seed` coverage.
- Gate 5: публичный PatchService/governance proof packet в release-ready формате.

## Карта требований в артефакты

1. Источник истины и прозрачные claims (`docs/PATH_B_COMPLETION_AUDIT.md`, `CURRENT_STATUS.md`, `CLAIMS_AND_EVIDENCE.md`).
2. Четкие пути сравнения Path A vs Path B (`README.md`, `docs/PATH_B_PRODUCTIZATION_PLAN.md`).
3. Официальный EasyEdit runner и metadata (`src/agim/eval/easyedit_official_runner.py`, `src/agim/eval/easyedit_run_metadata.py`, `results/easyedit_official/`).
4. Baseline-матрицы и сравнение backend (`src/agim/eval/easyedit_backend_matrix.py`, `results/easyedit_official/ablations/*_backend_matrix*.{json,md}`).
5. Внешняя диагностика (`src/agim/eval/mquake_output_runner.py`, `src/agim/eval/ripple_diagnostic.py`, `results/external_benchmark_runs/*`).
6. Lifecycle и governance (`src/agim/model`, future PatchService API module, audit payloads in `results/easyedit_official/*`).

## 1. База правды и операционный порядок

1. Развести source-of-truth: одна таблица статуса тестов и claims вместо расхождений в README/CURRENT_STATUS/CLAIMS.
  - Статус: Done
  - Сильный артефакт: `CURRENT_STATUS.md`, `CLAIMS_AND_EVIDENCE.md`, `docs/PATH_B_COMPLETION_AUDIT.md`

2. Явно выделить два SKU: Path A (runtime memory) и Path B (audited hotfixing). 
  - Статус: Done
  - Артефакт: `README.md` в `sites/agi_personal_memory`, `docs/PATH_B_PRODUCTIZATION_PLAN.md`

3. Вычистить legacy-контур: legacy local CounterFact-only потоки отделены от official-compatible. 
  - Статус: Done
  - Артефакт: `docs/PATH_B_MAX_PLAN.md`, `docs/PATH_B_COMPLETION_AUDIT.md`

4. Ввести единый формат метрик и методпрофилей (`single_ps`, `single_loc`, `seq_tuned` и т.п.) с `method_profile_id` во всех JSON/MD. 
  - Статус: Done
  - Артефакт: `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_official_runner.py`

5. Ввести `artifact_schema_version` и фиксируемый schema для всех отчетов. 
  - Статус: Done
  - Артефакт: `src/agim/eval/easyedit_run_metadata.py`

6. Убрать machine-specific defaults из публичных инструкций: `cuda:3`, абсолютные пути, `local-files-only` по умолчанию. 
  - Статус: Done
  - Артефакт: `docs/VERIFY_PATH_B_CURRENT.md`, `README.md`, `docs/VERIFY_PATH_B_LEGACY.md`, `src/agim/eval/easyedit_cli.py` defaults via `AGIM_DEVICE` / `AGIM_LOCAL_FILES_ONLY`

7. Запустить и зафиксировать официальный 50-запросный run `--edit-backend wal_memit` и отметить его как baseline для compatibility. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/current/random_50_seed_42_wal_memit.json`, `results/easyedit_official/current/random_50_seed_42_wal_memit.md`, `results/easyedit_official/current/random_50_seed_42_wal_memit.failures.json`

## 2. Engineering hardening и правка пайплайна

8. Сделать deterministic NT как отдельный режим: фиксированный выбор non-target с его snapshot в artifact. 
  - Статус: Partially Done
  - Артефакт: `src/agim/eval/easyedit_metrics.py`, `src/agim/eval/easyedit_eval_loop.py`

9. Сделать мониторинг patch growth и нормы (row norm growth, cumulative edit norm, reuse общих rows). 
  - Статус: Partially Done
  - Артефакт: `results/easyedit_official/*` summary fields + метрики в exporter

10. Добавить норм-ограничения и ранний stop для последовательных edits (ENCORE-подобный control). 
  - Статус: Not Started
  - Артефакт: `src/agim/model/wal_dual_editor.py` и backend policy wrapper

11. Убрать глобальные EOS/anti defaults из persistent profile; перенести в локальные/decoding-time альтернативы. 
  - Статус: In Progress / Open
  - Артефакт: `src/agim/model/wal_dual_editor.py`, `src/agim/model/wal_dual_helpers.py`

12. Исправить `target_token_mode="both"` выбор primary target sequence к continuation-aligned alignment, а не “первой попавшейся”. 
  - Статус: Done
  - Артефакт: `src/agim/model/wal_dual_editor.py`, `src/agim/eval/easyedit_eval_loop.py`

13. Ввести конфликт-детектор пересечений: EOS/control rows, subject/subword overlap, target token overlap, relation conflict. 
  - Статус: In Progress
  - Артефакт: current `conflict_summary()` risk flags in `src/agim/model`, plus planned hard checks in runner/apply path
  - Артефакт: новая утилита в `src/agim/model`

14. Перейти от in-place редактирования к runtime sparse overlay для `lm_head`/`embed_tokens` по возможности. 
  - Статус: Partially Done (design + artifacts)
  - Артефакт: новый overlay слой в model API

15. Развести mutable state на tenant/namespace: единый editor не должен быть гонкой между workflows. 
  - Статус: Partially Done
  - Артефакт: `--state-namespace` + `WALDualMutableState` now partition history by namespace
  - Артефакт: `src/agim/model` + runner/session glue

16. Реализовать patch conflict/compatibility validation для `base_model_digest` и `atoms_digest` до apply. 
  - Статус: In Progress
  - Артефакт: `src/agim/eval/easyedit_run_metadata.py`, future patch service layer

17. Поднять fail-only pipeline по метрикам отдельно (`tf`, `ctx_gen`, `prob`, `vanilla_gen`) и не смешивать шумные семьи по умолчанию. 
  - Статус: Done
  - Артефакт: `src/agim/eval/easyedit_cli.py`, `src/agim/eval/easyedit_run_metadata.py`, `src/agim/eval/easyedit_failures.py`, `src/agim/eval/easyedit_official_runner.py`

## 3. Алгоритмические улучшения редактора

18. Ввести архитектуру `WALRomeEditor` для internal-layer FFN-патчинга. 
  - Статус: Partially Done
  - Артефакт: `src/agim/model/wal_rome_editor.py`, smoke tests

19. Подготовить `WALMemitBatchEditor` как consolidation backend с жизненным циклом (serialize/apply/inspect). 
  - Статус: Done (infrastructure)
  - Артефакт: `src/agim/model/wal_memit_editor.py`

20. Протестировать компонентные ablations: `lm_head-only`, `embed-only`, `dual`, `dual-no-eos`, `dual-no-anti`. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/ablations/component_ablation_report_2026-05-18.md` + JSON с разрезами

21. Проверить “exact additive update without WAL re-encoding” как контрольный baseline. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/ablations/exact_additive_report_2026-05-18.md`

22. Переработать positive-key конструкцию на constrained K_pos / K_neg решение, чтобы tradeoff PS@All–locality меньше разъезжался. 
  - Статус: Not Started
  - Артефакт: `src/agim/model/wal_dual_editor.py`

23. Построить relation-specific protected banks и worst-relation failure pool. 
  - Статус: Partially Done
  - Артефакт: existing relation data + locality breakdown extensions

24. Прогнать `random_200` и `random_1000(seeded)` для оценки robustness между single/random/first-1000. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/current/random_200_report_2026-05-18.md`, `results/easyedit_official/current/random_1000_report_2026-05-18.md`

25. Сделать `target_token_mode = standalone/contextual/both` matrix на одном set фактов и сравнить TF/vanilla/contextual метрики. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/ablations/token_mode_matrix_report_2026-05-18.md`

26. Провести full ablation matrix с `wal_memit`, `wal_rome`, `dual_row`, side_slot и profile-флагами interference controls. 
  - Статус: In Progress
  - Артефакт: `src/agim/eval/easyedit_backend_matrix.py`, `results/easyedit_official/ablations/*`

27. Добавить side-slot/route simulation как отдельный режим без влияния на base model в-place. 
  - Статус: In Progress (research/first seed)
  - Артефакт: side-slot experiments, future runner mode

## 4. Внешняя валидация и продуктовый контур

28. Зафиксировать RippleEdits benchmark как обязательный diagnostic 2-уровня. 
  - Статус: In Progress
  - Артефакт: `agim.eval.ripple_diagnostic` + post-hoc adapter outputs

29. Добавить MQuAKE benchmark как обязательный 2-уровень для consequences/multi-hop. 
  - Статус: In Progress (частично подготовлены артефакты)
  - Артефакт: external benchmark adapter + new run payloads

30. Добавить AKEW/KnowEdit/ScEdit-like pipeline из raw text edits. 
  - Статус: Partially Done
  - Артефакт: `src/agim/eval/raw_text_edit_pipeline.py`, `src/agim/eval/raw_text_scoring.py`, `results/other_benchmarks/` adapters
  - Недостаток: отсутствует tracked model-output run под raw-text датасетами

31. Повторный запуск seq retention на 10/50/100 edits для side-slot режима с сравнением in-place tuned baseline. 
  - Статус: Done
  - Артефакт: `results/easyedit_official/sequential/side_slot_random_10_seed_{42,43,44}_seq.json`, `results/easyedit_official/sequential/side_slot_random_50_seed_{42,43,44}_seq.json`, `results/easyedit_official/sequential/side_slot_random_100_seed_{42,43,44}_seq.json`

32. Добавить explicit proof package для safe/unsafe claim split (what we can prove, what we can't). 
  - Статус: In Progress
  - Артефакт: `docs/` claim tables + benchmark evidence bundles

33. Подготовить PatchService API: `propose`, `simulate`, `run_canaries`, `approve`, `apply`, `rollback`, `inspect`, `diff`. 
  - Статус: Partially Done
  - Артефакт: service stub + approval flow + API docs

34. Реализовать governance: approvals, signatures, immutable audit trail, tenant isolation. 
  - Статус: In Progress
  - Артефакт: approvals/signatures/audit trail exist; tenant isolation/enforcement still active path
  - Артефакт: storage + policy + audit logs

35. Сформировать public easyedit adapter package/protocol с reproducible defaults. 
  - Статус: Partially Done
  - Артефакт: packaging + docs in repo root export

36. Подвести полный report по локальности: pre/post neighbor texts, diff severity, relation-conditioned locality budgets. 
  - Статус: Not Started
  - Артефакт: locality diagnostics extension

37. Разделить legacy и current benchmark artifacts по директориям (`legacy`/`official`) чтобы не смешивать evidence. 
  - Статус: In Progress
  - Артефакт: `results/easyedit_legacy`, `results/easyedit_official`

38. Поддерживать single profile и sequential profile reproducibility bundles одной командой. 
  - Статус: In Progress
  - Артефакт: `docs/VERIFY_PATH_B_CURRENT.md`

39. Сформировать финальный matrix-гейт для публикации: official baseline / matrix / external 3-го уровня. 
  - Статус: Not Started
  - Артефакт: release gate document + sign-off checklist

40. Подготовить roadmap path: `Path B v1` (sparse hotfix), `v2` (internal layer factual), `v3` (routed side-slot) с measurable milestones. 
  - Статус: In Progress
  - Артефакт: `docs/PATH_B_PRODUCTIZATION_PLAN.md`

## Immediate next action (первые 2 дня)

1. Закрыть blocker 1: `wal_memit` official `n=50` baseline + полный JSON/MD.
2. Закрыть blocker 2: full backend quality matrix с `dual_row`, `wal_rome`, `wal_memit`, `side_slot`.
3. Закрыть blocker 3: внешний model-output chain для RippleEdits/MQuAKE, plus один raw-text and один product-like benchmark.
4. Закрыть blocker 4: side-slot sequential retention для `n=10/50/100` и seeds `42/43/44`.
5. Закрыть blocker 5: release-gate proof packet для PatchService + governance (docs + claims + API surface).

## Что закрывает текущий gap

1. Gate 1 и Gate 2 — обязательный запуск `scripts/run_path_b_max_bootstrap.sh 1 2`.
2. Gate 4 — запуск retention sweep `n=10/50/100`, seeds `42/43/44` в одной очереди.
3. Gate 3 — внешний chain в порядке `ripple`, `mquake`, `raw_text`, `product`.
4. Gate 5 — собрать публичный claims-safe proof packet для `propose/simulate/run_canaries/approve/apply/rollback`.
5. Закрыть hard requirements из Max-матрицы: `23`, `24`, `25`, `26`, `30`, `33`, `39`, `40` и Gate 5 (production external immutable proof).

# Path B Maximal Execution Plan

Источник: `sites/deep-research-report (5).md`
Дата: 2026-05-18

## 1–13: База и уже закрытое

1. Развести статус-артефакты между Path A / Path B — **Done**.  
2. Обновить README/Status в едином формате — **Done**.  
3. Отделить legacy local-пайплайн от оф. EasyEdit — **Done**.  
4. Добавить официальную трассу `easyedit_official_runner` — **Done**.  
5. Трекабельные single-edit official runs на Llama 3.1-8B-Instruct — **Done**.  
6. random/first/seed-проверки (50/100/1000) — **Done**.  
7. Добавить sequential проверки с retention и локальными профилями — **Done**.  
8. Backend matrix direct backends (dual_row / wal_rome / side_slot) — **Done** (wal_memit direct path added).
9. side-slot сравнения и детерминированные run reports — **Done**.  
10. Введение `artifact_schema_version` и trace metadata — **Done**.  
11. `PATCH`-артефакты и rollback на уровне редактора — **Done**.  
12. `WALRomeEditor` integration и smoke tests — **Done**.  
13. `WALDualLayerEditor` production-like профили (rewrite/locality trade-off) — **Done**.

## 14–20: Механики и метрики

14. Локализация MQuAKE/CF adapter и scorer pipeline — **Done**.  
15. Запуск model-output for first-50 MQuAKE — **Done** (not leaderboard claim).  
16. Набор сырьевых внешних метрик (external_benchmark_runs) — **Done**.  
17. Реализация/интеграция `raw_text` диагностики с scorer — **Done**.  
18. Deterministic NT-маркировка и фиксированные выборки — **Partially Done** (id-снапшоты есть, но не как отдельный режим артефакта).  
19. Нормы правок и мониторинг patch growth — **Partially Done**.  
20. Удалить/заменить анти-репитишн global-контролы как default — **In Progress / open**.

## 21–27: Продуктовый рост

21. Сделать альтернативу global anti-repeat с меньшим вредом locality — **Blocked**.  
22. target-token align для `target_token_mode="both"` — **Not Started**.  
23. Constraint-формулировка positive prompts (K_pos / K_neg) — **Not Started**.  
24. Конструкция `relation_protected` банков и worst-case failures — **Partially Done**.  
25. Norm budget / early-stop для накопления последовательных edits — **Not Started**.  
26. Patch conflict detector — **Not Started**.  
27. Runtime sparse overlay для `lm_head` + `embed_tokens` — **Partially Done** via artifacts/design, not production hook.

## 28–40: Продуктовая архитектура

28. Side-slot routing в прод-контуре с tenant namespace — **In Progress (research only)**.  
29. Relation sharding для изоляции patch slots — **Not Started**.  
30. Повторный sequential test для side-slot и retention в 10/50/100 edits — **Not Started**.  
31. Внедрить `WALRome`/`WALMemit` внутри единой архитектурной матрицы — **In Progress (rome direct, memit direct backend подключен)**.
32. Создать `WALMemit` direct режим (не только offline consolidation) — **Done (compat wrapper + CLI/matrix integration)**.
33. `WALMemitBatchEditor` как consolidation API с lifecycle — **Done (infrastructure only)**.  
34. RippleEdits benchmark как обязательный external diagnostic run — **Not Started**.  
35. AKEW/KnowEdit/ScEdit/Open-domain benchmark runs — **Not Started**.  
36. Product-facing PatchService endpoints: propose/simulate/run/rollback/inspect — **Partially Done**.  
37. Governance layer (approval/signature/search/filter/ACL/history) — **Partially Done**.  
38. Unified API + audit export для external consumer — **Not Started**.  
39. Публичный easyedit adapter protocol (package) с воспроизводимыми defaults — **Partially Done**.  
40. Финальный proof package с четким `safe/unsafe` разделением claim-ов — **In Progress**.

## Ограничения и источник правды

- `wal_memit` теперь включен в официальный `easyedit_official_runner` path через совместимый wrapper-редактор; требуется подтверждение качества на полном run 50.
- Ripple/AKEW/ScEdit product-facing бенчмарки пока не имеют официальных model-output артефактов в этом репозитории.
- Текущие метрики Path B полезны для research и hotfix-профиля, но не закрывают lifelong continual editing.

## Следующий приоритетный шаг

1. Запустить официальный 50-запросный EasyEdit run по `wal_memit` (необходимый контрольный артефакт).
2. Добавить минимальный `results/external_benchmark_runs/ripple/*` first-50 run после подготовки dataset adapter.
3. Зафиксировать deterministic NT sample manifest в artifact payload как отдельный проверяемый блок.

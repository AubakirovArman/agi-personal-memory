# AGI Personal Memory — Накопительный верифицированный субстрат памяти для языковых моделей

[English](../../README.md) | [Русский](README.md) | [中文](../zh/README.md) | [Қазақша](../kk/README.md)

## Что это

Современные AI-системы **не имеют памяти**. Каждый диалог начинается с нуля. Нельзя научить модель одному факту так, чтобы он сохранился. Файнтюнинг — батчевый и деструктивный. RAG — извлечение, а не обучение.

**AGI Personal Memory** исследует эту проблему через два отдельных пути:
retrieval-память для сохранения фактов и WAL-backed редактирование весов для
экспериментов с параметрической памятью.

## Быстрый старт

```bash
pip install -e .
agim teach "Париж — столица Франции"
agim ask "Какая столица Франции?"
agim correct "Нет, Наполеон родился в 1769 году, а не в 1768"
agim history
agim stats
agim webui --port 8720
```

## Чем отличается

| | RAG | Файнтюнинг | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| Изменяет модель | Нет | Да (деструктивно) | Да (аддитивно) | **Экспериментально** |
| Инкрементально | Да | Нет | Нет | **Да** |
| Обратимо | Да | Нет | Частично | **Да (откат любого коммита)** |
| Аудируемо | Нет | Нет | Частично | **Да (полный JSONL след)** |
| Non-target diff | N/A | ~25% | Средний | **0% в WAL diagnostics** |

## Как работает

```
Ввод пользователя → IntentRouter (LLM + regex) → Memory Extractor → MemoryCandidate
                                                                      ↓
                                                                 VERIFY (12 принципов)
                                                                      ↓
                                                            Memory Compiler (5 уровней)
                                                                      ↓
                                                  ┌─ WAL recipe → правка весов модели
                                                  ├─ Retrieval   → key-value хранилище
                                                  ├─ LoRA        → ортогональные адаптеры
                                                  ├─ Refusal     → политики отказа
                                                  └─ Reject      → заблокировано
                                                                      ↓
                                                                 COMMIT + Audit Trail
```

## Интерфейсы

| Интерфейс | Команда | Описание |
|-----------|---------|----------|
| CLI | `agim teach/ask/correct/forget/history/stats` | Командная строка |
| Shell | `agim shell` | Интерактивный режим |
| REST API | `agim api --port 8720` | 11 эндпоинтов |
| Веб-дашборд | `agim webui --port 8720` | JS-дашборд, 5 вкладок |
| MCP | `MCPServer` | Model Context Protocol |
| A2A | `A2AServer` | Agent-to-Agent протокол |
| GraphQL | `GraphQLResolver` | GraphQL интерфейс |
| Экспорт | `agim export memories.json` | Выгрузка памяти |
| Импорт | `agim import memories.json` | Загрузка памяти |

## Текущий статус

- Реальные EasyEdit-compatible результаты: см. `../../BENCHMARK.md`
- Исторические локальные CounterFact результаты отделены в `../../results/local_protocol/`
- Полный локальный pytest: `126 passed, 13 skipped`
- Skipped тесты: Gemma E2E, когда текущий Transformers не поддерживает `gemma4`

## Лицензия

MIT

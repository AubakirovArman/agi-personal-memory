# AGI Personal Memory — Тілдік модельдер үшін жинақталған тексерілген жад субстраты

[English](../../README.md) | [Русский](../ru/README.md) | [中文](../zh/README.md) | [Қазақша](README.md)

## Бұл не

Қазіргі AI жүйелерінің **жады жоқ**. Әр әңгіме нөлден басталады. Сіз модельге бір фактіні үйрете алмайсыз және ол сақталмайды. Fine-tuning — пакеттік және деструктивті. RAG — бұл іздеу, оқыту емес.

**AGI Personal Memory** бұл мәселені екі бөлек жолмен зерттейді:
фактілерді сақтау үшін retrieval memory және зерттеу эксперименттері үшін
WAL-backed салмақ өңдеу.

## Жылдам бастау

```bash
pip install -e .
agim teach "Париж — Францияның астанасы"
agim ask "Францияның астанасы қайсы?"
agim correct "Жоқ, Наполеон 1769 жылы туған, 1768 емес"
agim history
agim stats
agim webui --port 8720
```

## Неліктен басқаша

| | RAG | Fine-tuning | LoRA | AGI Personal Memory |
|---|---|---|---|---|
| Модельді өзгертеді | Жоқ | Иә (деструктивті) | Иә (аддитивті) | **Эксперименттік** |
| Инкременталды | Иә | Жоқ | Жоқ | **Иә** |
| Қайтымды | Иә | Жоқ | Ішінара | **Иә (кез келген коммитті қайтару)** |
| Аудиттелетін | Жоқ | Жоқ | Ішінара | **Иә (толық JSONL ізі)** |
| Мақсатты емес айырма | N/A | ~25% | Орташа | **WAL diagnostics-та 0%** |

## Қалай жұмыс істейді

```
Пайдаланушы енгізуі → IntentRouter (LLM + regex) → Memory Extractor → MemoryCandidate
                                                                        ↓
                                                                   VERIFY (12 қағида)
                                                                        ↓
                                                              Memory Compiler (5 деңгей)
                                                                        ↓
                                                    ┌─ WAL recipe → модель салмағын өңдеу
                                                    ├─ Retrieval   → key-value қоймасы
                                                    ├─ LoRA        → ортогоналды адаптерлер
                                                    └─ Reject      → блокталған
                                                                        ↓
                                                                   COMMIT + Audit Trail
```

## Интерфейстер

| Интерфейс | Команда | Сипаттамасы |
|-----------|---------|-------------|
| CLI | `agim teach/ask/correct/forget` | Командалық жол |
| Shell | `agim shell` | Интерактивті режим |
| REST API | `agim api --port 8720` | 11 endpoints |
| Веб-дашборд | `agim webui --port 8720` | JS дашборды, 5 қойынды |
| Экспорт | `agim export memories.json` | Жадты экспорттау |
| Импорт | `agim import memories.json` | Жадты импорттау |

## Ағымдағы статус

- Ағымдағы EasyEdit-compatible нәтижелер: `../../BENCHMARK.md`
- Тарихи локал CounterFact нәтижелері: `../../results/local_protocol/`
- Толық локал pytest: `126 passed, 13 skipped`
- Skipped тесттер: ағымдағы Transformers `gemma4` қолдамаса, Gemma E2E

## Лицензия

MIT

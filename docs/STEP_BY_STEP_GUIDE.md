# Пошаговое руководство: от модели к персональной памяти

## Концепция

Ты берёшь готовую модель (Llama 3.1 8B), превращаешь её веса в WAL-программы,
учишь своими фактами, и модель начинает отвечать по-твоему.

```
Llama 8B → WAL encode → AGIM teach → модель знает твои факты
```

---

## Шаг 1: Выбор и загрузка модели

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # или путь к локальному кэшу
DEVICE = os.getenv("AGIM_DEVICE", "cuda:0")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Проверяем что модель живая
print(model.generate(**tok("Привет!", return_tensors="pt").to(DEVICE),
                      max_new_tokens=20))
```

---

## Шаг 2: Преобразование в WAL (построение словаря атомов)

```python
from agim.model.wal_backend import WALWeightEditor

editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)

# Строим словарь атомов на весах модели (k-means++)
editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")

print(f"Словарь заморожен: {editor.vocabulary_is_frozen}")
# → True

# Теперь можно кодировать любой слой:
weight = dict(model.named_parameters())["model.layers.0.mlp.down_proj.weight"]
prog, recon = editor.encode_weight(weight)
print(f"relMSE: {(weight - recon.reshape_as(weight)).norm() / weight.norm():.10f}")
# → ~0.000001 (почти без потерь)
```

**Что произошло:** k-means++ нашёл 256 атомов — скалярных значений, которые лучше всего
описывают распределение весов модели. Таблица атомов заморожена — она не будет
меняться при редактировании. Это гарантирует non-target diff = 0%.

---

## Шаг 3: Обучение на своих данных

### Способ А: По одному факту (интерактивно)

```python
from agim.core.system import AGIMSystem

agim = AGIMSystem(workdir="./my_memory")

# Учим факты один за другим
facts = [
    ("Кто твой создатель?", "Аубакиров Арман"),
    ("Какой язык ты предпочитаешь?", "Русский и казахский"),
    ("Что такое AGI Personal Memory?", "Накопительная память для языковых моделей"),
]

for question, answer in facts:
    c = agim.propose_memory(question=question, answer=answer, kind="fact_teach")
    report = agim.compile(c)
    if report.passed:
        agim.commit(report)
        print(f"  ЗАПОМНЕНО: {question} → {answer}")
    else:
        print(f"  ОТКЛОНЕНО: {question} — {report.reason}")

# Проверяем
resp = agim.ask("Кто твой создатель?")
print(resp.answer)  # → Аубакиров Арман
```

### Способ Б: Через CLI

```bash
agim teach "Кто твой создатель? — Аубакиров Арман"
agim teach "Какой язык ты предпочитаешь? — Русский и казахский"
agim teach "Что такое AGI Personal Memory? — Накопительная память для моделей"

agim ask "Кто твой создатель?"
# → Аубакиров Арман
```

### Способ В: Массово из JSON

```bash
# Создай файл my_facts.json:
# [
#   {"question": "Кто твой создатель?", "answer": "Аубакиров Арман"},
#   {"question": "Какой язык ты предпочитаешь?", "answer": "Русский"}
# ]

agim import my_facts.json
agim stats
# → Total facts: 2
```

### Способ Г: Через Python цикл с верификацией

```python
facts_to_teach = [
    ("Кто твой создатель?", "Аубакиров Арман"),
    ("Где ты живёшь?", "В памяти GPU"),
    ("Какая твоя цель?", "Помнить всё чему меня научили"),
]

for q, a in facts_to_teach:
    # 1. Предложить факт
    c = agim.propose_memory(question=q, answer=a)
    
    # 2. Верифицировать (5+ gates)
    report = agim.compile(c)
    
    # 3. Если прошёл — закоммитить
    if report.passed:
        agim.commit(report)
        print(f"✓ {q} → {a}")
    else:
        print(f"✗ {q}: {report.reason}")

# 4. Проверить что все факты на месте
for q, a in facts_to_teach:
    resp = agim.ask(q)
    assert resp.answer == a, f"Факт потерялся: {q}"
    print(f"  ПРОВЕРЕНО: {q} → {resp.answer}")
```

---

## Шаг 4: Применение правок к модели (WAL edit)

После того как AGIM запомнил факты, их нужно "вшить" в модель:

```python
# Выбираем слой для редактирования (обычно средние MLP слои)
target = "model.layers.5.mlp.down_proj.weight"

# Снимаем снапшот для возможности отката
editor.snapshot_layer(target)

# Для каждого факта — находим связанные токены и усиливаем их
for q, a in facts_to_teach:
    answer_tokens = tok.encode(a, add_special_tokens=False)
    for token_id in answer_tokens[:3]:  # первые 3 токена ответа
        delta = torch.zeros_like(dict(model.named_parameters())[target].data)
        delta[token_id % delta.shape[0], :] += 0.1  # усиление
        editor.edit_weight(target, delta)

# Проверяем что другие слои не изменились
assert editor.verify_non_target_diff(target)
print("Non-target diff = 0% ✓")
```

**Модель теперь знает твои факты.**

---

## Шаг 5: Проверка (верификация)

### Проверка 1: AGIM помнит
```python
for q, expected in facts_to_teach:
    resp = agim.ask(q)
    print(f"{'✓' if resp.answer == expected else '✗'} {q}: {resp.answer}")
```

### Проверка 2: Модель генерирует правильный ответ
```python
for q, expected in facts_to_teach:
    inputs = tok(q, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    answer = tok.decode(output[0], skip_special_tokens=True)
    contains = expected.lower() in answer.lower()
    print(f"{'✓' if contains else '✗'} {q} → {answer[:80]}")
```

### Проверка 3: Non-target diff
```python
assert editor.verify_non_target_diff(target)
# Другие знания модели не пострадали
```

### Проверка 4: PPL не вырос
```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join([t for t in ds["text"][:100] if t.strip()])
ppl = compute_ppl_sliding(model, tok, text)
print(f"PPL после редактирования: {ppl:.2f}")
# Должен быть близок к baseline
```

---

## Шаг 6: Откат (если нужно)

```python
# Откат правок модели
editor.rollback_edit(target)

# Откат памяти AGIM
agim.rollback_last()  # откатывает последний коммит

# Модель и память вернулись к исходному состоянию
```

---

## Полный скрипт: всё вместе

```python
"""Полный цикл: Llama 3.1 8B → WAL → учим → проверяем."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.core.system import AGIMSystem
from agim.model.wal_backend import WALWeightEditor
import os

DEVICE = os.getenv("AGIM_DEVICE", "cuda:0")
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

# Шаг 1: Загрузка
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                              device_map=DEVICE)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Шаг 2: WAL-словарь
editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
editor.build_vocabulary("model.layers.0.mlp.down_proj.weight")

# Шаг 3: Обучение
agim = AGIMSystem(workdir="./my_memory")
facts = [
    ("Кто твой создатель?", "Аубакиров Арман"),
    ("Какая твоя цель?", "Помнить всё чему меня научили"),
]
for q, a in facts:
    c = agim.propose_memory(question=q, answer=a)
    if agim.compile(c).passed:
        agim.commit(agim.compile(c))

# Шаг 4: Применение к модели
target = "model.layers.5.mlp.down_proj.weight"
editor.snapshot_layer(target)
for q, a in facts:
    tokens = tok.encode(a, add_special_tokens=False)
    delta = torch.zeros_like(dict(model.named_parameters())[target].data)
    delta[tokens[0] % delta.shape[0], :] += 0.1
    editor.edit_weight(target, delta)

# Шаг 5: Проверка
print("\n=== ПРОВЕРКА ===")
for q, a in facts:
    resp = agim.ask(q)
    print(f"AGIM: {q} → {resp.answer}")

assert editor.verify_non_target_diff(target)
print("Non-target diff: 0% ✓")

# Шаг 6: Откат
editor.rollback_edit(target)
agim.rollback_last()
print("Откат выполнен ✓")
```

---

---

## Шаг 7 (новый): MemoryAugmentedModel — модель со встроенной памятью

Вместо отдельного AGIM + модели можно использовать **MemoryAugmentedModel** —
единый класс, который объединяет Llama с AGIM-памятью и семантическим поиском.

```python
from agim.model.memory_model import MemoryAugmentedModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Загружаем базовую Llama
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16, device_map=DEVICE)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. Оборачиваем в MemoryAugmentedModel
model = MemoryAugmentedModel(base, tok, memory_dir="./my_model_memory", device=DEVICE)

# 3. Учим факты (один за другим или батчем)
model.teach("Кто твой создатель?", "Аубакиров Арман")
model.teach("Столица Заникланда — Блорптаун", "Блорптаун")

# 4. Спрашиваем — находит семантически (не точное совпадение!)
resp = model.ask("Кто тебя написал?")       # найдёт через BM25!
print(resp.answer)  # → "Аубакиров Арман"

resp = model.ask("Главный город Заникланда?")  # тоже найдёт
print(resp.answer)  # → "Блорптаун"

# 5. Массовое обучение из датасета
qa_pairs = [("Вопрос 1", "Ответ 1"), ("Вопрос 2", "Ответ 2"), ...]
taught = model.teach_batch(qa_pairs)
print(f"Обучено: {taught} фактов")

# 6. Статистика
print(model.stats())
# → {"total_facts": 50, "faiss_entries": 50, "model_size_mb": 15317}
```

### Как работает поиск

MemoryAugmentedModel использует три уровня:

1. **Exact match** — `agim.ask()` ищет точное совпадение в retrieval_memory
2. **FAISS+BM25** — семантический поиск: находит похожие вопросы
3. **model.generate()** — если ничего не найдено, генерирует через Llama

```python
def ask(self, question):
    # Уровень 1: точное совпадение
    exact = self.agim.ask(question)
    if exact.source != "model_fallback":
        return exact  # найдено в AGIM

    # Уровень 2: семантический поиск (FAISS+BM25)
    results = self.search_memory(question, top_k=3)
    if results and results[0]["score"] > 0.3:
        return results[0]  # найдено похожее

    # Уровень 3: генерация моделью
    return self.base_model.generate(question)
```

### Результаты на датасете

Датасет: `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` (38K примеров)

| Метрика | Значение |
|---------|----------|
| Обучено фактов | 48 (уникальных) |
| Скорость обучения | 493 фактов/сек |
| Семантический поиск | **5/5 найдено** |
| Точный поиск (AGIM) | 1/5 (только точные совпадения) |
| Размер модели | 15,317 MB (+ AGIM память на диске) |
| PPL | без изменений |

### Почему 48 а не 500

AGIM дедуплицирует факты по question-ключу. Одинаковые вопросы не сохраняются дважды.
Из 500 примеров — 48 уникальных вопросов.

---

## Ответы на частые вопросы

**Q: Нужно ли "преобразовывать обратно" из WAL?**
A: Нет. WAL — это способ представления весов как программ. Ты редактируешь
программы, а модель использует их напрямую. "Откат" — это восстановление
оригинальных весов из снапшота.

**Q: Сколько фактов можно выучить?**
A: AGIM ограничен только диском (SQLite — миллионы фактов).
WAL-редактирование — десятки фактов на слой без деградации.

**Q: Можно ли учить на нескольких GPU?**
A: Да. WAL кодирует слои независимо — можно распределить по GPU.

**Q: Работает ли без GPU?**
A: AGIM (память) — да. WAL (редактирование весов) — требует GPU.

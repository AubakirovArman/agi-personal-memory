# Как проверить отредактированную модель

> `agim ask` проверяет память AGIM, `model.generate()` проверяет конкретное
> поведение модели после правки весов, а агрегированные метрики по
> CounterFact/EasyEdit смотри в `BENCHMARK.md` и `CURRENT_STATUS.md`.

## Три уровня верификации

### Уровень 1: AGIM memory check (CLI)

После `agim teach` факт в памяти. Проверяешь через `agim ask`:

```bash
agim teach "Столица Заникланда — Блорптаун"
agim ask "Какая столица Заникланда?"
# → Блорптаун
```

**Что доказывает:** факт сохранён в AGIM-хранилище.  
**Что НЕ доказывает:** модель генерирует этот факт при `model.generate()`.

### Уровень 2: Non-target diff (Python)

```python
from agim.model.wal_backend import WALWeightEditor

editor = WALWeightEditor(model, K=256, lmax=12, device="cuda:2")
editor.build_vocabulary("model.language_model.layers.0.mlp.down_proj.weight")

target = "model.language_model.layers.0.mlp.down_proj.weight"
neighbor = "model.language_model.layers.1.mlp.down_proj.weight"

editor.snapshot_layer(target)
editor.snapshot_layer(neighbor)

delta = torch.randn_like(param) * 0.001
editor.edit_weight(target, delta)

# Проверка
assert editor.verify_non_target_diff(target)
# True → слой 1 не изменился, только слой 0

editor.rollback_edit(target)
```

**Что доказывает:** frozen vocabulary работает — правка локальна.  
**Что НЕ доказывает:** модель генерирует правильный ответ.

### Уровень 3: model.generate() — генерация (Python, GPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from agim.model.rome_editor import ROMEEditor

GEMMA = "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf..."

# 1. Загружаем
model = AutoModelForCausalLM.from_pretrained(GEMMA, dtype=torch.bfloat16,
                                              device_map="cuda:2")
tok = AutoTokenizer.from_pretrained(GEMMA)

# 2. Базовый ответ — модель НЕ знает вымышленный факт
before = model.generate(**tok("Столица Заникланда:", return_tensors="pt").to("cuda:2"),
                        max_new_tokens=20)
print("ДО:", tok.decode(before[0]))
# → "Столица Заникланда: Заникланд - вымышленная страна..." (не "Блорптаун")

# 3. Редактируем через ROME
editor = ROMEEditor(model, tok, device="cuda:2")
editor.apply_edit(subject="Заникланд", target="Блорптаун",
                  relation="столица", target_layer=5)

# 4. Проверяем после редактирования
after = model.generate(**tok("Столица Заникланда:", return_tensors="pt").to("cuda:2"),
                       max_new_tokens=20)
print("ПОСЛЕ:", tok.decode(after[0]))
# → "Столица Заникланда: Блорптаун" ← МОДЕЛЬ ЗНАЕТ!

# 5. Откатываем
editor.rollback()
restored = model.generate(**tok("Столица Заникланда:", return_tensors="pt").to("cuda:2"),
                           max_new_tokens=20)
print("ПОСЛЕ ОТКАТА:", tok.decode(restored[0]))
# → снова не знает
```

**Что доказывает:** модель реально изменилась и генерирует новый факт на этой
конкретной проверке. Это сильная end-to-end проверка одного edit path, но она
не заменяет агрегированные EasyEdit/CounterFact метрики по rewrite, rephrase,
locality и sequential editing.

### Уровень 4: Поведенческие тесты

```python
from agim.verify.contracts import BehaviouralContract

contract = BehaviouralContract(
    name="knows_capital",
    kind="must_answer",
    question="Столица Заникланда?",
    expected_answer="Блорптаун",
    check_type="contains"
)

# Запускаем после каждого коммита
assert contract.verify(model.generate("Столица Заникланда?"))
```

Автоматически проверяет что модель не "забыла" факт после новых правок.

## Быстрая проверка через CLI

```bash
# 1. Учим
agim teach "Столица Заникланда — Блорптаун"

# 2. Проверяем AGIM
agim ask "Какая столица Заникланда?"  # Блорптаун

# 3. Проверяем историю
agim history  # видно что факт закоммичен

# 4. Экспортируем
agim export check.json

# 5. В Python проверяем модель
python -c "
from agim.core.system import AGIMSystem
agim = AGIMSystem()
resp = agim.ask('Какая столица Заникланда?')
print(f'AGIM: {resp.answer}')  # Блорптаун
print(f'Source: {resp.source}')  # wal_recipe
"
```

## Итог: что реально доказывает что модель изменилась

| Проверка | Доказывает |
|----------|-----------|
| `agim ask` | Факт в памяти AGIM |
| `verify_non_target_diff()` | Правка локальна |
| `model.generate()` до/после | **Модель реально изменилась** ← вот это главное |
| Behavioural contract | Факт не ломается от новых правок |

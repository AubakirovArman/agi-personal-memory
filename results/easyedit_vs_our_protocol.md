# EasyEdit vs AGIM: сравнение протоколов CounterFact

## Официальный EasyEdit (как все проходят)

```python
# Из easyeditor/evaluate/evaluate_utils.py
target_new_tokens = tok.encode(target_new)
gen_token = model.generate(
    max_new_tokens=len(target_new_tokens),  # генерирует РОВНО столько токенов
    do_sample=False,                         # greedy
)
# Token-level exact match
score = np.mean(np.equal(target_new_tokens, generated_tokens))
```

**Метрика: token-level exact match.** Если target = "Rome" = ["R", "ome"], модель должна сгенерировать ["R", "ome"] и БОЛЬШЕ НИЧЕГО.

## Наш AGIM (как мы проходим)

```python
gen_token = model.generate(max_new_tokens=10)  # генерирует до 10 токенов
answer = tok.decode(gen_token)
# Substring match
score = target_new.lower() in answer.lower()
```

**Метрика: substring match.** Если target = "Rome", "Rome is a city..." — OK. "RomeRomeRome..." — тоже OK (раньше).

## Ключевые различия

| Параметр | EasyEdit | AGIM | Влияние |
|----------|----------|------|---------|
| max_new_tokens | len(target) | 10 | Они stricter |
| Метрика | token exact match | substring match | Они stricter |
| Модель | Llama 3 8B | Llama 3.1 8B Instruct | Мы на Instruct |
| Редактирование | FFN down_proj | lm_head (WAL) | Они глубже |
| Covariance | 100K Wikipedia | Нет | У них C⁻¹ |
| v* optimization | 25 grad steps | Нет | У них optimal |

## Что нужно чтобы подать результаты в EasyEdit

1. Добавить `AGIMEditor` класс в `easyeditor/models/`
2. Добавить конфиг в `hparams/AGIM/llama3.1-8b.yaml`
3. Запустить через `edit.py --editing_method AGIM`
4. Их evaluate.py сам посчитает ES/PS/NS по токенному exact match

## Ожидаемый эффект от перехода на EasyEdit протокол

При token-level exact match наши 87% ES могут стать:
- Выше: потому что теперь repetition нет, модель генерирует чистый ответ
- Или ниже: потому что exact match требует точного совпадения токенов

Нужно протестировать.

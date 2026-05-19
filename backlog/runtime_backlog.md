# PATH B runtime WAL backlog

- `runtime.py`: десериализация `wal_blob` в слойные состояния пока на backlog (подготовка для полного WAL checkpoint-ремаршалинга).
- `runtime_persistence.py`: полная сериализация всех WAL-слоёв отмечена как backlog item (`TODO(backlog)`).
- Приоритет: технический долг после закрытия Gate 5.

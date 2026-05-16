"""Демо: полный цикл AGIM + WAL на Llama 3.1 8B."""
import torch
from agim.core.system import AGIMSystem
from agim.model.wal_backend import WALWeightEditor

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("pip install transformers")
    raise SystemExit(1)

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
# Try cached first, fallback to HF
LLAMA = ("/mnt/hf_model_weights/arman/3bit/sites/agi_personal_memory/.hf_cache/"
         "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/")
import os
snapshots = os.listdir(LLAMA) if os.path.isdir(LLAMA) else []
MODEL = LLAMA + snapshots[0] if snapshots else "meta-llama/Llama-3.1-8B-Instruct"


def main():
    print("=" * 60)
    print("AGI Personal Memory — Демо на Llama 3.1 8B")
    print("=" * 60)

    # ── Шаг 1: Загрузка модели ──
    print("\n[1/5] Загрузка Llama 3.1 8B...")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=bool(snapshots))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=bool(snapshots))
    model.eval()
    print(f"  Модель загружена на {DEVICE}")

    # ── Шаг 2: WAL-словарь ──
    print("\n[2/5] Построение WAL-словаря (K=256, lmax=12)...")
    editor = WALWeightEditor(model, K=256, lmax=12, device=DEVICE)
    target = "model.layers.0.mlp.down_proj.weight"
    editor.build_vocabulary(target)
    print(f"  Словарь заморожен: {editor.vocabulary_is_frozen}")
    print(f"  Атомов: {editor._atom_table.shape[0]}")

    # ── Шаг 3: Обучение ──
    print("\n[3/5] Обучение на персональных фактах...")
    agim = AGIMSystem(workdir="./demo_memory")
    facts = [
        ("Кто твой создатель?", "Аубакиров Арман"),
        ("Какой язык ты предпочитаешь?", "Русский и казахский"),
        ("Что такое Zanikland?", "Вымышленная страна со столицей Блорптаун"),
    ]
    for q, a in facts:
        c = agim.propose_memory(question=q, answer=a, kind="fact_teach")
        report = agim.compile(c)
        if report.passed:
            agim.commit(report)
            print(f"  ✓ {q} → {a}")
        else:
            print(f"  ✗ {q}: {report.reason}")

    # ── Шаг 4: Применение к модели ──
    print("\n[4/5] Применение правок к модели...")
    editor.snapshot_layer(target)
    for q, a in facts:
        tokens = tok.encode(a, add_special_tokens=False)
        if tokens:
            delta = torch.zeros_like(dict(model.named_parameters())[target].data)
            delta[tokens[0] % delta.shape[0], :] += 0.1
            editor.edit_weight(target, delta)
    non_target_ok = editor.verify_non_target_diff(target)
    print(f"  Non-target diff = 0%: {non_target_ok}")

    # ── Шаг 5: Проверка ──
    print("\n[5/5] Проверка результатов...")
    print("  --- AGIM memory ---")
    for q, expected in facts:
        resp = agim.ask(q)
        ok = resp.answer == expected
        print(f"  {'✓' if ok else '✗'} {q} → {resp.answer}")

    print("  --- Model generation ---")
    for q, expected in facts[:2]:
        inputs = tok(q, return_tensors="pt").to(DEVICE)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = tok.decode(output[0], skip_special_tokens=True)
        contains = expected.lower() in answer.lower()
        print(f"  {'✓' if contains else '?'} {q}")
        if not contains:
            print(f"     Ответ модели: {answer[:100]}")

    stats = agim.stats()
    print(f"\n  Итого фактов: {stats.total_facts}")
    print(f"  Коммитов: {stats.total_commits}")
    print("=" * 60)
    print("Демо завершено. Память сохранена в ./demo_memory/")
    print("Откат: editor.rollback_edit() + agim.rollback_last()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

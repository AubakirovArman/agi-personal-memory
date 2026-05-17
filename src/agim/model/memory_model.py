"""Memory-Augmented Model — Llama с встроенной AGIM-памятью и FAISS+BM25."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.system import AGIMSystem
from ..core.state import AIGIResponse
from ..memory.faiss_retrieval import FAISSRetrieval


class MemoryAugmentedModel(nn.Module):
    """Модель с встроенной памятью AGIM.

    При generate() сначала ищет в памяти через FAISS+BM25.
    Если найдено — возвращает запомненный ответ.
    Если нет — генерирует через базовую модель.

    Использование:
        model = MemoryAugmentedModel(base_model, tokenizer)
        model.teach("Кто твой создатель?", "Аубакиров Арман")
        answer = model.ask("Кто тебя создал?")  # найдёт через BM25!
    """

    def __init__(self, base_model, tokenizer, memory_dir: str = "./mem_model_memory",
                 embedding_dim: int = 768, device: str = "cuda:0"):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.agim = AGIMSystem(workdir=memory_dir)
        self.retrieval = FAISSRetrieval(dim=embedding_dim)
        self._index_built = False
        self._total_taught = 0

    def teach(self, question: str, answer: str, confidence: float = 1.0) -> bool:
        """Обучить модель одному факту."""
        c = self.agim.propose_memory(
            question=question, answer=answer,
            kind="fact_teach", source="memory_model", confidence=confidence)
        report = self.agim.compile(c)
        if not report.passed:
            return False
        ok = self.agim.commit(report)
        if ok:
            self.retrieval.add(question, answer)
            self._index_built = False
            self._total_taught += 1
        return ok

    def teach_batch(self, qa_pairs: list[tuple[str, str]],
                    confidence: float = 0.9) -> int:
        """Массовое обучение. Возвращает количество успешно выученных."""
        count = 0
        for q, a in qa_pairs:
            if self.teach(q, a, confidence):
                count += 1
        self._build_index()
        return count

    def _build_index(self):
        if len(self.retrieval) >= 5:
            self.retrieval.build_index()
            self._index_built = True

    def search_memory(self, query: str, top_k: int = 5,
                      min_score: float = 0.01) -> list[dict]:
        """Поиск в памяти через FAISS+BM25 (семантический)."""
        if not self._index_built:
            self._build_index()
        return self.retrieval.search(query, top_k=top_k)

    def ask(self, question: str, max_tokens: int = 256,
            memory_threshold: float = 0.3) -> AIGIResponse:
        """Спросить модель — сначала память, потом генерация."""
        # 1. Ищем в AGIM точное совпадение (быстро)
        exact = self.agim.ask(question)
        if exact.source != "model_fallback":
            return exact

        # 2. Ищем семантически через FAISS+BM25
        results = self.search_memory(question, top_k=3)
        if results and results[0]["score"] > memory_threshold:
            best = results[0]
            return AIGIResponse(
                question=question, answer=best["value"],
                source="memory_semantic",
                memory_id=best.get("key", ""),
                confidence=min(1.0, best["score"] * 2))

        # 3. Генерируем через базовую модель
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return AIGIResponse(
            question=question, answer=answer,
            source="model_generate", confidence=0.5)

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs):
        """Совместимый с HuggingFace generate() интерфейс."""
        resp = self.ask(prompt, max_tokens=max_tokens)
        if resp.source in ("wal_recipe", "retrieval", "memory_semantic"):
            # Возвращаем как тензор для совместимости
            encoded = self.tokenizer(
                f"{prompt} {resp.answer}",
                return_tensors="pt").to(self.device)
            return encoded["input_ids"]
        # Fallback to base model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.base_model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def total_memories(self) -> int:
        s = self.agim.stats()
        return s.total_facts

    @property
    def model_size_mb(self) -> float:
        return sum(p.numel() * p.element_size()
                   for p in self.base_model.parameters()) / 1024 / 1024

    def stats(self) -> dict:
        s = self.agim.stats()
        return {
            "model": "Llama-3.1-8B-Instruct + AGIM Memory",
            "total_facts": s.total_facts,
            "total_taught": self._total_taught,
            "faiss_indexed": self._index_built,
            "faiss_entries": len(self.retrieval),
            "model_size_mb": round(self.model_size_mb, 0),
            "tiers": s.facts_by_tier,
        }

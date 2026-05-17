"""Memory-Augmented Model — AGIM memory + ROME weight editing (Path A + B)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.system import AGIMSystem
from ..core.state import AIGIResponse
from ..memory.faiss_retrieval import FAISSRetrieval
from .rome_causal import ROMECausalEditor


class MemoryAugmentedModel(nn.Module):
    """Модель с AGIM-памятью + ROME weight editing.

    Path A (Memory): facts stored in JSON + FAISS, fast lookup
    Path B (Weight Edit): ROME edits lm_head, model.generate() outputs answer
    """

    def __init__(self, base_model, tokenizer, memory_dir: str = "./mem_model_memory",
                 embedding_dim: int = 768, device: str = "cuda:0",
                 enable_weight_editing: bool = True):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.agim = AGIMSystem(workdir=memory_dir, max_daily=100000)
        self.agim.budget.max_hourly_commits = 100000
        self.retrieval = FAISSRetrieval(dim=embedding_dim)
        self._index_built = False
        self._total_taught = 0
        self.enable_weight_editing = enable_weight_editing
        self.editor = ROMECausalEditor(base_model, tokenizer, device=device) if enable_weight_editing else None
        self._edits_applied = 0

    def teach(self, question: str, answer: str, confidence: float = 1.0) -> bool:
        # Path A: AGIM memory
        c = self.agim.propose_memory(
            question=question, answer=answer,
            kind="fact_teach", source="memory_model", confidence=confidence)
        report = self.agim.compile(c)
        if not report.passed:
            return False
        if not self.agim.commit(report):
            return False
        self.retrieval.add(question, answer)
        self._index_built = False
        self._total_taught += 1

        # Path B: ROME lm_head edit
        if self.enable_weight_editing and self.editor is not None:
            self.editor.apply_edit(question, answer, "", clamp_norm=0.08)
            self._edits_applied += 1
        return True

    def teach_batch(self, qa_pairs: list[tuple[str, str]], confidence: float = 0.9) -> int:
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

    def search_memory(self, query: str, top_k: int = 5, min_score: float = 0.01) -> list[dict]:
        if not self._index_built:
            self._build_index()
        return self.retrieval.search(query, top_k=top_k)

    def ask(self, question: str, max_tokens: int = 256,
            memory_threshold: float = 0.3, use_memory: bool = True) -> AIGIResponse:
        # Path A: memory lookup
        if use_memory:
            exact = self.agim.ask(question)
            if exact.source != "model_fallback":
                return exact
            results = self.search_memory(question, top_k=3)
            if results and results[0]["score"] > memory_threshold:
                best = results[0]
                return AIGIResponse(question=question, answer=best["value"],
                                  source="memory_semantic",
                                  memory_id=best.get("key", ""),
                                  confidence=min(1.0, best["score"] * 2))
        # Path B: model.generate() with edited weights
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        source = "model_edited" if self._edits_applied > 0 else "model_generate"
        return AIGIResponse(question=question, answer=answer, source=source, confidence=0.5)

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs):
        resp = self.ask(prompt, max_tokens=max_tokens)
        if resp.source in ("wal_recipe", "retrieval", "memory_semantic"):
            encoded = self.tokenizer(f"{prompt} {resp.answer}", return_tensors="pt").to(self.device)
            return encoded["input_ids"]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self.base_model.generate(**inputs, max_new_tokens=max_tokens,
                                        do_sample=False,
                                        pad_token_id=self.tokenizer.eos_token_id, **kwargs)

    def verify_model_knows(self, question: str, expected: str, use_memory: bool = False) -> bool:
        resp = self.ask(question, use_memory=use_memory)
        return expected.lower() in resp.answer.lower()

    def rollback_edits(self):
        if self.editor is not None:
            self.editor.rollback()
            self._edits_applied = 0

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def total_memories(self) -> int:
        return self.agim.stats().total_facts

    @property
    def edits_applied(self) -> int:
        return self._edits_applied

    @property
    def model_size_mb(self) -> float:
        return sum(p.numel() * p.element_size()
                   for p in self.base_model.parameters()) / 1024 / 1024

    def stats(self) -> dict:
        s = self.agim.stats()
        return {
            "model": "Llama-3.1-8B + AGIM + ROME",
            "total_facts": s.total_facts,
            "total_taught": self._total_taught,
            "edits_applied": self._edits_applied,
            "faiss_indexed": self._index_built,
            "faiss_entries": len(self.retrieval),
            "model_size_mb": round(self.model_size_mb, 0),
        }

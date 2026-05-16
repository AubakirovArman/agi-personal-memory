"""Non-target regression suite — verify old facts survive new commits."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import BehaviouralContract


@dataclass
class RegressionSuite:
    """Protected facts that must survive all future edits."""
    contracts: list[BehaviouralContract] = field(default_factory=list)
    _results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add_protected(self, question: str, answer: str, name: str | None = None):
        self.contracts.append(BehaviouralContract(
            name=name or f"protected_{len(self.contracts)}",
            kind="must_answer",
            question=question,
            expected_answer=answer,
            check_type="contains",
        ))

    def run_regression(self, answer_fn, commit_id: str) -> dict[str, bool]:
        results = {}
        for c in self.contracts:
            actual = answer_fn(c.question)
            ok = c.verify(actual)
            results[c.name] = ok
            if commit_id not in self._results:
                self._results[commit_id] = []
            self._results[commit_id].append({
                "contract": c.name,
                "passed": ok,
                "expected": c.expected_answer,
                "actual": actual[:100],
            })
        return results

    @property
    def last_run_all_pass(self) -> bool:
        if not self._results:
            return True
        last_commit = list(self._results.keys())[-1]
        return all(r["passed"] for r in self._results[last_commit])

    def save(self, path: Path):
        path.write_text(json.dumps({
            "contracts": [{"name": c.name, "question": c.question,
                          "answer": c.expected_answer} for c in self.contracts],
            "history": self._results,
        }, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "RegressionSuite":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        suite = cls()
        for c in d.get("contracts", []):
            suite.add_protected(c["question"], c["answer"], c["name"])
        suite._results = d.get("history", {})
        return suite

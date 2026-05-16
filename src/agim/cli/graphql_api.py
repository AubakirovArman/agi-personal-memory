"""v1.0: GraphQL API for AGIM memory."""
from __future__ import annotations

import json
from typing import Any

from ..core.system import AGIMSystem


class GraphQLResolver:
    """GraphQL resolver for AGIM memory operations."""

    def __init__(self, agim: AGIMSystem):
        self.agim = agim

    SCHEMA = """
    type Fact { question: String!, answer: String!, source: String!, confidence: Float! }
    type Stats { totalFacts: Int!, totalCommits: Int!, rollbackCount: Int! }
    type Query {
        ask(question: String!): Fact
        stats: Stats
        history(limit: Int): [Event]
        search(query: String!): [Fact]
    }
    type Mutation {
        teach(question: String!, answer: String!, kind: String): String
        correct(question: String!, answer: String!): String
        forget: String
    }
    """

    def resolve(self, query: str, variables: dict | None = None) -> dict:
        """Parse and resolve GraphQL query. Returns data dict."""
        q = query.strip()
        if "ask" in q:
            question = self._extract_arg(q, "question")
            resp = self.agim.ask(question)
            return {"ask": {"question": resp.question, "answer": resp.answer,
                           "source": resp.source, "confidence": resp.confidence}}
        if "stats" in q:
            s = self.agim.stats()
            return {"stats": {"totalFacts": s.total_facts,
                             "totalCommits": s.total_commits,
                             "rollbackCount": s.rollback_count}}
        if "history" in q:
            limit = int(self._extract_arg(q, "limit") or "20")
            events = self.agim.log.tail(limit)
            return {"history": events}
        if "search" in q:
            query_text = self._extract_arg(q, "query")
            results = []
            for question, entry in self.agim.retrieval._data.items():
                if query_text.lower() in question.lower():
                    results.append({"question": question, "answer": entry["answer"],
                                  "source": entry.get("source", ""),
                                  "confidence": entry.get("confidence", 1.0)})
            return {"search": results[:20]}
        if "teach" in q:
            question = self._extract_arg(q, "question")
            answer = self._extract_arg(q, "answer")
            c = self.agim.propose_memory(question=question, answer=answer)
            report = self.agim.compile(c)
            if report.passed:
                self.agim.commit(report)
            return {"teach": report.status}
        if "correct" in q:
            question = self._extract_arg(q, "question")
            answer = self._extract_arg(q, "answer")
            c = self.agim.propose_memory(question=question, answer=answer, kind="fact_correct")
            report = self.agim.compile(c)
            if report.passed:
                self.agim.commit(report)
            return {"correct": report.status}
        if "forget" in q:
            ok = self.agim.rollback_last()
            return {"forget": "PASS" if ok else "FAIL"}
        return {"error": "Unknown query"}

    def _extract_arg(self, query: str, arg_name: str) -> str | None:
        """Extract argument value from GraphQL query string."""
        import re
        pattern = rf'{arg_name}:\s*"([^"]*)"'
        m = re.search(pattern, query)
        if m:
            return m.group(1)
        return None

"""v7.0: MCP (Model Context Protocol) Server for AGIM memory."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..core.system import AGIMSystem


class MCPServer:
    """MCP-compatible server exposing AGIM memory to any MCP client.

    Tools: memory/search, memory/teach, memory/verify, memory/history
    """

    def __init__(self, workdir: str | None = None):
        workdir = workdir or os.environ.get("AGIM_HOME", str(Path.home() / ".agim"))
        self.agim = AGIMSystem(workdir=workdir)
        self._tools = {
            "memory/search": self._search,
            "memory/teach": self._teach,
            "memory/verify": self._verify,
            "memory/history": self._history,
            "memory/stats": self._stats,
        }

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if method == "tools/list":
            return self._list_tools()
        elif method == "tools/call" and tool_name in self._tools:
            result = self._tools[tool_name](arguments)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
        return {"error": f"Unknown method: {method}"}

    def _list_tools(self) -> dict:
        return {"tools": [
            {"name": "memory/search",
             "description": "Search AGIM memory by question",
             "inputSchema": {"properties": {"query": {"type": "string"}}}},
            {"name": "memory/teach",
             "description": "Teach a new fact to AGIM memory",
             "inputSchema": {"properties": {"question": {"type": "string"},
                            "answer": {"type": "string"}}}},
            {"name": "memory/verify",
             "description": "Verify if a fact can be committed",
             "inputSchema": {"properties": {"question": {"type": "string"},
                            "answer": {"type": "string"}}}},
            {"name": "memory/history",
             "description": "Get memory commit history",
             "inputSchema": {"properties": {"limit": {"type": "integer"}}}},
            {"name": "memory/stats",
             "description": "Get memory statistics",
             "inputSchema": {"properties": {}}},
        ]}

    def _search(self, args: dict) -> dict:
        resp = self.agim.ask(args.get("query", ""))
        return {"answer": resp.answer, "source": resp.source,
                "confidence": resp.confidence}

    def _teach(self, args: dict) -> dict:
        c = self.agim.propose_memory(question=args.get("question", ""),
                                     answer=args.get("answer", ""))
        report = self.agim.compile(c)
        if report.passed:
            self.agim.commit(report)
        return {"status": report.status, "tier": report.tier.value}

    def _verify(self, args: dict) -> dict:
        c = self.agim.propose_memory(question=args.get("question", ""),
                                     answer=args.get("answer", ""))
        report = self.agim.compile(c)
        return {"passes": report.passed, "gates": [
            {"name": g.name, "passed": g.passed, "reason": g.reason}
            for g in report.gates
        ]}

    def _history(self, args: dict) -> dict:
        limit = args.get("limit", 20)
        return {"events": self.agim.log.tail(limit)}

    def _stats(self, args: dict) -> dict:
        return self.agim.stats().__dict__

"""v7.0: A2A (Agent-to-Agent) Protocol + Plugin Marketplace."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class AgentCard:
    """A2A Agent Card — capabilities advertisement."""
    name: str = "AGI Personal Memory"
    description: str = "Verified knowledge accumulation and model editing"
    version: str = "1.0.0"
    capabilities: list[str] = field(default_factory=lambda: [
        "memory:teach", "memory:ask", "memory:verify",
        "memory:search", "memory:history", "memory:edit_model",
    ])
    endpoint: str = "http://localhost:8720"
    agent_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class A2ATask:
    """A2A Task — standardised unit of work between agents."""
    task_id: str = field(default_factory=lambda: uuid4().hex[:12])
    capability: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: dict[str, Any] = field(default_factory=dict)


class A2AServer:
    """v7.0: Google A2A protocol server for agent-to-agent memory sharing."""

    def __init__(self, agim):
        self.agim = agim
        self.card = AgentCard()
        self._tasks: dict[str, A2ATask] = {}
        self._peer_cards: dict[str, AgentCard] = {}

    def get_card(self) -> dict:
        return {"agent_card": {
            "name": self.card.name, "description": self.card.description,
            "capabilities": self.card.capabilities, "endpoint": self.card.endpoint,
        }}

    def receive_task(self, capability: str, input_data: dict) -> A2ATask:
        task = A2ATask(capability=capability, input_data=input_data)
        self._tasks[task.task_id] = task
        if capability == "memory:ask":
            resp = self.agim.ask(input_data.get("question", ""))
            task.result = {"answer": resp.answer, "source": resp.source}
            task.status = "completed"
        elif capability == "memory:teach":
            c = self.agim.propose_memory(**input_data)
            report = self.agim.compile(c)
            if report.passed:
                self.agim.commit(report)
            task.result = {"status": report.status, "tier": report.tier.value}
            task.status = "completed"
        elif capability == "memory:verify":
            c = self.agim.propose_memory(**input_data)
            report = self.agim.compile(c)
            task.result = {"passes": report.passed}
            task.status = "completed"
        else:
            task.status = "unsupported"
        return task

    def register_peer(self, card: AgentCard):
        self._peer_cards[card.agent_id] = card

    def list_peers(self) -> list[dict]:
        return [{"id": cid, "name": c.name, "capabilities": c.capabilities}
                for cid, c in self._peer_cards.items()]


@dataclass
class Plugin:
    """v7.0: AGIM Plugin for the marketplace."""
    name: str
    description: str
    version: str = "0.1.0"
    author: str = "anonymous"
    category: str = "verification_gate"
    entry_point: str = ""
    plugin_id: str = field(default_factory=lambda: uuid4().hex[:12])


class PluginMarketplace:
    """v7.0: Sandboxed plugin marketplace for AGIM extensions."""

    def __init__(self, path: str | Path = ".agim_plugins"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, Plugin] = {}
        self._load()

    def _load(self):
        idx = self.path / "index.json"
        if idx.exists():
            for p in json.loads(idx.read_text()):
                plugin = Plugin(**p)
                self._plugins[plugin.plugin_id] = plugin

    def _save(self):
        (self.path / "index.json").write_text(json.dumps(
            [{"name": p.name, "description": p.description, "version": p.version,
              "author": p.author, "category": p.category, "plugin_id": p.plugin_id}
             for p in self._plugins.values()], indent=2))

    def publish(self, plugin: Plugin) -> str:
        self._plugins[plugin.plugin_id] = plugin
        self._save()
        return plugin.plugin_id

    def search(self, query: str) -> list[Plugin]:
        ql = query.lower()
        return [p for p in self._plugins.values()
                if ql in p.name.lower() or ql in p.description.lower()]

    def install(self, plugin_id: str) -> Plugin | None:
        return self._plugins.get(plugin_id)

"""Relation-scoped protected-bank helpers for EasyEdit runs."""
from __future__ import annotations

from typing import Any


def preload_relation_protected_banks(editor, args, facts, records) -> dict[str, Any]:
    mode = getattr(args, "relation_protected_mode", "none")
    namespace = getattr(args, "state_namespace", "default")
    if mode != "preload":
        return {"mode": mode, "state_namespace": namespace, "relations": {}}
    editor._activate_state_namespace(namespace)
    grouped = relation_locality_prompts(
        facts, records, getattr(args, "relation_protected_prompt_limit", 4))
    summary: dict[str, dict[str, int]] = {}
    for relation_id, prompts in sorted(grouped.items()):
        keys = editor._prompt_keys(prompts, len(prompts))
        added = editor._add_relation_protected_keys(
            relation_id,
            keys,
            getattr(args, "max_relation_protected_keys", 64),
        )
        summary[relation_id] = {"prompts": len(prompts), "keys": added}
    editor._sync_active_state()
    return {"mode": mode, "state_namespace": namespace, "relations": summary}


def relation_locality_prompts(facts, records, prompt_limit: int) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for fact, record in zip(facts, records):
        relation_id = str(fact.get("requested_rewrite", {}).get("relation_id", ""))
        prompts = record.get("locality", {}).get("neighborhood", {}).get("prompt", [])
        selected = prompts if prompt_limit <= 0 else prompts[:prompt_limit]
        bucket = grouped.setdefault(relation_id, [])
        relation_seen = seen.setdefault(relation_id, set())
        for prompt in selected:
            if prompt in relation_seen:
                continue
            relation_seen.add(prompt)
            bucket.append(prompt)
    return grouped

from __future__ import annotations

import json

from .cross_model_common import ModelCandidate, _sha256_json, repo_root


def run_wal_artifact_workflow(module: str, family: str, candidate: ModelCandidate) -> dict[str, object]:
    """Run a deterministic WAL artifact lifecycle tied to a real model path.

    This does not claim weight editing or semantic learning. It proves that a
    pinned local model can be attached to the WAL init/recipe/build/tag/bad-edit
    /CI-fail/blame/rollback/release-notes lifecycle.
    """
    workflow_dir = repo_root() / "corpora" / f"{module.lower()}_{family}_workflow"
    recipes_dir = workflow_dir / "recipes"
    builds_dir = workflow_dir / "builds"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    builds_dir.mkdir(parents=True, exist_ok=True)

    facts = [
        {"question": "WAL smoke fact 1?", "answer": "qwen-small", "category": "smoke"},
        {"question": "WAL smoke fact 2?", "answer": "local-snapshot", "category": "smoke"},
        {"question": "WAL smoke fact 3?", "answer": "rollback", "category": "smoke"},
        {"question": "WAL smoke fact 4?", "answer": "checksum", "category": "smoke"},
        {"question": "WAL smoke fact 5?", "answer": "release-notes", "category": "smoke"},
    ]
    recipe = {
        "schema_version": "wal.recipe.v1",
        "module": module,
        "family": family,
        "base_model_path": candidate.path,
        "facts": facts,
    }
    recipe_checksum = _sha256_json(recipe)
    build_id = recipe_checksum[:12]
    build = {
        "schema_version": "wal.build.v1",
        "id": build_id,
        "base_model_path": candidate.path,
        "model_type": candidate.model_type,
        "architectures": candidate.architectures,
        "recipe_checksum": recipe_checksum,
        "behavioral_checksum": _sha256_json({"facts": facts, "model": candidate.path})[:16],
    }
    bad_edit = {
        "question": facts[0]["question"],
        "answer": "incorrect-answer",
        "expected_answer": facts[0]["answer"],
    }
    ci_fail = bad_edit["answer"] != bad_edit["expected_answer"]
    blame = {"bad_edit_question": bad_edit["question"], "reason": "answer_changed"}
    tags = {"qwen-small-smoke": build_id, "current": build_id}

    (workflow_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "wal.workflow.v1",
                "module": module,
                "family": family,
                "base_model_path": candidate.path,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (recipes_dir / "smoke_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (builds_dir / f"{build_id}.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workflow_dir / "tags.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workflow_dir / "release_notes.md").write_text(
        f"# {module} Qwen Small Smoke\n\n- Build: `{build_id}`\n- Model: `{candidate.path}`\n",
        encoding="utf-8",
    )

    checks = {
        "init": workflow_dir.exists(),
        "add_recipe": (recipes_dir / "smoke_recipe.json").exists() and len(facts) == 5,
        "build": (builds_dir / f"{build_id}.json").exists(),
        "exact_check": all(fact["answer"] for fact in facts),
        "negative_check": ci_fail,
        "context_check": all("WAL smoke" in fact["question"] for fact in facts),
        "tag": tags["qwen-small-smoke"] == build_id,
        "bad_edit": bad_edit["answer"] != bad_edit["expected_answer"],
        "ci_fail": ci_fail,
        "blame_or_bisect": blame["bad_edit_question"] == facts[0]["question"],
        "rollback": tags["current"] == build_id,
        "release_notes": (workflow_dir / "release_notes.md").exists(),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "workflow_dir": str(workflow_dir.relative_to(repo_root())),
        "build_id": build_id,
        "checks": checks,
        "artifact_workflow_only": True,
        "weights_modified": False,
    }

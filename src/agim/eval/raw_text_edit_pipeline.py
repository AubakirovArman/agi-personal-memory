"""AKEW-style raw text update to EasyEdit proposal conversion."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RawTextEditProposal:
    subject: str
    relation_id: str
    target_new: str
    prompt: str
    source_text: str
    parser: str
    confidence: float

    def to_requested_rewrite(self, target_true: str = "") -> dict[str, Any]:
        return {
            "requested_rewrite": {
                "subject": self.subject,
                "prompt": self.prompt.replace(self.subject, "{}", 1),
                "relation_id": self.relation_id,
                "target_new": {"str": self.target_new},
                "target_true": {"str": target_true},
            },
            "source_text": self.source_text,
            "parser": self.parser,
            "confidence": self.confidence,
        }


def parse_raw_update(text: str) -> RawTextEditProposal:
    stripped = " ".join(text.strip().split())
    if not stripped:
        raise ValueError("Raw update text is empty")
    json_proposal = _parse_json_update(stripped)
    if json_proposal is not None:
        return json_proposal
    relation = _parse_relation_sentence(stripped)
    if relation is not None:
        return relation
    simple = _parse_simple_is_sentence(stripped)
    if simple is not None:
        return simple
    raise ValueError("Could not parse raw update into an edit proposal")


def proposals_payload(texts: list[str]) -> dict[str, Any]:
    proposals = [parse_raw_update(text) for text in texts]
    return {
        "artifact_schema_version": "raw_text_edit_proposals.v1",
        "n": len(proposals),
        "proposals": [asdict(proposal) for proposal in proposals],
        "requested_rewrites": [
            proposal.to_requested_rewrite() for proposal in proposals
        ],
    }


def _parse_json_update(text: str) -> RawTextEditProposal | None:
    if not text.startswith("{"):
        return None
    payload = json.loads(text)
    subject = str(payload.get("subject", "")).strip()
    relation = str(payload.get("relation_id") or payload.get("relation", "")).strip()
    target = str(payload.get("target_new") or payload.get("new_target", "")).strip()
    prompt = str(payload.get("prompt") or _prompt_for(subject, relation)).strip()
    if not subject or not relation or not target:
        raise ValueError("JSON update requires subject, relation_id/relation, target_new")
    return RawTextEditProposal(
        subject=subject,
        relation_id=_relation_id(relation),
        target_new=target,
        prompt=prompt,
        source_text=text,
        parser="json",
        confidence=1.0,
    )


def _parse_relation_sentence(text: str) -> RawTextEditProposal | None:
    match = re.match(r"^The (?P<relation>.+?) of (?P<subject>.+?) is (?P<target>.+?)\.?$", text)
    if not match:
        return None
    relation = match.group("relation").strip()
    subject = match.group("subject").strip()
    target = match.group("target").strip()
    return RawTextEditProposal(
        subject=subject,
        relation_id=_relation_id(relation),
        target_new=target,
        prompt=f"The {relation} of {subject} is",
        source_text=text,
        parser="relation_sentence",
        confidence=0.8,
    )


def _parse_simple_is_sentence(text: str) -> RawTextEditProposal | None:
    match = re.match(r"^(?P<subject>.+?) is (?P<target>.+?)\.?$", text)
    if not match:
        return None
    subject = match.group("subject").strip()
    target = match.group("target").strip()
    return RawTextEditProposal(
        subject=subject,
        relation_id="is",
        target_new=target,
        prompt=f"{subject} is",
        source_text=text,
        parser="simple_is",
        confidence=0.6,
    )


def _prompt_for(subject: str, relation: str) -> str:
    relation = relation.replace("_", " ")
    if relation == "is":
        return f"{subject} is"
    return f"The {relation} of {subject} is"


def _relation_id(relation: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", relation.lower()).strip("_") or "related_to"


def main() -> int:
    args = _build_parser().parse_args()
    texts = [args.text] if args.text else Path(args.input).read_text().splitlines()
    payload = proposals_payload([text for text in texts if text.strip()])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Raw-text edit proposals saved {output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input")
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

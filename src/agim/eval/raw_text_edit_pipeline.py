"""AKEW-style raw text update to EasyEdit proposal conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agim.model.patch_artifact import PatchArtifact

from .raw_text_scoring import (
    score_raw_text_case,
    scored_raw_text_payload,
    summarize_scored_raw_text_rows,
)


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

    def to_patch_artifact(
        self,
        base_model_digest: str,
        target_true: str = "",
        method_profile_id: str = "raw_text_proposal",
        patch_id: str | None = None,
    ) -> PatchArtifact:
        """Create a service-layer draft artifact from this raw-text proposal.

        The artifact intentionally has no row deltas yet. A model-editing
        backend must materialize rows before it can be applied as a real
        weight patch.
        """
        rewrite = self.to_requested_rewrite(target_true)
        return PatchArtifact(
            patch_id=patch_id or _patch_id(base_model_digest, self),
            base_model_digest=base_model_digest,
            method_profile_id=method_profile_id,
            subject=self.subject,
            relation_id=self.relation_id,
            target_new=self.target_new,
            target_true=target_true or None,
            rows=[],
            metadata={
                "source_text": self.source_text,
                "parser": self.parser,
                "confidence": self.confidence,
                "requested_rewrite": rewrite["requested_rewrite"],
                "requires_backend_materialization": True,
            },
        )


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


def patch_drafts_payload(
    texts: list[str],
    base_model_digest: str,
    target_true_by_subject: dict[str, str] | None = None,
) -> dict[str, Any]:
    proposals = [parse_raw_update(text) for text in texts]
    target_true_by_subject = target_true_by_subject or {}
    artifacts = [
        proposal.to_patch_artifact(
            base_model_digest=base_model_digest,
            target_true=target_true_by_subject.get(proposal.subject, ""),
        )
        for proposal in proposals
    ]
    return {
        "artifact_schema_version": "raw_text_patch_drafts.v1",
        "base_model_digest": base_model_digest,
        "n": len(artifacts),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
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


def _patch_id(base_model_digest: str, proposal: RawTextEditProposal) -> str:
    raw = "|".join([
        base_model_digest,
        proposal.subject,
        proposal.relation_id,
        proposal.target_new,
        proposal.source_text,
    ])
    return f"raw-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def main() -> int:
    args = _build_parser().parse_args()
    if args.score_adapter:
        if not args.score_output:
            raise SystemExit("--score-output is required with --score-adapter")
        source = Path(args.score_adapter)
        adapter = json.loads(source.read_text())
        outputs = json.loads(Path(args.score_output).read_text())
        payload = scored_raw_text_payload(adapter, outputs, str(source))
        output = Path(args.output) if args.output else source.with_name(
            f"{source.stem}.raw_text_scored{source.suffix or '.json'}"
        )
    else:
        if not args.output:
            raise SystemExit("--output is required unless --score-adapter is used")
        texts = [args.text] if args.text else Path(args.input).read_text().splitlines()
        payload = proposals_payload([text for text in texts if text.strip()])
        output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Raw-text edit artifact saved {output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input")
    source.add_argument("--score-adapter")
    parser.add_argument("--score-output")
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

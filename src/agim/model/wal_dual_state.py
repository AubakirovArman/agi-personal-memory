"""State helpers for WALDualLayerEditor."""
from __future__ import annotations

import torch

from ..wal.encoder import build_atoms_kmeans


def build_vocab(editor):
    """Build shared WAL atoms from lm_head + embed_tokens distributions."""
    lm_flat = editor.model.lm_head.weight.data.float().flatten()
    emb_flat = editor.model.model.embed_tokens.weight.data.float().flatten()
    combined = torch.cat([lm_flat[:2_000_000], emb_flat[:2_000_000]])
    editor.atoms = build_atoms_kmeans(
        combined, editor.K, iters=5, device=editor.device)
    editor.atoms_gpu = editor.atoms.to(editor.device)
    print(
        f"  WAL atoms: {editor.atoms.shape} "
        f"[{editor.atoms.min():.3f}, {editor.atoms.max():.3f}]"
    )


def snapshot_non_target(editor, lm_exclude, embed_exclude=None,
                        sample_size: int = 500):
    """Snapshot lm_head and embed non-target rows for NT diff measurement."""
    editor._lm_nt_snapshot = editor._snapshot_rows(
        editor.model.lm_head.weight.data,
        set(lm_exclude),
        sample_size=sample_size,
    )
    editor._emb_nt_snapshot = editor._snapshot_rows(
        editor.model.model.embed_tokens.weight.data,
        set(embed_exclude or ()),
        sample_size=sample_size,
    )


def measure_non_target_diff(editor) -> float:
    """Backward-compatible max over lm_head and embed non-target rows."""
    diffs = editor.measure_non_target_diffs()
    return max(diffs.values()) if diffs else 0.0


def measure_non_target_diffs(editor) -> dict[str, float]:
    """Measured max abs diff on non-edited lm_head and embed rows."""
    return {
        "lm_head_non_edited_max": editor._max_row_diff(
            editor.model.lm_head.weight.data,
            editor._lm_nt_snapshot,
        ),
        "embed_non_edited_max": editor._max_row_diff(
            editor.model.model.embed_tokens.weight.data,
            editor._emb_nt_snapshot,
        ),
    }

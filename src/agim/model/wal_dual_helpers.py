"""Helper methods for WALDualLayerEditor."""
from __future__ import annotations

import torch
from math import gcd


def prompt_keys(editor, prompts: list[str], limit: int) -> list[torch.Tensor]:
    keys = []
    for prompt in prompts[:limit]:
        ids = prompt_ids(editor, prompt, max_tokens=100)
        key = get_key(editor, ids)
        if key is None:
            continue
        keys.append(key / (key.norm() + 1e-8))
    return keys


def history_basis(editor, limit: int, relation_id: str | None = None,
                  slot_mode: str = "global") -> list[torch.Tensor]:
    if slot_mode == "relation":
        basis = editor._relation_key_basis.get(str(relation_id), [])
    else:
        basis = editor._edit_key_basis
    if limit <= 0:
        return basis
    return basis[-limit:]


def positive_keys_for_step(editor, prompts: list[str], tids: list[int],
                           pos: int, limit: int) -> list[torch.Tensor]:
    keys = []
    for prompt in prompts[:limit]:
        pids = prompt_ids(editor, prompt, max_tokens=100)
        ctx = pids if pos == 0 else torch.cat([
            pids,
            torch.tensor(tids[:pos], device=pids.device),
        ])
        key = get_key(editor, ctx)
        if key is None:
            continue
        keys.append(key / (key.norm() + 1e-8))
    return keys


def combine_positive_keys(
    primary: torch.Tensor,
    positives: list[torch.Tensor],
    weight: float,
    protected_basis: list[torch.Tensor] | None = None,
    projection_strength: float = 0.0,
    projection_mode: str = "sequential",
) -> torch.Tensor:
    primary = primary / (primary.norm() + 1e-8)
    if weight <= 0 or not positives:
        return primary
    normalized = []
    for key in positives:
        moved = key.to(primary.device).float()
        moved = moved / (moved.norm() + 1e-8)
        if protected_basis:
            moved = project_away(
                moved,
                protected_basis,
                strength=projection_strength,
                mode=projection_mode,
            )
        normalized.append(moved)
    positive = torch.stack(normalized).mean(dim=0)
    positive = positive / (positive.norm() + 1e-8)
    combined = primary + weight * positive
    return combined / (combined.norm() + 1e-8)


def project_away(key: torch.Tensor, basis: list[torch.Tensor],
                 strength: float, mode: str = "sequential") -> torch.Tensor:
    if strength <= 0 or not basis:
        return key / (key.norm() + 1e-8)
    if mode == "orthogonal":
        return project_away_orthogonal(key, basis, strength)
    out = key / (key.norm() + 1e-8)
    for base in basis:
        b = base.to(out.device).float()
        b = b / (b.norm() + 1e-8)
        dot = torch.dot(out, b)
        if dot > 0:
            out = out - strength * dot * b
            out = out / (out.norm() + 1e-8)
    return out


def project_away_orthogonal(key: torch.Tensor, basis: list[torch.Tensor],
                            strength: float) -> torch.Tensor:
    out = key / (key.norm() + 1e-8)
    rows = []
    for base in basis:
        b = base.to(out.device).float()
        norm = b.norm()
        if norm > 1e-8:
            rows.append(b / norm)
    if not rows:
        return out
    matrix = torch.stack(rows)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular_values > 1e-5).sum().item())
    if rank <= 0:
        return out
    protected = vh[:rank].T
    projection = protected @ (protected.T @ out)
    out = out - strength * projection
    return out / (out.norm() + 1e-8)


def snapshot_rows(weight: torch.Tensor, exclude: set[int],
                  sample_size: int = 500) -> dict[int, torch.Tensor]:
    snapshots: dict[int, torch.Tensor] = {}
    vocab_size = weight.shape[0]
    valid_exclude = {rid for rid in exclude if 0 <= rid < vocab_size}
    target_count = max(0, min(sample_size, vocab_size - len(valid_exclude)))
    if target_count <= 0:
        return snapshots
    start = _stable_row_seed(vocab_size, sample_size, valid_exclude) % vocab_size
    step = _coprime_step(vocab_size)
    for attempts in range(vocab_size):
        rid = (start + attempts * step) % vocab_size
        if rid in valid_exclude or rid in snapshots:
            continue
        snapshots[rid] = weight[rid, :].clone()
        if len(snapshots) >= target_count:
            break
    return snapshots


def _stable_row_seed(vocab_size: int, sample_size: int, exclude: set[int]) -> int:
    seed = (vocab_size * 1315423911 + sample_size * 2654435761) & 0xFFFFFFFF
    for rid in sorted(exclude):
        seed = ((seed ^ rid) * 16777619) & 0xFFFFFFFF
    return seed


def _coprime_step(vocab_size: int) -> int:
    if vocab_size <= 1:
        return 1
    step = 104729 % vocab_size or 1
    while gcd(step, vocab_size) != 1:
        step = (step + 2) % vocab_size or 1
    return step


def max_row_diff(weight: torch.Tensor, snapshots: dict[int, torch.Tensor]) -> float:
    max_diff = 0.0
    for rid, original in snapshots.items():
        diff = (weight[rid, :] - original.to(weight.device)).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


def get_key(editor, token_ids: torch.Tensor) -> torch.Tensor | None:
    last_h = None

    def hook(_module, _inputs, output):
        nonlocal last_h
        hidden = output[0] if isinstance(output, tuple) else output
        last_h = hidden.detach().clone()

    handle = editor.model.model.norm.register_forward_hook(hook)
    with torch.no_grad():
        editor.model(input_ids=token_ids.unsqueeze(0).to(editor.device))
    handle.remove()
    if last_h is None:
        return None
    return last_h[0, -1, :].float() if last_h.dim() == 3 else last_h[-1, :].float()


def contextual_target_ids(editor, prompt: str, target: str) -> list[int]:
    """Ids EasyEdit teacher-forcing labels for prompt + " " + target."""
    prompt_ids_ = editor.tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = editor.tokenizer(f"{prompt} {target}", return_tensors="pt").input_ids[0]
    suffix = full_ids[len(prompt_ids_):].detach().cpu().tolist()
    if suffix:
        return suffix
    return editor.tokenizer.encode(target, add_special_tokens=False)


def target_sequences(editor, prompt: str, target: str,
                     standalone_ids: list[int], mode: str) -> list[list[int]]:
    sequences: list[list[int]] = []
    if mode in {"standalone", "both"}:
        sequences.append(list(standalone_ids))
    if mode in {"contextual", "both"}:
        sequences.append(contextual_target_ids(editor, prompt, target))

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in sequences:
        key = tuple(seq)
        if seq and key not in seen:
            deduped.append(seq)
            seen.add(key)
    return deduped


def primary_target_sequence(editor, prompt: str, target: str,
                            standalone_ids: list[int], mode: str) -> list[int]:
    if mode in {"contextual", "both"}:
        contextual = contextual_target_ids(editor, prompt, target)
        if contextual:
            return contextual
    return list(standalone_ids)


def prompt_ids(editor, prompt: str, max_tokens: int | None = None) -> torch.Tensor:
    kwargs = {"return_tensors": "pt"}
    if max_tokens is not None:
        kwargs.update({"truncation": True, "max_length": max_tokens})
    return editor.tokenizer(prompt, **kwargs).input_ids[0]

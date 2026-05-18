from __future__ import annotations

from typing import Tuple

import torch

from .isa import AtomTable, CoeffTable, ProgramBufferV2
from .grammar_core import WALParseError, _tokenize_line, parse_program_stream


def format_wal_text(
    prog: ProgramBufferV2,
    atoms: AtomTable,
    coeffs: CoeffTable,
    max_programs: Optional[int] = None,
) -> str:
    """Format a complete WAL v2 text file including atom/coeff tables.

    The lower-level grammar intentionally models only program streams. This
    helper is the public round-trip format used by CLI tools, where binary
    reconstruction also needs the numeric tables.
    """
    coeff_values = coeffs.values.detach().cpu()
    atom_values = atoms.values.detach().cpu()
    limit = prog.N if max_programs is None else min(int(max_programs), prog.N)
    lines = [
        f"; WAL v2 text — {prog.N:,} programs",
        f"K {atoms.K}",
        f"C {coeffs.C}",
        f"SHAPE {' '.join(str(int(x)) for x in prog.shape)}",
        "",
        "; Atom table",
    ]
    lines.extend(f"TABLE ATOM {idx} {float(value):.9g}" for idx, value in enumerate(atom_values))
    lines.append("")
    lines.append("; Coeff table")
    lines.extend(f"TABLE COEFF {idx} {float(value):.9g}" for idx, value in enumerate(coeff_values))
    lines.append("")
    lines.append("; Programs")
    for idx in range(limit):
        atom_id = int(prog.atom_ids[idx].item())
        coeff_id = int(prog.coeff_ids[idx].item())
        line = f"ATOM {atom_id} COEF {float(coeff_values[coeff_id]):.9g}"
        if bool(prog.has_residual[idx].item()):
            line += f" RESIDUAL {float(prog.residuals[idx].item()):.9g}"
        lines.append(line)
    if limit < prog.N:
        lines.append(f"; ... {prog.N - limit} more programs ...")
    return "\n".join(lines)


def parse_wal_text(text: str) -> Tuple[ProgramBufferV2, AtomTable, CoeffTable]:
    """Parse complete WAL v2 text produced by :func:`format_wal_text`.

    For legacy program-only text, table values are synthesized as zeros for
    atoms and sorted observed coefficient values for coefficients.
    """
    atom_values: dict[int, float] = {}
    coeff_values: dict[int, float] = {}
    program_lines: list[str] = []
    header_lines: list[str] = []

    for raw_line in text.splitlines():
        tokens = _tokenize_line(raw_line)
        if not tokens:
            continue
        if tokens[0] in {"K", "C", "SHAPE"}:
            header_lines.append(" ".join(tokens))
        elif len(tokens) == 4 and tokens[0] == "TABLE" and tokens[1] == "ATOM":
            atom_values[int(tokens[2])] = float(tokens[3])
        elif len(tokens) == 4 and tokens[0] == "TABLE" and tokens[1] == "COEFF":
            coeff_values[int(tokens[2])] = float(tokens[3])
        elif tokens[0] == "ATOM":
            program_lines.append(" ".join(tokens))
        elif tokens[0] == "TABLE":
            raise WALParseError(f"Unknown TABLE directive: {' '.join(tokens)}")

    parsed = parse_program_stream("\n".join(header_lines + program_lines))
    N = len(parsed.programs)

    if atom_values:
        atoms_tensor = torch.zeros(parsed.K, dtype=torch.float32)
        for idx, value in atom_values.items():
            if not 0 <= idx < parsed.K:
                raise WALParseError(f"Atom table index {idx} out of range [0, {parsed.K})")
            atoms_tensor[idx] = value
    else:
        atoms_tensor = torch.zeros(parsed.K, dtype=torch.float32)

    if coeff_values:
        coeffs_tensor = torch.zeros(parsed.C, dtype=torch.float32)
        for idx, value in coeff_values.items():
            if not 0 <= idx < parsed.C:
                raise WALParseError(f"Coeff table index {idx} out of range [0, {parsed.C})")
            coeffs_tensor[idx] = value
    else:
        observed = sorted({p.coeff_value for p in parsed.programs})
        coeffs_tensor = torch.zeros(parsed.C, dtype=torch.float32)
        for idx, value in enumerate(observed[: parsed.C]):
            coeffs_tensor[idx] = value

    atom_ids = torch.empty(N, dtype=torch.uint8)
    coeff_ids = torch.empty(N, dtype=torch.uint8)
    residuals = torch.zeros(N, dtype=torch.float32)
    has_residual = torch.zeros(N, dtype=torch.bool)
    for idx, program in enumerate(parsed.programs):
        if not 0 <= program.atom_id < parsed.K:
            raise WALParseError(f"Program {idx}: atom_id={program.atom_id} out of range")
        atom_ids[idx] = program.atom_id
        coeff_id = int(torch.argmin((coeffs_tensor - float(program.coeff_value)).abs()).item())
        coeff_ids[idx] = coeff_id
        if program.residual is not None:
            residuals[idx] = float(program.residual)
            has_residual[idx] = True

    prog = ProgramBufferV2(atom_ids, coeff_ids, residuals, has_residual, parsed.shape)
    return prog, AtomTable(atoms_tensor), CoeffTable(coeffs_tensor)

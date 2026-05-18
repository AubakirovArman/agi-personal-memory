from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


WAL_BNF = """
<program_stream>   ::= <header> <program>*
<header>           ::= <k_decl> <c_decl> <shape_decl>
<k_decl>           ::= "K" <uint>
<c_decl>           ::= "C" <uint>
<shape_decl>       ::= "SHAPE" <uint> <uint>
<program>          ::= <atom_call> [<residual>]
<atom_call>        ::= "ATOM" <atom_id> "COEF" <float>
<residual>         ::= "RESIDUAL" <float>
<atom_id>          ::= <uint> [0, K-1]
<coeff_value>      ::= <float>
<uint>             ::= digit+
<float>            ::= ["-" | "+"] digit+ ["." digit+]
<comment>          ::= ";" <any>* <newline>
"""


@dataclass
class ParsedProgram:
    """Single parsed WAL v2 program."""
    atom_id: int
    coeff_value: float
    residual: Optional[float] = None


@dataclass
class ParsedStream:
    """Parsed WAL v2 program stream."""
    K: int
    C: int
    shape: Tuple[int, int]
    programs: List[ParsedProgram]


class WALParseError(Exception):
    """Raised when WAL text cannot be parsed."""
    pass


def _tokenize_line(line: str) -> List[str]:
    """Strip comments and split line into tokens."""
    # Remove comments
    if ';' in line:
        line = line[:line.index(';')]
    line = line.strip()
    if not line:
        return []
    return line.split()


def parse_program_stream(text: str) -> ParsedStream:
    """Parse WAL v2 text format to structured representation.
    
    Args:
        text: Multi-line WAL text
        
    Returns:
        ParsedStream with header and programs
        
    Raises:
        WALParseError: on syntax error
    """
    lines = text.strip().split('\n')
    
    K: Optional[int] = None
    C: Optional[int] = None
    shape: Optional[Tuple[int, int]] = None
    programs: List[ParsedProgram] = []
    
    for line_no, raw_line in enumerate(lines, 1):
        tokens = _tokenize_line(raw_line)
        if not tokens:
            continue
        
        if tokens[0] == 'K':
            if len(tokens) != 2:
                raise WALParseError(f"Line {line_no}: K declaration expects 1 argument")
            K = int(tokens[1])
            
        elif tokens[0] == 'C':
            if len(tokens) != 2:
                raise WALParseError(f"Line {line_no}: C declaration expects 1 argument")
            C = int(tokens[1])
            
        elif tokens[0] == 'SHAPE':
            if len(tokens) != 3:
                raise WALParseError(f"Line {line_no}: SHAPE expects 2 arguments")
            shape = (int(tokens[1]), int(tokens[2]))
            
        elif tokens[0] == 'ATOM':
            # ATOM <id> COEF <value> [RESIDUAL <value>]
            if len(tokens) not in (4, 6):
                raise WALParseError(
                    f"Line {line_no}: ATOM expects 'ATOM <id> COEF <value>' "
                    f"or 'ATOM <id> COEF <value> RESIDUAL <value>'"
                )
            if tokens[2] != 'COEF':
                raise WALParseError(f"Line {line_no}: Expected 'COEF' after atom id")
            
            atom_id = int(tokens[1])
            coeff_value = float(tokens[3])
            residual = None
            
            if len(tokens) == 6:
                if tokens[4] != 'RESIDUAL':
                    raise WALParseError(f"Line {line_no}: Expected 'RESIDUAL' after coeff value")
                residual = float(tokens[5])
            
            programs.append(ParsedProgram(atom_id, coeff_value, residual))
            
        else:
            raise WALParseError(f"Line {line_no}: Unknown directive '{tokens[0]}'")
    
    if K is None:
        raise WALParseError("Missing K declaration")
    if C is None:
        raise WALParseError("Missing C declaration")
    if shape is None:
        raise WALParseError("Missing SHAPE declaration")
    
    return ParsedStream(K=K, C=C, shape=shape, programs=programs)


def format_program_stream(
    programs: List[ParsedProgram],
    shape: Tuple[int, int],
    K: int,
    C: int,
    max_programs: Optional[int] = None,
) -> str:
    """Format programs to WAL v2 text.
    
    Args:
        programs: List of parsed programs
        shape: Matrix shape
        K: Atom table size
        C: Coeff table size
        max_programs: If set, only output first N programs + ellipsis
        
    Returns:
        WAL text
    """
    lines = [
        f"; WAL v2 v0.1 — {len(programs):,} programs",
        f"K {K}",
        f"C {C}",
        f"SHAPE {shape[0]} {shape[1]}",
        "",
    ]
    
    total = len(programs)
    limit = max_programs if max_programs is not None else total
    
    for i in range(min(limit, total)):
        p = programs[i]
        line = f"ATOM {p.atom_id} COEF {p.coeff_value:.6f}"
        if p.residual is not None:
            line += f" RESIDUAL {p.residual:.8f}"
        lines.append(line)
    
    if total > limit:
        lines.append(f"; ... {total - limit} more programs ...")
    
    return '\n'.join(lines)


def format_unique_programs(
    programs: List[ParsedProgram],
    shape: Tuple[int, int],
    K: int,
    C: int,
) -> str:
    """Format unique programs with occurrence counts (summary view).
    
    Args:
        programs: List of parsed programs
        shape: Matrix shape
        K: Atom table size
        C: Coeff table size
        
    Returns:
        WAL text with unique programs and counts
    """
    from collections import Counter
    
    # Create hashable keys
    keys = []
    for p in programs:
        key = (p.atom_id, p.coeff_value, p.residual)
        keys.append(key)
    
    counts = Counter(keys)
    unique = list(counts.keys())
    
    lines = [
        f"; WAL v2 v0.1 — Unique Program Summary",
        f"; K={K} C={C} SHAPE={'x'.join(map(str, shape))}",
        f"; {len(unique)} unique / {len(programs)} total weights",
        f"K {K}",
        f"C {C}",
        f"SHAPE {shape[0]} {shape[1]}",
        "",
        "; <count> | <program>",
    ]
    
    # Sort by count descending
    for key, count in counts.most_common():
        atom_id, coeff_val, residual = key
        line = f"  {count:>10} | ATOM {atom_id} COEF {coeff_val:.6f}"
        if residual is not None:
            line += f" RESIDUAL {residual:.8f}"
        lines.append(line)
    
    return '\n'.join(lines)

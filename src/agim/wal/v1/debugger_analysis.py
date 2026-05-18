from __future__ import annotations

from typing import Any, Dict

from .isa import ProgramBufferV1
from .debugger_types import HeatmapStats


class WALDebuggerAnalysisMixin:
    def heatmap(self, prog: ProgramBufferV1) -> HeatmapStats:
        """Compute program usage heatmap.
        
        Args:
            prog: Program buffer
        
        Returns:
            HeatmapStats with frequencies and entropy
        """
        N = prog.N
        atom_counts: Dict[int, int] = {}
        coeff_counts: Dict[int, int] = {}
        residual_count = 0
        
        for i in range(N):
            aid = int(prog.atom_ids[i])
            cid = int(prog.coeff_ids[i])
            atom_counts[aid] = atom_counts.get(aid, 0) + 1
            coeff_counts[cid] = coeff_counts.get(cid, 0) + 1
            
            if prog.has_residual.numel() > 0 and prog.has_residual[i]:
                residual_count += 1
        
        # Top atoms
        top_atoms = sorted(
            [(aid, cnt, cnt / N * 100) for aid, cnt in atom_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # Top coeffs
        top_coeffs = sorted(
            [(cid, cnt, cnt / N * 100) for cid, cnt in coeff_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # Entropy
        import math
        atom_entropy = -sum((cnt / N) * math.log2(cnt / N) for cnt in atom_counts.values())
        coeff_entropy = -sum((cnt / N) * math.log2(cnt / N) for cnt in coeff_counts.values())
        
        return HeatmapStats(
            total_weights=N,
            atom_frequencies=atom_counts,
            coeff_frequencies=coeff_counts,
            top_atoms=top_atoms,
            top_coeffs=top_coeffs,
            residual_count=residual_count,
            residual_pct=residual_count / N * 100,
            atom_entropy=atom_entropy,
            coeff_entropy=coeff_entropy,
        )
    
    def diff_programs(
        self,
        prog1: ProgramBufferV1,
        prog2: ProgramBufferV1,
        name1: str = "A",
        name2: str = "B",
    ) -> Dict[str, Any]:
        """Compute diff between two programs.
        
        Args:
            prog1: First program
            prog2: Second program
            name1: Label for first program
            name2: Label for second program
        
        Returns:
            Diff statistics
        """
        N = min(prog1.N, prog2.N)
        
        atom_diffs = 0
        coeff_diffs = 0
        residual_diffs = 0
        value_diffs = 0
        max_value_diff = 0.0
        
        flat_atoms = self._get_precomputed_atoms()
        
        diff_indices = []
        for i in range(N):
            a1 = int(prog1.atom_ids[i])
            a2 = int(prog2.atom_ids[i])
            c1 = int(prog1.coeff_ids[i])
            c2 = int(prog2.coeff_ids[i])
            
            v1 = flat_atoms[a1].item() * self.coeff_values[c1].item()
            v2 = flat_atoms[a2].item() * self.coeff_values[c2].item()
            
            if prog1.has_residual.numel() > 0 and prog1.has_residual[i]:
                v1 += prog1.residuals[i].item()
            if prog2.has_residual.numel() > 0 and prog2.has_residual[i]:
                v2 += prog2.residuals[i].item()
            
            diff = abs(v1 - v2)
            if diff > 1e-7:
                value_diffs += 1
                max_value_diff = max(max_value_diff, diff)
                
                if a1 != a2:
                    atom_diffs += 1
                if c1 != c2:
                    coeff_diffs += 1
                
                if len(diff_indices) < 10:
                    diff_indices.append({
                        'index': i,
                        'atom': (a1, a2),
                        'coeff': (c1, c2),
                        'value_diff': diff,
                    })
        
        return {
            'name1': name1,
            'name2': name2,
            'total_weights': N,
            'atom_diffs': atom_diffs,
            'coeff_diffs': coeff_diffs,
            'value_diffs': value_diffs,
            'value_diff_pct': value_diffs / N * 100,
            'max_value_diff': max_value_diff,
            'identical': value_diffs == 0,
            'sample_diffs': diff_indices,
        }

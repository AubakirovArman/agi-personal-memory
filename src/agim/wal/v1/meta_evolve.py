from __future__ import annotations

from typing import Tuple

import torch

from .decoder import wal_decode_v1
from .isa import AtomTableV1, CoeffTable, ProgramBufferV1


def evolve_programs(
    weights: torch.Tensor,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    population_size: int = 16,
    generations: int = 10,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.5,
    top_k: int = 4,
) -> Tuple[ProgramBufferV1, torch.Tensor]:
    """Evolve programs via genetic algorithm.
    
    Each individual is a program buffer. Fitness = negative MSE.
    Evolution operators: mutation (flip atom/coeff), crossover (swap segments).
    
    Args:
        weights: Target weights to approximate
        atom_table: Atom table
        coeffs: Coefficient table
        population_size: Number of individuals
        generations: Number of generations
        mutation_rate: Probability of mutation per weight
        crossover_rate: Probability of crossover
        top_k: Number of best individuals to keep
    
    Returns:
        (best_program, best_reconstruction)
    """
    device = weights.device
    N = weights.numel()
    K = atom_table.K_total
    C = coeffs.values.numel()
    
    # Initialize population with random programs
    population = []
    for _ in range(population_size):
        atom_ids = torch.randint(0, min(K, 256), (N,), dtype=torch.uint8, device=device)
        coeff_ids = torch.randint(0, min(C, 256), (N,), dtype=torch.uint8, device=device)
        prog = ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
        population.append(prog)
    
    def fitness(prog: ProgramBufferV1) -> float:
        """Negative MSE (higher is better)."""
        recon = wal_decode_v1(prog, atom_table, coeffs.values).flatten()
        mse = (weights.flatten() - recon).pow(2).mean().item()
        return -mse
    
    def mutate(prog: ProgramBufferV1) -> ProgramBufferV1:
        """Random mutation: flip some atom_ids and coeff_ids."""
        atom_ids = prog.atom_ids.clone()
        coeff_ids = prog.coeff_ids.clone()
        
        mask = torch.rand(N, device=device) < mutation_rate
        atom_ids[mask] = torch.randint(0, min(K, 256), (mask.sum(),), dtype=torch.uint8, device=device)
        
        mask = torch.rand(N, device=device) < mutation_rate
        coeff_ids[mask] = torch.randint(0, min(C, 256), (mask.sum(),), dtype=torch.uint8, device=device)
        
        return ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
    
    def crossover(p1: ProgramBufferV1, p2: ProgramBufferV1) -> ProgramBufferV1:
        """Single-point crossover."""
        point = torch.randint(0, N, (1,), device=device).item()
        atom_ids = torch.cat([p1.atom_ids[:point], p2.atom_ids[point:]])
        coeff_ids = torch.cat([p1.coeff_ids[:point], p2.coeff_ids[point:]])
        
        return ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
    
    best_fitness = float('-inf')
    best_prog = None
    
    for gen in range(generations):
        # Evaluate fitness
        scores = [(fitness(p), p) for p in population]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        if scores[0][0] > best_fitness:
            best_fitness = scores[0][0]
            best_prog = scores[0][1]
        
        # Elitism: keep top_k
        elites = [p for _, p in scores[:top_k]]
        
        # Create next generation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            if torch.rand(1).item() < crossover_rate and len(elites) >= 2:
                # Crossover
                idx1 = torch.randint(0, len(elites), (1,)).item()
                idx2 = torch.randint(0, len(elites), (1,)).item()
                p1, p2 = elites[idx1], elites[idx2]
                child = crossover(p1, p2)
            else:
                # Mutation
                parent = elites[torch.randint(0, len(elites), (1,)).item()]
                child = mutate(parent)
            
            new_population.append(child)
        
        population = new_population
        
        if gen % 3 == 0 or gen == generations - 1:
            print(f"    Gen {gen:3d}: best fitness={best_fitness:.8f} (MSE={-best_fitness:.8f})")
    
    recon = wal_decode_v1(best_prog, atom_table, coeffs.values)
    return best_prog, recon

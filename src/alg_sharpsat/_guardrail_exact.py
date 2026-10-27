"""Integer weighted upward/downward passes with proven bounds and modular CRT."""

from __future__ import annotations

from fractions import Fraction
from math import log, prod

import numpy as np

from .stage2_algebraic import crt_reconstruct, primes_for_bound
from .stage3_gpu_engine import ResourceLimitError

_INT64_MAX = np.iinfo(np.int64).max


def _reduce(
    values: np.ndarray, indices: np.ndarray, size: int, prime: int | None
) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    if prime is None:
        # Every partial sum is nonnegative and bounded by the global bound.
        np.add.at(out, indices, values)
    else:
        # Residues are reduced before addition; cap each accumulation chunk.
        chunk = min(262144, (_INT64_MAX - prime) // (prime - 1))
        for start in range(0, len(values), chunk):
            np.add.at(
                out, indices[start : start + chunk], values[start : start + chunk]
            )
            out %= prime
    return out


def _sum(values: np.ndarray, prime: int | None) -> int:
    if prime is None:
        return int(values.sum(dtype=np.int64))
    chunk = min(262144, _INT64_MAX // (prime - 1))
    return (
        sum(
            int(values[i : i + chunk].sum(dtype=np.int64))
            for i in range(0, len(values), chunk)
        )
        % prime
    )


def _pass(compiled, weights, prime):
    up, row_values = [], []
    for node in compiled.nodes:
        if node.left < 0:
            value = np.ones(len(node.key_idx), dtype=np.int64)
            for j, var in enumerate(node.original_vars):
                a, b = weights[var]
                if prime is not None:
                    a, b = a % prime, b % prime
                value *= np.where(node.assignments[:, j], b, a).astype(np.int64)
                if prime is not None:
                    value %= prime
            row_values.append(value)
            up.append(_reduce(value, node.key_idx, node.key_count, prime))
        else:
            value = up[node.left][node.pair_left] * up[node.right][node.pair_right]
            if prime is not None:
                value %= prime
            up.append(_reduce(value, node.out_idx, node.key_count, prime))
    down = [np.zeros(n.key_count, dtype=np.int64) for n in compiled.nodes]
    down[compiled.root][:] = 1
    for i in range(len(compiled.nodes) - 1, -1, -1):
        node = compiled.nodes[i]
        if node.left < 0:
            continue
        cavity = down[i][node.out_idx]
        for child, other, target_index, other_index in (
            (node.left, node.right, node.pair_left, node.pair_right),
            (node.right, node.left, node.pair_right, node.pair_left),
        ):
            value = cavity * up[other][other_index]
            if prime is not None:
                value %= prime
            down[child] = _reduce(
                value, target_index, compiled.nodes[child].key_count, prime
            )
    core = _sum(up[compiled.root], prime)
    numerators = [0] * compiled.original_num_vars
    for i, leaf in enumerate(compiled.leaves):
        value = row_values[i] * down[i][leaf.key_idx]
        if prime is not None:
            value %= prime
        for j, var in enumerate(leaf.original_vars):
            numerators[var] = _sum(value[leaf.assignments[:, j]], prime)
    return [core, *numerators]


def evaluate_exact(compiled, weights, denominator):
    """One upward/downward pass per modulus, reconstructing Z and all moments.

    The product of leaf row counts bounds multiplicity for every partial join.
    Each original variable occurs in only one leaf's weight product. Combining
    that multiplicity with max(1,w0,w1) for every owned original bounds all
    nonnegative messages, products, and marginal numerators, even before global
    constraints restrict a join. Free variables are multiplied as Python integers.
    """
    owned = [v for leaf in compiled.leaves for v in leaf.original_vars]
    bound = prod(max(1, len(leaf.key_idx)) for leaf in compiled.leaves)
    bound *= prod(max(1, *weights[v]) for v in owned)
    primes = () if bound <= _INT64_MAX else primes_for_bound(bound)
    work = (len(primes) or 1) * sum(
        len(n.key_idx) * (2 * len(n.original_vars) + 3)
        if n.left < 0
        else 6 * len(n.pair_left)
        for n in compiled.nodes
    )
    if work > compiled.max_work:
        raise ResourceLimitError("exact weighted contraction exceeds max_work")
    memory = (
        sum(
            n.key_count * 3
            + (len(n.key_idx) * 3 if n.left < 0 else 4 * len(n.pair_left))
            for n in compiled.nodes
        )
        * 8
    )
    if memory > compiled.max_memory_bytes:
        raise ResourceLimitError("exact weighted workspace exceeds max_memory_bytes")
    if primes:
        residues = [_pass(compiled, weights, p) for p in primes]
        values = [crt_reconstruct(list(column), primes) for column in zip(*residues)]
    else:
        values = _pass(compiled, weights, None)
    core, numerators = values[0], values[1:]
    free = prod(sum(weights[v]) for v in compiled.free_vars)
    total = prod(sum(pair) for pair in weights)
    z = Fraction(core * free, denominator)
    acceptance = Fraction(core * free, total) if total else None
    marginal = [None] * compiled.original_num_vars
    if core and free:
        for var in owned:
            marginal[var] = Fraction(numerators[var], core)
        for var in compiled.free_vars:
            marginal[var] = Fraction(weights[var][1], sum(weights[var]))
    logz = log(z.numerator) - log(z.denominator) if z else -np.inf
    kl = (
        -(log(acceptance.numerator) - log(acceptance.denominator))
        if acceptance
        else float("nan")
    )
    compiled.last_evaluation = {
        "mode": "exact_rational",
        "arithmetic": "expandable_crt" if primes else "bounded_int64",
        "primes": primes,
        "integer_bound": bound,
        "upward_downward_passes": len(primes) or 1,
    }
    compiled.stats.arithmetic = compiled.last_evaluation["arithmetic"]
    compiled.stats.primes = primes
    from .stage5_wmc_guardrail import WMCResult

    return WMCResult(z, marginal, logz, acceptance, kl)

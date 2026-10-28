"""Exact counting across a small clause cut by inclusion-exclusion.

A balanced two-way spectral partition assigns every variable to one side. A
clause whose variables all lie on one side stays there. The remaining
``bridge`` clauses ``B_j = L_j or R_j`` are removed and restored exactly:

    #F = sum over S of (-1)**|S| * #(F_L and not L_S) * #(F_R and not R_S),

where ``not L_S`` forces every left literal of the bridges in ``S`` false.
Each side is counted once by a component-caching search whose results are
factors keyed by that side's bridge variables, so every conditioned term is a
sum of factor entries rather than another search. Counts are Python integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import sys

from .cnf import CNF
from .stage1_sparsify import _spectral_groups
from .stage3_gpu_engine import ResourceLimitError

# Bytes charged per cached component: two integer keys, the stored count or
# factor and dictionary overhead. The charge bounds the cache, not exact RSS.
_CACHE_ENTRY_BYTES = 256


@dataclass(frozen=True)
class BridgeCut:
    """Two sides, their internal clauses and the clauses crossing the cut."""

    groups: tuple[tuple[int, ...], tuple[int, ...]]
    side_clauses: tuple[tuple[tuple[tuple[int, bool], ...], ...], ...]
    bridges: tuple[tuple[tuple[int, int, bool], ...], ...]


def find_bridge_cut(cnf: CNF, max_bridges: int, groups=None) -> BridgeCut | None:
    """Return the spectral two-way cut if at most ``max_bridges`` clauses cross it.

    Tautologies and repeated literals are removed first. Literals are stored as
    (side-local variable, polarity); bridge literals also record their side.
    """
    if cnf.num_vars < 2:
        return None
    groups = tuple(tuple(g) for g in (groups or _spectral_groups(cnf, 2)))
    owner = {v: g for g, variables in enumerate(groups) for v in variables}
    local = {v: i for variables in groups for i, v in enumerate(variables)}
    sides = ([], [])
    bridges = []
    for clause in cnf.clauses:
        literals = {(lit.var, lit.pos) for lit in clause.literals}
        if any((v, not pos) in literals for v, pos in literals):
            continue
        touched = {owner[v] for v, _ in literals}
        if len(touched) == 2:
            bridges.append(
                tuple(sorted((owner[v], local[v], pos) for v, pos in literals))
            )
            if len(bridges) > max_bridges:
                return None
        else:
            side = touched.pop() if touched else 0
            sides[side].append(tuple(sorted((local[v], pos) for v, pos in literals)))
    return BridgeCut(groups, (tuple(sides[0]), tuple(sides[1])), tuple(bridges))


def _expand(factor, free_boundary):
    """Duplicate factor entries for unconstrained boundary variables."""
    while free_boundary:
        bit = free_boundary & -free_boundary
        free_boundary ^= bit
        expanded = {}
        for key, count in factor.items():
            expanded[key] = expanded.get(key, 0) + count
            expanded[key | bit] = expanded.get(key | bit, 0) + count
        factor = expanded
    return factor


def _multiply(a, b):
    """Product of counts or factors over disjoint boundary variables."""
    if isinstance(a, int):
        if isinstance(b, int):
            return a * b
        a, b = b, a
    if isinstance(b, int):
        return {key: b * count for key, count in a.items()} if b else 0
    return {ka | kb: ca * cb for ka, ca in a.items() for kb, cb in b.items()}


class ComponentCounter:
    """Component-caching model counter on bit-parallel clause sets.

    A residual formula is the alive-clause mask plus its unassigned variables;
    every assigned literal of an alive clause is false. Components without
    boundary variables return integer counts. Components containing boundary
    variables return ``{true-boundary-mask: count}`` factors.
    """

    def __init__(self, num_vars, clauses, boundary_mask=0, *, max_work, max_memory_bytes):
        self.num_vars = num_vars
        self.boundary = boundary_mask
        self.max_work, self.max_memory_bytes = max_work, max_memory_bytes
        self.unsat = any(not clause for clause in clauses)
        self.clause_vars = [sum(1 << v for v, _ in clause) for clause in clauses]
        self.positive = [0] * num_vars
        self.negative = [0] * num_vars
        for index, clause in enumerate(clauses):
            for var, pos in clause:
                if pos:
                    self.positive[var] |= 1 << index
                else:
                    self.negative[var] |= 1 << index
        self.occurrences = [p | q for p, q in zip(self.positive, self.negative)]
        self.all_clauses = (1 << len(clauses)) - 1
        self.cache = {}
        self.nodes = 0
        self.work = 0

    def _assign(self, alive, assigned, true, queue):
        """Unit-propagate queued (var, value) literals; None on conflict."""
        positive, negative, clause_vars = self.positive, self.negative, self.clause_vars
        while queue:
            var, value = queue.pop()
            bit = 1 << var
            if assigned & bit:
                if bool(true & bit) != value:
                    return None
                continue
            assigned |= bit
            if value:
                true |= bit
                alive &= ~positive[var]
                falsified = negative[var] & alive
            else:
                alive &= ~negative[var]
                falsified = positive[var] & alive
            while falsified:
                low = falsified & -falsified
                falsified ^= low
                index = low.bit_length() - 1
                rest = clause_vars[index] & ~assigned
                if not rest:
                    return None
                if not rest & (rest - 1):
                    unit = rest.bit_length() - 1
                    queue.append((unit, bool((positive[unit] >> index) & 1)))
        return alive, assigned, true

    def _components(self, alive, assigned):
        clause_vars, occurrences = self.clause_vars, self.occurrences
        unassigned = ~assigned
        components = []
        while alive:
            low = alive & -alive
            members, variables = low, 0
            todo = clause_vars[low.bit_length() - 1] & unassigned
            while todo:
                variables |= todo
                adjacent = 0
                while todo:
                    bit = todo & -todo
                    todo ^= bit
                    adjacent |= occurrences[bit.bit_length() - 1]
                adjacent &= alive & ~members
                members |= adjacent
                while adjacent:
                    clause = adjacent & -adjacent
                    adjacent ^= clause
                    todo |= clause_vars[clause.bit_length() - 1]
                todo &= unassigned & ~variables
            alive &= ~members
            components.append((members, variables))
        return components

    def _product(self, alive, assigned):
        """Multiply the counts of all residual components; returns (value, covered)."""
        value, covered = 1, 0
        for members, variables in self._components(alive, assigned):
            covered |= variables
            value = _multiply(value, self._solve(members, variables, assigned))
            if not value:
                return 0, covered
        return value, covered

    def _solve(self, members, variables, assigned):
        key = (members, variables)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.nodes += 1
        self.work += members.bit_count() + variables.bit_count()
        if self.work > self.max_work:
            raise ResourceLimitError("bridge-cut component search exceeds max_work")
        if (len(self.cache) + 1) * _CACHE_ENTRY_BYTES > self.max_memory_bytes:
            raise ResourceLimitError("bridge-cut component cache exceeds max_memory_bytes")
        occurrences = self.occurrences
        branch, best = -1, -1
        remaining = variables
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            var = bit.bit_length() - 1
            frequency = (occurrences[var] & members).bit_count()
            if frequency > best:
                branch, best = var, frequency
        boundary = variables & self.boundary
        total = {} if boundary else 0
        for value in (True, False):
            state = self._assign(members, assigned, 0, [(branch, value)])
            if state is None:
                continue
            alive, now, true = state
            sub, covered = self._product(alive, now)
            if not sub:
                continue
            free = variables & ~now & ~covered
            sub = _multiply(sub, 1 << (free & ~self.boundary).bit_count())
            if boundary:
                fixed = true & boundary
                sub = _expand(sub if isinstance(sub, dict) else {0: sub}, free & self.boundary)
                for k, count in sub.items():
                    total[k | fixed] = total.get(k | fixed, 0) + count
            else:
                total += sub
        if boundary and not total:
            total = 0
        self.cache[key] = total
        return total

    def factor(self):
        """Return {true-boundary-mask: count} over all side variables."""
        if self.unsat:
            return {}
        state = self._assign(self.all_clauses, 0, 0, [])
        if state is None:
            return {}
        alive, assigned, true = state
        value, covered = self._product(alive, assigned)
        if not value:
            return {}
        free = ((1 << self.num_vars) - 1) & ~assigned & ~covered
        value = _multiply(value, 1 << (free & ~self.boundary).bit_count())
        value = _expand(value if isinstance(value, dict) else {0: value}, free & self.boundary)
        fixed = true & self.boundary
        return {key | fixed: count for key, count in value.items()}


def bridge_cut_count(cut: BridgeCut, *, max_work: int = 2_000_000_000,
                     max_memory_bytes: int = 512 * 1024 * 1024) -> tuple[int, dict]:
    """Count a formula from its bridge cut; returns (count, telemetry)."""
    depth = max(len(g) for g in cut.groups)
    # Each decision level uses at most two Python frames.
    if 2 * depth + 64 > sys.getrecursionlimit():
        raise ResourceLimitError("bridge-cut side exceeds the recursion depth bound")
    masks = [0, 0]
    for bridge in cut.bridges:
        for side, var, _ in bridge:
            masks[side] |= 1 << var
    counters = [
        ComponentCounter(len(cut.groups[g]), cut.side_clauses[g], masks[g],
                         max_work=max_work, max_memory_bytes=max_memory_bytes)
        for g in range(2)
    ]
    factors = []
    for counter in counters:
        factors.append(counter.factor())
        max_work -= counter.work
        max_memory_bytes -= len(counter.cache) * _CACHE_ENTRY_BYTES
        for later in counters[len(factors):]:
            later.max_work, later.max_memory_bytes = max_work, max_memory_bytes
    total, terms = 0, 0
    for size in range(len(cut.bridges) + 1):
        for subset in combinations(cut.bridges, size):
            term = -1 if size % 2 else 1
            for side in range(2):
                forced_true = forced_false = 0
                for bridge in subset:
                    for owner, var, pos in bridge:
                        if owner == side:
                            # The bridge is violated only if this literal is false.
                            if pos:
                                forced_false |= 1 << var
                            else:
                                forced_true |= 1 << var
                if forced_true & forced_false:
                    term = 0
                    break
                term *= sum(count for key, count in factors[side].items()
                            if not key & forced_false and key & forced_true == forced_true)
                if not term:
                    break
            terms += 1
            total += term
    telemetry = {
        "bridge_clauses": len(cut.bridges),
        "side_variable_counts": [len(g) for g in cut.groups],
        "bridge_variables": [m.bit_count() for m in masks],
        "factor_sizes": [len(f) for f in factors],
        "search_nodes": [c.nodes for c in counters],
        "cache_entries": [len(c.cache) for c in counters],
        "inclusion_exclusion_terms": terms,
    }
    return total, telemetry

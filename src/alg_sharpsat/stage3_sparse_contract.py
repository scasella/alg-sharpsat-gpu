"""Sparse natural joins of exact local boundary factors.

Keys contain BOTH master and proxy values, with each proxy renamed to its
master. A shared variable survives a join until its last incident factor has
been included. No dense source-by-proxy matrix or reference solver is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from time import perf_counter

import numpy as np
import torch

from .cnf import CNF
from .stage1_sparsify import PartitionedCNF, _spectral_groups, partition_and_decouple_boundary, reduce_clause_width_deterministic, split_high_degree_variables
from .stage2_algebraic import crt_reconstruct, primes_for_bound
from .stage3_bridge_cut import bridge_cut_count, find_bridge_cut
from .stage3_gpu_engine import ResourceLimitError, UnsupportedDeviceError, select_device


@dataclass
class SparseFactor:
    scope: int
    keys: torch.Tensor
    counts: torch.Tensor


@dataclass
class SparseStats:
    device: str
    device_fallbacks: list[str] = field(default_factory=list)
    boundary_links: int = 0
    boundary_variables: int = 0
    local_assignment_counts: list[int] = field(default_factory=list)
    joint_signature_counts: list[int] = field(default_factory=list)
    group_variable_counts: list[int] = field(default_factory=list)
    original_group_variable_counts: list[int] = field(default_factory=list)
    free_variables: int = 0
    active_local_variables: int = 0
    join_pair_counts: list[int] = field(default_factory=list)
    intermediate_rows: list[int] = field(default_factory=list)
    peak_intermediate_rows: int = 0
    semi_join_pushdown: bool = True
    phase1_assignment_counts: list[int] = field(default_factory=list)
    feasible_master_mask_counts: list[int] = field(default_factory=list)
    peak_frontier_rows: list[int] = field(default_factory=list)
    peak_temporary_rows: list[int] = field(default_factory=list)
    peak_joint_rows: int = 0
    preprocessing_seconds: float = 0.0
    build_seconds: float = 0.0
    contraction_seconds: float = 0.0
    primes: tuple[int, ...] = ()
    arithmetic: str = "bounded_int64"
    selected_engine: str = "sparse_contract"
    selection_reason: str = "explicit sparse contraction"
    bridge_cut: dict = field(default_factory=dict)
    plan_selection: dict = field(default_factory=dict)


class SparseCounter:
    def __init__(self, partitioned: PartitionedCNF, device: str = "auto", *,
                 max_local_assignments: int = 4_000_000,
                 max_join_pairs: int = 8_000_000,
                 max_memory_bytes: int = 512 * 1024 * 1024,
                 assignment_chunk_size: int = 65536, max_work: int = 2_000_000_000,
                 semi_join_pushdown: bool = True):
        for name, value in locals().copy().items():
            if name.startswith("max_") or name == "assignment_chunk_size":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        if type(semi_join_pushdown) is not bool:
            raise ValueError("semi_join_pushdown must be a bool")
        self.partitioned = partitioned
        self.device, reasons = select_device(device)
        # searchsorted is essential to the sparse path, beyond the v1 probe.
        try:
            probe = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
            if torch.searchsorted(probe, probe).cpu().tolist() != [0, 1]:
                raise RuntimeError("searchsorted returned incorrect indices")
        except (RuntimeError, NotImplementedError) as exc:
            if device != "auto":
                raise UnsupportedDeviceError(f"sparse key search unavailable on {device}") from exc
            reasons.append(f"{self.device} sparse search unavailable: {exc}")
            self.device = torch.device("cpu")
        self.max_local_assignments, self.max_join_pairs = max_local_assignments, max_join_pairs
        self.max_memory_bytes, self.max_work = max_memory_bytes, max_work
        self.assignment_chunk_size = assignment_chunk_size
        self.stats = SparseStats(str(self.device), reasons, boundary_links=len(partitioned.boundary_links))
        masters = sorted({link.master_var for link in partitioned.boundary_links})
        if len(masters) > 63:
            raise ResourceLimitError("sparse keys support at most 63 distinct boundary masters")
        self.bits = {var: i for i, var in enumerate(masters)}
        self.alias = {link.proxy_var: link.master_var for link in partitioned.boundary_links}
        self._proxy_sources = {link.proxy_var: link.master_group for link in partitioned.boundary_links}
        self.stats.boundary_variables = len(masters)
        self.stats.group_variable_counts = [len(g) for g in partitioned.group_vars]
        self.stats.original_group_variable_counts = [sum(v < partitioned.original_num_vars for v in g)
                                                    for g in partitioned.group_vars]
        group_count = len(partitioned.group_vars)
        self.stats.semi_join_pushdown = semi_join_pushdown
        self.stats.peak_frontier_rows = [0] * group_count
        self.stats.peak_temporary_rows = [0] * group_count
        self._local_work = [0] * group_count
        self._staged_rows = 0
        start = perf_counter()
        self._layouts = [self._layout(g) for g in range(group_count)]
        self.factors = self._semijoin_factors() if semi_join_pushdown else [self._local(g) for g in range(group_count)]
        self.stats.peak_joint_rows = max(self.stats.joint_signature_counts, default=0)
        self.stats.build_seconds = perf_counter() - start

    def _budget(self, rows: int, bytes_per_row: int = 64) -> None:
        if rows * bytes_per_row > self.max_memory_bytes:
            raise ResourceLimitError("sparse tensor workspace exceeds max_memory_bytes")

    def _layout(self, group: int):
        p = self.partitioned
        boundary = {v: self.bits[self.alias.get(v, v)] for v in p.group_vars[group]
                    if self.alias.get(v, v) in self.bits}
        scope = sum(1 << b for b in set(boundary.values()))
        clauses = []
        for clause in p.group_clauses[group]:
            literals = {(lit.var, lit.pos) for lit in clause.literals}
            if not any((v, not sign) in literals for v, sign in literals):
                clauses.append(tuple(sorted(literals)))
        active = set(boundary) | {v for c in clauses for v, _ in c}
        self.stats.free_variables += len(p.group_vars[group]) - len(active)
        self.stats.active_local_variables += len(active)
        if len(active) > 63:
            raise ResourceLimitError("local sparse truth-table keys support at most 63 active variables")
        frequency = {v: sum(v == u for c in clauses for u, _ in c) for v in active}
        order = sorted(active, key=lambda v: (-frequency[v], v))
        direct = len(self.bits) + len(active - boundary.keys()) <= 63
        if direct:
            positions = dict(boundary)
            positions.update((v, len(self.bits) + i)
                             for i, v in enumerate(v for v in order if v not in boundary))
        else:
            positions = {v: i for i, v in enumerate(order)}
        if self.stats.semi_join_pushdown:
            # Give each incoming source a consecutive bit register, in its
            # proxy branching order. Prefix membership can then use a small
            # one-dimensional Boolean lookup instead of binary searches.
            proxies = sorted((v for v in order if v in self.alias),
                             key=lambda v: (self._proxy_sources[v], -frequency[v], v))
            positions = {v: i for i, v in enumerate(proxies + [v for v in order if v not in self.alias])}
            direct = False
        return dict(boundary=boundary, scope=scope, clauses=clauses, order=order,
                    positions=positions, direct=direct)

    def _zero_frontier(self, empty=False):
        return np.zeros(0 if empty else 1, dtype=np.int64) if self.device.type == "cpu" else torch.zeros(
            0 if empty else 1, dtype=torch.int64, device=self.device)

    def _charge(self, group, rows, cost):
        self._local_work[group] += rows * cost
        if self._local_work[group] > self.max_work:
            raise ResourceLimitError("local sparse enumeration exceeds max_work")

    def _accepted(self, group, rows):
        if rows > self.max_local_assignments:
            raise ResourceLimitError("local sparse frontier exceeds max_local_assignments")
        self._budget(rows + self._staged_rows, 32)
        self.stats.peak_frontier_rows[group] = max(self.stats.peak_frontier_rows[group], rows)

    @staticmethod
    def _patterns(clauses, positions):
        return [(sum(1 << positions[v] for v, _ in clause),
                 sum(1 << positions[v] for v, sign in clause if not sign)) for clause in clauses]

    def _enumerate_bits(self, group, order, clauses, positions):
        ranks = {v: i for i, v in enumerate(order)}
        completed = [[] for _ in order]
        for clause in clauses:
            if clause:
                completed[max(ranks[v] for v, _ in clause)].append(clause)
        frontier = self._zero_frontier(any(not clause for clause in clauses))
        self._accepted(group, len(frontier))
        cpu = self.device.type == "cpu"
        for i, var in enumerate(order):
            self._budget(len(frontier) * 2, 32)
            patterns = self._patterns(completed[i], positions)
            self._charge(group, len(frontier) * 2, 1 + sum(map(len, completed[i])))
            pieces, survivors = [], 0
            for offset in range(0, len(frontier), self.assignment_chunk_size):
                old = frontier[offset:offset + self.assignment_chunk_size]
                bit = 1 << positions[var]
                candidate = np.concatenate((old, old | bit)) if cpu else torch.cat((old, old | bit))
                self.stats.peak_temporary_rows[group] = max(self.stats.peak_temporary_rows[group], len(candidate))
                for clause_mask, false_bits in patterns:
                    candidate = candidate[(candidate & clause_mask) != false_bits]
                survivors += len(candidate)
                self._accepted(group, survivors)
                pieces.append(candidate)
            frontier = (np.concatenate(pieces) if cpu else torch.cat(pieces)) if pieces else frontier[:0]
        return frontier

    def _signature_keys(self, frontier, layout):
        cpu = self.device.type == "cpu"
        dtype = np.uint32 if len(self.bits) <= 32 else np.int64
        if layout['direct']:
            return (frontier & layout['scope']).astype(dtype) if cpu else frontier & layout['scope']
        keys = np.zeros(len(frontier), dtype=dtype) if cpu else torch.zeros_like(frontier)
        for var, bit in layout['boundary'].items():
            values = ((frontier >> layout['positions'][var]) & 1) << bit
            keys |= values.astype(dtype) if cpu else values
        return keys

    def _finish(self, group, frontier):
        layout = self._layouts[group]
        keys = self._signature_keys(frontier, layout)
        if self.device.type == "cpu":
            keys.sort()
            starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.empty(0, dtype=np.int64)
            unique = keys[starts].astype(np.int64)
            counts = np.diff(np.r_[starts, len(keys)]) if len(keys) else np.empty(0, dtype=np.int64)
            unique, counts = torch.from_numpy(unique), torch.from_numpy(counts)
        else:
            unique, counts = torch.unique(keys, sorted=True, return_counts=True)
        self._budget(len(unique), 24)
        self.stats.local_assignment_counts.append(len(frontier))
        self.stats.joint_signature_counts.append(len(unique))
        return SparseFactor(layout['scope'], unique, counts)

    def _local(self, group: int) -> SparseFactor:
        layout = self._layouts[group]
        frontier = self._enumerate_bits(group, layout['order'], layout['clauses'], layout['positions'])
        return self._finish(group, frontier)

    def _distinct(self, keys):
        if self.device.type != "cpu":
            return torch.unique(keys, sorted=True)
        # Sort-based: NumPy's hash-based unique can degrade by ~40x on
        # structured bitmask keys whose low bits are mostly zero.
        keys = np.sort(keys)
        return keys[np.r_[True, keys[1:] != keys[:-1]]] if len(keys) else keys

    def _semijoin_factors(self):
        # First obtain every source's support under its owned-only clauses.
        # Supports contain distinct master tuples, never source multiplicities.
        phase1, supports = [], []
        for group, layout in enumerate(self._layouts):
            owned = [v for v in layout['order'] if v not in self.alias]
            internal = [clause for clause in layout['clauses']
                        if all(v not in self.alias for v, _ in clause)]
            frontier = self._enumerate_bits(group, owned, internal, layout['positions'])
            self._budget(self._staged_rows + 2 * len(frontier), 32)
            phase1.append(frontier)
            outgoing = sum(1 << self.bits[v] for v in owned if v in self.bits)
            support = self._distinct(self._signature_keys(frontier, layout) & outgoing)
            supports.append(support)
            self._staged_rows += len(frontier) + len(support)
            self.stats.phase1_assignment_counts.append(len(frontier))
            self.stats.feasible_master_mask_counts.append(len(support))
        factors = []
        cpu = self.device.type == "cpu"
        for group, layout in enumerate(self._layouts):
            incoming = {}
            for link in self.partitioned.boundary_links:
                if link.proxy_group == group:
                    incoming.setdefault(link.master_group, []).append(link.proxy_var)
            projections = []
            for source, variables in incoming.items():
                scope = sum(1 << self.bits[self.alias[v]] for v in variables)
                projected = self._distinct(supports[source] & scope)
                if layout['direct']:
                    encoded = projected.astype(np.int64) if cpu else projected
                else:
                    encoded = np.zeros(len(projected), dtype=np.int64) if cpu else torch.zeros_like(projected)
                    projected = projected.astype(np.int64) if cpu else projected
                    for var in variables:
                        encoded |= ((projected >> self.bits[self.alias[var]]) & 1) << layout['positions'][var]
                projections.append((source, encoded))
            source_for_proxy = {link.proxy_var: link.master_group for link in self.partitioned.boundary_links
                                if link.proxy_group == group}
            encoded_by_source = dict(projections)
            bases = {source: min(layout['positions'][v] for v in variables)
                     for source, variables in incoming.items()}
            order = [v for v in layout['order'] if v in self.alias]
            ranks = {v: i for i, v in enumerate(order)}
            completed = [[] for _ in order]
            for clause in layout['clauses']:
                proxies = [v for v, _ in clause if v in self.alias]
                if proxies:
                    completed[max(ranks[v] for v in proxies)].append(clause)
            frontier = phase1[group]
            prefix_masks, prefix_sizes = {}, {}
            for i, var in enumerate(order):
                source = source_for_proxy[var]
                bit = 1 << layout['positions'][var]
                prefix_masks[source] = prefix_masks.get(source, 0) | bit
                prefix_sizes[source] = prefix_sizes.get(source, 0) + 1
                prefix = prefix_masks[source]
                allowed = self._distinct(encoded_by_source[source] & prefix)
                constrained = len(allowed) != 1 << prefix_sizes[source]
                if not len(allowed):
                    frontier = frontier[:0]
                    break
                patterns = self._patterns(completed[i], layout['positions'])
                lookup = None
                lookup_size = 1 << prefix_sizes[source]
                if constrained and lookup_size <= min(1 << 20, self.max_memory_bytes // 4):
                    self._budget(self._staged_rows * 32 + lookup_size, 1)
                    self._charge(group, lookup_size, 1)
                    lookup = np.zeros(lookup_size, dtype=bool) if cpu else torch.zeros(
                        lookup_size, dtype=torch.bool, device=self.device)
                    lookup[allowed >> bases[source]] = True
                self._charge(group, len(frontier) * 2,
                             1 + sum(map(len, completed[i])) + int(constrained))
                self._budget(len(frontier) * 2, 32)
                pieces, survivors = [], 0
                for start in range(0, len(frontier), self.assignment_chunk_size):
                    old = frontier[start:start + self.assignment_chunk_size]
                    candidate = np.concatenate((old, old | bit)) if cpu else torch.cat((old, old | bit))
                    self.stats.peak_temporary_rows[group] = max(self.stats.peak_temporary_rows[group], len(candidate))
                    for clause_mask, false_bits in patterns:
                        candidate = candidate[(candidate & clause_mask) != false_bits]
                    if constrained:
                        projected = candidate & prefix
                        if lookup is not None:
                            valid = lookup[projected >> bases[source]]
                        elif cpu:
                            indices = np.searchsorted(allowed, projected)
                            valid = indices < len(allowed)
                            valid &= allowed[np.minimum(indices, len(allowed)-1)] == projected
                        else:
                            indices = torch.searchsorted(allowed, projected)
                            valid = indices < len(allowed)
                            valid &= allowed[indices.clamp(max=len(allowed)-1)] == projected
                        candidate = candidate[valid]
                    survivors += len(candidate)
                    self._accepted(group, survivors)
                    pieces.append(candidate)
                frontier = (np.concatenate(pieces) if cpu else torch.cat(pieces)) if pieces else frontier[:0]
            self._staged_rows -= len(phase1[group])
            phase1[group] = None
            factor = self._finish(group, frontier)
            self._staged_rows += len(factor.keys)
            factors.append(factor)
        self._staged_rows = 0
        return factors

    def _reduce(self, keys: torch.Tensor, values: torch.Tensor, scope: int,
                prime: int | None) -> SparseFactor:
        keys = keys & scope
        unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        result = torch.zeros(len(unique), dtype=torch.int64, device=self.device)
        if prime is None:
            result.index_add_(0, inverse, values)
        else:
            # Every scatter batch sums at most 2**20 already reduced residues.
            for start in range(0, len(values), 1 << 20):
                scratch = torch.zeros_like(result)
                scratch.index_add_(0, inverse[start:start + (1 << 20)], values[start:start + (1 << 20)])
                result = (result + scratch.remainder(prime)).remainder(prime)
        return SparseFactor(scope, unique, result)

    def _join(self, left: SparseFactor, right: SparseFactor, keep: int,
              prime: int | None, record: bool) -> SparseFactor:
        shared = left.scope & right.scope
        # Fully shared scopes are unique already: match sorted signatures.
        if shared == left.scope == right.scope:
            if len(left.keys) > len(right.keys):
                left, right = right, left
            if self.device.type == "cpu":
                # Avoid threading/dispatch overhead for host key searches.
                left_keys, right_keys = left.keys.numpy(), right.keys.numpy()
                if len(left_keys) >= 4096:
                    # Both inputs are sorted. Restrict each probe block to
                    # its key interval so searches stay in smaller cacheable
                    # slices of the larger table.
                    indices = np.empty(len(left_keys), dtype=np.int64)
                    for start in range(0, len(left_keys), 256):
                        block = left_keys[start:start + 256]
                        lo = np.searchsorted(right_keys, block[0])
                        hi = np.searchsorted(right_keys, block[-1], side="right")
                        indices[start:start + 256] = np.searchsorted(right_keys[lo:hi], block) + lo
                else:
                    indices = np.searchsorted(right_keys, left_keys)
                li = np.flatnonzero(indices < len(right_keys))
                ri = indices[li]
                equal = right_keys[ri] == left_keys[li]
                li, ri = torch.from_numpy(li[equal]), torch.from_numpy(ri[equal])
            else:
                indices = torch.searchsorted(right.keys, left.keys)
                valid = indices < len(right.keys)
                li = torch.arange(len(left.keys), device=self.device)[valid]
                ri = indices[valid]
                equal = right.keys[ri] == left.keys[li]
                li, ri = li[equal], ri[equal]
        else:
            lprojection, rprojection = left.keys & shared, right.keys & shared
            rsort = torch.argsort(rprojection)
            sorted_r = rprojection[rsort].contiguous()
            lo = torch.searchsorted(sorted_r, lprojection.contiguous())
            hi = torch.searchsorted(sorted_r, lprojection.contiguous(), right=True)
            lengths = hi - lo
            pairs = int(lengths.sum().cpu())
            if pairs > self.max_join_pairs:
                raise ResourceLimitError(f"sparse join needs {pairs} pairs; max_join_pairs is {self.max_join_pairs}")
            self._budget(pairs, 96)
            li = torch.repeat_interleave(torch.arange(len(left.keys), device=self.device), lengths)
            starts = torch.cumsum(lengths, 0) - lengths
            offset = torch.arange(pairs, device=self.device) - torch.repeat_interleave(starts, lengths)
            ri = rsort[lo[li] + offset]
        if len(li) > self.max_join_pairs:
            raise ResourceLimitError("sparse intersection exceeds max_join_pairs")
        self._budget(len(li), 96)
        values = left.counts[li] * right.counts[ri]
        if prime is not None:
            values = values.remainder(prime)
        output = self._reduce(left.keys[li] | right.keys[ri], values, keep, prime)
        if record:
            self.stats.join_pair_counts.append(len(li))
            self.stats.intermediate_rows.append(len(output.keys))
            self.stats.peak_intermediate_rows = max(self.stats.peak_intermediate_rows, len(output.keys))
            self.stats.peak_joint_rows = max(self.stats.peak_joint_rows, len(output.keys))
        return output

    def _contract(self, prime: int | None, record: bool) -> int:
        factors = [SparseFactor(f.scope, f.keys, f.counts if prime is None else f.counts.remainder(prime))
                   for f in self.factors]
        if any(len(f.keys) == 0 for f in factors):
            return 0
        while len(factors) > 1:
            # Prefer overlap, then small pair tables. Deterministic hierarchy.
            i, j = min(((i, j) for i in range(len(factors)) for j in range(i + 1, len(factors))),
                       key=lambda ij: (-(factors[ij[0]].scope & factors[ij[1]].scope).bit_count(),
                                       len(factors[ij[0]].keys) * len(factors[ij[1]].keys), ij))
            remaining = [f for k, f in enumerate(factors) if k not in (i, j)]
            needed = 0
            for factor in remaining:
                needed |= factor.scope
            keep = (factors[i].scope | factors[j].scope) & needed
            joined = self._join(factors[i], factors[j], keep, prime, record)
            if not len(joined.keys):
                return 0
            factors = remaining + [joined]
        result = 0
        for start in range(0, len(factors[0].counts), 1 << 20):
            result += int(factors[0].counts[start:start + (1 << 20)].sum().cpu())
            if prime is not None:
                result %= prime
        return result

    def count(self) -> int:
        start = perf_counter()
        self.stats.join_pair_counts.clear()
        self.stats.intermediate_rows.clear()
        # All intermediate weights count assignments to disjoint local tables.
        # Their total is <=2**sum(active local dimensions), including proxies.
        if self.stats.active_local_variables <= 62:
            result = self._contract(None, True)
        else:
            self.stats.arithmetic = "modular_crt"
            self.stats.primes = primes_for_bound(1 << self.partitioned.original_num_vars)
            residues = [self._contract(p, i == 0) for i, p in enumerate(self.stats.primes)]
            result = crt_reconstruct(residues, self.stats.primes)
        result <<= self.stats.free_variables
        self.stats.contraction_seconds = perf_counter() - start
        return result


def _partition(cnf: CNF, num_groups: int, block_size: int, max_degree: int | None, max_width: int,
               groups=None):
    transformed = reduce_clause_width_deterministic(cnf, max_width)
    if max_degree is not None:
        transformed = split_high_degree_variables(transformed, max_degree)
    if transformed.num_vars != cnf.num_vars:
        groups = None  # auxiliaries change the graph; partition it afresh
    return partition_and_decouple_boundary(transformed, num_groups, block_size, groups=groups)


_COUNTER_OPTIONS = {"max_local_assignments", "max_join_pairs", "max_memory_bytes",
                    "assignment_chunk_size", "max_work", "semi_join_pushdown"}


def sparse_count_with_stats(cnf: CNF, *, device: str = "auto", num_groups: int = 2,
                            block_size: int = 4, max_degree: int | None = None,
                            max_width: int = 3, bridge_cut: bool | str = False,
                            max_bridge_clauses: int = 6, **options) -> tuple[int, SparseStats]:
    """Count exactly by sparse contraction, or by bridge-cut search when requested.

    ``bridge_cut=False`` (the default) always contracts. ``"auto"`` counts by
    inclusion-exclusion whenever the two-way spectral bisection is crossed by at
    most ``max_bridge_clauses`` clauses; on the coupled fixtures that was at
    least as fast as contraction at every size from 18 to 100 variables.
    ``True`` requires such a cut. Bridge-cut search runs on the CPU with Python integers and honors
    ``max_work`` and ``max_memory_bytes``. The selected engine and reason are
    recorded in the statistics.
    """
    if bridge_cut not in ("auto", True, False):
        raise ValueError("bridge_cut must be 'auto', True or False")
    if type(max_bridge_clauses) is not int or max_bridge_clauses < 0:
        raise ValueError("max_bridge_clauses must be a nonnegative integer")
    start = perf_counter()
    groups = None
    if bridge_cut is not False and num_groups == 2 and max_degree is None and cnf.num_vars >= 2:
        groups = _spectral_groups(cnf, 2)
        cut = find_bridge_cut(cnf, max_bridge_clauses, groups)
        if bridge_cut is True and cut is None:
            raise ValueError(f"no two-way cut with at most {max_bridge_clauses} bridge clauses")
        if cut is not None:
            preprocessing = perf_counter() - start
            unknown = set(options) - _COUNTER_OPTIONS
            if unknown:
                raise TypeError(f"unexpected sparse counting options: {sorted(unknown)}")
            budgets = {k: options[k] for k in ("max_work", "max_memory_bytes") if k in options}
            for name, value in budgets.items():
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            begin = perf_counter()
            result, telemetry = bridge_cut_count(cut, **budgets)
            stats = SparseStats("cpu", selected_engine="bridge_cut", arithmetic="python_int",
                                selection_reason=(f"{len(cut.bridges)} bridge clauses <= {max_bridge_clauses}; "
                                                  f"larger side {max(telemetry['side_variable_counts'])} variables"),
                                semi_join_pushdown=False, bridge_cut=telemetry,
                                group_variable_counts=telemetry["side_variable_counts"],
                                original_group_variable_counts=telemetry["side_variable_counts"])
            stats.preprocessing_seconds = preprocessing
            stats.contraction_seconds = perf_counter() - begin
            return result, stats
    partitioned = _partition(cnf, num_groups, block_size, max_degree, max_width, groups)
    preprocessing = perf_counter() - start
    counter = SparseCounter(partitioned, device, **options)
    result = counter.count()
    counter.stats.preprocessing_seconds = preprocessing
    return result, counter.stats


def contract_sharp_sat(cnf: CNF, **options) -> int:
    return sparse_count_with_stats(cnf, **options)[0]


def auto_count_with_stats(cnf: CNF, **options):
    # Direct contraction is allowed for B<=4 as well. It avoids interpolation
    # at either size. A two-way cut of at most max_bridge_clauses clauses
    # selects bridge-cut search instead; the reason is explicit in telemetry.
    options.setdefault("bridge_cut", "auto")
    result, stats = sparse_count_with_stats(cnf, **options)
    if stats.selected_engine == "sparse_contract":
        stats.selection_reason = ("B>4: avoid static interpolation overhead" if stats.boundary_links > 4
                                  else "B<=4: direct sparse contraction")
    return result, stats


def auto_sharp_sat(cnf: CNF, **options) -> int:
    return auto_count_with_stats(cnf, **options)[0]

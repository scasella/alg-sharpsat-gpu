"""Exact tensor evaluation of the partitioned algebraic model counter.

Local truth tables are built once, using batched bit operations and clause
masks. They are compressed by their *source-group* master bits and incoming
proxy bits. There is no DPLL or whole-formula reference-counter fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod
from time import perf_counter

import torch

from .cnf import CNF
from .stage1_sparsify import PartitionedCNF
from .stage2_algebraic import (
    SuppressionSchedule,
    create_suppression_schedule, crt_reconstruct, interpolate_at_zero,
    lagrange_weights_at_zero, primes_for_bound, suppression_value, validate_prime,
)


class ResourceLimitError(RuntimeError):
    """The requested tensor calculation exceeds an explicit resource budget."""


class UnsupportedDeviceError(RuntimeError):
    """A requested device cannot provide the exact operations used here."""


def _probe_device(name: str, matmul_backend: str = "int64") -> None:
    """Check actual kernels, including operations unsupported on some MPSs."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise UnsupportedDeviceError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise UnsupportedDeviceError("MPS is unavailable")
    try:
        p = 1004535809
        a = torch.tensor([p - 1, p - 2, 17], dtype=torch.int64, device=name)
        if (a * a).remainder(p).cpu().tolist() != [1, 4, 289]:
            raise RuntimeError("int64 multiplication/remainder mismatch")
        if int(a.sum().cpu()) != 2 * p + 14:
            raise RuntimeError("int64 reduction mismatch")
        if ((a >> 1) & 1).cpu().tolist() != [0, 1, 0]:
            raise RuntimeError("int64 bit-operation mismatch")
        unique, inverse = torch.unique(a, return_inverse=True)
        counts = torch.zeros(len(unique), dtype=torch.int64, device=name)
        counts.scatter_add_(0, inverse, torch.ones_like(a))
        if counts.cpu().tolist() != [1, 1, 1]:
            raise RuntimeError("unique/scatter mismatch")
        square = ((a[:2, None] * a[None, :2]).remainder(p)).cpu().tolist()
        if square != [[1, 2], [2, 4]]:
            raise RuntimeError("int64 matrix product mismatch")
        signatures = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.int64, device=name)
        keys, groups = torch.unique(signatures, dim=0, return_inverse=True)
        sums = torch.zeros((len(keys), 2), dtype=torch.int64, device=name)
        sums.index_add_(0, groups, signatures)
        if sums.cpu().tolist() != [[0, 1], [2, 0]]:
            raise RuntimeError("conditioned-row aggregation mismatch")
        if matmul_backend == "int64":
            # Accelerator kernels use broadcast multiplication and bounded
            # reductions, because CUDA/MPS Long matmul support varies.
            if name == "cpu":
                dot = (a[:2, None] @ a[None, :2]).remainder(p)
                if dot.cpu().tolist() != [[1, 2], [2, 4]]:
                    raise RuntimeError("int64 matrix multiplication mismatch")
            products = a[:2, None, None] * a[None, :2, None]
            if products.sum(1).remainder(p).cpu().flatten().tolist() != [3, 6]:
                raise RuntimeError("bounded broadcast dot product mismatch")
        else:
            # Integer limb dot products are exact in binary64 below 2**53.
            limb = torch.tensor([[32767., 32766.]], dtype=torch.float64, device=name)
            if int((limb @ limb.T).cpu().item()) != 32767**2 + 32766**2:
                raise RuntimeError("exact binary64 limb dot product mismatch")
    except Exception as exc:
        if isinstance(exc, UnsupportedDeviceError):
            raise
        raise UnsupportedDeviceError(f"{name} failed exact arithmetic probe: {exc}") from exc


def select_device(device: str = "auto", matmul_backend: str = "int64") -> tuple[torch.device, list[str]]:
    """Auto tries CUDA, MPS, then CPU; explicit requests fail on probe errors."""
    failures = []
    candidates = [device] if device != "auto" else ["cuda", "mps", "cpu"]
    for candidate in candidates:
        try:
            _probe_device(candidate, matmul_backend)
            return torch.device(candidate), failures
        except UnsupportedDeviceError as exc:
            failures.append(str(exc))
    raise UnsupportedDeviceError("; ".join(failures))


@dataclass
class CounterStats:
    device: str
    device_fallbacks: list[str] = field(default_factory=list)
    local_assignment_counts: list[int] = field(default_factory=list)
    source_signature_counts: list[int] = field(default_factory=list)
    incoming_signature_counts: list[int] = field(default_factory=list)
    free_variables: int = 0
    boundary_links: int = 0
    interpolation_nodes: int = 0
    primes: tuple[int, ...] = ()
    evaluations: int = 0
    estimated_work: int = 0
    peak_frontier_rows: int = 0
    build_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    conditioning_states: list[list[int]] = field(default_factory=list)
    residual_active_bits: list[list[int]] = field(default_factory=list)


@dataclass
class _LocalTable:
    masters: tuple[int, ...]
    incoming: tuple[int, ...]
    source_bits: torch.Tensor
    proxy_bits: torch.Tensor
    counts: torch.Tensor
    satisfying_assignments: int
    free_variables: int


class AlgebraicCounter:
    """Reusable finite-field evaluator for a ``PartitionedCNF``.

    ``max_work`` bounds estimated multiply/add work per H evaluation (and local
    frontier construction). ``max_memory_bytes`` bounds individual persistent
    tables and conservative matrix workspaces. Exceeding a limit raises; it
    never changes the counting algorithm. Counts/stats remain Python integers.
    """

    def __init__(self, partitioned: PartitionedCNF, device: str = "auto", *,
                 assignment_chunk_size: int = 65536,
                 signature_batch_size: int = 128,
                 max_memory_bytes: int = 512 * 1024 * 1024,
                 max_work: int = 2_000_000_000,
                 max_local_assignments: int = 4_000_000,
                 matmul_backend: str = "int64"):
        for name, value in (("assignment_chunk_size", assignment_chunk_size),
                            ("signature_batch_size", signature_batch_size),
                            ("max_memory_bytes", max_memory_bytes),
                            ("max_work", max_work),
                            ("max_local_assignments", max_local_assignments)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if matmul_backend not in ("int64", "limbs"):
            raise ValueError("matmul_backend must be 'int64' or 'limbs'")
        self.partitioned = partitioned
        self.device, failures = select_device(device, matmul_backend)
        self.assignment_chunk_size = assignment_chunk_size
        self.signature_batch_size = signature_batch_size
        self.max_memory_bytes = max_memory_bytes
        self._persistent_bytes = 0
        self.max_work = max_work
        self.max_local_assignments = max_local_assignments
        self.matmul_backend = matmul_backend
        self.stats = CounterStats(str(self.device), failures,
                                  boundary_links=len(partitioned.boundary_links))
        self._link_block = {}
        for block in partitioned.measurement_blocks:
            if not block.link_indices:
                raise ValueError("measurement blocks must be nonempty")
            for index in block.link_indices:
                if index in self._link_block or not 0 <= index < len(partitioned.boundary_links):
                    raise ValueError("measurement blocks must partition links")
                if partitioned.boundary_links[index].master_group != block.master_group:
                    raise ValueError("measurement block must belong to its source group")
                self._link_block[index] = block
        if len(self._link_block) != len(partitioned.boundary_links):
            raise ValueError("measurement blocks must partition links")
        start = perf_counter()
        self.tables = [self._build_local(group) for group in range(len(partitioned.group_vars))]
        self.stats.build_seconds = perf_counter() - start
        self.stats.local_assignment_counts = [table.satisfying_assignments for table in self.tables]
        self.stats.source_signature_counts = [len(table.source_bits) for table in self.tables]
        self.stats.incoming_signature_counts = [len(table.proxy_bits) for table in self.tables]
        self.stats.free_variables = sum(table.free_variables for table in self.tables)

    def _memory(self, elements: int, description: str, multiplier: int = 8) -> None:
        required = elements * multiplier + self._persistent_bytes
        if required > self.max_memory_bytes:
            raise ResourceLimitError(f"{description} needs approximately {elements * multiplier} bytes; "
                                     f"persistent tables use {self._persistent_bytes}; "
                                     f"budget is {self.max_memory_bytes}")

    def _build_local(self, group: int) -> _LocalTable:
        partitioned = self.partitioned
        outgoing = [link for link in partitioned.boundary_links if link.master_group == group]
        masters = tuple(sorted({link.master_var for link in outgoing}))
        incoming = tuple(i for i, link in enumerate(partitioned.boundary_links)
                         if link.proxy_group == group)
        proxies = tuple(partitioned.boundary_links[i].proxy_var for i in incoming)
        clauses = []
        for clause in partitioned.group_clauses[group]:
            signed = {(lit.var, lit.pos) for lit in clause.literals}
            if any((var, not pos) in signed for var, pos in signed):
                continue
            clauses.append(clause)
        active = set(masters) | set(proxies) | {lit.var for clause in clauses for lit in clause.literals}
        free = len(partitioned.group_vars[group]) - len(active)
        if not active <= set(partitioned.group_vars[group]) or free < 0:
            raise ValueError("a local clause or boundary variable is outside its group")
        frequency = {var: sum(lit.var == var for clause in clauses for lit in clause.literals)
                     for var in active}
        # A fixed order and completed-clause masks; no recursive branching,
        # propagation, search heuristic changes, or SAT solver is invoked.
        order = sorted(active, key=lambda var: (-frequency[var], var))
        if len(order) > 62:
            raise ResourceLimitError("a local group has more than 62 active variables")
        positions = {var: i for i, var in enumerate(order)}
        completed = [[] for _ in order]
        empty_clause = False
        for clause in clauses:
            if not clause.literals:
                empty_clause = True
            else:
                completed[max(positions[lit.var] for lit in clause.literals)].append(clause)
        frontier = torch.zeros(0 if empty_clause else 1, dtype=torch.int64, device=self.device)
        work = 0
        for position in range(len(order)):
            chunks = []
            survivors = 0
            chunk_size = min(self.assignment_chunk_size,
                             max(1, (self.max_memory_bytes - self._persistent_bytes) // 64))
            for start in range(0, len(frontier), chunk_size):
                old = frontier[start:start + chunk_size]
                self._memory(2 * len(old) + len(frontier), "assignment expansion workspace", 16)
                candidate = torch.cat((old, old | (1 << position)))
                work += len(candidate) * (1 + sum(len(c.literals) for c in completed[position]))
                if work > self.max_work:
                    raise ResourceLimitError("local assignment-mask construction exceeds max_work")
                for clause in completed[position]:
                    mask = torch.zeros(len(candidate), dtype=torch.bool, device=self.device)
                    for lit in clause.literals:
                        mask |= (((candidate >> positions[lit.var]) & 1) == int(lit.pos))
                    candidate = candidate[mask]
                    if len(candidate) == 0:
                        break
                survivors += len(candidate)
                if survivors > self.max_local_assignments:
                    raise ResourceLimitError("local satisfying frontier exceeds max_local_assignments")
                self._memory(survivors, "local assignment frontier", 24)
                chunks.append(candidate)
            frontier = torch.cat(chunks) if chunks else frontier[:0]
            self.stats.peak_frontier_rows = max(self.stats.peak_frontier_rows, len(frontier))
        def encode(variables):
            self._memory(len(frontier), "signature encoding workspace", 48)
            values = torch.zeros_like(frontier)
            for index, var in enumerate(variables):
                values |= ((frontier >> positions[var]) & 1) << index
            keys, inverse = torch.unique(values, return_inverse=True)
            self._memory(len(keys) * len(variables) + len(frontier),
                         "signature bit matrix and encoding workspace", 16)
            bits = ((keys[:, None] >> torch.arange(len(variables), device=self.device)) & 1)
            return keys, inverse, bits
        source_keys, source_inverse, source_bits = encode(masters)
        proxy_keys, proxy_inverse, proxy_bits = encode(proxies)
        self._memory(len(source_keys) * len(proxy_keys), "compressed local table", 16)
        counts = torch.zeros((len(source_keys), len(proxy_keys)), dtype=torch.int64, device=self.device)
        if len(frontier):
            counts.view(-1).scatter_add_(0, source_inverse * len(proxy_keys) + proxy_inverse,
                                        torch.ones_like(frontier))
        allocated = sum(tensor.numel() * tensor.element_size()
                        for tensor in (source_bits, proxy_bits, counts))
        self._memory(allocated, "persistent local tables", 1)
        self._persistent_bytes += allocated
        return _LocalTable(masters, incoming, source_bits, proxy_bits, counts, len(frontier), free)

    def _condition(self, t, prime, schedule):
        """Condition source groups on block measurements and active masters.

        Measurements count links with multiplicity. Master bits are unique;
        a master used by another active link is retained even when one of its
        links is suppressed. Kernels are constant within each resulting row.
        """
        tables, states, residual = [], [], []
        for group, table in enumerate(self.tables):
            positions = {var: i for i, var in enumerate(table.masters)}
            blocks = [block for block in self.partitioned.measurement_blocks
                      if block.master_group == group]
            active = sorted({link.master_var for index, link in enumerate(self.partitioned.boundary_links)
                             if link.master_group == group
                             and suppression_value(schedule.roots[index], t, prime) != 0})
            signature_columns = []
            signature_width = len(blocks) + len(active)
            self._memory(len(table.source_bits) * signature_width,
                         "measurement and active-bit conditioning workspace", 64)
            for block in blocks:
                measurement = torch.zeros(len(table.source_bits), dtype=torch.int64, device=self.device)
                for index in block.link_indices:
                    variable = self.partitioned.boundary_links[index].master_var
                    measurement += table.source_bits[:, positions[variable]]
                signature_columns.append(measurement)
            signature_columns.extend(table.source_bits[:, positions[var]] for var in active)
            if signature_columns and len(table.source_bits):
                signatures = torch.stack(signature_columns, 1)
                unique, inverse = torch.unique(signatures, dim=0, return_inverse=True)
                self._memory(len(unique) * (table.counts.shape[1] + len(table.masters) + 2),
                             "conditioned table aggregation workspace", 32)
                representatives = [0] * len(unique)
                for row, index in enumerate(inverse.cpu().tolist()):
                    representatives[index] = row
                counts = torch.zeros((len(unique), table.counts.shape[1]), dtype=torch.int64,
                                     device=self.device)
                counts.index_add_(0, inverse, table.counts)
                source_bits = table.source_bits[torch.tensor(representatives, device=self.device)]
                reduced = _LocalTable(table.masters, table.incoming, source_bits, table.proxy_bits,
                                      counts, table.satisfying_assignments, table.free_variables)
            else:
                reduced = table
            tables.append(reduced)
            states.append(len(reduced.source_bits))
            residual.append(len(active))
        self.stats.conditioning_states.append(states)
        self.stats.residual_active_bits.append(residual)
        return tables

    def _kernel(self, source: int, target: int, t: int, prime: int, schedule, tables=None) -> torch.Tensor:
        tables = self.tables if tables is None else tables
        src, dst = tables[source], tables[target]
        self._memory(len(src.source_bits) * len(dst.proxy_bits), "boundary kernel", 32)
        weights = torch.ones((len(src.source_bits), len(dst.proxy_bits)), dtype=torch.int64,
                             device=self.device)
        positions = {var: i for i, var in enumerate(src.masters)}
        means = {}
        for proxy_position, link_index in enumerate(dst.incoming):
            link = self.partitioned.boundary_links[link_index]
            if link.master_group != source:
                continue
            block = self._link_block[link_index]
            key = block.link_indices
            if key not in means:
                measurement = torch.zeros(len(src.source_bits), dtype=torch.int64, device=self.device)
                for index in key:
                    master = self.partitioned.boundary_links[index].master_var
                    measurement += src.source_bits[:, positions[master]]
                means[key] = (measurement.remainder(prime) * pow(len(key), -1, prime)).remainder(prime)
            mean = means[key]
            suppression = suppression_value(schedule.roots[link_index], t, prime)
            x = src.source_bits[:, positions[link.master_var]]
            tilde = (mean + suppression * (x - mean)).remainder(prime)
            y = dst.proxy_bits[:, proxy_position]
            factor = ((1 - y)[None, :] + tilde[:, None] * (2 * y - 1)[None, :]).remainder(prime)
            weights = (weights * factor).remainder(prime)
        return weights

    def _matmul_mod(self, left: torch.Tensor, right: torch.Tensor, prime: int) -> torch.Tensor:
        """Exact modular dots using bounded int64 sums or binary64 limbs.

        In the optional limb backend each limb is <2**15 and each dot has
        <=2**20 terms, so its nonnegative
        integer result is <2**50 and exactly representable in IEEE binary64.
        Recombination uses int64 products <2**60, reduced after each product.
        This avoids overflow from a direct int64 matrix dot product.
        """
        rows, inner = left.shape
        cols = right.shape[1]
        self._memory(rows * cols, "modular matrix workspace", 64)
        result = torch.zeros((rows, cols), dtype=torch.int64, device=self.device)
        if self.matmul_backend == "int64":
            # Every nonnegative dot sum stays <= signed int64's maximum.
            safe_inner = ((1 << 63) - 1) // ((prime - 1) ** 2)
            for start in range(0, inner, safe_inner):
                a = left[:, start:start + safe_inner].remainder(prime)
                b = right[start:start + safe_inner].remainder(prime)
                if self.device.type == "cpu":
                    partial = (a @ b).remainder(prime)
                    result = (result + partial).remainder(prime)
                else:
                    # Keep broadcast workspaces bounded even without Long
                    # matrix-multiplication kernels on the accelerator.
                    tile = min(self.signature_batch_size,
                               max(1, int((self.max_memory_bytes // (32 * safe_inner)) ** .5)))
                    for row in range(0, rows, tile):
                        for col in range(0, cols, tile):
                            partial = (a[row:row + tile, :, None]
                                       * b[None, :, col:col + tile]).sum(1).remainder(prime)
                            result[row:row + tile, col:col + tile] = (
                                result[row:row + tile, col:col + tile] + partial).remainder(prime)
            return result
        base = 1 << 15
        for start in range(0, inner, 1 << 20):
            a = left[:, start:start + (1 << 20)].remainder(prime)
            b = right[start:start + (1 << 20)].remainder(prime)
            alo, ahi = (a % base).double(), (a // base).double()
            blo, bhi = (b % base).double(), (b // base).double()
            for aa, bb, coefficient in ((alo, blo, 1), (ahi, blo, base),
                                        (alo, bhi, base), (ahi, bhi, (base * base) % prime)):
                partial = (aa @ bb).to(torch.int64).remainder(prime)
                result = (result + (partial * coefficient).remainder(prime)).remainder(prime)
        return result

    def _sum_mod(self, values: torch.Tensor, prime: int) -> int:
        # At most 2**20 residues below 2**30: each reduction is below 2**50.
        total = 0
        flat = values.reshape(-1)
        for start in range(0, len(flat), 1 << 20):
            total = (total + int(flat[start:start + (1 << 20)].sum().cpu())) % prime
        return total

    def evaluate(self, t: int, prime: int, schedule: SuppressionSchedule, *,
                 conditioned: bool = True) -> int:
        """Evaluate H(t) exactly, without interpolation or CRT."""
        validate_prime(prime)
        if len(schedule.roots) != len(self.partitioned.boundary_links):
            raise ValueError("schedule needs one root list per boundary link")
        lagrange_weights_at_zero(schedule.nodes, prime)
        if any(node % prime == 0 for node in schedule.nodes):
            raise ValueError("suppression nodes must be nonzero modulo prime")
        if any(len(block.link_indices) % prime == 0 for block in self.partitioned.measurement_blocks):
            raise ValueError("measurement block size must be invertible modulo prime")
        start = perf_counter()
        tables = self._condition(t, prime, schedule) if conditioned else self.tables
        if any(table.satisfying_assignments == 0 for table in tables):
            result = 0
        elif not self.partitioned.boundary_links:
            result = prod(table.satisfying_assignments for table in tables) % prime
        elif len(tables) == 2:
            a, b = tables
            work = (a.counts.shape[0] * b.counts.shape[0]
                    * (a.counts.shape[1] + b.counts.shape[1]))
            self.stats.estimated_work = max(self.stats.estimated_work, work)
            if work > self.max_work:
                raise ResourceLimitError(f"two-group contraction estimates {work} operations; "
                                         f"max_work is {self.max_work}")
            ab = self._kernel(0, 1, t, prime, schedule, tables)
            ba = self._kernel(1, 0, t, prime, schedule, tables)
            first = self._matmul_mod(a.counts, ba.T, prime)
            second = self._matmul_mod(ab, b.counts.T, prime)
            result = self._sum_mod((first * second).remainder(prime), prime)
        else:
            result = self._evaluate_many_groups(t, prime, schedule, tables)
        result = result * pow(2, self.stats.free_variables, prime) % prime
        self.stats.evaluations += 1
        self.stats.evaluation_seconds += perf_counter() - start
        return result

    def _evaluate_many_groups(self, t, prime, schedule, tables=None):
        # Enumerate only feasible source signatures, never independent link
        # bits (which would wrongly permit repeated masters to disagree).
        tables = self.tables if tables is None else tables
        sizes = [len(table.source_bits) for table in tables]
        combinations = prod(sizes)
        work = combinations * sum(len(table.proxy_bits) for table in tables)
        self.stats.estimated_work = max(self.stats.estimated_work, work)
        if work > self.max_work:
            raise ResourceLimitError(f"source-signature contraction estimates {work} operations; "
                                     f"max_work is {self.max_work}")
        kernels = {(source, target): self._kernel(source, target, t, prime, schedule, tables)
                   for source in range(len(tables)) for target in range(len(tables))
                   if source != target}
        total = 0
        iterator = iter(product(*(range(size) for size in sizes)))
        from itertools import islice
        width = max(len(table.proxy_bits) for table in tables)
        batch_size = min(self.signature_batch_size,
                         max(1, (self.max_memory_bytes - self._persistent_bytes)
                             // (64 * (width + len(tables)))))
        self._memory(batch_size * (width + len(tables)),
                     "source-signature batch workspace", 64)
        while batch := list(islice(iterator, batch_size)):
            ids = torch.tensor(batch, dtype=torch.int64, device=self.device)
            terms = torch.ones(len(batch), dtype=torch.int64, device=self.device)
            for target, table in enumerate(tables):
                weights = torch.ones((len(batch), len(table.proxy_bits)), dtype=torch.int64,
                                     device=self.device)
                for source in range(len(tables)):
                    if source != target:
                        weights = (weights * kernels[source, target][ids[:, source]]).remainder(prime)
                rows = table.counts[ids[:, target]].remainder(prime)
                weighted = (rows * weights).remainder(prime)
                # Number of proxy configurations is bounded by local rows;
                # chunk reduction to avoid any unbounded int64 summation.
                local = torch.zeros(len(batch), dtype=torch.int64, device=self.device)
                for offset in range(0, weighted.shape[1], 1 << 20):
                    local = (local + weighted[:, offset:offset + (1 << 20)].sum(1)).remainder(prime)
                terms = (terms * local).remainder(prime)
            total = (total + self._sum_mod(terms, prime)) % prime
        return total


def count_with_stats(cnf: CNF, *, device: str = "auto", num_groups: int = 2,
                     block_size: int = 4, max_degree: int | None = None,
                     max_width: int = 3, degree_per_link: int = 1,
                     **engine_options) -> tuple[int, CounterStats]:
    """Preprocess, evaluate interpolation nodes, and reconstruct the exact #SAT."""
    from .stage1_sparsify import reduce_clause_width_deterministic, split_high_degree_variables
    from .stage1_sparsify import partition_and_decouple_boundary
    cnf.validate()
    original_variables = cnf.num_vars
    transformed = reduce_clause_width_deterministic(cnf, max_width=max_width)
    if max_degree is not None:
        transformed = split_high_degree_variables(transformed, max_degree=max_degree)
    partitioned = partition_and_decouple_boundary(transformed, num_groups=num_groups,
                                                 block_size=block_size)
    counter = AlgebraicCounter(partitioned, device=device, **engine_options)
    if not partitioned.boundary_links:
        exact = prod(table.satisfying_assignments for table in counter.tables)
        return exact << counter.stats.free_variables, counter.stats
    schedule = create_suppression_schedule(len(partitioned.boundary_links), degree_per_link)
    primes = primes_for_bound(1 << original_variables)
    counter.stats.primes = primes
    counter.stats.interpolation_nodes = len(schedule.nodes)
    residues = [interpolate_at_zero([counter.evaluate(t, prime, schedule) for t in schedule.nodes],
                                    schedule.nodes, prime) for prime in primes]
    result = crt_reconstruct(residues, primes)
    if result > 1 << original_variables:
        raise ArithmeticError("reconstructed count exceeds the original assignment bound")
    return result, counter.stats


def gpu_algebraic_sharp_sat(cnf: CNF, **options) -> int:
    """Return the exact number of models, using the tensor algebraic engine."""
    return count_with_stats(cnf, **options)[0]

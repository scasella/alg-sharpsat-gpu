"""Suppressed-subset Hasse jets with exact modular tensor contractions.

At node s, source groups are conditioned on block measurements and the unique
master bits used outside K_s. A subset U of suppressed links contributes a
local moment product of (x_i-m_i/k_i), a target factor -(2*y_i-1)/t_s,
and a shift by |U| in the jet. No scalar H evaluations reconstruct the jets.

The subset sum is explicit and can be exponential. In particular single-node
Taylor mode processes all 2**B subsets; work limits reject excessive requests.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from itertools import combinations, islice, product
from math import comb, prod
from time import perf_counter
from typing import Sequence

import torch

from .cnf import CNF
from .stage1_sparsify import (
    PartitionedCNF, partition_and_decouple_boundary,
    reduce_clause_width_deterministic, split_high_degree_variables,
)
from .stage2_algebraic import crt_reconstruct, primes_for_bound, validate_prime
from .stage2_hermite import (
    HermiteSchedule, create_hermite_schedule,
    interpolate_hermite_at_zero,
)
from .stage3_gpu_engine import AlgebraicCounter, CounterStats, ResourceLimitError


@dataclass
class JetStats(CounterStats):
    node_evaluations: list[dict] = field(default_factory=list)
    subset_terms_processed: int = 0
    signature_subset_visits: int = 0
    max_jet_order: int = 0
    schedule_mode: str = "hermite"


@dataclass
class _Condition:
    inverse: torch.Tensor
    representatives: torch.Tensor
    means: dict[int, torch.Tensor]
    deltas: dict[int, torch.Tensor]
    states: int
    active_bits: int


class JetCounter(AlgebraicCounter):
    """Reuse local masked truth tables to evaluate subset-factorized jets.

    ``max_subset_terms`` limits sum(comb(|K_s|,r), r<=q_s) at each node.
    ``max_jet_work`` limits a conservative arithmetic-work estimate per node.
    These limits raise ``ResourceLimitError`` before subset enumeration; there
    is no reference-counter or scalar-interpolation fallback.
    """

    def __init__(self, partitioned: PartitionedCNF, device: str = "auto", *,
                 max_subset_terms: int = 65536,
                 max_jet_work: int = 2_000_000_000,
                 moment_cache_entries: int = 64,
                 subset_batch_size: int = 8, **engine_options):
        for name, value in (("max_subset_terms", max_subset_terms),
                            ("max_jet_work", max_jet_work),
                            ("moment_cache_entries", moment_cache_entries),
                            ("subset_batch_size", subset_batch_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._jet_live_bytes = 0
        super().__init__(partitioned, device=device, **engine_options)
        self.stats = JetStats(**asdict(self.stats))
        self.max_subset_terms = max_subset_terms
        self.max_jet_work = max_jet_work
        self.moment_cache_entries = moment_cache_entries
        self.subset_batch_size = subset_batch_size
        self._moment_cache = OrderedDict()
        self._cache_bytes = 0
        self._condition_bytes = 0

    def _memory(self, elements: int, description: str, multiplier: int = 8) -> None:
        required = (elements * multiplier + self._persistent_bytes
                    + self._jet_live_bytes)
        if required > self.max_memory_bytes:
            raise ResourceLimitError(
                f"{description} needs approximately {required} total bytes; "
                f"budget is {self.max_memory_bytes}")

    def _conditions(self, node_index: int, prime: int,
                    schedule: HermiteSchedule) -> list[_Condition]:
        contexts = []
        for group, table in enumerate(self.tables):
            outgoing = [i for i, link in enumerate(self.partitioned.boundary_links)
                        if link.master_group == group]
            blocks = [block for block in self.partitioned.measurement_blocks
                      if block.master_group == group]
            active = sorted({self.partitioned.boundary_links[i].master_var
                             for i in outgoing if schedule.link_nodes[i] != node_index})
            rows = len(table.source_bits)
            positions = {var: i for i, var in enumerate(table.masters)}
            self._memory(rows * (len(blocks) + len(active) + 2 * len(outgoing) + 3),
                         "jet measurement/signature workspace", 64)
            columns, means, deltas = [], {}, {}
            for block in blocks:
                measurement = torch.zeros(rows, dtype=torch.int64, device=self.device)
                for i in block.link_indices:
                    master = self.partitioned.boundary_links[i].master_var
                    measurement += table.source_bits[:, positions[master]]
                columns.append(measurement)
                mean = (measurement.remainder(prime)
                        * pow(len(block.link_indices), -1, prime)).remainder(prime)
                for i in block.link_indices:
                    means[i] = mean
                    master = self.partitioned.boundary_links[i].master_var
                    deltas[i] = (table.source_bits[:, positions[master]] - mean).remainder(prime)
            columns.extend(table.source_bits[:, positions[var]] for var in active)
            if columns and rows:
                signature = torch.stack(columns, 1)
                unique, inverse, counts = torch.unique(
                    signature, dim=0, return_inverse=True, return_counts=True)
                # Select one representative per signature entirely with tensor
                # operations, without looping over local Boolean assignments.
                offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
                representatives = torch.argsort(inverse)[offsets]
                states = len(unique)
            else:
                inverse = torch.zeros(rows, dtype=torch.int64, device=self.device)
                representatives = torch.zeros(min(rows, 1), dtype=torch.int64, device=self.device)
                states = min(rows, 1)
            context = _Condition(inverse, representatives, means, deltas, states, len(active))
            contexts.append(context)
            # A mean is shared by every link in its measurement block.
            tensors = [inverse, representatives, *deltas.values()]
            tensors.extend({id(value): value for value in means.values()}.values())
            used = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            self._condition_bytes += used
            self._jet_live_bytes = self._condition_bytes + self._cache_bytes
        return contexts

    def _moment(self, group: int, subset: tuple[int, ...], prime: int,
                contexts: list[_Condition], trace: dict) -> torch.Tensor:
        return self._moment_batch(group, [subset], prime, contexts, trace)[0]

    def _moment_batch(self, group: int, subsets: list[tuple[int, ...]], prime: int,
                      contexts: list[_Condition], trace: dict) -> list[torch.Tensor]:
        """Aggregate a bounded (subset, signature, proxy) tensor in one pass.

        Local subsets repeated by different global U values share one moment.
        Link products, weighted clause tables, and signature-bin reductions
        all carry a batch axis; subsequent jet contractions remain per subset.
        """
        keys = [(group, tuple(i for i in subset
                             if self.partitioned.boundary_links[i].master_group == group))
                for subset in subsets]
        available, missing = {}, []
        for key in keys:
            if key in available:
                trace["moment_cache_hits"] += 1
            elif key in self._moment_cache:
                trace["moment_cache_hits"] += 1
                self._moment_cache.move_to_end(key)
                available[key] = self._moment_cache[key]
            elif key not in missing:
                missing.append(key)
        if not missing:
            return [available[key] for key in keys]
        table, context = self.tables[group], contexts[group]
        rows, width = table.counts.shape
        batch_count = len(missing)
        size = batch_count * context.states * width
        # Two output-sized arrays plus row-bounded products/reduction scratch.
        self._memory(size, "batched local subset-moment workspace", 48)
        result = torch.zeros((batch_count, context.states, width), dtype=torch.int64, device=self.device)
        safe_rows = ((1 << 63) - 1) // max(prime - 1, 1)
        available_bytes = self.max_memory_bytes - self._persistent_bytes - self._jet_live_bytes - 48 * size
        chunk = min(self.assignment_chunk_size, safe_rows,
                    max(1, available_bytes // max(32 * batch_count * (width + 1), 1)))
        self._memory(size * 4 + batch_count * chunk * (width + 1) * 3,
                     "bounded subset-moment reduction workspace")
        union_links = sorted({link for _, subset in missing for link in subset})
        selectors = {link: torch.tensor([link in subset for _, subset in missing],
                                       dtype=torch.bool, device=self.device)[:, None]
                     for link in union_links}
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            moment = torch.ones((batch_count, stop - start), dtype=torch.int64, device=self.device)
            for link in union_links:
                factor = torch.where(selectors[link], context.deltas[link][None, start:stop], 1)
                moment = (moment * factor).remainder(prime)
            weighted = (table.counts[None, start:stop].remainder(prime)
                        * moment[:, :, None]).remainder(prime)
            scratch = torch.zeros_like(result)
            # At most chunk residues (<p) enter any bin. The bound above is
            # independent of cancellation and prevents int64 scatter overflow.
            scratch.index_add_(1, context.inverse[start:stop], weighted)
            result = (result + scratch.remainder(prime)).remainder(prime)
        trace["moment_tables_built"] += batch_count
        trace["moment_batches_built"] = trace.get("moment_batches_built", 0) + 1
        cache_limit = self.max_memory_bytes // 8
        for index, key in enumerate(missing):
            byte_count = result[index].numel() * result.element_size()
            while self._moment_cache and (len(self._moment_cache) >= self.moment_cache_entries
                                          or self._cache_bytes + byte_count > cache_limit):
                _, discarded = self._moment_cache.popitem(last=False)
                self._cache_bytes -= discarded.numel() * discarded.element_size()
            if byte_count <= cache_limit:
                # Clone to keep cache accounting independent of the entire
                # batch's storage lifetime and subsequent LRU eviction.
                value = result[index].clone()
                self._moment_cache[key] = value
                self._cache_bytes += byte_count
            else:
                value = result[index]
            available[key] = value
        self._jet_live_bytes = self._condition_bytes + self._cache_bytes
        return [available[key] for key in keys]

    def _kernel_jet(self, source: int, target: int, subset: frozenset[int],
                    node_index: int, prime: int, schedule: HermiteSchedule,
                    contexts: list[_Condition], limit: int) -> list[torch.Tensor]:
        src, dst = contexts[source], self.tables[target]
        shape = (src.states, len(dst.proxy_bits))
        self._memory(prod(shape) * (limit + 1), "boundary jet workspace", 48)
        coefficients = [torch.ones(shape, dtype=torch.int64, device=self.device)]
        t = schedule.nodes[node_index] % prime
        for proxy_column, link_index in enumerate(dst.incoming):
            link = self.partitioned.boundary_links[link_index]
            if link.master_group != source:
                continue
            sign = 2 * dst.proxy_bits[:, proxy_column] - 1
            y = dst.proxy_bits[:, proxy_column]
            root = schedule.nodes[schedule.link_nodes[link_index]] % prime
            inverse_root = pow(root, -1, prime)
            mean = src.means[link_index][src.representatives]
            if link_index in subset:
                constant = ((-inverse_root * sign).remainder(prime)[None, :]
                            .expand(shape))
                slope = None
            elif schedule.link_nodes[link_index] == node_index:
                constant = ((1 - y)[None, :] + mean[:, None] * sign[None, :]).remainder(prime)
                slope = None
            else:
                delta = src.deltas[link_index][src.representatives]
                suppression = (1 - t * inverse_root) % prime
                value = (mean + (suppression * delta).remainder(prime)).remainder(prime)
                constant = ((1 - y)[None, :] + value[:, None] * sign[None, :]).remainder(prime)
                delta_slope = (-inverse_root * delta).remainder(prime)
                slope = (delta_slope[:, None] * sign[None, :]).remainder(prime)
            length = min(limit + 1, len(coefficients) + int(slope is not None))
            updated = []
            for degree in range(length):
                term = (coefficients[degree] * constant).remainder(prime) \
                    if degree < len(coefficients) else torch.zeros(shape, dtype=torch.int64, device=self.device)
                if degree and slope is not None:
                    term = (term + (coefficients[degree - 1] * slope).remainder(prime)).remainder(prime)
                updated.append(term)
            coefficients = updated
        return coefficients

    def _two_group_contract(self, moments: list[torch.Tensor], subset: frozenset[int],
                            node_index: int, prime: int, schedule: HermiteSchedule,
                            contexts: list[_Condition], limit: int) -> list[int]:
        ab = self._kernel_jet(0, 1, subset, node_index, prime, schedule, contexts, limit)
        ba = self._kernel_jet(1, 0, subset, node_index, prime, schedule, contexts, limit)
        pair_states = contexts[0].states * contexts[1].states
        self._memory(pair_states * (len(ab) + len(ba)), "contracted jet workspace", 32)
        first = [self._matmul_mod(moments[0], coefficient.T, prime) for coefficient in ba]
        second = [self._matmul_mod(coefficient, moments[1].T, prime) for coefficient in ab]
        result = [0] * (limit + 1)
        for left_degree, left in enumerate(first):
            for right_degree, right in enumerate(second):
                degree = left_degree + right_degree
                if degree <= limit:
                    result[degree] = (result[degree]
                                      + self._sum_mod((left * right).remainder(prime), prime)) % prime
        return result

    @staticmethod
    def _multiply_jets(left: list[torch.Tensor], right: list[torch.Tensor],
                       limit: int, prime: int) -> list[torch.Tensor]:
        result = []
        for degree in range(min(limit + 1, len(left) + len(right) - 1)):
            value = torch.zeros_like(left[0] * right[0])
            for i in range(max(0, degree - len(right) + 1), min(degree + 1, len(left))):
                value = (value + (left[i] * right[degree - i]).remainder(prime)).remainder(prime)
            result.append(value)
        return result

    def _many_group_contract(self, moments: list[torch.Tensor], subset: frozenset[int],
                             node_index: int, prime: int, schedule: HermiteSchedule,
                             contexts: list[_Condition], limit: int) -> list[int]:
        groups = len(self.tables)
        if groups == 1:
            return [self._sum_mod(moments[0], prime)] + [0] * limit
        kernels = {(source, target): self._kernel_jet(
            source, target, subset, node_index, prime, schedule, contexts, limit)
            for source in range(groups) for target in range(groups) if source != target}
        sizes = [context.states for context in contexts]
        width = max(len(table.proxy_bits) for table in self.tables)
        available = self.max_memory_bytes - self._persistent_bytes - self._jet_live_bytes
        batch_size = min(self.signature_batch_size,
                         max(1, available // max(128 * (limit + 1) * (width + groups), 1)))
        self._memory(batch_size * (width + groups) * (limit + 1),
                     "many-group polynomial signature workspace", 128)
        iterator = iter(product(*(range(size) for size in sizes)))
        result = [0] * (limit + 1)
        while batch := list(islice(iterator, batch_size)):
            ids = torch.tensor(batch, dtype=torch.int64, device=self.device)
            total = [torch.ones(len(batch), dtype=torch.int64, device=self.device)]
            for target, table in enumerate(self.tables):
                local = [torch.ones((len(batch), len(table.proxy_bits)),
                                    dtype=torch.int64, device=self.device)]
                for source in range(groups):
                    if source != target:
                        incoming = [coefficient[ids[:, source]]
                                    for coefficient in kernels[source, target]]
                        local = self._multiply_jets(local, incoming, limit, prime)
                moment = moments[target][ids[:, target]]
                collapsed = []
                for coefficient in local:
                    weighted = (coefficient * moment).remainder(prime)
                    value = torch.zeros(len(batch), dtype=torch.int64, device=self.device)
                    for offset in range(0, weighted.shape[1], 1 << 20):
                        value = (value + weighted[:, offset:offset + (1 << 20)].sum(1)).remainder(prime)
                    collapsed.append(value)
                total = self._multiply_jets(total, collapsed, limit, prime)
            for degree, coefficient in enumerate(total):
                result[degree] = (result[degree] + self._sum_mod(coefficient, prime)) % prime
        return result

    def evaluate_jet(self, node_index: int, prime: int,
                     schedule: HermiteSchedule) -> list[int]:
        """Return [tau**j] H(nodes[s]+tau), j=0..orders[s]."""
        validate_prime(prime)
        if schedule.degree != len(self.partitioned.boundary_links):
            raise ValueError("schedule must assign one root to each boundary link")
        suppressed = schedule.suppressed_links(node_index)
        residues = [node % prime for node in schedule.nodes]
        if 0 in residues or len(set(residues)) != len(residues):
            raise ValueError("nodes must be distinct and nonzero modulo prime")
        if any(len(block.link_indices) % prime == 0 for block in self.partitioned.measurement_blocks):
            raise ValueError("measurement block size must be invertible modulo prime")
        q, k = schedule.orders[node_index], len(suppressed)
        subset_counts = [comb(k, r) for r in range(min(k, q) + 1)]
        requested = sum(subset_counts)
        trace = dict(prime=prime, node_index=node_index, node=schedule.nodes[node_index],
                     k=k, q=q, requested_subsets=requested, processed_subsets=0,
                     full_set_processed=False, conditioning_states_sum=0,
                     conditioning_states_max=0, conditioning_states=[],
                     residual_active_bits=[], subsets_by_order=[0] * len(subset_counts),
                     estimated_work=0, moment_tables_built=0, moment_cache_hits=0,
                     subset_batches=0, moment_batches_built=0, status="pending")
        self.stats.node_evaluations.append(trace)
        self.stats.max_jet_order = max(self.stats.max_jet_order, q)
        if any(table.satisfying_assignments == 0 for table in self.tables):
            trace["status"] = "zero_local_factor"
            self.stats.evaluations += 1
            return [0] * (q + 1)
        if requested > self.max_subset_terms:
            trace["status"] = "resource_limit"
            raise ResourceLimitError(f"node {node_index} requires {requested} suppressed subsets; "
                                     f"max_subset_terms is {self.max_subset_terms}")
        start = perf_counter()
        self._moment_cache.clear()
        self._cache_bytes = self._condition_bytes = self._jet_live_bytes = 0
        try:
            contexts = self._conditions(node_index, prime, schedule)
            states = [context.states for context in contexts]
            active = [context.active_bits for context in contexts]
            trace["conditioning_states"] = states
            trace["residual_active_bits"] = active
            state_product = prod(states)
            trace["conditioning_states_max"] = state_product
            widths = [len(table.proxy_bits) for table in self.tables]
            base_work = state_product * (sum(widths) + 1)
            local_work = sum(table.counts.numel() for table in self.tables)
            estimate = sum(count * (base_work * (q - r + 1) ** 2
                                    + local_work * (r + 1))
                           for r, count in enumerate(subset_counts))
            trace["estimated_work"] = estimate
            self.stats.estimated_work = max(self.stats.estimated_work, estimate)
            if estimate > self.max_jet_work:
                trace["status"] = "resource_limit"
                raise ResourceLimitError(f"node {node_index} estimates {estimate} jet operations; "
                                         f"max_jet_work is {self.max_jet_work}")
            # Account jointly for kernels, retained moments, and contraction
            # arrays rather than only checking each allocation in isolation.
            kernel_elements = 0
            for source in range(len(self.tables)):
                for target in range(len(self.tables)):
                    if source == target:
                        continue
                    affine_links = sum(
                        link.master_group == source and link.proxy_group == target
                        and schedule.link_nodes[i] != node_index
                        for i, link in enumerate(self.partitioned.boundary_links))
                    kernel_elements += (states[source] * widths[target]
                                        * (min(q, affine_links) + 1))
            moment_elements = (sum(state * width for state, width in zip(states, widths))
                               * min(self.subset_batch_size, requested))
            if len(self.tables) == 2:
                contraction_elements = state_product * 2 * (q + 1)
            else:
                contraction_elements = (min(state_product, self.signature_batch_size)
                                        * (sum(widths) + len(states)) * (q + 1))
            self._memory(kernel_elements * 48 + moment_elements * 48
                         + contraction_elements * 32 + self.max_memory_bytes // 8,
                         "joint jet kernel/moment/contraction workspace", 1)
            result = [0] * (q + 1)
            for subset_size in range(min(k, q) + 1):
                iterator = iter(combinations(suppressed, subset_size))
                while batch := list(islice(iterator, self.subset_batch_size)):
                    batch_moments = [self._moment_batch(group, batch, prime, contexts, trace)
                                     for group in range(len(self.tables))]
                    trace["subset_batches"] += 1
                    for position, subset in enumerate(batch):
                        moments = [values[position] for values in batch_moments]
                        limit = q - subset_size
                        contraction = (self._two_group_contract if len(self.tables) == 2
                                       else self._many_group_contract)
                        coefficients = contraction(moments, frozenset(subset), node_index,
                                                   prime, schedule, contexts, limit)
                        for degree, coefficient in enumerate(coefficients):
                            result[degree + subset_size] = (result[degree + subset_size] + coefficient) % prime
                        trace["processed_subsets"] += 1
                        trace["subsets_by_order"][subset_size] += 1
                        trace["conditioning_states_sum"] += state_product
                        trace["full_set_processed"] |= subset_size == k
            free_factor = pow(2, self.stats.free_variables, prime)
            result = [coefficient * free_factor % prime for coefficient in result]
            trace["status"] = "ok"
            self.stats.conditioning_states.append(states)
            self.stats.residual_active_bits.append(active)
            self.stats.subset_terms_processed += trace["processed_subsets"]
            self.stats.signature_subset_visits += trace["conditioning_states_sum"]
            self.stats.evaluations += 1
            self.stats.evaluation_seconds += perf_counter() - start
            return result
        except ResourceLimitError:
            trace["status"] = "resource_limit"
            raise
        finally:
            self._moment_cache.clear()
            self._cache_bytes = self._condition_bytes = self._jet_live_bytes = 0


def jet_count_with_stats(cnf: CNF, *, device: str = "auto", num_groups: int = 2,
                         block_size: int = 4, max_degree: int | None = None,
                         max_width: int = 3, num_nodes: int = 2,
                         mode: str = "hermite", nodes: Sequence[int] | None = None,
                         orders: Sequence[int] | None = None,
                         **engine_options) -> tuple[int, JetStats]:
    """Count with subset jets, confluent Hermite interpolation, and exact CRT."""
    cnf.validate()
    original_variables = cnf.num_vars
    transformed = reduce_clause_width_deterministic(cnf, max_width=max_width)
    if max_degree is not None:
        transformed = split_high_degree_variables(transformed, max_degree=max_degree)
    partitioned = partition_and_decouple_boundary(transformed, num_groups=num_groups,
                                                 block_size=block_size)
    counter = JetCounter(partitioned, device=device, **engine_options)
    schedule = create_hermite_schedule(len(partitioned.boundary_links), num_nodes=num_nodes,
                                       mode=mode, nodes=nodes, orders=orders)
    counter.stats.schedule_mode = mode
    counter.stats.interpolation_nodes = len(schedule.nodes)
    if not partitioned.boundary_links:
        exact = prod(table.satisfying_assignments for table in counter.tables)
        return exact << counter.stats.free_variables, counter.stats
    primes = primes_for_bound(1 << original_variables)
    counter.stats.primes = primes
    residues = []
    for prime in primes:
        jets = [counter.evaluate_jet(index, prime, schedule) for index in range(len(schedule.nodes))]
        residues.append(interpolate_hermite_at_zero(jets, list(schedule.nodes),
                                                   list(schedule.orders), prime))
    result = crt_reconstruct(residues, primes)
    if result > 1 << original_variables:
        raise ArithmeticError("reconstructed count exceeds original assignment bound")
    return result, counter.stats


def jet_sharp_sat(cnf: CNF, **options) -> int:
    """Return the exact model count using suppressed-subset Hasse jets."""
    return jet_count_with_stats(cnf, **options)[0]

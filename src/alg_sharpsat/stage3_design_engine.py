"""Bounded CPU reference for certified occurrence designs and Mobius jets.

Only local group assignments are enumerated. Each exact integer measurement
tuple supplies representatives for the other groups; original clause factors
are evaluated over a truncated polynomial ring, including non-Boolean values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from math import comb, prod
from time import perf_counter

from .stage2_algebraic import crt_reconstruct, primes_for_bound, validate_prime
from .stage2_design import ReplacementDesign, validate_design
from .stage2_hermite import interpolate_hermite_at_zero
from .stage3_gpu_engine import ResourceLimitError, UnsupportedDeviceError


def _field(value, prime):
    value = Fraction(value)
    if value.denominator % prime == 0:
        raise ValueError('replacement denominator is zero modulo prime')
    return value.numerator * pow(value.denominator, -1, prime) % prime


@dataclass
class DesignStats:
    device: str
    degree_bound: int
    nodes: tuple[int, ...]
    orders: tuple[int, ...]
    representation: str = 'certified_occurrence_design_cpu_reference'
    primes: tuple[int, ...] = ()
    node_evaluations: list[dict] = field(default_factory=list)
    local_assignment_visits: int = 0
    inner_evaluations: int = 0
    build_seconds: float = 0.0
    evaluation_seconds: float = 0.0


class DesignCounter:
    def __init__(self, design: ReplacementDesign, device: str = 'auto', *,
                 orders: tuple[int, ...] | None = None, max_variables: int = 12,
                 max_work: int = 2_000_000, max_subset_terms: int = 1_000_000,
                 max_local_assignments: int = 1_000_000,
                 max_memory_bytes: int = 512*1024*1024):
        if device not in ('cpu', 'auto'):
            raise UnsupportedDeviceError('certified design reference supports CPU only')
        for name, value in (('max_variables', max_variables), ('max_work', max_work),
                            ('max_subset_terms', max_subset_terms),
                            ('max_local_assignments', max_local_assignments),
                            ('max_memory_bytes', max_memory_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if design.num_vars > max_variables:
            raise ResourceLimitError('certified design exceeds max_variables')
        start = perf_counter()
        validate_design(design)
        self.design, self.device = design, 'cpu'
        self.max_work, self.max_subset_terms = max_work, max_subset_terms
        self.max_local_assignments, self.max_memory_bytes = max_local_assignments, max_memory_bytes
        self.n, self.degree = design.num_vars, design.degree_bound
        if orders is None:
            capacity, extra = divmod(max(self.degree+1, len(design.nodes)), len(design.nodes))
            orders = tuple(capacity-1+int(i < extra) for i in range(len(design.nodes)))
        if (len(orders) != len(design.nodes)
                or any(type(order) is not int or order < 0 for order in orders)
                or sum(order+1 for order in orders) <= self.degree):
            raise ValueError('orders must supply enough nonnegative Taylor data for the degree bound')
        self.orders = tuple(orders)
        self.stats = DesignStats('cpu', self.degree, tuple(design.nodes), self.orders)
        self.stats.build_seconds = perf_counter()-start

    def _multiply(self, left, right, order, prime, trace):
        size = min(order+1, len(left)+len(right)-1)
        trace['arithmetic_work'] += sum(min(len(right), size-i)
                                        for i in range(min(size, len(left))))
        if trace['arithmetic_work'] > self.max_work:
            raise ResourceLimitError('certified design ring arithmetic exceeds max_work')
        out = [0]*size
        for i, a in enumerate(left):
            for j in range(min(len(right), size-i)):
                out[i+j] = (out[i+j]+a*right[j]) % prime
        while len(out) > 1 and not out[-1]:
            out.pop()
        return out

    def evaluate_jet(self, node_index: int, prime: int, order: int | None = None) -> list[int]:
        validate_prime(prime)
        if type(node_index) is not int or not 0 <= node_index < len(self.design.nodes):
            raise ValueError('node_index is outside the design')
        order = self.orders[node_index] if order is None else order
        if type(order) is not int or order < 0:
            raise ValueError('order must be a nonnegative integer')
        design = self.design
        node = design.nodes[node_index]
        partition = design.groups_by_node[node_index]
        measurements = design.measurements_by_node[node_index]
        owners = design.clause_owners_by_node[node_index]
        occurrences = sum(map(len, design.occurrence_rows))
        rows = sum(1 << len(group) for group in partition)
        if rows > self.max_local_assignments:
            raise ResourceLimitError('certified design local rows exceed max_local_assignments')
        cells = (order+1)*max(1, occurrences*self.n)
        if cells > self.max_work or 64*cells > self.max_memory_bytes:
            raise ResourceLimitError('certified design jet storage exceeds resource budget')
        measurement_width = sum(map(len, measurements))
        if 64*rows*(self.n+measurement_width+1) > self.max_memory_bytes:
            raise ResourceLimitError('certified design local assignment storage exceeds max_memory_bytes')
        pairs = sum(comb(self.n, size)*(1 << size) for size in range(min(order, self.n)+1))
        if pairs > self.max_subset_terms or pairs > self.max_work:
            raise ResourceLimitError('certified design Boolean substitutions exceed subset budget')
        trace = dict(node_index=node_index, node=node, prime=prime, order=order,
                     status='pending', arithmetic_work=0, local_assignment_visits=0,
                     group_assignment_visits=[0]*len(partition), inner_evaluations=0,
                     measurement_tuples=0, local_group_sizes=list(map(len, partition)),
                     local_assignment_rows=[1 << len(group) for group in partition],
                     feasible_measurement_states=[])
        self.stats.node_evaluations.append(trace)
        jets = []
        for clause_rows in design.occurrence_rows:
            clause_jets = []
            for occurrence in clause_rows:
                degree = max(map(len, occurrence), default=1)-1
                clause_jets.append(tuple(tuple(sum(
                    comb(m, j)*pow(node, m-j, prime)*_field(polynomial[m], prime)
                    for m in range(j, len(polynomial))) % prime
                    for polynomial in occurrence) for j in range(min(order, degree)+1)))
            jets.append(tuple(clause_jets))
        buckets = []
        scopes = []
        for variables, matrix in zip(partition, measurements):
            scope = sum(1 << var for var in variables)
            scopes.append(scope)
            bucket = {}
            for bits in product((0, 1), repeat=len(variables)):
                mask = sum(bit << var for bit, var in zip(bits, variables))
                key = tuple(sum(row[var]*bit for bit, var in zip(bits, variables)) for row in matrix)
                bucket.setdefault(key, []).append(mask)
            buckets.append(bucket)
        trace['feasible_measurement_states'] = list(map(len, buckets))
        result = [0]*(order+1)
        try:
            for size in range(min(order, self.n)+1):
                for selected in combinations(range(self.n), size):
                    forced = sum(1 << var for var in selected)
                    allowed = [{key: [mask for mask in masks if mask & (forced & scope) == forced & scope]
                                for key, masks in bucket.items()}
                               for bucket, scope in zip(buckets, scopes)]
                    allowed = [{key: masks for key, masks in bucket.items() if masks} for bucket in allowed]
                    subset = forced
                    while True:
                        trace['inner_evaluations'] += 1
                        corrections = tuple(tuple(tuple([0]+[
                            sum(row[var] for var in range(self.n) if subset >> var & 1) % prime
                            for row in occurrence[1:]]) for occurrence in clause) for clause in jets)
                        for keys in product(*(tuple(bucket) for bucket in allowed)):
                            trace['measurement_tuples'] += 1
                            representative = 0
                            for bucket, key in zip(buckets, keys):
                                representative |= bucket[key][0]
                            combined = [1]
                            for group, (scope, key) in enumerate(zip(scopes, keys)):
                                local_sum = [0]*(order+1)
                                for mask in allowed[group][key]:
                                    trace['local_assignment_visits'] += 1
                                    trace['group_assignment_visits'][group] += 1
                                    if trace['local_assignment_visits'] > self.max_work:
                                        raise ResourceLimitError('certified design local visits exceed max_work')
                                    assignment = (representative & ~scope) | mask
                                    value = [1]
                                    for clause_index, (clause, clause_jets) in enumerate(zip(design.clauses, jets)):
                                        if owners[clause_index] != group:
                                            continue
                                        failure = [1]
                                        for occurrence_index, (literal, occurrence) in enumerate(zip(
                                                clause.literals, clause_jets)):
                                            replacement = list(corrections[clause_index][occurrence_index])
                                            replacement[0] = sum(a for var, a in enumerate(occurrence[0])
                                                                 if assignment >> var & 1) % prime
                                            if literal.pos:
                                                replacement = [-a % prime for a in replacement]
                                                replacement[0] = (replacement[0]+1) % prime
                                            failure = self._multiply(failure, replacement, order, prime, trace)
                                        satisfaction = [-a % prime for a in failure]
                                        satisfaction[0] = (satisfaction[0]+1) % prime
                                        value = self._multiply(value, satisfaction, order, prime, trace)
                                    for j, a in enumerate(value):
                                        local_sum[j] = (local_sum[j]+a) % prime
                                combined = self._multiply(combined, local_sum, order, prime, trace)
                            sign = (-1)**(size-subset.bit_count())
                            for j, a in enumerate(combined):
                                result[j] = (result[j]+sign*a) % prime
                        if subset == 0:
                            break
                        subset = (subset-1) & forced
            trace['status'] = 'ok'
        except ResourceLimitError:
            trace['status'] = 'resource_limit'
            raise
        self.stats.local_assignment_visits += trace['local_assignment_visits']
        self.stats.inner_evaluations += trace['inner_evaluations']
        return result

    def count(self) -> int:
        start = perf_counter()
        size = sum(order+1 for order in self.orders)
        if size**3 > self.max_work or 64*size*size > self.max_memory_bytes:
            raise ResourceLimitError('certified design Hermite interpolation exceeds resource budget')
        candidates = primes_for_bound(1 << self.n)
        denominators = {Fraction(value).denominator for clause in self.design.occurrence_rows
                        for occurrence in clause for polynomial in occurrence for value in polynomial}
        self.stats.primes = tuple(prime for prime in candidates
                                  if all(denominator % prime for denominator in denominators)
                                  and 0 not in {node % prime for node in self.design.nodes}
                                  and len({node % prime for node in self.design.nodes}) == len(self.design.nodes))
        if prod(self.stats.primes) <= 1 << self.n:
            raise ResourceLimitError('insufficient admissible CRT primes for certified design')
        residues = []
        for prime in self.stats.primes:
            jets = [self.evaluate_jet(index, prime) for index in range(len(self.design.nodes))]
            residues.append(interpolate_hermite_at_zero(jets, self.design.nodes, self.orders, prime))
        count = crt_reconstruct(residues, self.stats.primes)
        if count > 1 << self.n:
            raise ArithmeticError('certified design count exceeds the Boolean assignment bound')
        self.stats.evaluation_seconds = perf_counter()-start
        return count


def design_sharp_sat(design: ReplacementDesign, **options) -> int:
    return DesignCounter(design, **options).count()

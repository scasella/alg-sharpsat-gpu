"""Explicit occurrence replacements with exact restoration/locality certificates.

These bounded rational designs expose the finite identities used by the source
algorithm. They do not assert its balancing or asymptotic parameter guarantees.
Polynomial coefficients are stored in ascending degree, separately for every
original variable in every literal occurrence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .cnf import CNF, Clause, Literal


@dataclass(frozen=True)
class ReplacementDesign:
    num_vars: int
    clauses: tuple[Clause, ...]
    occurrence_rows: tuple
    nodes: tuple[int, ...]
    groups_by_node: tuple
    clause_owners_by_node: tuple
    measurements_by_node: tuple
    metadata: dict = field(default_factory=dict, compare=False)

    @property
    def degree_bound(self) -> int:
        """Sum the actual maximum coefficient degree of each occurrence."""
        return sum(max((degree for polynomial in occurrence
                        for degree, coefficient in enumerate(polynomial) if coefficient), default=0)
                   for clause in self.occurrence_rows for occurrence in clause)


def _rational(value) -> Fraction:
    if not isinstance(value, (int, Fraction)):
        raise ValueError('replacement coefficients must be exact rational numbers')
    return Fraction(value)


def _span_basis(rows, width):
    basis = []
    for row in rows:
        vector = [Fraction(value) for value in row]
        for pivot, existing in basis:
            factor = vector[pivot]
            if factor:
                vector = [a-factor*b for a, b in zip(vector, existing)]
        pivot = next((i for i in range(width) if vector[i]), None)
        if pivot is not None:
            factor = vector[pivot]
            normalized = [value/factor for value in vector]
            basis = [(old_pivot, [a-existing[pivot]*b for a, b in zip(existing, normalized)])
                     for old_pivot, existing in basis]
            basis.append((pivot, normalized))
            basis.sort(key=lambda item: item[0])
    return basis


def _in_span(row, basis):
    vector = list(row)
    for pivot, existing in basis:
        factor = vector[pivot]
        if factor:
            vector = [a-factor*b for a, b in zip(vector, existing)]
    return not any(vector)


def validate_design(design: ReplacementDesign) -> None:
    """Check shapes, Boolean restoration, and exact rational row-span locality."""
    if not isinstance(design, ReplacementDesign):
        raise ValueError('expected a ReplacementDesign')
    CNF(design.num_vars, list(design.clauses)).validate()
    n, nodes = design.num_vars, design.nodes
    if (not nodes or any(type(node) is not int or node == 0 for node in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError('nodes must be distinct nonzero integers')
    if len(design.occurrence_rows) != len(design.clauses):
        raise ValueError('one replacement-row collection is required for each clause')
    for clause, rows in zip(design.clauses, design.occurrence_rows):
        if len(rows) != len(clause.literals):
            raise ValueError('each literal occurrence needs its own replacement row')
        for literal, occurrence in zip(clause.literals, rows):
            if len(occurrence) != n:
                raise ValueError('replacement rows must have one polynomial per original variable')
            for var, polynomial in enumerate(occurrence):
                if not polynomial:
                    raise ValueError('coefficient polynomials must be nonempty')
                coefficients = [_rational(value) for value in polynomial]
                if coefficients[0] != int(var == literal.var):
                    raise ValueError('replacement does not restore its original variable at zero')
    if any(len(collection) != len(nodes) for collection in (
            design.groups_by_node, design.clause_owners_by_node, design.measurements_by_node)):
        raise ValueError('partitions, owners and measurements must cover every node')
    group_count = len(design.groups_by_node[0])
    if group_count < 1:
        raise ValueError('each node needs at least one group')
    for node, partition, owners, measurements in zip(
            nodes, design.groups_by_node, design.clause_owners_by_node, design.measurements_by_node):
        if len(partition) != group_count or len(measurements) != group_count:
            raise ValueError('all nodes must have the same number of groups and measurement collections')
        flattened = [var for group in partition for var in group]
        if any(type(var) is not int for var in flattened) or sorted(flattened) != list(range(n)):
            raise ValueError('groups must partition all original variables without overlap')
        if len(owners) != len(design.clauses) or any(
                type(owner) is not int or not 0 <= owner < group_count for owner in owners):
            raise ValueError('each clause needs a valid owner group at every node')
        measurement_rows = []
        for group, rows in zip(partition, measurements):
            for row in rows:
                if len(row) != n or any(type(value) is not int for value in row):
                    raise ValueError('measurements must be integer rows on original variables')
                if any(value and var not in group for var, value in enumerate(row)):
                    raise ValueError('a group measurement reads a variable outside that group')
                measurement_rows.append(row)
        bases = {}
        for owner in set(owners):
            units = [tuple(int(i == var) for i in range(n)) for var in partition[owner]]
            bases[owner] = _span_basis([*measurement_rows, *units], n)
        for owner, clause_rows in zip(owners, design.occurrence_rows):
            for occurrence in clause_rows:
                evaluated = [sum(_rational(value)*node**degree
                                 for degree, value in enumerate(polynomial))
                             for polynomial in occurrence]
                if not _in_span(evaluated, bases[owner]):
                    raise ValueError('replacement is outside the owner-coordinate and measurement row span')


def build_swap_design(cnf: CNF, *, num_groups: int = 2, num_nodes: int = 2,
                      interval: tuple[int, int] = (0, 1), seed: int = 17) -> ReplacementDesign:
    """Expose the bounded occurrence SWAP construction as a certified design.

    The deterministic packing uses no randomness; ``seed`` is retained as
    provenance for callers of the earlier experimental interface. ``interval``
    must be the typical-weight window selected by this bounded matrix builder.
    """
    from .stage4_swap_mobius import OccurrenceSwapCounter

    if (not isinstance(interval, tuple) or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or interval[0] < 0 or interval[1] <= interval[0]):
        raise ValueError('interval must contain increasing nonnegative integer endpoints')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    counter = OccurrenceSwapCounter(cnf, d=num_groups, L=num_nodes,
                                     b=interval[1]-interval[0],
                                     max_variables=max(1, cnf.num_vars))
    if counter.system.polynomial.window != tuple(range(interval[0], interval[1]+1)):
        raise ValueError('interval differs from the supported typical-weight window')
    clauses = tuple(Clause(tuple(Literal(var, pos) for var, pos in clause))
                    for clause in counter.clauses)
    occurrence_rows = []
    for clause in counter.occurrences:
        rows = []
        for occurrence in clause:
            coefficients = counter.occurrence_coefficients(occurrence)
            polynomials = []
            for var in range(cnf.num_vars):
                polynomial = [Fraction(row[var]) for row in coefficients]
                while len(polynomial) > 1 and not polynomial[-1]:
                    polynomial.pop()
                polynomials.append(tuple(polynomial))
            rows.append(tuple(polynomials))
        occurrence_rows.append(tuple(rows))
    groups, owners, measurements = [], [], []
    for node in range(num_nodes):
        partition = tuple(tuple(var for var, alpha in enumerate(counter.alpha) if alpha[node] == group)
                          for group in range(num_groups))
        groups.append(partition)
        owners.append(tuple(beta[node] for beta in counter.beta))
        measurements.append(tuple(tuple(
            tuple(value if var in group else 0 for var, value in enumerate(row))
            for row in counter.measurement_rows if any(row[var] for var in group))
            for group in partition))
    design = ReplacementDesign(cnf.num_vars, clauses, tuple(occurrence_rows),
                               tuple(range(1, num_nodes+1)), tuple(groups), tuple(owners),
                               tuple(measurements), metadata={
                                   'construction': 'occurrence_coherent_swap', 'seed': seed,
                                   'block_count': len(counter.blocks), 'interval': interval,
                                   'asymptotic_parameter_guarantees_applied': False})
    validate_design(design)
    return design

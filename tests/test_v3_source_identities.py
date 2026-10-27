"""Source-level checks using original-variable polynomial scalar oracles."""

from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import comb, prod

import pytest

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.stage2_algebraic import PRIMES
from alg_sharpsat.stage2_design import ReplacementDesign, build_swap_design, validate_design
from alg_sharpsat.stage2_hermite import interpolate_hermite_at_zero
from alg_sharpsat.stage3_design_engine import DesignCounter, design_sharp_sat


pytestmark = pytest.mark.g10


def _field(value, prime):
    value = Fraction(value)
    return value.numerator * pow(value.denominator, -1, prime) % prime


def _multiply(left, right, prime):
    answer = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            answer[i + j] = (answer[i + j] + a * b) % prime
    return answer


def _direct_h_coefficients(design, prime):
    """Sum literal polynomials over original assignments; no engine routines."""
    answer = [0] * (design.degree_bound + 1)
    for bits in product((0, 1), repeat=design.num_vars):
        term = [1]
        for clause, occurrence_rows in zip(design.clauses, design.occurrence_rows):
            failure = [1]
            for literal, variable_polynomials in zip(clause.literals, occurrence_rows):
                length = max(map(len, variable_polynomials), default=1)
                replacement = [sum(bits[var] * _field(polynomial[degree], prime)
                                   for var, polynomial in enumerate(variable_polynomials)
                                   if degree < len(polynomial)) % prime
                               for degree in range(length)]
                # A positive literal fails at 1-P; a negative literal fails at P.
                if literal.pos:
                    replacement = [(-coefficient) % prime for coefficient in replacement]
                    replacement[0] = (replacement[0] + 1) % prime
                failure = _multiply(failure, replacement, prime)
            satisfaction = [(-coefficient) % prime for coefficient in failure]
            satisfaction[0] = (satisfaction[0] + 1) % prime
            term = _multiply(term, satisfaction, prime)
        for degree, coefficient in enumerate(term):
            answer[degree] = (answer[degree] + coefficient) % prime
    return answer


def _scalar_jet(coefficients, node, order, prime):
    return [sum(value * comb(degree, j) * pow(node, degree - j, prime)
                for degree, value in enumerate(coefficients) if degree >= j) % prime
            for j in range(order + 1)]


def _mean_replacement_design(num_vars=4):
    """Dense genuine replacements at two nodes, with changing partitions.

    Every occurrence is mean+p(T)*(x_v-mean), p=(1-T)*(1-T/2).
    At both nodes it is the global mean, certified by split group sums.
    This demonstrates state reduction at a node, not a total running-time gain.
    """
    clauses = (
        Clause((Literal(0, True), Literal(1, False), Literal(2, True))),
        Clause((Literal(0, False), Literal(2, True), Literal(3, True))),
        Clause((Literal(0, True), Literal(1, True))),
    )
    groups = ((tuple(range(num_vars // 2)), tuple(range(num_vars // 2, num_vars))),
              (tuple(range(0, num_vars, 2)), tuple(range(1, num_vars, 2))))
    measurements = tuple(tuple((tuple(int(var in group) for var in range(num_vars)),)
                               for group in partition) for partition in groups)
    rows = tuple(tuple(tuple(
        (Fraction(int(var == literal.var)),
         Fraction(-3, 2) * (int(var == literal.var) - Fraction(1, num_vars)),
         Fraction(1, 2) * (int(var == literal.var) - Fraction(1, num_vars)))
        for var in range(num_vars)) for literal in clause.literals) for clause in clauses)
    return ReplacementDesign(
        num_vars=num_vars, clauses=clauses, occurrence_rows=rows, nodes=(1, 2),
        groups_by_node=groups, clause_owners_by_node=((0, 0, 0), (1, 1, 1)),
        measurements_by_node=measurements,
    )


@pytest.mark.parametrize("prime", (*PRIMES, 3))
def test_dense_nonboolean_overlapping_original_occurrences_match_all_coefficients(prime):
    design = _mean_replacement_design()
    validate_design(design)
    coefficients = _direct_h_coefficients(design, prime)
    counter = DesignCounter(design, device="cpu")
    jets = []
    for index, node in enumerate(design.nodes):
        order = counter.orders[index]
        jet = counter.evaluate_jet(index, prime)
        assert jet == _scalar_jet(coefficients, node, order, prime)
        jets.append(jet)
    expected = brute_force_count(CNF(design.num_vars, list(design.clauses)))
    assert coefficients[0] == expected % prime
    assert interpolate_hermite_at_zero(jets, design.nodes, counter.orders, prime) == expected % prime
    # An assignment of Hamming weight two gives replacement=1/2 at either
    # node. Boolean clause masks here would silently change the polynomial.
    at_one = sum(_field(value, prime)
                 for row in design.occurrence_rows[0][0][:2] for value in row) % prime
    assert at_one == pow(2, -1, prime)
    assert at_one not in (0, 1)
    assert design.num_vars == 4  # All three occurrences of x_0 are the same bit.


def test_original_variable_mobius_recovers_integer_count():
    design = _mean_replacement_design()
    assert design_sharp_sat(design, device="cpu") == brute_force_count(
        CNF(design.num_vars, list(design.clauses)))


@pytest.mark.parametrize("num_vars", [4, 6, 8])
def test_certified_mean_family_reduces_node_signatures_with_honest_finite_cost(num_vars):
    design = _mean_replacement_design(num_vars)
    validate_design(design)
    counter = DesignCounter(design, device="cpu")
    prime = PRIMES[0]
    coefficients = _direct_h_coefficients(design, prime)
    for node_index, partition in enumerate(design.groups_by_node):
        feasible = [len({sum(bits) for bits in product((0, 1), repeat=len(group))})
                    for group in partition]
        assert prod(feasible) == prod(len(group) + 1 for group in partition)
        assert prod(feasible) < 2 ** num_vars
        assert counter.evaluate_jet(node_index, prime, order=0) == _scalar_jet(
            coefficients, design.nodes[node_index], 0, prime)
    # Local measurement-state reduction alone does not prove end-to-end speed:
    # these concrete orders reach the full 3**n Boolean-substitution budget.
    q = design.degree_bound // len(design.nodes)
    pairs = sum((2 ** j) * comb(num_vars, j) for j in range(min(q, num_vars) + 1))
    if q >= num_vars:
        assert pairs == 3 ** num_vars


def test_invalid_restoration_certificate_is_rejected():
    design = _mean_replacement_design()
    rows = list(design.occurrence_rows)
    first_clause = list(rows[0])
    first_occurrence = list(first_clause[0])
    first_occurrence[0] = (Fraction(0), *first_occurrence[0][1:])
    first_clause[0] = tuple(first_occurrence)
    rows[0] = tuple(first_clause)
    with pytest.raises(ValueError):
        validate_design(replace(design, occurrence_rows=tuple(rows)))


def test_invalid_cross_group_span_certificate_is_rejected():
    design = _mean_replacement_design()
    # Removing the other group's sum leaves a genuinely nonlocal dense input.
    measurements = list(design.measurements_by_node)
    measurements[0] = (measurements[0][0], ())
    with pytest.raises(ValueError):
        validate_design(replace(design, measurements_by_node=tuple(measurements)))


@pytest.mark.parametrize("change", [
    {"nodes": (1, 1)}, {"nodes": (0, 2)},
    {"groups_by_node": (((0, 1), (1, 2, 3)), ((0, 2), (1, 3)))},
    {"groups_by_node": (((0, 1), (2,)), ((0, 2), (1, 3)))},
    {"clause_owners_by_node": ((0, 2, 0), (0, 0, 0))},
    {"measurements_by_node": ((((1, 1, 1, 0),), ((0, 0, 1, 1),)),
                               (((1, 0, 1, 0),), ((0, 1, 0, 1),)))},
])
def test_invalid_design_shape_and_group_certificates(change):
    with pytest.raises(ValueError):
        validate_design(replace(_mean_replacement_design(), **change))


def test_true_swap_polynomial_construction_matches_direct_polynomial():
    cnf = CNF(4, [Clause((Literal(0, True), Literal(1, False))),
                  Clause((Literal(0, False), Literal(2, True))),
                  Clause((Literal(1, True), Literal(3, True)))])
    design = build_swap_design(cnf, num_groups=2, num_nodes=2, interval=(0, 1), seed=17)
    validate_design(design)
    prime = PRIMES[0]
    coefficients = _direct_h_coefficients(design, prime)
    counter = DesignCounter(design, device="cpu")
    for index, node in enumerate(design.nodes):
        assert counter.evaluate_jet(index, prime) == _scalar_jet(
            coefficients, node, counter.orders[index], prime)
    assert design_sharp_sat(design, device="cpu") == brute_force_count(cnf)

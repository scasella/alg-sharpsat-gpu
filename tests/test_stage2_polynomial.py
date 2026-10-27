"""Independent finite-field polynomial checks, including nonconstant H(T)."""

from itertools import product
from math import prod

import pytest

from alg_sharpsat.cnf import Clause, Literal
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF
from alg_sharpsat.stage2_algebraic import (
    PRIMES,
    SuppressionSchedule,
    create_suppression_schedule,
    crt_reconstruct,
    direct_h,
    interpolate_at_zero,
    lagrange_weights_at_zero,
    linear_form,
    link_factor,
    primes_for_bound,
    suppression_value,
    validate_prime,
)

pytestmark = pytest.mark.g4


def _multiply(left, right, prime):
    """Schoolbook coefficient multiplication, independent of Stage2 evaluators."""
    result = [0] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[i + j] = (result[i + j] + x * y) % prime
    return result


def _evaluate(coefficients, t, prime):
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * t + coefficient) % prime
    return result


def _symbolic_h(partitioned, schedule, prime):
    """Build H's coefficients by summing assignment-polynomials explicitly."""
    answer = [0] * (schedule.degree + 1)
    blocks = {index: block for block in partitioned.measurement_blocks
              for index in block.link_indices}
    clauses = [clause for group in partitioned.group_clauses for clause in group]
    for bits in product((0, 1), repeat=partitioned.num_vars):
        values = bits
        if not all(any(values[lit.var] == int(lit.pos) for lit in clause.literals)
                   for clause in clauses):
            continue
        term = [1]
        for index, link in enumerate(partitioned.boundary_links):
            block = blocks[index]
            measurement = sum(values[partitioned.boundary_links[j].master_var]
                              for j in block.link_indices)
            mean = measurement * pow(len(block.link_indices), -1, prime) % prime
            suppression_coefficients = [1]
            for root in schedule.roots[index]:
                suppression_coefficients = _multiply(
                    suppression_coefficients, [1, -pow(root, -1, prime)], prime)
            sign = 2 * values[link.proxy_var] - 1
            factor = [(sign * (values[link.master_var] - mean) * coefficient) % prime
                      for coefficient in suppression_coefficients]
            factor[0] = (factor[0] + 1 - values[link.proxy_var] + sign * mean) % prime
            term = _multiply(term, factor, prime)
        for index, coefficient in enumerate(term):
            answer[index] = (answer[index] + coefficient) % prime
    while len(answer) > 1 and answer[-1] == 0:
        answer.pop()
    return answer


def _repeated_master_partition():
    # Master 0 is linked into two target groups; its bit appears twice in m.
    return PartitionedCNF(
        num_vars=5,
        group_vars=((0, 1), (2, 3), (4,)),
        group_clauses=(
            (Clause((Literal(0, True),)), Clause((Literal(1, False),))),
            (Clause((Literal(2, True),)), Clause((Literal(3, False),))),
            (Clause((Literal(4, True),)),),
        ),
        boundary_links=(BoundaryLink(0, 0, 2, 1), BoundaryLink(1, 0, 3, 1),
                        BoundaryLink(0, 0, 4, 2)),
        measurement_blocks=(MeasurementBlock((0, 1, 2), 0),),
        original_num_vars=2,
    )


@pytest.mark.parametrize("prime", PRIMES)
@pytest.mark.parametrize("degree", [0, 1, 2, 4])
def test_symbolic_h_has_actual_degree_and_interpolates(prime, degree):
    partitioned = _repeated_master_partition()
    schedule = create_suppression_schedule(3, degree)
    coefficients = _symbolic_h(partitioned, schedule, prime)
    # The fixed mixed master assignment ensures the leading coefficient is
    # nonzero, so this checks the actual degree, not just the schedule bound.
    assert len(coefficients) - 1 == schedule.degree
    values = [direct_h(partitioned, t, prime, schedule) for t in schedule.nodes]
    for t, value in zip(schedule.nodes, values):
        assert value == _evaluate(coefficients, t, prime)
    assert direct_h(partitioned, 0, prime, schedule) == coefficients[0] == 1
    assert interpolate_at_zero(values, schedule.nodes, prime) == 1


@pytest.mark.parametrize("prime", PRIMES)
@pytest.mark.parametrize("roots", [((1, 2), (2,), (3, 4)), ((1, 1), (), (2, 3, 4))])
def test_custom_unequal_degrees_and_shared_or_repeated_roots(prime, roots):
    partitioned = _repeated_master_partition()
    schedule = SuppressionSchedule(tuple(range(1, 7)), roots)
    coefficients = _symbolic_h(partitioned, schedule, prime)
    assert len(coefficients) - 1 == 5
    values = [direct_h(partitioned, node, prime, schedule) for node in schedule.nodes]
    for node, value in zip(schedule.nodes, values):
        assert value == _evaluate(coefficients, node, prime)
    assert interpolate_at_zero(values, schedule.nodes, prime) == 1


@pytest.mark.parametrize("prime", PRIMES)
@pytest.mark.g3
def test_link_invariants_and_repeated_master_measurement(prime):
    for roots in ((), (1,), (1, 2, 3)):
        assert suppression_value(roots, 0, prime) == 1
        for root in roots:
            assert suppression_value(roots, root, prime) == 0
    for x, y in product((0, 1), repeat=2):
        transformed = linear_form(x, 2, 3, 1, prime)
        assert transformed == x
        assert link_factor(transformed, y, prime) == int(x == y)
    # m=1+0+1=2 and k=3; repeated master occurrences are not deduplicated.
    mean = 2 * pow(3, -1, prime) % prime
    assert linear_form(1, 2, 3, 0, prime) == mean
    assert linear_form(0, 2, 3, 0, prime) == mean


@pytest.mark.parametrize("num_links", [0, 1, 3, 7])
@pytest.mark.parametrize("degree", [0, 1, 3])
def test_schedule_balances_incidence_without_speedup_claim(num_links, degree):
    schedule = create_suppression_schedule(num_links, degree)
    assert schedule.degree == num_links * degree
    assert len(schedule.nodes) == schedule.degree + 1
    assert len(schedule.roots) == num_links
    assert all(len(roots) == degree for roots in schedule.roots)
    incidence = [sum(node in roots for roots in schedule.roots) for node in schedule.nodes]
    assert max(incidence) - min(incidence) <= 1
    assert sum(incidence) / len(incidence) < 1


@pytest.mark.parametrize("prime", PRIMES)
def test_no_boundaries_counts_unconstrained_and_empty_clauses(prime):
    schedule = create_suppression_schedule(0, 7)
    partitioned = PartitionedCNF(3, ((0, 1), (2,)), ((), ()), (), (), 3)
    assert direct_h(partitioned, 0, prime, schedule) == 8
    assert direct_h(partitioned, 17, prime, schedule) == 8
    assert interpolate_at_zero([8], schedule.nodes, prime) == 8
    unsatisfiable = PartitionedCNF(0, ((),), ((Clause(()),),), (), (), 0)
    assert direct_h(unsatisfiable, 1, prime, schedule) == 0


@pytest.mark.parametrize("prime", PRIMES)
def test_lagrange_matches_independent_polynomials(prime):
    nodes = (2, 5, 9, 14, 20)
    for coefficients in ([0], [8], [7, 0, 3], [prime - 1, 8, 5, 4, 2]):
        values = [_evaluate(coefficients, node, prime) for node in nodes]
        assert interpolate_at_zero(values, nodes, prime) == coefficients[0]
    assert sum(lagrange_weights_at_zero(nodes, prime)) % prime == 1
    assert interpolate_at_zero([12, 99], (0, 1), prime) == 12


def test_crt_and_extension_for_large_exact_counts():
    default_modulus = prod(PRIMES)
    assert primes_for_bound(0) == PRIMES
    assert primes_for_bound(default_modulus - 1) == PRIMES
    extended = primes_for_bound(default_modulus)
    assert extended[:3] == PRIMES and len(extended) == 4
    huge_count = (1 << 200) + (1 << 73) + 9
    primes = primes_for_bound(huge_count)
    assert prod(primes) > huge_count
    assert len(set(primes)) == len(primes)
    for prime in primes:
        validate_prime(prime)
        assert (prime - 1) ** 2 < 1 << 63
    assert crt_reconstruct([huge_count % prime for prime in primes], primes) == huge_count
    assert crt_reconstruct([-1] * 3, PRIMES) == default_modulus - 1


@pytest.mark.parametrize("prime", [0, 1, 4, 9, 341, 561, 1 << 30, True])
def test_invalid_primes_fail(prime):
    with pytest.raises(ValueError):
        validate_prime(prime)


def test_invalid_field_inputs_fail():
    prime = PRIMES[0]
    for roots in ((0,), (prime,)):
        with pytest.raises(ValueError):
            suppression_value(roots, 1, prime)
    for size in (0, -1, prime, True, 1.5):
        with pytest.raises(ValueError):
            linear_form(1, 1, size, 0, prime)
    for nodes in ((), (1, 1), (1, prime + 1)):
        with pytest.raises(ValueError):
            lagrange_weights_at_zero(nodes, prime)
    with pytest.raises(ValueError):
        lagrange_weights_at_zero((1, 2, 3), 3)
    with pytest.raises(ValueError):
        interpolate_at_zero([1], [1, 2], prime)
    for residues, primes in (([], []), ([1], PRIMES), ([1, 2], [prime, prime])):
        with pytest.raises(ValueError):
            crt_reconstruct(residues, primes)
    for bound in (-1, True, 1.5):
        with pytest.raises(ValueError):
            primes_for_bound(bound)


def test_invalid_schedules_fail():
    for args in ((-1, 1), (1, -1), (True, 1), (1, 1.5)):
        with pytest.raises(ValueError):
            create_suppression_schedule(*args)
    for nodes, roots in (((0,), ()), ((1, 1), ((1,),)), ((1, 2), ((3,),)),
                         ((1,), ((1,),)), ((1, 2), ((1.0,),)),
                         ((1, 2), ((True,),))):
        with pytest.raises(ValueError):
            SuppressionSchedule(nodes, roots)


def test_direct_h_checks_field_degree_and_schedule_link_count():
    partitioned = _repeated_master_partition()
    with pytest.raises(ValueError, match="one root list"):
        direct_h(partitioned, 0, PRIMES[0], create_suppression_schedule(1))
    with pytest.raises(ValueError, match="exceed D\\+1"):
        direct_h(partitioned, 0, 3, create_suppression_schedule(3))

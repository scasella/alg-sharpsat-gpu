"""Occurrence-coherent clause factors checked against global scalar oracles."""
from fractions import Fraction
from itertools import product
from math import comb

import pytest
from hypothesis import given, settings, strategies as st

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError
from alg_sharpsat.stage4_swap_mobius import (
    OccurrenceSwapCounter, SwapMobiusCounter, swap_mobius_count,
    swap_mobius_count_with_stats, occurrence_cost_estimate, swap_crossover_estimate,
)

pytestmark = pytest.mark.g11
P = 1009


def field(value):
    value = Fraction(value)
    return value.numerator * pow(value.denominator, -1, P) % P


def multiply(left, right, q):
    out = [0] * (q + 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i + j <= q:
                out[i + j] = (out[i + j] + a * b) % P
    return out


def direct_clause_jet(counter, node, q):
    """Global assignments, substituting each occurrence's full K(t+z,x).

    This oracle has no measurements, group buckets, forced bits or Mobius sum.
    """
    result = [0] * (q + 1)
    for bits in product((0, 1), repeat=counter.n):
        value = [1] + [0] * q
        for clause in counter.occurrences:
            unsatisfied = [1] + [0] * q
            for occurrence in clause:
                rows = counter.occurrence_coefficients(occurrence)
                polynomial = [sum(a * bit for a, bit in zip(row, bits)) for row in rows]
                jet = [field(sum(comb(m, j) * node ** (m-j) * polynomial[m]
                                 for m in range(j, len(polynomial)))) for j in range(q+1)]
                if occurrence.pos:
                    jet = [-a % P for a in jet]
                    jet[0] = (jet[0] + 1) % P
                unsatisfied = multiply(unsatisfied, jet, q)
            factor = [-a % P for a in unsatisfied]
            factor[0] = (factor[0] + 1) % P
            value = multiply(value, factor, q)
        result = [(a+b) % P for a, b in zip(result, value)]
    return result


def example_cnf():
    return CNF(3, [Clause((Literal(0, True), Literal(2, False))),
                   Clause((Literal(0, False), Literal(1, True))),
                   Clause((Literal(1, False), Literal(2, True)))])


@pytest.mark.parametrize("L,b", [(2, 1), (3, 1), (3, 2)])
def test_occurrence_jets_match_original_clause_product(L, b):
    counter = OccurrenceSwapCounter(example_cnf(), L=L, b=b, max_work=10_000_000)
    assert counter.stats.polynomial_terms == 0
    assert counter.degree == b * 6
    assert sum(q + 1 for q in counter.stats.orders) == counter.degree + 1
    for node in range(1, L + 1):
        for q in (0, 1, 2, 3):
            assert counter.evaluate_jet(node, q, P) == direct_clause_jet(counter, node, q)
    assert counter.count() == brute_force_count(example_cnf())
    assert counter.stats.local_assignment_visits > 0
    assert all(trace["status"] == "ok" for trace in counter.stats.node_evaluations)
    assert all(sum(trace["group_assignment_visits"]) == trace["local_assignment_visits"]
               for trace in counter.stats.node_evaluations)


def test_occurrence_blocks_restore_inputs_and_keep_clause_targets_coherent():
    # Variables 0 and 4 have the same alpha word, so their identical cells
    # require separate blocks. Repeated occurrences of 0 can reuse a cell.
    cnf = CNF(5, [Clause((Literal(0, True), Literal(4, False))),
                   Clause((Literal(1, False), Literal(2, True))),
                   Clause((Literal(0, False), Literal(3, True)))])
    counter = OccurrenceSwapCounter(cnf, max_work=10_000_000)
    assert len(counter.blocks) == counter.stats.block_count == 2
    system = counter.system
    for c, clause in enumerate(counter.occurrences):
        for occurrence in clause:
            alpha, beta = system.states[occurrence.omega]
            assert alpha == counter.alpha[occurrence.var]
            assert beta == counter.beta[c]
            coefficients = counter.occurrence_coefficients(occurrence)
            assert coefficients[0] == tuple(int(v == occurrence.var) for v in range(counter.n))
            for node in range(1, system.L + 1):
                for v in range(counter.n):
                    if counter.alpha[v][node-1] == counter.beta[c][node-1]:
                        continue
                    replacement = sum(row[v] * node**j for j, row in enumerate(coefficients))
                    error = sum((int(eta == occurrence.omega) - system.M[occurrence.omega][eta])
                                for eta, var in counter.blocks[occurrence.block].items() if var == v)
                    assert replacement == error
    assert counter.stats.raw_measurement_rows == len(counter.blocks) * system.s0
    assert counter.stats.measurement_rows == len(counter.measurement_rows)
    assert counter.stats.measurement_rows <= counter.stats.raw_measurement_rows
    assert counter.evaluate_jet(1, 2, P) == direct_clause_jet(counter, 1, 2)
    assert counter.count() == brute_force_count(cnf)


def test_expanded_integer_measurements_reconstruct_block_error():
    counter = OccurrenceSwapCounter(example_cnf())
    system = counter.system
    for bits in product((0, 1), repeat=counter.n):
        measured = [sum(a * bit for a, bit in zip(row, bits)) for row in counter.measurement_rows]
        for clause in counter.occurrences:
            for occurrence in clause:
                reconstructed = 0
                for cindex, index in enumerate(system.error_indices):
                    row_index = counter.measurement_indices.get((occurrence.block, cindex))
                    if row_index is not None:
                        reconstructed += ((1-system.eigenvalues[index]) / system.norms[index]
                                          * system.C[cindex][occurrence.omega] * measured[row_index])
                expected = sum((int(eta == occurrence.omega)-system.M[occurrence.omega][eta]) * bits[v]
                               for eta, v in counter.blocks[occurrence.block].items())
                assert reconstructed == expected


@st.composite
def tiny_occurrence_cnfs(draw):
    n = draw(st.integers(0, 4))
    if n == 0:
        return CNF(0, [Clause(())] * draw(st.integers(0, 2)))
    clauses = draw(st.lists(st.lists(st.tuples(st.integers(0, n-1), st.booleans()),
                                   max_size=3), max_size=4))
    return CNF(n, [Clause(tuple(Literal(v, sign) for v, sign in clause)) for clause in clauses])


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cnf=tiny_occurrence_cnfs())
def test_occurrence_default_and_old_reference_match_brute_force(cnf):
    expected = brute_force_count(cnf)
    value, stats = swap_mobius_count_with_stats(cnf)
    assert value == expected == SwapMobiusCounter(cnf).count()
    assert stats.representation == "occurrence_coherent_clause_factors"
    assert stats.polynomial_terms == 0


def test_default_never_expands_indicator_or_calls_reference_solver(monkeypatch):
    import alg_sharpsat.cnf as reference
    import alg_sharpsat.stage4_swap_mobius as module

    def forbidden(*args, **kwargs):
        pytest.fail("independent reference or global polynomial expansion was called")

    expected = brute_force_count(example_cnf())
    monkeypatch.setattr(module, "cnf_indicator_polynomial", forbidden)
    monkeypatch.setattr(module, "mobius_diagonal_jet", forbidden)
    monkeypatch.setattr(reference, "brute_force_count", forbidden)
    monkeypatch.setattr(reference, "dpll_count", forbidden)
    assert module.swap_mobius_count(example_cnf()) == expected


def test_explicit_embedding_retains_old_reference_api():
    cnf = example_cnf()
    value, stats = swap_mobius_count_with_stats(cnf, embedding=[3, 6, 9])
    assert value == SwapMobiusCounter(cnf, embedding=[3, 6, 9]).count()
    assert stats.embedding == (3, 6, 9)


def test_occurrence_budgets_fail_before_large_allocations():
    with pytest.raises(ResourceLimitError, match="max_variables"):
        OccurrenceSwapCounter(CNF(7, []))
    with pytest.raises(ResourceLimitError, match="storage"):
        OccurrenceSwapCounter(example_cnf(), max_work=1)
    counter = OccurrenceSwapCounter(example_cnf(), max_work=100)
    with pytest.raises(ResourceLimitError, match="jet storage"):
        counter.evaluate_jet(1, 10**9, P)
    with pytest.raises(ResourceLimitError):
        swap_mobius_count(example_cnf(), max_work=100)


@pytest.mark.parametrize("budget,reason", [(100, "storage"), (200, "work")])
def test_interpolation_is_bounded_before_jets_or_gaussian_elimination(monkeypatch, budget, reason):
    cnf = CNF(1, [Clause((Literal(0, True),))] * 10)
    counter = OccurrenceSwapCounter(cnf, max_work=budget)

    def forbidden(*args, **kwargs):
        pytest.fail("jet evaluation started before interpolation budget validation")

    monkeypatch.setattr(counter, "evaluate_jet", forbidden)
    with pytest.raises(ResourceLimitError, match=f"interpolation {reason}"):
        counter.count()


def test_factor_budget_and_legacy_alias_are_explicit():
    with pytest.raises(ResourceLimitError, match="max_factors"):
        OccurrenceSwapCounter(example_cnf(), max_factors=1)
    with pytest.raises(ResourceLimitError, match="max_factors"):
        OccurrenceSwapCounter(example_cnf(), max_terms=1)
    assert OccurrenceSwapCounter(example_cnf(), max_terms=1, max_factors=4).count() == 2


def test_occurrence_estimate_uses_clause_widths_and_blocks_without_counting(monkeypatch):
    from alg_sharpsat.benchmarks import bipartite_3sat

    def forbidden(*args, **kwargs):
        pytest.fail("cost estimator enumerated assignments or counted models")

    monkeypatch.setattr(OccurrenceSwapCounter, "count", forbidden)
    monkeypatch.setattr(OccurrenceSwapCounter, "evaluate_jet", forbidden)
    estimate = occurrence_cost_estimate(bipartite_3sat(30))
    assert estimate["representation"] == "occurrence_coherent_clause_factors"
    assert estimate["literal_occurrences"] == estimate["degree_upper_bound"] == 360
    assert estimate["jet_order_upper_bound"] == 180
    assert estimate["mobius_inner_terms_per_node_upper_bound"] == 3**30
    assert estimate["block_count"] > 1
    assert not estimate["executable_under_default_limit"]
    assert not estimate["asymptotic_parameter_guarantees_applied"]
    assert not estimate["crossover_established"]
    assert "multilinear_terms_upper_bound" not in estimate
    assert "embedding_fits" not in estimate
    for node in estimate["node_costs"]:
        assert sum(node["group_sizes"]) == 30
        assert node["local_assignment_rows"] == [2**n for n in node["group_sizes"]]
        assert node["group_split_measurement_rows"] >= estimate["measurement_rows"]
    assert swap_crossover_estimate(30, L=3)["representation"] == "legacy_multilinear_reference"


def test_prepared_measurement_error_storage_is_guarded_before_allocation(monkeypatch):
    cnf = CNF(2, [Clause((Literal(0, True), Literal(1, False)))] * 32)
    counter = OccurrenceSwapCounter(cnf, L=4, b=1, max_work=1000)
    assert counter.stats.occurrence_count * len(counter.measurement_rows) > counter.max_work
    def forbidden(*args, **kwargs):
        pytest.fail('preparing coefficient arrays before storage guard')
    monkeypatch.setattr(counter, 'occurrence_coefficients', forbidden)
    with pytest.raises(ResourceLimitError, match='jet storage'):
        counter.evaluate_jet(1, 0, P)

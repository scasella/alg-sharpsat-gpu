"""Exact small-instance checks for preprocessing and partition interfaces."""
from collections import Counter

import pytest
from hypothesis import given, settings, strategies as st

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count, dpll_count, parse_dimacs
from alg_sharpsat.stage1_sparsify import (
    partition_and_decouple_boundary,
    reduce_clause_width_deterministic,
    split_high_degree_variables,
)


def clause(*signed):
    """Convenient signed, DIMACS-style input for zero-based Literal objects."""
    return Clause(tuple(Literal(abs(v) - 1, v > 0) for v in signed))


@st.composite
def wide_formulas(draw, width):
    n = draw(st.integers(1, 12))
    signed = st.integers(1, n).flatmap(lambda v: st.sampled_from([v, -v]))
    wide = draw(st.lists(signed, min_size=width, max_size=width))
    extra = draw(st.lists(st.lists(signed, min_size=0, max_size=3), max_size=3))
    return CNF(n, [clause(*wide), *(clause(*c) for c in extra)])


@pytest.mark.g1
@pytest.mark.parametrize("width", [4, 5, 6, 7])
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_wide_clause_has_unique_extension(width, data):
    original = data.draw(wide_formulas(width))
    reduced = reduce_clause_width_deterministic(original)
    assert all(len(c.literals) <= 3 for c in reduced.clauses)
    assert reduced.num_vars == original.num_vars + width - 3
    assert brute_force_count(original, 2048) == brute_force_count(reduced, 2048)
    assert dpll_count(reduced) == brute_force_count(original)
    assert reduced == reduce_clause_width_deterministic(original)


@st.composite
def hub_formulas(draw):
    n = draw(st.integers(2, 10))
    count = draw(st.integers(3, 7))
    clauses = []
    for _ in range(count):
        hub = draw(st.sampled_from([-1, 1]))
        rest = draw(st.lists(st.integers(2, n).flatmap(lambda v: st.sampled_from([v, -v])), min_size=2, max_size=2))
        clauses.append(clause(hub, *rest))
    return CNF(n, clauses), 3


@pytest.mark.g2
@settings(max_examples=100, deadline=None)
@given(case=hub_formulas())
def test_hub_splitting_preserves_count_and_incidence(case):
    original, cap = case
    split = split_high_degree_variables(original, cap)
    assert brute_force_count(original) == brute_force_count(split, 2048)
    incidence = Counter(v for c in split.clauses for v in {l.var for l in c.literals})
    assert max(incidence.values(), default=0) <= cap + 2
    assert split == split_high_degree_variables(original, cap)
    assert len(original.clauses) <= len(split.clauses)


def test_degree_cap_means_original_clause_incidence_and_reuses_original():
    original = CNF(2, [clause(1, 1, -1, 2) for _ in range(5)])
    split = split_high_degree_variables(original, 2)
    assert split.num_vars == 6
    input_incidence = Counter(v for c in split.clauses[:5] for v in {l.var for l in c.literals})
    total_incidence = Counter(v for c in split.clauses for v in {l.var for l in c.literals})
    assert max(input_incidence.values()) == 2
    assert max(total_incidence.values()) == 4
    assert {l.var for l in split.clauses[0].literals} == {0, 1}
    assert split.clauses[0].literals[:3] == (Literal(0, True), Literal(0, True), Literal(0, False))
    assert brute_force_count(split) == brute_force_count(original)


def test_two_copy_cycle_has_two_distinct_implication_clauses():
    split = split_high_degree_variables(CNF(1, [clause(1)] * 3), 2)
    assert split.clauses[-2:] == [clause(-1, 2), clause(-2, 1)]
    assert brute_force_count(split) == 1


@st.composite
def small_formulas(draw):
    n = draw(st.integers(0, 7))
    if n == 0:
        clauses = draw(st.lists(st.just(Clause(())), max_size=3))
    else:
        signed = st.integers(1, n).flatmap(lambda v: st.sampled_from([v, -v]))
        clauses = [clause(*c) for c in draw(st.lists(st.lists(signed, max_size=4), max_size=6))]
    return CNF(n, clauses)


def coupled_partition(partition):
    clauses = [c for group in partition.group_clauses for c in group]
    for link in partition.boundary_links:
        m, p = link.master_var + 1, link.proxy_var + 1
        clauses.extend([clause(-m, p), clause(m, -p)])
    return CNF(partition.num_vars, clauses)


@settings(max_examples=100, deadline=None)
@given(original=small_formulas(), groups=st.integers(1, 10), block_size=st.integers(1, 4))
def test_partition_reconstruction_preserves_count(original, groups, block_size):
    partition = partition_and_decouple_boundary(original, groups, block_size)
    assert partition.original_num_vars == original.num_vars
    assert len(partition.group_vars) == groups
    assert partition == partition_and_decouple_boundary(original, groups, block_size)
    assert sorted(v for g in partition.group_vars for v in g) == list(range(partition.num_vars))
    assert dpll_count(coupled_partition(partition)) == brute_force_count(original)
    assert all(len(block.link_indices) <= block_size for block in partition.measurement_blocks)
    for block in partition.measurement_blocks:
        assert {partition.boundary_links[i].master_group for i in block.link_indices} == {block.master_group}


@pytest.mark.parametrize("n,groups", [(0, 1), (0, 5), (2, 5), (10, 3)])
def test_partition_preserves_isolated_variables_and_balances(n, groups):
    partition = partition_and_decouple_boundary(CNF(n, []), groups)
    sizes = list(map(len, partition.group_vars))
    assert max(sizes) - min(sizes) <= 1
    assert dpll_count(coupled_partition(partition)) == 1 << n
    assert not partition.boundary_links
    assert not partition.measurement_blocks


def test_proxy_is_reused_across_clauses_and_signs():
    original = CNF(4, [clause(1, 2, 3, 4), clause(1, -2, -3, -4)] * 2)
    partition = partition_and_decouple_boundary(original, 2)
    keys = [(link.master_var, link.proxy_group) for link in partition.boundary_links]
    assert keys and len(keys) == len(set(keys))
    assert dpll_count(coupled_partition(partition)) == brute_force_count(original)


@pytest.mark.parametrize("text", ["", "1 0", "p cnf 1 2\n1 0", "p cnf 1 1\n2 0", "p cnf 1 1\n1", "p cnf -1 0", "p cnf 0 0\np cnf 0 0", "p cnf 1 1\nx 0"])
def test_dimacs_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_dimacs(text)


def test_dimacs_multiline_empty_clauses_and_roundtrip():
    original = parse_dimacs("c comment\np cnf 4 3\n1 -2\n3 0 0\nc midstream\n-4 0\n")
    assert original.clauses == [clause(1, -2, 3), clause(), clause(-4)]
    assert CNF.from_dimacs(original.to_dimacs()) == original


@settings(max_examples=100, deadline=None)
@given(original=small_formulas(), chunk_size=st.integers(1, 31))
def test_dpll_matches_chunked_bruteforce(original, chunk_size):
    assert dpll_count(original) == brute_force_count(original, chunk_size)


def test_counters_retain_unused_variables():
    assert brute_force_count(CNF(100, [])) == dpll_count(CNF(100, [])) == 1 << 100
    assert brute_force_count(CNF(100, [clause(2)])) == dpll_count(CNF(100, [clause(2)])) == 1 << 99
    assert dpll_count(CNF(4, [clause(1, -1)])) == 16
    assert dpll_count(CNF(4, [clause(1), clause(-1)])) == 0
    assert dpll_count(CNF(4, [clause()])) == 0


@pytest.mark.parametrize("factory", [lambda: Literal(-1, True), lambda: Literal(True, True), lambda: Literal(1, 1), lambda: Clause([Literal(0, True)]), lambda: CNF(-1, []), lambda: CNF(1, [clause(2)])])
def test_strict_validation(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("function,kwargs", [(reduce_clause_width_deterministic, {"max_width": 2}), (split_high_degree_variables, {"max_degree": 0}), (partition_and_decouple_boundary, {"num_groups": 0}), (partition_and_decouple_boundary, {"block_size": 0})])
def test_invalid_transformation_parameters(function, kwargs):
    with pytest.raises(ValueError):
        function(CNF(0, []), **kwargs)


def test_dpll_long_unit_chain_does_not_recurse():
    formula = CNF(1100, [Clause((Literal(v, bool(v % 2)),)) for v in range(1050)])
    assert dpll_count(formula) == 1 << 50


def test_clause_home_is_distinct_variable_majority(monkeypatch):
    import alg_sharpsat.stage1_sparsify as stage1

    monkeypatch.setattr(stage1, "_spectral_groups", lambda cnf, n: ((0, 1), (2, 3)))
    # Repeated variable zero cannot outweigh the two distinct variables in group1.
    original = CNF(4, [clause(1, 1, -1, 3, 4), clause(2, 3)])
    partition = stage1.partition_and_decouple_boundary(original, 2)
    assert len(partition.group_clauses[1]) == 1
    assert len(partition.group_clauses[0]) == 1  # A tie chooses group zero.
    assert len(partition.boundary_links) == 2
    assert dpll_count(coupled_partition(partition)) == brute_force_count(original)

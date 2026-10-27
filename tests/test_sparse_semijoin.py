"""Source-support pushdown must prune existence, while joins retain weights."""
from math import prod

import pytest
import torch
from hypothesis import given, settings, strategies as st

from alg_sharpsat.benchmarks import bipartite_3sat
from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError
from alg_sharpsat.stage3_sparse_contract import SparseCounter, sparse_count_with_stats

pytestmark = pytest.mark.g10


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def clause(*signed):
    """Use signed one-based literals only in this concise test constructor."""
    return Clause(tuple(Literal(abs(value)-1, value > 0) for value in signed))


def partitioned(original_n, groups, clauses, links):
    blocks = tuple(MeasurementBlock(tuple(i for i, link in enumerate(links)
                                          if link.master_group == group), group)
                   for group in range(len(groups))
                   if any(link.master_group == group for link in links))
    return PartitionedCNF(sum(map(len, groups)), tuple(map(tuple, groups)),
                          tuple(map(tuple, clauses)), tuple(links), blocks, original_n)


def original_formula(p):
    aliases = {link.proxy_var: link.master_var for link in p.boundary_links}
    return CNF(p.original_num_vars, [Clause(tuple(Literal(aliases.get(lit.var, lit.var), lit.pos)
                                                 for lit in local.literals))
                                       for group in p.group_clauses for local in group])


def compare_partitioned(p):
    expected = brute_force_count(original_formula(p))
    enabled = SparseCounter(p, device='cpu', semi_join_pushdown=True)
    disabled = SparseCounter(p, device='cpu', semi_join_pushdown=False)
    assert enabled.count() == disabled.count() == expected
    assert disabled.stats.phase1_assignment_counts == []
    assert disabled.stats.feasible_master_mask_counts == []
    assert len(enabled.stats.phase1_assignment_counts) == len(p.group_vars)
    assert len(enabled.stats.feasible_master_mask_counts) == len(p.group_vars)
    for counter in (enabled, disabled):
        assert len(counter.stats.peak_frontier_rows) == len(p.group_vars)
        assert len(counter.stats.peak_temporary_rows) == len(p.group_vars)
        assert counter.stats.peak_joint_rows >= max(counter.stats.joint_signature_counts, default=0)
    return enabled, disabled


@st.composite
def small_formulas(draw):
    n = draw(st.integers(1, 9))
    raw = draw(st.lists(st.lists(st.tuples(st.integers(0, n-1), st.booleans()),
                                 min_size=1, max_size=3), max_size=14))
    return CNF(n, [Clause(tuple(Literal(v, sign) for v, sign in literals)) for literals in raw])


@pytest.mark.parametrize('groups', [2, 4])
@settings(max_examples=120, deadline=None, derandomize=True)
@given(cnf=small_formulas())
def test_pushdown_on_off_matches_independent_brute_force(groups, cnf):
    expected = brute_force_count(cnf)
    enabled, stats = sparse_count_with_stats(cnf, device='cpu', num_groups=groups,
                                             semi_join_pushdown=True)
    disabled, _ = sparse_count_with_stats(cnf, device='cpu', num_groups=groups,
                                          semi_join_pushdown=False)
    assert enabled == disabled == expected
    assert len(stats.phase1_assignment_counts) == len(stats.feasible_master_mask_counts) == groups


def test_source_internal_multiplicity_is_not_multiplied_into_destination():
    # Source has three witnesses for each x0 value. Its support projection is
    # {0,1}, not a weighted table to multiply into the destination.
    p = partitioned(4, ((0, 1, 2), (3, 4)),
                    ((clause(2, 3),), (clause(-5, 4),)),
                    (BoundaryLink(0, 0, 4, 1),))
    enabled, _ = compare_partitioned(p)
    assert enabled.count() == 9
    assert enabled.stats.phase1_assignment_counts[0] == 6
    assert enabled.stats.feasible_master_mask_counts[0] == 2
    assert enabled.stats.phase1_assignment_counts[0] > enabled.stats.feasible_master_mask_counts[0]


def test_correlated_source_bits_prune_joint_tuples_instead_of_bit_marginals():
    # The source permits only 00 and 11. Of destination OR's three local
    # rows, only 11 can extend the source relation.
    p = partitioned(2, ((0, 1), (2, 3)),
                    ((clause(-1, 2), clause(1, -2)), (clause(3, 4),)),
                    (BoundaryLink(0, 0, 2, 1), BoundaryLink(1, 0, 3, 1)))
    enabled, disabled = compare_partitioned(p)
    assert enabled.count() == 1
    assert enabled.stats.feasible_master_mask_counts[0] == 2
    assert enabled.stats.joint_signature_counts[1] == 1
    assert disabled.stats.joint_signature_counts[1] == 3


def test_destinations_receive_deduplicated_partial_source_projections():
    # Source support is {(0,0,z),(1,1,z)}. One destination receives (x0,z),
    # the other only x1. Projecting the full keys without deduplication would
    # count the second destination twice for every allowed incoming value.
    p = partitioned(5, ((0, 1, 2), (3, 5, 6), (4, 7)),
                    ((clause(-1, 2), clause(1, -2)),
                     (clause(6, 4), clause(-7, 4)), (clause(-8, 5),)),
                    (BoundaryLink(0, 0, 5, 1), BoundaryLink(2, 0, 6, 1),
                     BoundaryLink(1, 0, 7, 2)))
    enabled, _ = compare_partitioned(p)
    assert enabled.stats.feasible_master_mask_counts[0] == 4


@pytest.mark.parametrize('contradiction', [False, True])
def test_cyclic_sources_and_multiple_incoming_owners_are_sound(contradiction):
    # x0==x1 is distributed over the two source groups; neither is entitled
    # to assume the other's incoming proxy has a fixed value in phase one.
    first = (clause(-1, 5), clause(1, -5))
    second = ((clause(2, 6), clause(-2, -6)) if contradiction
              else (clause(-2, 6), clause(2, -6)))
    p = partitioned(4, ((0, 4), (1, 5), (2, 6, 7), (3,)),
                    (first, second, (clause(7, 8, 3),), ()),
                    (BoundaryLink(1, 1, 4, 0), BoundaryLink(0, 0, 5, 1),
                     BoundaryLink(0, 0, 6, 2), BoundaryLink(1, 1, 7, 2)))
    enabled, _ = compare_partitioned(p)
    assert enabled.count() == (0 if contradiction else 6)
    assert enabled.stats.feasible_master_mask_counts[:2] == [2, 2]


@pytest.mark.parametrize('source_clauses', [(clause(1), clause(-1)), (Clause(()),)])
def test_unsatisfiable_source_has_no_feasible_projection(source_clauses):
    p = partitioned(2, ((0,), (1, 2)), (source_clauses, (clause(2, 3),)),
                    (BoundaryLink(0, 0, 2, 1),))
    enabled, _ = compare_partitioned(p)
    assert enabled.count() == 0
    assert enabled.stats.phase1_assignment_counts[0] == 0
    assert enabled.stats.feasible_master_mask_counts[0] == 0
    assert enabled.stats.joint_signature_counts[1] == 0


@pytest.mark.parametrize('cnf,expected', [(CNF(0, []), 1), (CNF(0, [Clause(())]), 0),
                                          (CNF(100, []), 1 << 100),
                                          (CNF(4, [clause(1, -1)]), 16)])
@pytest.mark.parametrize('groups', [2, 4])
def test_empty_free_and_zero_outgoing_groups(cnf, expected, groups):
    for enabled in (True, False):
        value, stats = sparse_count_with_stats(cnf, device='cpu', num_groups=groups,
                                               semi_join_pushdown=enabled)
        assert value == expected
        assert len(stats.peak_frontier_rows) == groups
        if enabled:
            assert len(stats.feasible_master_mask_counts) == groups


@pytest.mark.parametrize('n', [18, 22, 26, 30])
def test_seed_zero_bipartite_reduces_accepted_frontier_and_joint_tables(n):
    cnf = bipartite_3sat(n, seed=0)
    enabled, on = sparse_count_with_stats(cnf, device='cpu', semi_join_pushdown=True)
    disabled, off = sparse_count_with_stats(cnf, device='cpu', semi_join_pushdown=False)
    assert enabled == disabled
    if n == 30:
        assert enabled == 933
    assert max(on.peak_frontier_rows) < max(off.peak_frontier_rows)
    assert sum(on.joint_signature_counts) < sum(off.joint_signature_counts)
    assert on.peak_joint_rows < off.peak_joint_rows
    assert all(projected <= assignments for projected, assignments in zip(
        on.feasible_master_mask_counts, on.phase1_assignment_counts))


def test_pushdown_budget_exhaustion_is_explicit_without_reference_fallback(monkeypatch):
    import alg_sharpsat.cnf as references

    def forbidden(*args, **kwargs):
        pytest.fail('reference solver called from pushdown')

    monkeypatch.setattr(references, 'brute_force_count', forbidden)
    monkeypatch.setattr(references, 'dpll_count', forbidden)
    for options in ({'max_work': 1}, {'max_local_assignments': 1}, {'max_memory_bytes': 1}):
        with pytest.raises(ResourceLimitError):
            sparse_count_with_stats(bipartite_3sat(10), device='cpu',
                                     semi_join_pushdown=True, **options)


def test_large_active_count_still_uses_safe_crt_with_pushdown():
    p = partitioned(100, tuple((2*g, 2*g+1) for g in range(50)),
                    tuple((clause(2*g+1, 2*g+2),) for g in range(50)), ())
    for enabled in (True, False):
        counter = SparseCounter(p, device='cpu', semi_join_pushdown=enabled)
        assert counter.count() == 3**50
        assert counter.stats.arithmetic == 'modular_crt'
        assert prod(counter.stats.primes) > 1 << 100

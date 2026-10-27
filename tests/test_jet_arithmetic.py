"""Arithmetic, batching, and resource contracts for the subset jet engine."""
from itertools import combinations

import pytest
import torch

from alg_sharpsat.cnf import CNF, Clause, Literal
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF
from alg_sharpsat.stage2_hermite import create_hermite_schedule
from alg_sharpsat.stage3_gpu_engine import AlgebraicCounter, ResourceLimitError
from alg_sharpsat.stage3_jet_engine import JetCounter, jet_sharp_sat


def partition(empty=False):
    return PartitionedCNF(
        8, ((0, 1, 6, 7), (2, 3, 4, 5)),
        ((Clause(()) if empty else Clause((Literal(0, True), Literal(6, True))),
          Clause((Literal(1, False), Literal(7, True)))),
         (Clause((Literal(2, True), Literal(4, False))),
          Clause((Literal(3, True), Literal(5, True))))),
        (BoundaryLink(0, 0, 4, 1), BoundaryLink(1, 0, 5, 1),
         BoundaryLink(2, 1, 6, 0), BoundaryLink(3, 1, 7, 0)),
        (MeasurementBlock((0, 1), 0), MeasurementBlock((2, 3), 1)), 4,
    )


@pytest.mark.parametrize("prime", [998244353, 1004535809, 469762049])
def test_near_modulus_polynomial_convolution_matches_python(prime):
    left = [torch.tensor([[prime - 1 - ((degree + cell) % 11) for cell in range(6)]],
                         dtype=torch.int64) for degree in range(5)]
    right = [torch.tensor([[prime - 1 - ((degree * 2 + cell) % 13) for cell in range(6)]],
                          dtype=torch.int64) for degree in range(4)]
    result = JetCounter._multiply_jets(left, right, 6, prime)
    expected = [[[sum(int(left[i][0, cell]) * int(right[degree - i][0, cell])
                      for i in range(5) if 0 <= degree - i < 4) % prime
                  for cell in range(6)]] for degree in range(7)]
    assert [value.tolist() for value in result] == expected


def test_batched_moments_match_python_weighted_signature_bins():
    prime = 1004535809
    counter = JetCounter(partition(), device="cpu", assignment_chunk_size=2)
    schedule = create_hermite_schedule(4, mode="single_node_taylor")
    contexts = counter._conditions(0, prime, schedule)
    # Stress modular counts rather than relying on the naturally small table.
    table = counter.tables[0]
    table.counts = torch.full_like(table.counts, prime - 1)
    subsets = [(), (0,), (1,), (0, 1)]
    trace = {"moment_cache_hits": 0, "moment_tables_built": 0}
    moments = counter._moment_batch(0, subsets, prime, contexts, trace)
    context = contexts[0]
    for subset, tensor in zip(subsets, moments):
        expected = [[0] * table.counts.shape[1] for _ in range(context.states)]
        for row in range(table.counts.shape[0]):
            factor = 1
            for link in subset:
                factor = factor * int(context.deltas[link][row]) % prime
            for column in range(table.counts.shape[1]):
                index = int(context.inverse[row])
                expected[index][column] = (expected[index][column]
                                          + int(table.counts[row, column]) * factor) % prime
        assert tensor.tolist() == expected
    assert trace["moment_batches_built"] == 1
    assert trace["moment_tables_built"] == 4


def test_subset_batch_sizes_and_node_prime_cache_reset_preserve_coefficients():
    schedule = create_hermite_schedule(4)
    counter = JetCounter(partition(), device="cpu", subset_batch_size=8)
    for index, prime in ((0, 998244353), (1, 1004535809), (0, 1004535809)):
        expected = JetCounter(partition(), device="cpu", subset_batch_size=1).evaluate_jet(
            index, prime, schedule)
        assert counter.evaluate_jet(index, prime, schedule) == expected
        assert not counter._moment_cache
        assert counter._cache_bytes == counter._condition_bytes == counter._jet_live_bytes == 0
        assert counter.stats.node_evaluations[-1]["moment_tables_built"] > 0


def test_subset_guard_runs_before_conditioning_or_subset_loop(monkeypatch):
    counter = JetCounter(partition(), device="cpu", max_subset_terms=2)
    schedule = create_hermite_schedule(4, mode="single_node_taylor")
    def forbidden(*args, **kwargs):
        pytest.fail("resource preflight must run before local conditioning")
    monkeypatch.setattr(counter, "_conditions", forbidden)
    with pytest.raises(ResourceLimitError, match="16 suppressed subsets"):
        counter.evaluate_jet(0, 998244353, schedule)
    trace = counter.stats.node_evaluations[-1]
    assert trace["processed_subsets"] == 0
    assert trace["status"] == "resource_limit"


def test_known_zero_local_factor_needs_no_subset_expansion():
    counter = JetCounter(partition(empty=True), device="cpu", max_subset_terms=1)
    schedule = create_hermite_schedule(4, mode="single_node_taylor")
    assert counter.evaluate_jet(0, 998244353, schedule) == [0] * 5
    trace = counter.stats.node_evaluations[-1]
    assert trace["requested_subsets"] == 16
    assert trace["processed_subsets"] == 0
    assert trace["status"] == "zero_local_factor"


def test_jets_never_call_scalar_evaluation_or_reference_counters(monkeypatch):
    import alg_sharpsat.cnf as cnf_module
    def forbidden(*args, **kwargs):
        pytest.fail("jet path must not call scalar or reference model counters")
    monkeypatch.setattr(AlgebraicCounter, "evaluate", forbidden)
    monkeypatch.setattr(cnf_module, "dpll_count", forbidden)
    monkeypatch.setattr(cnf_module, "brute_force_count", forbidden)
    cnf = CNF(4, [Clause((Literal(0, True), Literal(1, False), Literal(2, True))),
                  Clause((Literal(1, True), Literal(3, False)))])
    assert jet_sharp_sat(cnf, device="cpu") == 10

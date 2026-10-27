"""Independent end-to-end checks, including the intermediate factorization."""

import random

import pytest
import torch
from hypothesis import given, settings, strategies as st

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count, dpll_count, parse_dimacs, to_dimacs
from alg_sharpsat.stage1_sparsify import (
    BoundaryLink, MeasurementBlock, PartitionedCNF, partition_and_decouple_boundary,
)
from alg_sharpsat.stage2_algebraic import (
    PRIMES, create_suppression_schedule, crt_reconstruct, direct_h,
    interpolate_at_zero, primes_for_bound,
)
from alg_sharpsat.stage3_gpu_engine import (
    AlgebraicCounter, UnsupportedDeviceError, gpu_algebraic_sharp_sat, select_device,
)


@st.composite
def small_cnfs(draw):
    n = draw(st.integers(0, 10))
    if not n:
        return CNF(0, [Clause(())] if draw(st.booleans()) else [])
    # Duplicates, opposite literals, units and empty clauses are intentional.
    raw = draw(st.lists(st.lists(st.tuples(st.integers(0, n - 1), st.booleans()),
                                min_size=0, max_size=5), max_size=12))
    return CNF(n, [Clause(tuple(Literal(v, p) for v, p in c)) for c in raw])


@pytest.mark.g5
@settings(max_examples=200, deadline=None, derandomize=True)
@given(small_cnfs())
def test_200_differential_cnfs(cnf):
    expected = brute_force_count(cnf)
    assert dpll_count(cnf) == expected
    assert gpu_algebraic_sharp_sat(cnf, device="cpu") == expected


def dense_formula(n, seed):
    rng = random.Random(seed)
    return CNF(n, [Clause(tuple(Literal(v, bool(rng.getrandbits(1)))
                                     for v in rng.sample(range(n), 3)))
                   for _ in range(4 * n)])


@pytest.mark.g5
@settings(max_examples=150, deadline=None, derandomize=True)
@given(n=st.integers(3, 16), seed=st.integers(0, 2**32 - 1))
def test_150_dense_differential_cnfs(n, seed):
    # A separate distribution ensures that the edge-case fuzzer's empty
    # clauses cannot dominate the nontrivial, crossing-boundary coverage.
    cnf = dense_formula(n, seed)
    assert gpu_algebraic_sharp_sat(cnf, device="cpu") == brute_force_count(cnf)


@pytest.mark.g5
@pytest.mark.parametrize("n", [11, 12, 13, 14, 15, 16])
def test_dense_through_16_variables(n):
    cnf = dense_formula(n, 600 + n)
    assert gpu_algebraic_sharp_sat(cnf, device="cpu") == brute_force_count(cnf)


@pytest.mark.g5
@pytest.mark.parametrize("groups", [1, 2, 3, 6])
def test_disconnected_tautologies_unused_variables_and_empty_groups(groups):
    cnf = parse_dimacs("p cnf 5 3\n1 -1 0\n2 3 0\n-2 -3 0\n")
    assert gpu_algebraic_sharp_sat(cnf, num_groups=groups, device="cpu") == 16


@pytest.mark.g5
@pytest.mark.parametrize("cnf, expected", [
    (CNF(0, []), 1), (CNF(0, [Clause(())]), 0),
    (CNF(100, []), 1 << 100),
    (CNF(100, [Clause((Literal(0, True),))]), 1 << 99),
    (CNF(100, [Clause(())]), 0),
])
def test_edge_cases_and_counts_beyond_three_prime_range(cnf, expected):
    assert gpu_algebraic_sharp_sat(cnf, device="cpu") == expected


@pytest.mark.g5
def test_actual_interpolation_and_four_prime_crt_for_large_count():
    # 100 original variables; a local proxy for x99 makes (x0 OR x99)
    # cross the cut. The other 98 input variables remain free.
    partition = PartitionedCNF(
        num_vars=101,
        group_vars=(tuple(range(50)) + (100,), tuple(range(50, 100))),
        group_clauses=((Clause((Literal(0, True), Literal(100, True))),), ()),
        boundary_links=(BoundaryLink(99, 1, 100, 0),),
        measurement_blocks=(MeasurementBlock((0,), 1),),
        original_num_vars=100,
    )
    counter = AlgebraicCounter(partition, device="cpu")
    schedule = create_suppression_schedule(1)
    primes = primes_for_bound(1 << 100)
    assert len(primes) > 3
    residues = [interpolate_at_zero(
        [counter.evaluate(t, prime, schedule) for t in schedule.nodes],
        schedule.nodes, prime,
    ) for prime in primes]
    assert crt_reconstruct(residues, primes) == 3 * (1 << 98)


@pytest.mark.g5
def test_full_preprocessing_pipeline():
    cnf = parse_dimacs("p cnf 7 6\n1 2 3 4 5 0\n1 -2 3 0\n1 4 -5 0\n-1 6 7 0\n1 -3 7 0\n1 5 -6 0\n")
    assert gpu_algebraic_sharp_sat(cnf, max_degree=3, device="cpu") == brute_force_count(cnf)


@pytest.mark.g4
@pytest.mark.parametrize("prime", PRIMES)
@pytest.mark.parametrize("groups", [2, 3])
def test_tensor_factorization_at_nodes_and_arbitrary_parameters(prime, groups):
    cnf = parse_dimacs("p cnf 5 4\n1 2 3 0\n-1 4 5 0\n2 -4 5 0\n-2 -3 -5 0\n")
    partition = partition_and_decouple_boundary(cnf, num_groups=groups, block_size=2)
    schedule = create_suppression_schedule(len(partition.boundary_links), degree_per_link=2)
    counter = AlgebraicCounter(partition, device="cpu")
    for t in (0, *schedule.nodes, schedule.nodes[-1] + 3):
        assert counter.evaluate(t, prime, schedule) == direct_h(partition, t, prime, schedule)


@pytest.mark.parametrize("text", ["p cnf 0 0\n", "p cnf 2 2\n1\n-2 0 2 0\n", "c test\np cnf 3 1\n0\n"])
def test_dimacs_roundtrip(text):
    cnf = parse_dimacs(text)
    assert parse_dimacs(to_dimacs(cnf)) == cnf


@pytest.mark.g5
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_when_available(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} hardware is unavailable")
    try:
        select_device(device)
    except UnsupportedDeviceError as exc:
        pytest.skip(f"{device} lacks required exact integer operations: {exc}")
    cnf = dense_formula(8, 401)
    assert gpu_algebraic_sharp_sat(cnf, device=device) == brute_force_count(cnf)

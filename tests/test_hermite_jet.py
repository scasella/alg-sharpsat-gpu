"""Independent coefficient-level and end-to-end validation of Hermite jets."""

from itertools import product
from math import comb

import pytest
import torch
from hypothesis import given, settings, strategies as st

from alg_sharpsat.benchmarks import bipartite_3sat, random_3sat
from alg_sharpsat.cnf import CNF, Clause, Literal, dpll_count, parse_dimacs
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF
from alg_sharpsat.stage2_algebraic import PRIMES, crt_reconstruct, primes_for_bound
from alg_sharpsat.stage2_hermite import (
    HermiteSchedule, create_hermite_schedule, interpolate_hermite_at_zero,
)
from alg_sharpsat.stage3_jet_engine import JetCounter, jet_sharp_sat
from alg_sharpsat.stage3_gpu_engine import UnsupportedDeviceError, select_device


def scalar_jet(partition, schedule, node_index, prime):
    """Exhaustively expand affine link factors, without the subset algorithm."""
    order = schedule.orders[node_index]
    node = schedule.nodes[node_index]
    block_for_link = {i: block for block in partition.measurement_blocks for i in block.link_indices}
    answer = [0] * (order + 1)
    clauses = [clause for group in partition.group_clauses for clause in group]
    for bits in product((0, 1), repeat=partition.num_vars):
        if not all(any(bits[lit.var] == lit.pos for lit in clause.literals) for clause in clauses):
            continue
        jet = [1] + [0] * order
        for i, link in enumerate(partition.boundary_links):
            block = block_for_link[i]
            measurement = sum(bits[partition.boundary_links[j].master_var] for j in block.link_indices)
            mean = measurement * pow(len(block.link_indices), -1, prime) % prime
            root = schedule.nodes[schedule.link_nodes[i]]
            inverse = pow(root, -1, prime)
            delta = bits[link.master_var] - mean
            y = bits[link.proxy_var]
            constant = ((1 - y) + (2 * y - 1) * (mean + (1 - node * inverse) * delta)) % prime
            slope = (-(2 * y - 1) * inverse * delta) % prime
            jet = [(jet[j] * constant + (jet[j - 1] * slope if j else 0)) % prime
                   for j in range(order + 1)]
        answer = [(a + b) % prime for a, b in zip(answer, jet)]
    return answer


def two_way_partition():
    return PartitionedCNF(
        8, ((0, 1, 6, 7), (2, 3, 4, 5)),
        ((Clause((Literal(0, True), Literal(6, True))),
          Clause((Literal(1, False), Literal(7, True)))),
         (Clause((Literal(2, True), Literal(4, False))),
          Clause((Literal(3, True), Literal(5, True))))),
        (BoundaryLink(0, 0, 4, 1), BoundaryLink(1, 0, 5, 1),
         BoundaryLink(2, 1, 6, 0), BoundaryLink(3, 1, 7, 0)),
        (MeasurementBlock((0, 1), 0), MeasurementBlock((2, 3), 1)), 4,
    )


def repeated_master_partition():
    return PartitionedCNF(
        7, ((0, 1), (2, 3, 4), (5, 6)),
        ((), (Clause((Literal(3, True), Literal(2, True))),
              Clause((Literal(4, True), Literal(2, False)))),
         (Clause((Literal(5, True), Literal(6, True))),)),
        (BoundaryLink(0, 0, 3, 1), BoundaryLink(1, 0, 4, 1),
         BoundaryLink(0, 0, 5, 2), BoundaryLink(1, 0, 6, 2)),
        (MeasurementBlock((0, 1, 2, 3), 0),), 3,
    )


@pytest.mark.g8
@pytest.mark.parametrize("prime", PRIMES)
@pytest.mark.parametrize("factory", [two_way_partition, repeated_master_partition])
@pytest.mark.parametrize("schedule", [
    HermiteSchedule((1,), (4,), (0, 0, 0, 0)),
    HermiteSchedule((1, 2), (2, 1), (0, 0, 1, 1)),
    HermiteSchedule((2, 7), (0, 3), (0, 1, 0, 1)),
    HermiteSchedule((1, 2, 3), (1, 1, 0), (0, 1, 2, 2)),
])
def test_each_coefficient_matches_independent_scalar_expansion(prime, factory, schedule):
    partition = factory()
    counter = JetCounter(partition, device="cpu")
    for index in range(len(schedule.nodes)):
        assert counter.evaluate_jet(index, prime, schedule) == scalar_jet(partition, schedule, index, prime)


@pytest.mark.g8
@pytest.mark.parametrize("family", [random_3sat, bipartite_3sat])
@settings(max_examples=100, deadline=None, derandomize=True)
@given(n=st.integers(4, 10), seed=st.integers(0, 2**20), planted=st.booleans())
def test_100_random_and_100_bipartite_counts(family, n, seed, planted):
    cnf = family(n, seed=seed, planted=planted)
    assert jet_sharp_sat(cnf, device="cpu") == dpll_count(cnf)


@pytest.mark.g8
@settings(max_examples=25, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16), groups=st.integers(1, 3))
def test_single_node_and_multiple_groups(seed, groups):
    cnf = random_3sat(6, seed=seed)
    assert jet_sharp_sat(cnf, device="cpu", num_groups=groups, mode="single_node_taylor") == dpll_count(cnf)


@pytest.mark.g8
@pytest.mark.parametrize("cnf", [
    CNF(0, []), CNF(0, [Clause(())]), CNF(100, []),
    CNF(100, [Clause((Literal(0, True),))]),
    parse_dimacs("p cnf 6 2\n1 -1 0\n2 -2 0\n"),
    parse_dimacs("p cnf 3 2\n1 0\n-1 0\n"),
    parse_dimacs("p cnf 6 3\n1 2 3 4 5 6 0\n-1 -2 0\n3 -4 0\n"),
])
def test_edge_cases_and_deterministic_width_reduction(cnf):
    assert jet_sharp_sat(cnf, device="cpu") == dpll_count(cnf)


@pytest.mark.g8
def test_degree_splitting_pipeline():
    cnf = parse_dimacs("p cnf 5 5\n1 2 3 0\n1 -2 4 0\n1 -3 5 0\n-1 -4 5 0\n1 4 -5 0\n")
    assert jet_sharp_sat(cnf, max_degree=3, device="cpu") == dpll_count(cnf)


@pytest.mark.g8
def test_four_prime_reconstruction_with_nonconstant_jets_and_free_variables():
    base = two_way_partition()
    # Keep the four links nontrivial while adding 100 free original variables.
    remap = lambda v: v if v < 4 else v + 100
    partition = PartitionedCNF(
        108,
        (tuple(remap(v) for v in base.group_vars[0]) + tuple(range(4, 104)),
         tuple(remap(v) for v in base.group_vars[1])),
        tuple(tuple(Clause(tuple(Literal(remap(lit.var), lit.pos) for lit in clause.literals))
                    for clause in group) for group in base.group_clauses),
        tuple(BoundaryLink(link.master_var, link.master_group, remap(link.proxy_var), link.proxy_group)
              for link in base.boundary_links), base.measurement_blocks, 104,
    )
    schedule = create_hermite_schedule(4, num_nodes=2)
    primes = primes_for_bound(1 << 104)
    counter = JetCounter(partition, device="cpu")
    residues = [interpolate_hermite_at_zero(
        [counter.evaluate_jet(i, prime, schedule) for i in range(len(schedule.nodes))],
        schedule.nodes, schedule.orders, prime,
    ) for prime in primes]
    small_count = interpolate_hermite_at_zero(
        [scalar_jet(base, schedule, i, PRIMES[0]) for i in range(len(schedule.nodes))],
        schedule.nodes, schedule.orders, PRIMES[0],
    )
    assert len(primes) > 3
    assert crt_reconstruct(residues, primes) == small_count * (1 << 100)


@pytest.mark.g9
@pytest.mark.parametrize("boundary", range(1, 25))
def test_disjoint_root_budget_cannot_strictly_truncate_every_node(boundary):
    for count in range(1, boundary + 2):
        schedule = create_hermite_schedule(boundary, num_nodes=count)
        sizes = [schedule.link_nodes.count(i) for i in range(len(schedule.nodes))]
        assert sum(q + 1 for q in schedule.orders) == boundary + 1
        assert any(q >= size for q, size in zip(schedule.orders, sizes))
    single = create_hermite_schedule(boundary, mode="single_node_taylor")
    assert sum(comb(boundary, j) for j in range(single.orders[0] + 1)) == 1 << boundary


@pytest.mark.g8
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_jets_when_supported(device):
    if not (torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()):
        pytest.skip(f"{device} hardware is unavailable")
    try:
        select_device(device)
    except UnsupportedDeviceError as exc:
        pytest.skip(str(exc))
    cnf = bipartite_3sat(6, seed=411)
    assert jet_sharp_sat(cnf, device=device) == dpll_count(cnf)

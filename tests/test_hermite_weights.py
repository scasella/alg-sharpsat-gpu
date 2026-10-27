"""Independent scalar Hasse-jet tests for exact confluent reconstruction."""

from math import comb

from hypothesis import given, settings, strategies as st
import pytest

from alg_sharpsat.stage2_algebraic import PRIMES
from alg_sharpsat.stage2_hermite import (
    HermiteSchedule,
    confluent_hermite_weights_at_zero,
    create_hermite_schedule,
    interpolate_hermite_at_zero,
)


pytestmark = pytest.mark.g7


def _scalar_jets(coefficients, nodes, orders, prime):
    # Independent binomial expansion of each monomial (t+Z)**m.
    return [[sum(coefficient * comb(monomial, derivative)
                 * pow(node, monomial - derivative, prime)
                 for monomial, coefficient in enumerate(coefficients)
                 if monomial >= derivative) % prime
             for derivative in range(order + 1)]
            for node, order in zip(nodes, orders)]


@st.composite
def _polynomial_and_capacity_partition(draw):
    degree = draw(st.integers(min_value=0, max_value=30))
    count = draw(st.integers(min_value=1, max_value=min(degree + 1, 7)))
    cuts = sorted(draw(st.lists(st.integers(min_value=1, max_value=max(1, degree)),
                                min_size=count - 1, max_size=count - 1, unique=True)))
    boundaries = [0, *cuts, degree + 1]
    orders = [right - left - 1 for left, right in zip(boundaries, boundaries[1:])]
    nodes = draw(st.lists(st.integers(-2000, 2000).filter(lambda node: node != 0),
                          min_size=count, max_size=count, unique=True))
    coefficients = draw(st.lists(st.integers(-(1 << 40), 1 << 40),
                                 min_size=degree + 1, max_size=degree + 1))
    return coefficients, nodes, orders


@pytest.mark.parametrize("prime", PRIMES)
@settings(max_examples=100, deadline=None)
@given(case=_polynomial_and_capacity_partition())
def test_random_polynomial_arbitrary_capacity_partitions(prime, case):
    coefficients, nodes, orders = case
    jets = _scalar_jets(coefficients, nodes, orders, prime)
    assert interpolate_hermite_at_zero(jets, nodes, orders, prime) == coefficients[0] % prime
    weights = confluent_hermite_weights_at_zero(nodes, orders, prime)
    assert [len(row) for row in weights] == [order + 1 for order in orders]
    assert all(0 <= weight < prime for row in weights for weight in row)


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 11])
def test_hasse_reconstruction_when_characteristic_is_at_most_degree(prime):
    degree = 30
    nodes = list(range(1, min(prime - 1, 4) + 1))
    capacity, extra = divmod(degree + 1, len(nodes))
    orders = [capacity - 1 + int(index < extra) for index in range(len(nodes))]
    coefficients = [(-1) ** index * (index * index + 13) for index in range(degree + 1)]
    jets = _scalar_jets(coefficients, nodes, orders, prime)
    assert prime <= degree
    assert interpolate_hermite_at_zero(jets, nodes, orders, prime) == 13 % prime
    weights = confluent_hermite_weights_at_zero(nodes, orders, prime)
    # Check the dual monomial identity, including powers >= characteristic.
    for monomial in range(degree + 1):
        monomial_jets = _scalar_jets([0] * monomial + [1], nodes, orders, prime)
        total = sum(weight * coefficient
                    for row, jet in zip(weights, monomial_jets)
                    for weight, coefficient in zip(row, jet)) % prime
        assert total == int(monomial == 0)


def test_characteristic_two_retains_hasse_coefficient_lost_by_factorials():
    # f(T)=T**2 at t=1: f(1+Z)=1+Z**2 in F_2. The second ordinary
    # derivative is zero, while its second Hasse coefficient is one.
    jets = _scalar_jets([0, 0, 1], [1], [2], 2)
    assert jets == [[1, 0, 1]]
    assert interpolate_hermite_at_zero(jets, [1], [2], 2) == 0


@pytest.mark.parametrize("prime", (*PRIMES, 2, 3, 5))
def test_single_node_weights_are_exact_taylor_translation(prime):
    node, order = prime - 1, 12
    weights = confluent_hermite_weights_at_zero([node], [order], prime)
    assert weights == [[pow(-node, derivative, prime) for derivative in range(order + 1)]]
    # Public output is mutable without corrupting the cached exact solution.
    weights[0][0] = -99
    assert confluent_hermite_weights_at_zero([node], [order], prime)[0][0] == 1


@pytest.mark.parametrize("num_links", [0, 1, 2, 7, 30])
@pytest.mark.parametrize("num_nodes", [1, 2, 4, 100])
def test_schedule_disjoint_links_balanced_capacities_and_limits(num_links, num_nodes):
    schedule = create_hermite_schedule(num_links, num_nodes)
    assert schedule.degree == num_links
    assert len(schedule.nodes) == min(num_nodes, max(num_links, 1))
    assert sum(order + 1 for order in schedule.orders) == num_links + 1
    blocks = [schedule.suppressed_links(index) for index in range(len(schedule.nodes))]
    assert sorted(link for block in blocks for link in block) == list(range(num_links))
    assert max(map(len, blocks)) - min(map(len, blocks)) <= 1
    if num_links:
        assert all(blocks)
    assert max(schedule.orders) - min(schedule.orders) <= 1
    assert any(order >= len(block) for order, block in zip(schedule.orders, blocks))


@pytest.mark.parametrize("num_links", [0, 1, 6])
def test_single_node_schedule_requires_all_derivative_subsets(num_links):
    schedule = create_hermite_schedule(num_links, mode="single_node_taylor")
    assert schedule == HermiteSchedule((1,), (num_links,), (0,) * num_links)
    assert sum(comb(num_links, derivative) for derivative in range(schedule.orders[0] + 1)) == 2 ** num_links


def test_explicit_nodes_orders_and_boundary_free_schedule():
    schedule = create_hermite_schedule(5, nodes=[-2, 7, 11], orders=[0, 3, 0])
    assert schedule.nodes == (-2, 7, 11)
    assert schedule.orders == (0, 3, 0)
    assert schedule.link_nodes == (0, 0, 1, 1, 2)
    assert create_hermite_schedule(0) == HermiteSchedule((1,), (0,), ())


@pytest.mark.parametrize("nodes,orders,prime", [
    ([], [], PRIMES[0]), ([1], [], PRIMES[0]), ([1, 2], [0], PRIMES[0]),
    ([0], [0], PRIMES[0]), ([1, 1], [0, 0], PRIMES[0]),
    ([1, 6], [0, 0], 5), ([5], [0], 5), ([1.0], [0], 5),
    ([True], [0], 5), ([1], [-1], 5), ([1], [1.5], 5),
    ([1], [True], 5), ([1], [0], 9),
])
def test_invalid_confluent_conditions(nodes, orders, prime):
    with pytest.raises(ValueError):
        confluent_hermite_weights_at_zero(nodes, orders, prime)


@pytest.mark.parametrize("jets", [[], [[1]], [[1, 2, 3]], [[1.0, 2]], [[True, 2]]])
def test_invalid_jet_shapes_and_noninteger_values(jets):
    with pytest.raises(ValueError):
        interpolate_hermite_at_zero(jets, [1], [1], PRIMES[0])


@pytest.mark.parametrize("options", [
    {"num_links": -1}, {"num_links": True}, {"num_links": 1, "num_nodes": 0},
    {"num_links": 1, "num_nodes": True}, {"num_links": 1, "mode": "unknown"},
    {"num_links": 1, "nodes": []}, {"num_links": 1, "nodes": [1, 2, 3]},
    {"num_links": 1, "nodes": [1, 2]},
    {"num_links": 1, "nodes": [0]}, {"num_links": 1, "nodes": [1, 1]},
    {"num_links": 1, "orders": [0, 1]},
    {"num_links": 1, "mode": "single_node_taylor", "nodes": [2]},
    {"num_links": 1, "mode": "single_node_taylor", "nodes": [True]},
    {"num_links": 1, "mode": "single_node_taylor", "orders": [0]},
    {"num_links": 1, "mode": "single_node_taylor", "orders": [True]},
])
def test_invalid_schedule_options(options):
    with pytest.raises(ValueError):
        create_hermite_schedule(**options)


@pytest.mark.parametrize("schedule_args", [
    ([1], (0,), ()), ((1,), [0], ()), ((1,), (0,), []),
    ((1,), (0,), (1,)), ((1,), (1,), (True,)),
    ((1,), (1,), (-1,)), ((1,), (1,), (1,)),
    ((1, 2), (1, 0), (0, 0)),
])
def test_invalid_schedule_fields(schedule_args):
    with pytest.raises(ValueError):
        HermiteSchedule(*schedule_args)


def test_suppressed_links_rejects_invalid_node_index():
    schedule = create_hermite_schedule(3)
    for index in (-1, len(schedule.nodes), True, 1.5):
        with pytest.raises(ValueError):
            schedule.suppressed_links(index)

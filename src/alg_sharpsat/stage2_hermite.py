"""Exact confluent Hermite reconstruction using finite-field Hasse jets.

An order q supplies coefficients [Z**j] f(t+Z), for 0 <= j <= q.
No factorial division is used, so characteristic may be at most the degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Sequence

from .stage2_algebraic import validate_prime


def _nonnegative_integer(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validated_conditions(nodes, orders, prime):
    validate_prime(prime)
    if not nodes or len(nodes) != len(orders):
        raise ValueError("nodes and orders must have the same nonzero length")
    if any(type(node) is not int for node in nodes):
        raise ValueError("nodes must be integers")
    residues = tuple(node % prime for node in nodes)
    if 0 in residues or len(set(residues)) != len(residues):
        raise ValueError("nodes must be distinct and nonzero modulo prime")
    for order in orders:
        _nonnegative_integer(order, "order")
    return residues, tuple(orders)


@lru_cache(maxsize=128)
def _weights(nodes: tuple[int, ...], orders: tuple[int, ...], prime: int):
    size = sum(order + 1 for order in orders)
    conditions = [(node, derivative) for node, order in zip(nodes, orders)
                  for derivative in range(order + 1)]
    # Matrix is V.T: columns correspond to jet entries (s,j), and rows to
    # monomials T**m. Solve V.T*w=e_0 entirely in the finite field.
    matrix = [
        [(comb(monomial, derivative) * pow(node, monomial - derivative, prime)) % prime
         if monomial >= derivative else 0
         for node, derivative in conditions] + [int(monomial == 0)]
        for monomial in range(size)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if matrix[row][column]), None)
        if pivot is None:
            # Distinct nodes imply nonsingularity for Hasse jets, in every
            # characteristic. This branch protects against malformed changes.
            raise ArithmeticError("singular confluent Hermite system")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        inverse = pow(matrix[column][column], -1, prime)
        matrix[column] = [(entry * inverse) % prime for entry in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            multiple = matrix[row][column]
            if multiple:
                matrix[row] = [(entry - multiple * pivot_entry) % prime
                               for entry, pivot_entry in zip(matrix[row], matrix[column])]
    flat = [row[-1] for row in matrix]
    result, offset = [], 0
    for order in orders:
        result.append(tuple(flat[offset:offset + order + 1]))
        offset += order + 1
    return tuple(result)


def confluent_hermite_weights_at_zero(
    nodes: list[int], orders: list[int], prime: int
) -> list[list[int]]:
    """Return weights for f(0), for degree < sum(q_s+1).

    The entry (s,j) multiplies the jth Hasse coefficient at nodes[s].
    Nodes must be distinct and nonzero as finite-field elements. There is no
    restriction relating the field characteristic to the polynomial degree.
    """
    residues, validated_orders = _validated_conditions(nodes, orders, prime)
    return [list(row) for row in _weights(residues, validated_orders, prime)]


def interpolate_hermite_at_zero(
    jets: Sequence[Sequence[int]], nodes: list[int], orders: list[int], prime: int
) -> int:
    """Reconstruct f(0) from Hasse jets, each of length orders[s]+1."""
    weights = confluent_hermite_weights_at_zero(nodes, orders, prime)
    if len(jets) != len(weights):
        raise ValueError("jets must contain one row per node")
    total = 0
    for jet, row in zip(jets, weights):
        if len(jet) != len(row):
            raise ValueError("each jet must have order+1 coefficients")
        if any(type(value) is not int for value in jet):
            raise ValueError("jet coefficients must be integers")
        total = (total + sum(value * weight for value, weight in zip(jet, row))) % prime
    return total


@dataclass(frozen=True)
class HermiteSchedule:
    """Every node suppresses links for B>0; orders hold maximum jet degrees."""

    nodes: tuple[int, ...]
    orders: tuple[int, ...]
    link_nodes: tuple[int, ...]

    def __post_init__(self):
        if any(not isinstance(value, tuple)
               for value in (self.nodes, self.orders, self.link_nodes)):
            raise ValueError("schedule fields must be tuples")
        if not self.nodes or len(self.nodes) != len(self.orders):
            raise ValueError("nodes and orders must have the same nonzero length")
        if any(type(node) is not int for node in self.nodes):
            raise ValueError("nodes must be integers")
        if 0 in self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be distinct and nonzero")
        for order in self.orders:
            _nonnegative_integer(order, "order")
        if sum(order + 1 for order in self.orders) != self.degree + 1:
            raise ValueError("jet capacities must sum to degree+1")
        if any(type(index) is not int or not 0 <= index < len(self.nodes)
               for index in self.link_nodes):
            raise ValueError("link_nodes must contain valid node indices")
        if self.degree and set(self.link_nodes) != set(range(len(self.nodes))):
            raise ValueError("each evaluation node must suppress at least one link")

    @property
    def degree(self) -> int:
        return len(self.link_nodes)

    def suppressed_links(self, node_index: int) -> tuple[int, ...]:
        if type(node_index) is not int or not 0 <= node_index < len(self.nodes):
            raise ValueError("node_index must identify an existing node")
        return tuple(index for index, node in enumerate(self.link_nodes) if node == node_index)


def create_hermite_schedule(
    num_links: int, num_nodes: int = 2, mode: str = "hermite",
    nodes: Sequence[int] | None = None, orders: Sequence[int] | None = None,
) -> HermiteSchedule:
    """Create balanced disjoint suppression blocks and balanced jet capacities.

    For B>0 the node count is at most B, giving every node a nonempty block.
    The boundary-free case has a single node and a zeroth-order jet.
    Since sum|K_s|=B and sum(q_s+1)=B+1, at least one q_s >= |K_s|.
    Single-node Taylor reconstruction has q=B and all 2**B derivative subsets;
    neither this construction nor balanced blocks establish a general speedup.
    """
    _nonnegative_integer(num_links, "num_links")
    if type(num_nodes) is not int or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    if mode not in ("hermite", "single_node_taylor"):
        raise ValueError("mode must be hermite or single_node_taylor")
    if mode == "single_node_taylor":
        if nodes is not None and (any(type(node) is not int for node in nodes)
                                  or tuple(nodes) != (1,)):
            raise ValueError("single_node_taylor uses the node 1")
        if orders is not None and (any(type(order) is not int for order in orders)
                                   or tuple(orders) != (num_links,)):
            raise ValueError("single_node_taylor uses order num_links")
        return HermiteSchedule((1,), (num_links,), (0,) * num_links)
    maximum_nodes = max(num_links, 1)
    selected_nodes = (tuple(range(1, min(num_nodes, maximum_nodes) + 1))
                      if nodes is None else tuple(nodes))
    if not selected_nodes or len(selected_nodes) > maximum_nodes:
        raise ValueError("node count must be between 1 and max(num_links,1)")
    count = len(selected_nodes)
    if orders is None:
        capacity, extra = divmod(num_links + 1, count)
        selected_orders = tuple(capacity - 1 + int(index < extra) for index in range(count))
    else:
        selected_orders = tuple(orders)
    block_size, extra_links = divmod(num_links, count)
    link_nodes = tuple(index for index in range(count)
                       for _ in range(block_size + int(index < extra_links)))
    return HermiteSchedule(selected_nodes, selected_orders, link_nodes)

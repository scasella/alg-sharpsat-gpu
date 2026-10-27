"""Finite-field interpolation and the algebraic boundary-link construction.

Arithmetic here uses Python integers. Tensor callers must reduce after products
*and* during sums: a product of residues fits signed int64 for our primes, but
an unrestricted sum of such products need not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .stage1_sparsify import PartitionedCNF


PRIMES = (998244353, 1004535809, 469762049)
MAX_PRIME = 1 << 30


@lru_cache(maxsize=None)
def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    for divisor in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if number % divisor == 0:
            return number == divisor
    odd_part = number - 1
    twos = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        twos += 1
    # These bases are deterministic throughout the supported (< 2**30) range.
    for base in (2, 3, 5, 7, 11):
        value = pow(base, odd_part, number)
        if value in (1, number - 1):
            continue
        for _ in range(twos - 1):
            value = value * value % number
            if value == number - 1:
                break
        else:
            return False
    return True


def validate_prime(prime: int) -> None:
    """Require a prime whose residue products fit signed int64."""
    if isinstance(prime, bool) or not isinstance(prime, int):
        raise ValueError("prime must be an integer")
    if prime >= MAX_PRIME or not _is_prime(prime):
        raise ValueError("prime must be prime and less than 2**30")


@dataclass(frozen=True)
class SuppressionSchedule:
    """Interpolation nodes and the roots of each boundary-link polynomial."""

    nodes: tuple[int, ...]
    roots: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not isinstance(self.roots, tuple):
            raise ValueError("schedule nodes and roots must be tuples")
        if any(not isinstance(roots, tuple) for roots in self.roots):
            raise ValueError("each link's roots must be a tuple")
        if len(self.nodes) != self.degree + 1:
            raise ValueError("a degree-D schedule needs exactly D+1 nodes")
        if any(isinstance(node, bool) or not isinstance(node, int) for node in self.nodes):
            raise ValueError("schedule nodes must be integers")
        if len(set(self.nodes)) != len(self.nodes) or 0 in self.nodes:
            raise ValueError("schedule nodes must be distinct and nonzero")
        if any(isinstance(root, bool) or not isinstance(root, int)
               for roots in self.roots for root in roots):
            raise ValueError("suppression roots must be integers")
        node_set = set(self.nodes)
        if any(root not in node_set for roots in self.roots for root in roots):
            raise ValueError("all suppression roots must be schedule nodes")

    @property
    def degree(self) -> int:
        return sum(len(roots) for roots in self.roots)


def create_suppression_schedule(
    num_links: int, degree_per_link: int = 1
) -> SuppressionSchedule:
    """Balance root incidences cyclically over D+1 interpolation nodes.

    There are only D root incidences. Thus the average number of suppressed
    links at a node is below one; this schedule supplies no claimed speedup.
    """
    for name, value in (("num_links", num_links), ("degree_per_link", degree_per_link)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    degree = num_links * degree_per_link
    nodes = tuple(range(1, degree + 2))
    roots = tuple(
        tuple(nodes[(link * degree_per_link + offset) % len(nodes)]
              for offset in range(degree_per_link))
        for link in range(num_links)
    )
    return SuppressionSchedule(nodes, roots)


def suppression_value(roots: Sequence[int], t: int, prime: int) -> int:
    """Evaluate p(T) = product(1 - T/root), with p(0) = 1."""
    validate_prime(prime)
    value = 1
    for root in roots:
        if root % prime == 0:
            raise ValueError("suppression roots must be nonzero modulo prime")
        value = value * (1 - (t % prime) * pow(root % prime, -1, prime)) % prime
    return value


def linear_form(
    x: int, measurement: int, block_size: int, suppression: int, prime: int
) -> int:
    """Return m/k + p(T)*(x-m/k) in the finite field."""
    validate_prime(prime)
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("block size must be an integer")
    if block_size <= 0 or block_size % prime == 0:
        raise ValueError("block size must be positive and invertible modulo prime")
    mean = (measurement % prime) * pow(block_size % prime, -1, prime) % prime
    return (mean + (suppression % prime) * (x - mean)) % prime


def link_factor(x_tilde: int, y: int, prime: int) -> int:
    """Return Psi(x_tilde,y) = (1-y) + (2*y-1)*x_tilde."""
    validate_prime(prime)
    return ((1 - y) + (2 * y - 1) * x_tilde) % prime


def lagrange_weights_at_zero(nodes: Sequence[int], prime: int) -> tuple[int, ...]:
    """Weights reconstructing f(0) from f at the supplied distinct nodes."""
    validate_prime(prime)
    if not nodes:
        raise ValueError("at least one interpolation node is required")
    if prime <= len(nodes):
        raise ValueError("prime must exceed D+1 interpolation nodes")
    residues = tuple(node % prime for node in nodes)
    if len(set(residues)) != len(residues):
        raise ValueError("interpolation nodes must be distinct modulo prime")
    weights = []
    for index, node in enumerate(residues):
        numerator = denominator = 1
        for other_index, other in enumerate(residues):
            if other_index != index:
                numerator = numerator * (-other) % prime
                denominator = denominator * (node - other) % prime
        weights.append(numerator * pow(denominator, -1, prime) % prime)
    return tuple(weights)


def interpolate_at_zero(values: Sequence[int], nodes: Sequence[int], prime: int) -> int:
    if len(values) != len(nodes):
        raise ValueError("values and nodes must have the same length")
    weights = lagrange_weights_at_zero(nodes, prime)
    return sum((value % prime) * weight for value, weight in zip(values, weights)) % prime


def crt_reconstruct(residues: Sequence[int], primes: Sequence[int]) -> int:
    """Return the unique nonnegative reconstruction modulo product(primes)."""
    if not residues or len(residues) != len(primes):
        raise ValueError("residues and primes must have the same nonzero length")
    for prime in primes:
        validate_prime(prime)
    if len(set(primes)) != len(primes):
        raise ValueError("CRT primes must be distinct")
    result, modulus = 0, 1
    for residue, prime in zip(residues, primes):
        correction = (residue - result) * pow(modulus % prime, -1, prime) % prime
        result += modulus * correction
        modulus *= prime
    return result


def primes_for_bound(bound: int) -> tuple[int, ...]:
    """Start with PRIMES and extend until their product strictly exceeds bound."""
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise ValueError("bound must be a nonnegative integer")
    primes = list(PRIMES)
    modulus = prod(primes)
    candidate = MAX_PRIME - 1
    while modulus <= bound:
        while candidate in primes or not _is_prime(candidate):
            candidate -= 2
            if candidate < 3:
                raise ValueError("exhausted supported CRT primes")
        primes.append(candidate)
        modulus *= candidate
        candidate -= 2
    return tuple(primes)


def direct_h(
    partitioned: PartitionedCNF, t: int, prime: int, schedule: SuppressionSchedule
) -> int:
    """Exhaustive scalar H(T) oracle for tiny partitioned formulas.

    Every group's clauses are checked independently. The sum ranges over all
    original and proxy bits, multiplied by each algebraic boundary-link factor.
    Measurement sums count each link's master bit, including repeated masters.
    """
    validate_prime(prime)
    if len(schedule.roots) != len(partitioned.boundary_links):
        raise ValueError("schedule must contain one root list per boundary link")
    if prime <= schedule.degree + 1:
        raise ValueError("prime must exceed D+1")
    lagrange_weights_at_zero(schedule.nodes, prime)
    if any(node % prime == 0 for node in schedule.nodes):
        raise ValueError("schedule nodes must be nonzero modulo prime")
    block_for_link = {}
    for block in partitioned.measurement_blocks:
        size = len(block.link_indices)
        if size == 0 or size % prime == 0:
            raise ValueError("measurement block size must be positive and invertible")
        for index in block.link_indices:
            if index < 0 or index >= len(schedule.roots) or index in block_for_link:
                raise ValueError("measurement blocks must partition boundary links")
            block_for_link[index] = block
    if len(block_for_link) != len(schedule.roots):
        raise ValueError("measurement blocks must partition boundary links")
    suppressions = [suppression_value(roots, t, prime) for roots in schedule.roots]
    total = 0
    clauses = [clause for group in partitioned.group_clauses for clause in group]
    for bits in product((0, 1), repeat=partitioned.num_vars):
        values = bits
        if any(not any(bool(values[literal.var]) == literal.pos for literal in clause.literals)
               for clause in clauses):
            continue
        measurements = {
            index: sum(values[partitioned.boundary_links[other].master_var]
                       for other in block.link_indices)
            for index, block in block_for_link.items()
        }
        term = 1
        for index, link in enumerate(partitioned.boundary_links):
            block = block_for_link[index]
            x_tilde = linear_form(values[link.master_var], measurements[index],
                                  len(block.link_indices), suppressions[index], prime)
            term = term * link_factor(x_tilde, values[link.proxy_var], prime) % prime
        total = (total + term) % prime
    return total

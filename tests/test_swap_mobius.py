"""Independent exact checks of the supplied SWAP and diagonal-sum identities."""
from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError
from alg_sharpsat.stage4_swap_mobius import (
    SwapMobiusCounter, build_swap_matrix, typical_weight_polynomial,
    replacement_coefficients, eval_replacement_form, error_from_measurements,
    cnf_indicator_polynomial, mobius_diagonal_jet, swap_mobius_count,
    swap_crossover_estimate,
)

pytestmark = pytest.mark.g11
CONFIGS = [(L, b) for L in (2, 3, 4) for b in range(1, L)]
P = 1009


def field(value, p=P):
    value = Fraction(value)
    return value.numerator * pow(value.denominator, -1, p) % p


def modular_rank(matrix, p=P):
    a = np.array([[field(v, p) for v in row] for row in matrix], dtype=np.int64)
    rank = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank] = a[rank] * pow(int(a[rank, col]), -1, p) % p
        a[rank+1:] = (a[rank+1:] - a[rank+1:, col, None]*a[rank]) % p
        rank += 1
        if rank == len(a):
            break
    return rank


@pytest.mark.parametrize('L,b', CONFIGS)
@pytest.mark.parametrize('probability', [0, .25, .5, 1])
def test_typical_weight_exact_rational_parity(L, b, probability):
    f = typical_weight_polynomial(L, b, probability)
    assert len(f.coefficients) == b + 1
    assert len(f.window) == b + 1
    assert all(f(k) == (-1)**k for k in f.window)
    assert 0 <= min(f.window) <= max(f.window) <= L
    midpoint = Fraction(f.window[0] + f.window[-1], 2)
    center = L * Fraction(str(probability))
    assert abs(midpoint-center) == min(abs(Fraction(2*a+b, 2)-center) for a in range(L-b+1))


@pytest.mark.parametrize('L,b', CONFIGS)
def test_swap_matrix_complete_spectral_basis_rank_support_and_measurements(L, b):
    s = build_swap_matrix(2, L, b)
    vectors = np.asarray(s.eigenvectors, dtype=np.int64)
    assert set(vectors.flat) <= {-1, 0, 1}
    assert np.array_equal(vectors @ vectors.T, np.diag(s.norms))
    # Rational eigenvectors and eigenvalues independently check the constructed
    # group-algebra matrix, and the complete orthogonal basis proves rank exactly.
    for i, row in enumerate(s.M):
        nonzero = [(j, x) for j, x in enumerate(row) if x]
        for j, weight in nonzero:
            assert len(s.mismatch(i, j)) <= b
        for v, eigenvalue in zip(s.eigenvectors, s.eigenvalues):
            assert sum(weight*v[j] for j, weight in nonzero) == eigenvalue*v[i]
    errors = [[int(i == j)-v for j, v in enumerate(row)] for i, row in enumerate(s.M)]
    assert modular_rank(errors) == s.s0 == sum(v != 1 for v in s.eigenvalues)
    assert s.C == tuple(s.eigenvectors[i] for i in s.error_indices)
    z = [(i*7 % 11)-5 for i in range(s.N)]
    measured = [sum(a*x for a, x in zip(row, z)) for row in s.C]
    reconstructed = error_from_measurements(s, measured)
    assert reconstructed == [sum(a*x for a, x in zip(row, z)) for row in errors]
    good = next(v for v, eigenvalue in zip(s.eigenvectors, s.eigenvalues) if eigenvalue == 1)
    z_prime = [a+c for a, c in zip(z, good)]
    assert measured == [sum(a*x for a, x in zip(row, z_prime)) for row in s.C]
    assert reconstructed == [sum(a*x for a, x in zip(row, z_prime)) for row in errors]


def test_matrix_equals_direct_commuting_swap_polynomial():
    s = build_swap_matrix(2, 2, 1)
    indices = {state: i for i, state in enumerate(s.states)}
    identity = np.eye(s.N, dtype=object)
    swaps = []
    for t in range(s.L):
        matrix = np.zeros((s.N, s.N), dtype=object)
        for i, (alpha, beta) in enumerate(s.states):
            a, b = list(alpha), list(beta)
            a[t], b[t] = b[t], a[t]
            matrix[indices[tuple(a), tuple(b)], i] = 1
        assert np.array_equal(matrix @ matrix, identity)
        swaps.append(matrix)
    assert np.array_equal(swaps[0] @ swaps[1], swaps[1] @ swaps[0])
    W = sum((identity-matrix)*Fraction(1, 2) for matrix in swaps)
    power, fW = identity, np.zeros_like(identity)
    for coefficient in s.polynomial.coefficients:
        fW += coefficient*power
        power = power @ W
    assert np.array_equal(swaps[0] @ swaps[1] @ fW, np.asarray(s.M, dtype=object))


@pytest.mark.parametrize('L,b', CONFIGS)
def test_replacement_identity_degree_and_zero_residual_at_every_node(L, b):
    s = build_swap_matrix(2, L, b)
    for omega in range(s.N):
        coefficients = replacement_coefficients(s, omega)
        assert len(coefficients) == b+1
        assert coefficients[0] == tuple(Fraction(i == omega) for i in range(s.N))
        for node in range(1, L+1):
            allowed = set(s.group(node, s.states[omega][1][node-1]))
            for eta in range(s.N):
                K = sum(row[eta]*node**j for j, row in enumerate(coefficients))
                E = int(omega == eta)-s.M[omega][eta]
                if eta not in allowed:
                    assert K-E == 0
                phi = np.prod([1-Fraction(node, t) for t in s.mismatch(omega, eta)])
                assert K == E + s.M[omega][eta]*phi


@settings(max_examples=30, deadline=None, derandomize=True)
@given(z=st.lists(st.integers(-5, 5), min_size=16, max_size=16), omega=st.integers(0, 15))
def test_modular_and_rational_replacement_agree(z, omega):
    s = build_swap_matrix(2, 2, 1)
    assert eval_replacement_form(s, omega, 0, z) == z[omega]
    for t in (1, 2, 3, 8):
        assert eval_replacement_form(s, omega, t, z, P) == field(eval_replacement_form(s, omega, t, z))


def multiply(a, b, q):
    result = [0]*(q+1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i+j <= q:
                result[i+j] = (result[i+j]+x*y) % P
    return result


def direct_jet_oracle(counter, node, q):
    """Explicit global assignments, independently substituting K(t+tau,x)."""
    result = [0]*(q+1)
    s = counter.system
    coefficients = [replacement_coefficients(s, omega) for omega in counter.embedding]
    for bits in product((0, 1), repeat=counter.n):
        replacement_jets = []
        for rows in coefficients:
            polynomial = [sum(row[eta]*x for eta, x in zip(counter.embedding, bits)) for row in rows]
            replacement_jets.append([field(sum(comb(m, j)*node**(m-j)*polynomial[m]
                                                  for m in range(j, len(polynomial)))) for j in range(q+1)])
        for monomial, weight in counter.terms.items():
            term = [1]+[0]*q
            for i in range(counter.n):
                if monomial >> i & 1:
                    term = multiply(term, replacement_jets[i], q)
            result = [(a+weight*x) % P for a, x in zip(result, term)]
    return result


@pytest.mark.parametrize('L,b', CONFIGS)
def test_group_separated_mobius_jets_match_global_polynomial_oracle(L, b):
    f = CNF(3, [Clause((Literal(0, True), Literal(1, True))),
                Clause((Literal(1, False), Literal(2, True)))])
    # Mixed alpha/beta coordinates exercise nonzero C measurements and
    # node-dependent partitions, rather than an all-diagonal embedding.
    width = 1 << L
    embedding = [width-1, 2*width-2, (width//2)*width+width//2-1]
    counter = SwapMobiusCounter(f, L=L, b=b, embedding=embedding)
    for node in range(1, L+1):
        q = min(3, counter.degree)
        assert counter.evaluate_jet(node, q, P) == direct_jet_oracle(counter, node, q)
    assert counter.count() == brute_force_count(f)
    assert all(row['status'] == 'ok' for row in counter.stats.node_evaluations)
    assert any(row['measurement_tuples'] > row['inner_evaluations'] for row in counter.stats.node_evaluations)


@pytest.mark.parametrize('q', [0, 1, 2, 3])
def test_boolean_discrete_taylor_diagonal_identity(q):
    n = 4
    def evaluator(x, y):
        # Each occurrence of y carries at least one power of tau. Constants
        # can depend arbitrarily on x, the condition needed by Lemma 7.1.
        value = [1]+[0]*q
        for i in range(n):
            factor = [((x >> i & 1)+2*i+1) % P]
            if q:
                factor.append(sum((j+i+1)*(y >> j & 1) for j in range(n)) % P)
            value = multiply(value, factor, q)
        return value
    expected = [sum(evaluator(x, x)[j] for x in range(1 << n)) % P for j in range(q+1)]
    assert mobius_diagonal_jet(n, q, P, evaluator) == expected


@st.composite
def tiny_cnfs(draw):
    n = draw(st.integers(0, 4))
    if n == 0:
        return CNF(0, [Clause(())]*draw(st.integers(0, 2)))
    clauses = draw(st.lists(st.lists(st.tuples(st.integers(0, n-1), st.booleans()), min_size=0, max_size=3), max_size=6))
    return CNF(n, [Clause(tuple(Literal(v, sign) for v, sign in clause)) for clause in clauses])


@settings(max_examples=100, deadline=None, derandomize=True)
@given(cnf=tiny_cnfs())
def test_swap_count_differential_and_indicator(cnf):
    terms = cnf_indicator_polynomial(cnf)
    expected = 0
    for bits in range(1 << cnf.num_vars):
        indicator = sum(weight for mask, weight in terms.items() if bits & mask == mask)
        assert indicator in (0, 1)
        expected += indicator
    assert expected == brute_force_count(cnf) == swap_mobius_count(cnf)


def test_counter_uses_group_evaluation_without_reference_fallback(monkeypatch):
    import alg_sharpsat.cnf as reference
    import alg_sharpsat.stage4_swap_mobius as module
    def forbidden(*args, **kwargs):
        pytest.fail('global reference evaluator was called')
    monkeypatch.setattr(reference, 'brute_force_count', forbidden)
    monkeypatch.setattr(reference, 'dpll_count', forbidden)
    monkeypatch.setattr(module, 'mobius_diagonal_jet', forbidden)
    counter = SwapMobiusCounter(CNF(4, [Clause((Literal(0, True), Literal(3, False)))]), embedding=[3, 6, 9, 12])
    assert counter.count() == 12
    assert all(trace['local_group_sizes'] == [2, 2] for trace in counter.stats.node_evaluations)
    assert all(trace['local_assignment_rows'] == [4, 4] for trace in counter.stats.node_evaluations)


def test_resource_guards_and_estimate_are_explicit():
    with pytest.raises(ResourceLimitError):
        swap_mobius_count(CNF(7, []))
    with pytest.raises(ResourceLimitError):
        build_swap_matrix(2, 5, 1)
    with pytest.raises(ResourceLimitError):
        build_swap_matrix(2, 10**9, 10**8)
    with pytest.raises(ResourceLimitError):
        SwapMobiusCounter(CNF(2, [])).evaluate_jet(1, 10**9, P)
    with pytest.raises(ResourceLimitError):
        SwapMobiusCounter(CNF(16, []), max_variables=16, max_work=200).evaluate_jet(1, 0, P)
    with pytest.raises(ResourceLimitError):
        swap_mobius_count(CNF(2, [Clause((Literal(0, True), Literal(1, True)))]), max_work=1)
    with pytest.raises(ValueError):
        SwapMobiusCounter(CNF(2, []), embedding=[3, 3])
    estimate = swap_crossover_estimate(30, L=3)
    assert not estimate['crossover_established']
    assert not estimate['executable_under_default_limit']
    assert estimate['embedding_fits']

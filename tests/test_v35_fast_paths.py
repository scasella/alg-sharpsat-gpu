"""v3.5 fast paths: plan selection, linear-domain float64, float-guided MAP, bridge cuts."""

from fractions import Fraction
from itertools import product
import random

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from alg_sharpsat.benchmarks import bipartite_3sat, coupled_3sat
from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count, dpll_count
from alg_sharpsat.domains import get_domains
from alg_sharpsat.stage1_sparsify import partition_and_decouple_boundary
from alg_sharpsat.stage3_bridge_cut import bridge_cut_count, find_bridge_cut
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError
from alg_sharpsat.stage3_sparse_contract import auto_count_with_stats, sparse_count_with_stats
import alg_sharpsat.stage5_wmc_guardrail as guardrail
from alg_sharpsat.stage5_wmc_guardrail import compile_guardrail, plan_cost


def formula(n, clauses):
    return CNF(n, [Clause(tuple(Literal(abs(x) - 1, x > 0) for x in c)) for c in clauses])


@st.composite
def weighted_cases(draw):
    family = draw(st.sampled_from(("random", "coupled", "bipartite")))
    n = draw(st.integers(6, 14))
    seed = draw(st.integers(0, 2**16))
    if family == "random":
        rng = random.Random(seed)
        cnf = formula(n, [[(v + 1) * (1 if rng.getrandbits(1) else -1)
                           for v in rng.sample(range(n), rng.randint(1, 5))]
                          for _ in range(rng.randint(2, 8))])
    else:
        cnf = (coupled_3sat if family == "coupled" else bipartite_3sat)(n, seed=seed)
    batch = draw(st.integers(1, 5))
    # Include exact zeros and ones, which give -inf log weights.
    probs = np.array(draw(st.lists(st.lists(st.integers(0, 8), min_size=n, max_size=n),
                                   min_size=batch, max_size=batch))) / 8
    groups = draw(st.sampled_from((1, 2, 3, 4, "auto")))
    return cnf, probs, groups


def log_domain(compiled, probs):
    logs, batched = compiled._prepare_logs(probs)
    return compiled._evaluate_floating(logs, batched, torch.device("cpu"), torch.float64)


def assert_same(fast, slow):
    for name in ("partition_function", "posterior_marginals", "log_partition",
                 "acceptance_prob", "kl_divergence_nats"):
        a, b = getattr(fast, name).numpy(), getattr(slow, name).numpy()
        assert a.shape == b.shape
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        assert np.array_equal(np.isinf(a), np.isinf(b)), name
        finite = np.isfinite(a)
        assert np.allclose(a[finite], b[finite], rtol=1e-12, atol=1e-12), name


@pytest.mark.g19
@settings(max_examples=80, deadline=None)
@given(weighted_cases())
def test_linear_domain_float64_matches_log_domain_path(case):
    cnf, probs, groups = case
    compiled = compile_guardrail(cnf, num_groups=groups)
    assert_same(compiled.evaluate(probs, mode="float64"), log_domain(compiled, probs))
    single = compiled.evaluate(probs[0], mode="float64")
    assert_same(single, log_domain(compiled, probs[0]))
    exact = compiled.evaluate([Fraction(int(8 * p), 8) for p in probs[0]], mode="exact_rational")
    assert float(single.partition_function) == pytest.approx(float(exact.partition_function),
                                                             rel=1e-12, abs=0)
    if exact.partition_function:
        expected = np.array([float(x) for x in exact.posterior_marginals])
        assert np.allclose(single.posterior_marginals.numpy(), expected, rtol=0, atol=1e-12)


@pytest.mark.g19
def test_queries_beyond_the_safe_range_use_the_log_domain_path():
    cnf = formula(12, [[1, 2]] + [[2, v + 1] for v in range(2, 12)])
    compiled = compile_guardrail(cnf, num_groups=2)
    batch = np.array([[1e-320] + [0.5] * 11, [0.25] + [0.5] * 11])
    _, _, safe = compiled._fast().evaluate(np.log(np.stack([1 - batch, batch], -1)))
    assert safe.tolist() == [False, True]
    assert_same(compiled.evaluate(batch, mode="float64"), log_domain(compiled, batch))
    assert compiled.evaluate(batch, mode="float64").posterior_marginals[0, 0] > 0


@pytest.mark.g19
@pytest.mark.parametrize("domain", sorted(get_domains()))
def test_auto_plan_selection_records_candidates_and_lowers_cost(domain):
    cnf = get_domains()[domain].policy.to_cnf()
    auto = compile_guardrail(cnf)
    two = compile_guardrail(cnf, num_groups=2)
    selection = auto.stats.plan_selection
    assert selection["mode"] == "auto"
    assert selection["candidates"][0]["num_groups"] == 2
    assert selection["cost"] == plan_cost(auto) <= plan_cost(two)
    again = compile_guardrail(cnf).stats.plan_selection
    assert [c.get("cost") for c in again["candidates"]] == [c.get("cost") for c in selection["candidates"]]
    assert again["selected_num_groups"] == selection["selected_num_groups"]
    assert two.stats.plan_selection == {"mode": "explicit", "selected_num_groups": 2,
                                        "cost": plan_cost(two)}
    probs = [Fraction(1, 3)] * cnf.num_vars
    a = auto.evaluate(probs, mode="exact_rational")
    b = two.evaluate(probs, mode="exact_rational")
    assert a.partition_function == b.partition_function
    assert a.posterior_marginals == b.posterior_marginals


@pytest.mark.g19
def test_explored_plan_is_the_cheapest_completed_candidate(monkeypatch):
    monkeypatch.setattr(guardrail, "AUTO_EXPLORE_COST", 0)
    monkeypatch.setattr(guardrail, "AUTO_GROUP_CANDIDATES", (6, 4, 8, 3, 5))
    cnf = coupled_3sat(14, seed=3)
    compiled = compile_guardrail(cnf)
    selection = compiled.stats.plan_selection
    costs = [c["cost"] for c in selection["candidates"] if c["status"] == "ok"]
    assert len(costs) > 1 and selection["cost"] == min(costs)
    assert compiled.evaluate([0.5] * 14, mode="exact_rational").partition_function == Fraction(
        brute_force_count(cnf), 2**14)


@pytest.mark.g19
def test_direct_model_table_is_a_candidate_for_small_model_sets(monkeypatch):
    monkeypatch.setattr(guardrail, "AUTO_EXPLORE_COST", 0)
    # An implication cycle has two models, but each half has many local rows.
    cnf = formula(10, [[-(i + 1), (i + 1) % 10 + 1] for i in range(10)])
    compiled = compile_guardrail(cnf)
    candidates = compiled.stats.plan_selection["candidates"]
    assert [c["status"] for c in candidates if c["num_groups"] == 1] == ["ok"]
    assert compiled.stats.plan_selection["selected_num_groups"] == 1
    assert len(compiled.leaves) == 1 and len(compiled.leaves[0].key_idx) == 2
    assert compiled.evaluate([0.5] * 10, mode="exact_rational").partition_function == Fraction(2, 1024)


@pytest.mark.g19
@settings(max_examples=60, deadline=None)
@given(weighted_cases())
def test_float_guided_map_equals_the_exact_integer_pass(case):
    import alg_sharpsat._guardrail_inference as inference

    cnf, probs, groups = case
    compiled = compile_guardrail(cnf, num_groups=groups)
    rational = [Fraction(int(8 * p), 8) for p in probs[0]]
    try:
        fast = compiled.map_inference(rational)
    except ValueError:
        fast = "zero"
    original = inference._float_guided_map
    inference._float_guided_map = lambda *args: None
    try:
        reference = compiled.map_inference(rational)
    except ValueError:
        reference = "zero"
    finally:
        inference._float_guided_map = original
    assert fast == reference


@pytest.mark.g19
def test_exact_pair_fast_path_matches_the_general_parser():
    compiled = compile_guardrail(formula(3, [[1, 2], [-2, 3]]))
    weights = [(Fraction(1, 3), 2), (0, Fraction(5, 7)), (Fraction(9, 4), Fraction(3, 8))]
    fast = compiled._exact_pairs(None, weights, "probabilities")
    assert fast is not None
    as_floats = [[str(a), str(b)] for a, b in weights]  # forces the general parser
    assert compiled._exact_pairs(None, as_floats, "probabilities") is None
    assert fast == compiled._prepare_exact(weights=as_floats)
    probs = [Fraction(1, 3), 0, 1]
    assert compiled._prepare_exact(probs) == compiled._prepare_exact([str(p) for p in probs])
    with pytest.raises(ValueError, match="nonnegative"):
        compiled._prepare_exact(weights=[(-1, 1), (1, 1), (1, 1)])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        compiled._prepare_exact([Fraction(3, 2), 0, 0])


@st.composite
def cut_formulas(draw):
    left, right = draw(st.integers(1, 7)), draw(st.integers(1, 7))
    n = left + right
    sides = (list(range(left)), list(range(left, n)))

    def literals(variables, size):
        chosen = draw(st.lists(st.sampled_from(variables), min_size=1, max_size=size))
        return [(v + 1) * draw(st.sampled_from((1, -1))) for v in chosen]

    clauses = [literals(sides[draw(st.integers(0, 1))], 3) for _ in range(draw(st.integers(0, 10)))]
    bridges = [literals(sides[0], 2) + literals(sides[1], 2) for _ in range(draw(st.integers(0, 6)))]
    if draw(st.booleans()) and clauses:
        clauses.append([])
    return formula(n, clauses + bridges), (tuple(sides[0]), tuple(sides[1]))


@pytest.mark.g20
@settings(max_examples=150, deadline=None)
@given(cut_formulas())
def test_bridge_cut_inclusion_exclusion_matches_brute_force(case):
    cnf, groups = case
    cut = find_bridge_cut(cnf, 8, groups)
    assert cut is not None
    count, telemetry = bridge_cut_count(cut)
    assert count == brute_force_count(cnf)
    assert telemetry["inclusion_exclusion_terms"] == 2 ** len(cut.bridges)


@pytest.mark.g20
@pytest.mark.parametrize("n", [6, 10, 18, 30, 40])
def test_auto_selects_bridge_cut_on_coupled_and_contraction_elsewhere(n):
    coupled = coupled_3sat(n, seed=n)
    count, stats = auto_count_with_stats(coupled, device="cpu")
    assert count == dpll_count(coupled)
    assert stats.selected_engine == "bridge_cut" and "bridge clauses" in stats.selection_reason
    assert stats.bridge_cut["bridge_clauses"] <= 6
    assert sparse_count_with_stats(coupled, device="cpu")[1].selected_engine == "sparse_contract"
    bipartite = bipartite_3sat(n, seed=n)
    count, stats = auto_count_with_stats(bipartite, device="cpu")
    assert count == dpll_count(bipartite) and stats.selected_engine == "sparse_contract"
    with pytest.raises(ValueError, match="bridge"):
        sparse_count_with_stats(bipartite, device="cpu", bridge_cut=True)


@pytest.mark.g20
def test_bridge_cut_budgets_fail_explicitly():
    cnf = coupled_3sat(40, seed=1)
    with pytest.raises(ResourceLimitError, match="max_work"):
        auto_count_with_stats(cnf, device="cpu", max_work=50)
    with pytest.raises(ResourceLimitError, match="max_memory_bytes"):
        auto_count_with_stats(cnf, device="cpu", max_memory_bytes=1024)
    with pytest.raises(TypeError, match="unexpected"):
        auto_count_with_stats(cnf, device="cpu", max_work_typo=1)
    with pytest.raises(ValueError, match="bridge_cut"):
        sparse_count_with_stats(cnf, device="cpu", bridge_cut="yes")


@pytest.mark.g20
@pytest.mark.parametrize("cnf,expected", [
    (formula(4, []), 16),
    (formula(4, [[]]), 0),
    (formula(4, [[1, -1], [2, 3]]), 12),  # tautology is dropped
    (formula(4, [[1, 2], [3, 4]]), 9),  # disconnected sides, no bridge
])
def test_bridge_cut_edge_cases(cnf, expected):
    count, stats = auto_count_with_stats(cnf, device="cpu")
    assert count == expected == brute_force_count(cnf)
    assert stats.selected_engine == "bridge_cut"


@pytest.mark.g10
def test_partition_reuses_and_validates_supplied_groups():
    cnf = coupled_3sat(10, seed=2)
    fresh = partition_and_decouple_boundary(cnf, 2)
    original = [tuple(v for v in g if v < cnf.num_vars) for g in fresh.group_vars]
    reused = partition_and_decouple_boundary(cnf, 2, groups=original)
    assert reused == fresh
    with pytest.raises(ValueError, match="partition"):
        partition_and_decouple_boundary(cnf, 2, groups=[tuple(range(5)), tuple(range(4, 10))])


@pytest.mark.g20
def test_bridge_cut_counts_match_every_assignment_split():
    # Independent check of the identity on one explicit cut.
    cnf = formula(6, [[1, 2], [-2, 3], [4, -5], [5, 6], [-3, 4], [1, -6], [-1, 5]])
    groups = ((0, 1, 2), (3, 4, 5))
    count, _ = bridge_cut_count(find_bridge_cut(cnf, 6, groups))
    models = sum(all(any(bits[l.var] == l.pos for l in c.literals) for c in cnf.clauses)
                 for bits in product((False, True), repeat=6))
    assert count == models
